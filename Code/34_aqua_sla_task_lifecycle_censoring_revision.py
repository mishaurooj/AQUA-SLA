from __future__ import annotations

r"""
AQUA-SLA reviewer concern #1 closure:
task-lifecycle-aware censoring + corrected future-horizon target + predictor rerun.

This is a SEPARATE experiment. It does not overwrite prior V2/V3/V4 results.

What this script fixes
----------------------
The reviewer asked for:
  * a decision time Ti,
  * a future prediction horizon H,
  * task lifecycle handling,
  * censoring rules,
  * leakage-free decision-time features,
  * retraining after target correction.

The earlier V2 code already created a future-horizon target, but its censoring
used the GLOBAL trace end only. This script adds task-level lifecycle censoring.

For a decision row at Ti and horizon H:
  Positive:
    a later disruptive event {FAIL, LOST, EVICT, KILL} occurs for the same
    task in (Ti, Ti+H], before any earlier clean terminal FINISH event.

  Known negative:
    EITHER
      (a) the task has an explicit FINISH in (Ti, Ti+H] before any disruption,
          meaning the lifecycle ended cleanly,
    OR
      (b) the task is observed through Ti+H.

  Censored:
    no positive event occurs, no clean FINISH proves a negative outcome, and
    the task is not observed through Ti+H.

Rows at/after a prior disruptive event are excluded so a later lifecycle is
not silently mixed into the original task episode.

The script:
  1) reproduces the old global-only target for an audit,
  2) builds the strict task-lifecycle target for H={60,300,900}s,
  3) compares old-vs-strict eligibility/labels,
  4) performs chronological 60/20/20 splitting,
  5) retrains classical baselines on natural-prevalence data,
  6) retrains matched RBF, quantum-fidelity and HCQKL kernels,
  7) selects HCQKL alpha by validation only,
  8) calibrates/thresholds on later validation data,
  9) evaluates on untouched natural-prevalence temporal test data,
 10) saves a downstream-rerun recommendation.

Default project:
  E:\other\AQUA-SLA

Run:
  conda activate aqua-sla
  cd /d E:\other\AQUA-SLA\Code
  python 34_aqua_sla_task_lifecycle_censoring_revision.py

Smoke test:
  python 34_aqua_sla_task_lifecycle_censoring_revision.py --fast

Important
---------
Only FINISH is treated as an explicit non-disruptive terminal event by default.
If your trace encoding uses another clean terminal label, pass it explicitly:

  --clean-terminal-events FINISH,SUCCESS,COMPLETED

Do not add an event unless its semantics truly indicate clean task termination.
"""

import argparse
import json
import math
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(r"E:\other\AQUA-SLA")
PREPARED_DATA = PROJECT_DIR / "Dataset" / "processed" / "aqua_sla_prepared.csv"
OUTPUT_ROOT = PROJECT_DIR / "results" / "aqua_sla_lifecycle_censoring_v5"

TIME_COLUMN = "time_seconds"
EVENT_COLUMN = "event"
END_TIME_COLUMN = "end_time_seconds"
TASK_KEYS = ["collection_id", "instance_index"]

DISRUPTIVE_EVENTS = {"FAIL", "LOST", "EVICT", "KILL"}
DEFAULT_CLEAN_TERMINAL_EVENTS = {"FINISH"}

PRIMARY_HORIZON_SECONDS = 300.0
HORIZONS = [60.0, 300.0, 900.0]

EARLY_FEATURES = [
    "scheduling_class",
    "collection_type",
    "priority",
    "vertical_scaling",
    "scheduler",
    "requested_cpu",
    "requested_memory",
    "requested_cpu_missing",
    "requested_memory_missing",
]

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20

PCA_COMPONENTS = 6
ALPHA_GRID = np.linspace(0.0, 1.0, 11)

CLASSICAL_TRAIN_MAX = 100_000

KERNEL_TRAIN_MAX = 2_000
KERNEL_TRAIN_POSITIVE_MAX = 500
KERNEL_VAL_SELECT_MAX = 3_000
KERNEL_VAL_CAL_MAX = 3_000
KERNEL_TEST_MAX = 5_000

SEED = 42


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    root: Path
    manifest: Path
    audit: Path
    prediction: Path
    models: Path
    kernels: Path
    figures: Path

    @classmethod
    def create(cls, run_name: str) -> "Paths":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = OUTPUT_ROOT / f"{stamp}_{run_name}"
        if root.exists():
            raise FileExistsError(root)

        obj = cls(
            root=root,
            manifest=root / "00_manifest",
            audit=root / "01_target_lifecycle_audit",
            prediction=root / "02_prediction",
            models=root / "03_models",
            kernels=root / "04_kernel_artifacts",
            figures=root / "05_figures",
        )
        for p in obj.__dict__.values():
            if isinstance(p, Path):
                p.mkdir(parents=True, exist_ok=True)
        return obj


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_json(obj: Any, path: Path) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [conv(v) for v in x]
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(conv(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    x = df.loc[:, cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def natural_subsample(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    """
    Deterministic temporal spread without class balancing.
    Preserves natural prevalence approximately and covers the whole interval.
    """
    if len(df) <= max_n:
        return df.copy().reset_index(drop=True)
    ids = np.linspace(0, len(df) - 1, max_n, dtype=int)
    return df.iloc[ids].copy().reset_index(drop=True)


def class_enriched_train(
    df: pd.DataFrame,
    max_n: int,
    positive_max: int,
    target: str,
) -> pd.DataFrame:
    """
    Training-only enrichment for the O(n^2) matched-kernel experiment.
    Validation and test are NEVER enriched.
    """
    pos = df[df[target] == 1]
    neg = df[df[target] == 0]

    n_pos = min(len(pos), positive_max, max_n // 2)
    n_neg = min(len(neg), max_n - n_pos)

    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            "Kernel training subset contains only one class. "
            "Increase the available training interval or inspect target construction."
        )

    pos_s = natural_subsample(pos, n_pos)
    neg_s = natural_subsample(neg, n_neg)

    out = (
        pd.concat([pos_s, neg_s], axis=0)
        .sort_values(TIME_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    n = len(d)
    a = int(n * TRAIN_FRACTION)
    b = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return (
        d.iloc[:a].copy(),
        d.iloc[a:b].copy(),
        d.iloc[b:].copy(),
    )


def split_validation(
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = val.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    cut = len(v) // 2
    return v.iloc[:cut].copy(), v.iloc[cut:].copy()


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ans = 0.0
    for i in range(bins):
        if i == bins - 1:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            m = (p >= edges[i]) & (p < edges[i + 1])
        if not m.any():
            continue
        ans += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ans)


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """
    Validation-only F1 selection.
    """
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return 0.5

    candidates = np.unique(
        np.quantile(p, np.linspace(0.01, 0.99, 199))
    )
    best = (0.5, -np.inf)
    for t in candidates:
        pred = (p >= t).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best[1]:
            best = (float(t), float(score))
    return best[0]


def metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-8, 1.0 - 1e-8)
    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": (
            float(roc_auc_score(y, p))
            if len(np.unique(y)) > 1 else np.nan
        ),
        "pr_auc": (
            float(average_precision_score(y, p))
            if len(np.unique(y)) > 1 else np.nan
        ),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece10": float(ece(y, p, 10)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def prepare_target_frame(raw: pd.DataFrame) -> pd.DataFrame:
    required = TASK_KEYS + [TIME_COLUMN, EVENT_COLUMN]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing target-construction columns: {missing}")

    df = raw.copy()
    df["_source_row"] = np.arange(len(df), dtype=np.int64)
    df[TIME_COLUMN] = pd.to_numeric(df[TIME_COLUMN], errors="coerce")
    df[EVENT_COLUMN] = (
        df[EVENT_COLUMN]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if END_TIME_COLUMN in df.columns:
        df[END_TIME_COLUMN] = pd.to_numeric(
            df[END_TIME_COLUMN], errors="coerce"
        )
    else:
        df[END_TIME_COLUMN] = np.nan

    df = df.dropna(subset=[TIME_COLUMN]).copy()

    for c in TASK_KEYS:
        df[c] = df[c].astype(str)

    df = (
        df.sort_values(
            TASK_KEYS + [TIME_COLUMN, "_source_row"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return df


def lifecycle_annotations(
    df: pd.DataFrame,
    clean_terminal_events: set[str],
) -> pd.DataFrame:
    n = len(df)

    next_disruption = np.full(n, np.nan)
    next_clean_terminal = np.full(n, np.nan)
    prior_disruption = np.zeros(n, dtype=bool)
    task_last_event = np.full(n, np.nan)
    task_observation_end = np.full(n, np.nan)

    groups = df.groupby(TASK_KEYS, sort=False).indices

    for _, ids0 in groups.items():
        ids = np.asarray(ids0, dtype=int)
        times = df.loc[ids, TIME_COLUMN].to_numpy(float)
        events = df.loc[ids, EVENT_COLUMN].to_numpy(str)

        last_event = float(np.nanmax(times))

        end_values = pd.to_numeric(
            df.loc[ids, END_TIME_COLUMN],
            errors="coerce",
        ).to_numpy(float)
        valid_end = end_values[np.isfinite(end_values)]

        if len(valid_end):
            obs_end = max(last_event, float(np.nanmax(valid_end)))
        else:
            obs_end = last_event

        task_last_event[ids] = last_event
        task_observation_end[ids] = obs_end

        nd = np.nan
        nf = np.nan

        # Reverse scan: time of next disruptive and next clean terminal event.
        for pos in range(len(ids) - 1, -1, -1):
            gid = ids[pos]
            next_disruption[gid] = nd
            next_clean_terminal[gid] = nf

            ev = events[pos]
            t = times[pos]

            if ev in DISRUPTIVE_EVENTS:
                nd = t

            if ev in clean_terminal_events:
                nf = t

        seen_disruption = False
        for pos in range(len(ids)):
            gid = ids[pos]
            prior_disruption[gid] = seen_disruption
            if events[pos] in DISRUPTIVE_EVENTS:
                seen_disruption = True

    out = df.copy()
    out["next_disruption_time"] = next_disruption
    out["next_clean_terminal_time"] = next_clean_terminal
    out["prior_disruption"] = prior_disruption
    out["current_disruption"] = out[EVENT_COLUMN].isin(DISRUPTIVE_EVENTS)
    out["task_last_event_time"] = task_last_event
    out["task_observation_end_time"] = task_observation_end

    return out


def build_global_only_target(
    annotated: pd.DataFrame,
    horizon: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Reproduces the earlier V2 global-trace-end censoring for audit only.
    """
    df = annotated.copy()

    global_end = float(df[TIME_COLUMN].max())
    delta = df["next_disruption_time"] - df[TIME_COLUMN]

    df["future_sla_event"] = (
        df["next_disruption_time"].notna()
        & (delta > 0)
        & (delta <= horizon)
    ).astype(int)

    df["right_censored_global_only"] = (
        df[TIME_COLUMN] > global_end - horizon
    )

    eligible = df[
        ~df["current_disruption"]
        & ~df["prior_disruption"]
        & ~df["right_censored_global_only"]
    ].copy()

    meta = {
        "horizon_seconds": horizon,
        "eligible_rows": len(eligible),
        "positives": int(eligible["future_sla_event"].sum()),
        "prevalence": float(eligible["future_sla_event"].mean()),
        "censoring": "global trace end only",
    }
    return eligible.reset_index(drop=True), meta


def build_lifecycle_target(
    annotated: pd.DataFrame,
    horizon: float,
    clean_terminal_events: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = annotated.copy()

    ti = df[TIME_COLUMN].to_numpy(float)
    nd = df["next_disruption_time"].to_numpy(float)
    nf = df["next_clean_terminal_time"].to_numpy(float)
    obs_end = df["task_observation_end_time"].to_numpy(float)

    horizon_end = ti + float(horizon)
    global_end = float(df[TIME_COLUMN].max())

    disruption_in_horizon = (
        np.isfinite(nd)
        & (nd > ti)
        & (nd <= horizon_end)
    )

    clean_before_disruption = (
        np.isfinite(nf)
        & (nf > ti)
        & (nf <= horizon_end)
        & (~np.isfinite(nd) | (nf < nd))
    )

    # A positive disruption must occur before any earlier clean terminal.
    positive = disruption_in_horizon & ~clean_before_disruption

    observed_through_horizon = (
        np.isfinite(obs_end)
        & (obs_end >= horizon_end)
        & (horizon_end <= global_end)
    )

    known_clean_negative = clean_before_disruption

    # Positive outcomes are observed by definition. A negative is usable only
    # when full follow-up exists or a clean terminal event closes the lifecycle.
    lifecycle_censored = (
        ~positive
        & ~known_clean_negative
        & ~observed_through_horizon
    )

    df["horizon_end_time"] = horizon_end
    df["positive_disruption_in_horizon"] = positive
    df["known_clean_negative"] = known_clean_negative
    df["observed_through_horizon"] = observed_through_horizon
    df["right_censored_task_lifecycle"] = lifecycle_censored
    df["future_sla_event"] = positive.astype(int)

    eligible_mask = (
        ~df["current_disruption"].to_numpy(bool)
        & ~df["prior_disruption"].to_numpy(bool)
        & ~lifecycle_censored
    )

    eligible = df.loc[eligible_mask].copy().reset_index(drop=True)

    meta = {
        "horizon_seconds": float(horizon),
        "decision_time": TIME_COLUMN,
        "task_keys": TASK_KEYS,
        "positive_definition": (
            "later FAIL/LOST/EVICT/KILL for same task in (Ti,Ti+H], "
            "before any earlier clean terminal event"
        ),
        "negative_definition": (
            "no disruptive event in horizon AND either explicit clean terminal "
            "before horizon or task observed through Ti+H"
        ),
        "clean_terminal_events": sorted(clean_terminal_events),
        "censoring": (
            "task-level lifecycle right censoring; rows with insufficient "
            "follow-up are excluded unless a positive disruption or explicit "
            "clean terminal is observed"
        ),
        "source_rows": int(len(annotated)),
        "eligible_rows": int(len(eligible)),
        "positives": int(eligible["future_sla_event"].sum()),
        "negatives": int((eligible["future_sla_event"] == 0).sum()),
        "prevalence": float(eligible["future_sla_event"].mean()),
        "task_lifecycle_censored_rows": int(lifecycle_censored.sum()),
        "known_clean_negative_rows": int(known_clean_negative.sum()),
        "observed_through_horizon_rows": int(observed_through_horizon.sum()),
    }

    return eligible, meta


def compare_targets(
    old: pd.DataFrame,
    strict: pd.DataFrame,
) -> dict[str, Any]:
    old_key = old[["_source_row", "future_sla_event"]].rename(
        columns={"future_sla_event": "old_label"}
    )
    new_key = strict[["_source_row", "future_sla_event"]].rename(
        columns={"future_sla_event": "strict_label"}
    )

    both = old_key.merge(new_key, on="_source_row", how="outer", indicator=True)

    common = both[both["_merge"] == "both"].copy()
    changed = int(
        (
            common["old_label"].astype(int)
            != common["strict_label"].astype(int)
        ).sum()
    )

    return {
        "old_global_only_eligible": int(len(old)),
        "strict_lifecycle_eligible": int(len(strict)),
        "old_only_rows": int((both["_merge"] == "left_only").sum()),
        "strict_only_rows": int((both["_merge"] == "right_only").sum()),
        "common_rows": int((both["_merge"] == "both").sum()),
        "changed_labels_on_common_rows": changed,
        "old_positives": int(old["future_sla_event"].sum()),
        "strict_positives": int(strict["future_sla_event"].sum()),
        "old_prevalence": float(old["future_sla_event"].mean()),
        "strict_prevalence": float(strict["future_sla_event"].mean()),
    }


# ---------------------------------------------------------------------------
# Horizon sensitivity
# ---------------------------------------------------------------------------

def run_horizon_sensitivity(
    annotated: pd.DataFrame,
    clean_terminal_events: set[str],
    out: Paths,
    fast: bool,
) -> pd.DataFrame:
    rows = []

    use_horizons = [PRIMARY_HORIZON_SECONDS] if fast else HORIZONS

    for H in use_horizons:
        data, meta = build_lifecycle_target(
            annotated,
            H,
            clean_terminal_events,
        )

        train, val, test = temporal_split(data)
        feats = [c for c in EARLY_FEATURES if c in data.columns]

        train_use = natural_subsample(
            train,
            min(10_000 if fast else CLASSICAL_TRAIN_MAX, len(train)),
        )

        Xtr = numeric(train_use, feats)
        ytr = train_use["future_sla_event"].to_numpy(int)

        Xv = numeric(val, feats)
        yv = val["future_sla_event"].to_numpy(int)

        Xt = numeric(test, feats)
        yt = test["future_sla_event"].to_numpy(int)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=SEED,
            )),
        ])

        sw = compute_sample_weight("balanced", ytr)
        model.fit(Xtr, ytr, clf__sample_weight=sw)

        pv = model.predict_proba(Xv)[:, 1]
        t = choose_threshold(yv, pv)
        pt = model.predict_proba(Xt)[:, 1]

        m = metrics(yt, pt, t)

        rows.append({
            "horizon_seconds": H,
            "eligible_rows": meta["eligible_rows"],
            "positives_total": meta["positives"],
            "overall_prevalence": meta["prevalence"],
            **m,
        })

    ans = pd.DataFrame(rows)
    ans.to_csv(
        out.audit / "horizon_sensitivity_lifecycle.csv",
        index=False,
    )
    return ans


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

def run_classical_baselines(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out: Paths,
    fast: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = [c for c in EARLY_FEATURES if c in train.columns]
    if not feats:
        raise RuntimeError("No early decision-time features are available.")

    max_train = 20_000 if fast else CLASSICAL_TRAIN_MAX
    tr = natural_subsample(train, min(max_train, len(train)))

    Xtr = numeric(tr, feats)
    ytr = tr["future_sla_event"].to_numpy(int)
    Xv = numeric(val, feats)
    yv = val["future_sla_event"].to_numpy(int)
    Xt = numeric(test, feats)
    yt = test["future_sla_event"].to_numpy(int)

    models: dict[str, Any] = {
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=SEED,
            )),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400 if not fast else 150,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=SEED,
            )),
        ]),
        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-4,
                max_iter=250 if not fast else 100,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=SEED,
            )),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=350 if not fast else 120,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=SEED,
            )),
        ])
    except Exception:
        pass

    rows = []
    predictions = pd.DataFrame({
        "_source_row": test["_source_row"].to_numpy(),
        TIME_COLUMN: test[TIME_COLUMN].to_numpy(),
        "future_sla_event": yt,
    })

    for name, model in models.items():
        start = time.perf_counter()

        if name == "HistGradientBoosting":
            sw = compute_sample_weight("balanced", ytr)
            model.fit(Xtr, ytr, clf__sample_weight=sw)
        elif name == "XGBoost":
            pos = max(int(ytr.sum()), 1)
            neg = max(int((ytr == 0).sum()), 1)
            model.set_params(clf__scale_pos_weight=neg / pos)
            model.fit(Xtr, ytr)
        else:
            model.fit(Xtr, ytr)

        train_sec = time.perf_counter() - start

        pv = model.predict_proba(Xv)[:, 1]
        threshold = choose_threshold(yv, pv)

        infer_start = time.perf_counter()
        pt = model.predict_proba(Xt)[:, 1]
        infer_sec = time.perf_counter() - infer_start

        row = {
            "model": name,
            "train_rows": len(tr),
            "train_seconds": train_sec,
            "test_inference_seconds": infer_sec,
            **metrics(yt, pt, threshold),
        }
        rows.append(row)
        predictions[f"p_{name}"] = pt

        joblib.dump(
            {
                "model": model,
                "features": feats,
                "threshold": threshold,
                "target": "future_sla_event",
                "censoring": "task-lifecycle-aware",
            },
            out.models / f"{name}_lifecycle.joblib",
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        out.prediction / "classical_baseline_metrics_lifecycle.csv",
        index=False,
    )
    predictions.to_csv(
        out.prediction / "classical_test_predictions_lifecycle.csv",
        index=False,
    )

    return metrics_df, predictions


# ---------------------------------------------------------------------------
# Matched RBF / quantum / HCQKL
# ---------------------------------------------------------------------------

def statevectors(Z: np.ndarray) -> np.ndarray:
    try:
        from qiskit.quantum_info import Statevector
        try:
            from qiskit.circuit.library import zz_feature_map

            def make_map(d: int):
                return zz_feature_map(
                    feature_dimension=d,
                    reps=2,
                    entanglement="linear",
                )
        except Exception:
            from qiskit.circuit.library import ZZFeatureMap

            def make_map(d: int):
                return ZZFeatureMap(
                    feature_dimension=d,
                    reps=2,
                    entanglement="linear",
                )
    except Exception as exc:
        raise RuntimeError(
            "Qiskit is required for matched quantum-kernel evaluation."
        ) from exc

    fmap = make_map(Z.shape[1])
    states = []
    for row in Z:
        qc = fmap.assign_parameters(row, inplace=False)
        states.append(
            np.asarray(
                Statevector.from_instruction(qc).data,
                complex,
            )
        )
    return np.vstack(states)


def fidelity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs(A @ B.conj().T) ** 2


def fit_calibrator(
    scores: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Calibration subset has only one class. "
            "Increase validation size or inspect target prevalence."
        )
    lr = LogisticRegression(
        solver="lbfgs",
        random_state=SEED,
    )
    lr.fit(np.asarray(scores).reshape(-1, 1), y)
    return lr


def run_matched_kernels(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out: Paths,
    fast: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feats = [c for c in EARLY_FEATURES if c in train.columns]

    val_select_full, val_cal_full = split_validation(val)

    if fast:
        train_max = 600
        pos_max = 150
        val_select_max = 600
        val_cal_max = 600
        test_max = 1500
    else:
        train_max = KERNEL_TRAIN_MAX
        pos_max = KERNEL_TRAIN_POSITIVE_MAX
        val_select_max = KERNEL_VAL_SELECT_MAX
        val_cal_max = KERNEL_VAL_CAL_MAX
        test_max = KERNEL_TEST_MAX

    tr = class_enriched_train(
        train,
        train_max,
        pos_max,
        "future_sla_event",
    )
    vs = natural_subsample(
        val_select_full,
        min(val_select_max, len(val_select_full)),
    )
    vc = natural_subsample(
        val_cal_full,
        min(val_cal_max, len(val_cal_full)),
    )
    te = natural_subsample(
        test,
        min(test_max, len(test)),
    )

    for label, frame in [
        ("training", tr),
        ("validation-selection", vs),
        ("validation-calibration", vc),
        ("test", te),
    ]:
        if frame["future_sla_event"].nunique() < 2:
            raise RuntimeError(
                f"{label} matched-kernel subset contains only one class. "
                "Increase subset size before interpreting kernel results."
            )

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    pca = PCA(
        n_components=min(PCA_COMPONENTS, len(feats)),
        random_state=SEED,
    )
    angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))

    Xtr0 = numeric(tr, feats)
    Xvs0 = numeric(vs, feats)
    Xvc0 = numeric(vc, feats)
    Xte0 = numeric(te, feats)

    Xtr = imputer.fit_transform(Xtr0)
    Xvs = imputer.transform(Xvs0)
    Xvc = imputer.transform(Xvc0)
    Xte = imputer.transform(Xte0)

    Xtr = scaler.fit_transform(Xtr)
    Xvs = scaler.transform(Xvs)
    Xvc = scaler.transform(Xvc)
    Xte = scaler.transform(Xte)

    Ztr0 = pca.fit_transform(Xtr)
    Zvs0 = pca.transform(Xvs)
    Zvc0 = pca.transform(Xvc)
    Zte0 = pca.transform(Xte)

    Ztr = angle_scaler.fit_transform(Ztr0)
    Zvs = angle_scaler.transform(Zvs0)
    Zvc = angle_scaler.transform(Zvc0)
    Zte = angle_scaler.transform(Zte0)

    ytr = tr["future_sla_event"].to_numpy(int)
    yvs = vs["future_sla_event"].to_numpy(int)
    yvc = vc["future_sla_event"].to_numpy(int)
    yte = te["future_sla_event"].to_numpy(int)

    gamma = 1.0 / max(Ztr.shape[1], 1)

    Kr_tr = rbf_kernel(Ztr, Ztr, gamma=gamma)
    Kr_vs = rbf_kernel(Zvs, Ztr, gamma=gamma)
    Kr_vc = rbf_kernel(Zvc, Ztr, gamma=gamma)
    Kr_te = rbf_kernel(Zte, Ztr, gamma=gamma)

    Str = statevectors(Ztr)
    Svs = statevectors(Zvs)
    Svc = statevectors(Zvc)
    Ste = statevectors(Zte)

    Kq_tr = fidelity(Str, Str)
    Kq_vs = fidelity(Svs, Str)
    Kq_vc = fidelity(Svc, Str)
    Kq_te = fidelity(Ste, Str)

    np.save(out.kernels / "rbf_train.npy", Kr_tr)
    np.save(out.kernels / "rbf_test.npy", Kr_te)
    np.save(out.kernels / "quantum_train.npy", Kq_tr)
    np.save(out.kernels / "quantum_test.npy", Kq_te)

    alpha_rows = []

    for alpha in ALPHA_GRID:
        Kh_tr = alpha * Kr_tr + (1.0 - alpha) * Kq_tr
        Kh_vs = alpha * Kr_vs + (1.0 - alpha) * Kq_vs

        svc = SVC(
            kernel="precomputed",
            C=10,
            class_weight="balanced",
            probability=False,
            random_state=SEED,
        )
        svc.fit(Kh_tr, ytr)

        scores = svc.decision_function(Kh_vs)

        roc = (
            roc_auc_score(yvs, scores)
            if len(np.unique(yvs)) > 1 else np.nan
        )
        pr = (
            average_precision_score(yvs, scores)
            if len(np.unique(yvs)) > 1 else np.nan
        )
        selection = (
            0.5 * roc + 0.5 * pr
            if np.isfinite(roc) and np.isfinite(pr)
            else -np.inf
        )

        alpha_rows.append({
            "alpha_rbf": float(alpha),
            "alpha_quantum": float(1.0 - alpha),
            "validation_roc_auc": float(roc),
            "validation_pr_auc": float(pr),
            "selection_score": float(selection),
        })

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(
        out.prediction / "hcqkl_alpha_sweep_lifecycle.csv",
        index=False,
    )

    best_alpha = float(
        alpha_df.loc[
            alpha_df["selection_score"].idxmax(),
            "alpha_rbf",
        ]
    )

    model_specs = {
        "RBF_kernel": (
            Kr_tr,
            Kr_vc,
            Kr_te,
        ),
        "Quantum_fidelity_kernel": (
            Kq_tr,
            Kq_vc,
            Kq_te,
        ),
        "HCQKL": (
            best_alpha * Kr_tr + (1.0 - best_alpha) * Kq_tr,
            best_alpha * Kr_vc + (1.0 - best_alpha) * Kq_vc,
            best_alpha * Kr_te + (1.0 - best_alpha) * Kq_te,
        ),
    }

    metric_rows = []
    pred_df = pd.DataFrame({
        "_source_row": te["_source_row"].to_numpy(),
        TIME_COLUMN: te[TIME_COLUMN].to_numpy(),
        "future_sla_event": yte,
    })

    fitted_models = {}

    for name, (Ktr, Kvc, Kte) in model_specs.items():
        start = time.perf_counter()

        svc = SVC(
            kernel="precomputed",
            C=10,
            class_weight="balanced",
            probability=False,
            random_state=SEED,
        )
        svc.fit(Ktr, ytr)

        train_sec = time.perf_counter() - start

        dvc = svc.decision_function(Kvc)
        calibrator = fit_calibrator(dvc, yvc)
        pvc = calibrator.predict_proba(
            np.asarray(dvc).reshape(-1, 1)
        )[:, 1]
        threshold = choose_threshold(yvc, pvc)

        infer_start = time.perf_counter()
        dte = svc.decision_function(Kte)
        pte = calibrator.predict_proba(
            np.asarray(dte).reshape(-1, 1)
        )[:, 1]
        infer_sec = time.perf_counter() - infer_start

        metric_rows.append({
            "model": name,
            "alpha_rbf": (
                best_alpha if name == "HCQKL"
                else 1.0 if name == "RBF_kernel"
                else 0.0
            ),
            "alpha_quantum": (
                1.0 - best_alpha if name == "HCQKL"
                else 0.0 if name == "RBF_kernel"
                else 1.0
            ),
            "train_rows": len(tr),
            "validation_calibration_rows": len(vc),
            "test_rows": len(te),
            "train_seconds": train_sec,
            "test_inference_seconds": infer_sec,
            **metrics(yte, pte, threshold),
        })

        pred_df[f"p_{name}"] = pte

        fitted_models[name] = {
            "svc": svc,
            "calibrator": calibrator,
            "threshold": threshold,
        }

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        out.prediction / "matched_kernel_test_metrics_lifecycle.csv",
        index=False,
    )
    pred_df.to_csv(
        out.prediction / "matched_kernel_test_predictions_lifecycle.csv",
        index=False,
    )

    joblib.dump(
        {
            "target": "future_sla_event",
            "censoring": "task-lifecycle-aware",
            "horizon_seconds": PRIMARY_HORIZON_SECONDS,
            "features": feats,
            "imputer": imputer,
            "scaler": scaler,
            "pca": pca,
            "angle_scaler": angle_scaler,
            "z_train": Ztr,
            "statevectors_train": Str,
            "rbf_gamma": gamma,
            "best_alpha": best_alpha,
            "models": fitted_models,
            "training_source_rows": tr["_source_row"].to_numpy(),
        },
        out.models / "hcqkl_lifecycle_bundle.joblib",
    )

    subset_summary = pd.DataFrame([
        {
            "subset": "train_enriched",
            "rows": len(tr),
            "positives": int(tr["future_sla_event"].sum()),
            "prevalence": float(tr["future_sla_event"].mean()),
        },
        {
            "subset": "validation_select_natural",
            "rows": len(vs),
            "positives": int(vs["future_sla_event"].sum()),
            "prevalence": float(vs["future_sla_event"].mean()),
        },
        {
            "subset": "validation_calibration_natural",
            "rows": len(vc),
            "positives": int(vc["future_sla_event"].sum()),
            "prevalence": float(vc["future_sla_event"].mean()),
        },
        {
            "subset": "test_natural",
            "rows": len(te),
            "positives": int(te["future_sla_event"].sum()),
            "prevalence": float(te["future_sla_event"].mean()),
        },
    ])
    subset_summary.to_csv(
        out.prediction / "matched_kernel_subset_summary.csv",
        index=False,
    )

    return metrics_df, alpha_df, pred_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(
    audit_df: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    kernel_metrics: pd.DataFrame,
    out: Paths,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Target eligibility audit.
    if not audit_df.empty:
        z = audit_df.copy()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(z))
        width = 0.35
        ax.bar(
            x - width / 2,
            z["old_global_only_eligible"],
            width,
            label="Global-only",
        )
        ax.bar(
            x + width / 2,
            z["strict_lifecycle_eligible"],
            width,
            label="Task-lifecycle",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{int(v)} s" for v in z["horizon_seconds"]]
        )
        ax.set_ylabel("Eligible rows")
        ax.set_title("Target eligibility after task-lifecycle censoring")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            out.figures / "target_lifecycle_censoring_audit.png",
            dpi=300,
        )
        plt.close(fig)

    combined = []
    if not baseline_metrics.empty:
        for _, r in baseline_metrics.iterrows():
            combined.append({
                "model": r["model"],
                "roc_auc": r["roc_auc"],
                "pr_auc": r["pr_auc"],
            })
    if not kernel_metrics.empty:
        for _, r in kernel_metrics.iterrows():
            combined.append({
                "model": r["model"],
                "roc_auc": r["roc_auc"],
                "pr_auc": r["pr_auc"],
            })

    if combined:
        z = pd.DataFrame(combined).sort_values("roc_auc")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(z["model"], z["roc_auc"])
        ax.set_xlabel("ROC-AUC")
        ax.set_xlim(0.0, 1.0)
        ax.set_title("Lifecycle-censored future-horizon prediction")
        fig.tight_layout()
        fig.savefig(
            out.figures / "prediction_roc_auc_lifecycle.png",
            dpi=300,
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AQUA-SLA task-lifecycle censoring audit and prediction rerun."
        )
    )
    parser.add_argument(
        "--run-name",
        default="task_lifecycle_censoring",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=PRIMARY_HORIZON_SECONDS,
    )
    parser.add_argument(
        "--clean-terminal-events",
        default="FINISH",
        help=(
            "Comma-separated explicit NON-disruptive terminal events. "
            "Default: FINISH"
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
    )
    args = parser.parse_args()

    clean_terminal_events = {
        x.strip().upper()
        for x in args.clean_terminal_events.split(",")
        if x.strip()
    }

    if not clean_terminal_events:
        raise ValueError(
            "At least one explicit clean terminal event must be supplied."
        )

    if not PREPARED_DATA.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {PREPARED_DATA}"
        )

    out = Paths.create(args.run_name)
    print("AQUA-SLA lifecycle-censoring output:", out.root)

    raw = pd.read_csv(PREPARED_DATA, low_memory=False)
    prepared = prepare_target_frame(raw)
    annotated = lifecycle_annotations(
        prepared,
        clean_terminal_events,
    )

    # Save event vocabulary so the clean-terminal semantics are auditable.
    event_counts = (
        annotated[EVENT_COLUMN]
        .value_counts(dropna=False)
        .rename_axis("event")
        .reset_index(name="rows")
    )
    event_counts.to_csv(
        out.audit / "event_distribution.csv",
        index=False,
    )

    audit_rows = []
    strict_primary = None
    strict_meta_primary = None

    use_horizons = [args.horizon] if args.fast else HORIZONS

    for H in use_horizons:
        old, old_meta = build_global_only_target(
            annotated,
            H,
        )
        strict, strict_meta = build_lifecycle_target(
            annotated,
            H,
            clean_terminal_events,
        )

        comp = compare_targets(old, strict)
        comp["horizon_seconds"] = float(H)
        audit_rows.append(comp)

        if math.isclose(float(H), float(args.horizon)):
            strict_primary = strict
            strict_meta_primary = strict_meta

        # Save row-level censoring audit for each horizon.
        cols = [
            "_source_row",
            *TASK_KEYS,
            TIME_COLUMN,
            EVENT_COLUMN,
            "next_disruption_time",
            "next_clean_terminal_time",
            "task_last_event_time",
            "task_observation_end_time",
            "horizon_end_time",
            "positive_disruption_in_horizon",
            "known_clean_negative",
            "observed_through_horizon",
            "right_censored_task_lifecycle",
            "future_sla_event",
        ]
        available_cols = [c for c in cols if c in strict.columns]
        strict[available_cols].to_csv(
            out.audit / f"strict_target_h{int(H)}_eligible.csv",
            index=False,
        )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(
        out.audit / "global_vs_task_lifecycle_audit.csv",
        index=False,
    )

    if strict_primary is None:
        strict_primary, strict_meta_primary = build_lifecycle_target(
            annotated,
            args.horizon,
            clean_terminal_events,
        )

    strict_primary.to_csv(
        out.audit / "future_target_task_lifecycle_primary.csv",
        index=False,
    )
    save_json(
        strict_meta_primary,
        out.audit / "future_target_task_lifecycle_metadata.json",
    )

    # Split summary.
    train, val, test = temporal_split(strict_primary)
    split_summary = pd.DataFrame([
        {
            "split": name,
            "rows": len(d),
            "positives": int(d["future_sla_event"].sum()),
            "prevalence": float(d["future_sla_event"].mean()),
            "time_min": float(d[TIME_COLUMN].min()),
            "time_max": float(d[TIME_COLUMN].max()),
        }
        for name, d in [
            ("train", train),
            ("validation", val),
            ("test", test),
        ]
    ])
    split_summary.to_csv(
        out.audit / "temporal_split_summary_lifecycle.csv",
        index=False,
    )

    print("Running lifecycle-aware horizon sensitivity...")
    horizon_df = run_horizon_sensitivity(
        annotated,
        clean_terminal_events,
        out,
        args.fast,
    )

    print("Retraining classical baselines...")
    baseline_metrics, _ = run_classical_baselines(
        train,
        val,
        test,
        out,
        args.fast,
    )

    print("Retraining matched RBF / quantum / HCQKL kernels...")
    kernel_metrics, alpha_df, _ = run_matched_kernels(
        train,
        val,
        test,
        out,
        args.fast,
    )

    # Determine whether old downstream risk-dependent experiments should be
    # regenerated for one internally consistent final manuscript.
    primary_audit = audit_df[
        np.isclose(
            audit_df["horizon_seconds"].astype(float),
            float(args.horizon),
        )
    ]

    if primary_audit.empty:
        downstream_required = True
        reason = "Primary-horizon audit row unavailable."
    else:
        r = primary_audit.iloc[0]
        target_changed = (
            int(r["old_only_rows"]) > 0
            or int(r["strict_only_rows"]) > 0
            or int(r["changed_labels_on_common_rows"]) > 0
        )
        downstream_required = bool(target_changed)
        reason = (
            "Task-lifecycle censoring changes the eligible/labelled prediction "
            "dataset. For a single final experimental chain, regenerate any "
            "downstream scheduling/iQuantum results that consume the predictor."
            if target_changed
            else
            "Task-lifecycle audit did not alter eligibility or labels at the "
            "primary horizon; prior downstream results can remain."
        )

    decision = {
        "reviewer_concern": (
            "Review concern #1: task lifecycle and censoring rules."
        ),
        "primary_horizon_seconds": float(args.horizon),
        "clean_terminal_events": sorted(clean_terminal_events),
        "strict_target_rows": int(len(strict_primary)),
        "strict_target_positives": int(
            strict_primary["future_sla_event"].sum()
        ),
        "strict_target_prevalence": float(
            strict_primary["future_sla_event"].mean()
        ),
        "downstream_rerun_required_for_full_internal_consistency": (
            downstream_required
        ),
        "reason": reason,
    }
    save_json(
        decision,
        out.manifest / "downstream_rerun_decision.json",
    )

    make_figures(
        audit_df,
        baseline_metrics,
        kernel_metrics,
        out,
    )

    final_status = {
        "completed": True,
        "output": str(out.root),
        "source_dataset": str(PREPARED_DATA),
        "reviewer_concern_addressed": (
            "Task-lifecycle-aware right censoring of the future-horizon SLA target"
        ),
        "primary_horizon_seconds": float(args.horizon),
        "clean_terminal_events": sorted(clean_terminal_events),
        "source_rows": int(len(raw)),
        "strict_eligible_rows": int(len(strict_primary)),
        "strict_positives": int(
            strict_primary["future_sla_event"].sum()
        ),
        "strict_prevalence": float(
            strict_primary["future_sla_event"].mean()
        ),
        "classical_models": baseline_metrics["model"].tolist(),
        "kernel_models": kernel_metrics["model"].tolist(),
        "selected_alpha_rbf": float(
            alpha_df.loc[
                alpha_df["selection_score"].idxmax(),
                "alpha_rbf",
            ]
        ),
        "downstream_rerun_required": downstream_required,
        "notes": [
            "Validation and test retain natural prevalence.",
            "Only training may be class-enriched for the O(n^2) matched-kernel experiment.",
            "Decision-time predictor features exclude runtime telemetry, event labels, end time, and duration.",
            "Task-level censoring is based on observed follow-up or explicit clean terminal events.",
            "Previous V2/V3/V4 result folders are not modified.",
        ],
    }
    save_json(
        final_status,
        out.manifest / "final_status.json",
    )

    save_json(
        {
            "project": "AQUA-SLA",
            "version": "task-lifecycle-censoring-v5",
            "created": datetime.now().isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "output": str(out.root),
            "fast": bool(args.fast),
            "early_features": EARLY_FEATURES,
            "disruptive_events": sorted(DISRUPTIVE_EVENTS),
            "clean_terminal_events": sorted(clean_terminal_events),
            "primary_horizon_seconds": float(args.horizon),
            "target_metadata": strict_meta_primary,
        },
        out.manifest / "run_manifest.json",
    )

    print()
    print("Finished.")
    print(json.dumps(final_status, indent=2))


if __name__ == "__main__":
    main()
