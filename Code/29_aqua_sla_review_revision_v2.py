from __future__ import annotations

"""
AQUA-SLA reviewer-revision V2 experiment suite.

This file creates a new timestamped V2 result tree and never overwrites V1.
It implements the reviewer-driven experiments:
  1) future-horizon SLA event target,
  2) strict decision-time feature view,
  3) matched RBF / quantum-fidelity / HCQKL kernel comparison,
  4) classical baselines,
  5) natural-prevalence evaluation,
  6) capacity-aware multi-task scheduling,
  7) exact MILP reference,
  8) raw QAOA and repaired QAOA reported separately,
  9) external SLO evaluation with arrivals, durations, queues,
 10) temporal-block bootstrap/Wilcoxon/Holm statistics,
 11) independent-QASM iQuantum policy comparison.

Default project root:
    E:\\other\\AQUA-SLA

Recommended:
    conda activate aqua-sla
    cd /d E:\\other\\AQUA-SLA\\Code
    python 29_aqua_sla_review_revision_v2.py --mode full

Smoke test:
    python 29_aqua_sla_review_revision_v2.py --mode full --run-name reviewer_revision_v2_test --fast

Full reviewer run:
    python 29_aqua_sla_review_revision_v2.py --mode full --run-name reviewer_revision_v2
"""

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(r"E:\other\AQUA-SLA")
PREPARED_DATA = PROJECT_DIR / "Dataset" / "processed" / "aqua_sla_prepared.csv"
V2_ROOT = PROJECT_DIR / "results" / "aqua_sla_review_v2"

TIME_COLUMN = "time_seconds"
EVENT_COLUMN = "event"
TASK_KEYS = ["collection_id", "instance_index"]
DISRUPTIVE_EVENTS = {"FAIL", "LOST", "EVICT", "KILL"}

PRIMARY_HORIZON_SECONDS = 300.0
HORIZON_SENSITIVITY = [60.0, 300.0, 900.0]

EARLY_FEATURES = [
    "scheduling_class", "collection_type", "priority", "vertical_scaling",
    "scheduler", "requested_cpu", "requested_memory",
    "requested_cpu_missing", "requested_memory_missing",
]

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PCA_COMPONENTS = 6
ALPHA_GRID = np.linspace(0.0, 1.0, 11)

# Matched-kernel study. Training is deliberately class-enriched because the
# natural future-event prevalence is low; validation/test remain natural.
KERNEL_TRAIN_MAX = 2000
KERNEL_TRAIN_POSITIVE_MAX = 500
KERNEL_VAL_SELECT_MAX = 3000
KERNEL_VAL_CAL_MAX = 3000
KERNEL_TEST_MAX = 5000
CLASSICAL_TRAIN_MAX = 100_000

# Classical capacity-aware study uses larger instances. QAOA is evaluated on
# a separate small but multi-task / multi-resource capacity-aware setting.
SCHED_BATCH_SIZES = [4, 6, 8]
SCHED_BLOCKS = 20
SCHED_RESOURCES = 3
QAOA_BATCH_SIZE = 3
QAOA_RESOURCES = 2
QAOA_BLOCKS = 10
QAOA_CAPACITY_UNITS = 3
QAOA_REPS = 1
QAOA_SHOTS = 512
QAOA_MAXITER = 30
QAOA_PENALTY = 20.0

BOOTSTRAP_RESAMPLES = 5000
SEED = 42

IQUANTUM_HOME = Path(os.environ.get(
    "IQUANTUM_HOME", str(PROJECT_DIR / "third_party" / "iQuantum")
))
IQUANTUM_BRIDGE_CLASS = "org.iquantum.examples.experimental.AquaSlaIQuantumBridge"
OLD_IQUANTUM_WORKLOAD = (
    PROJECT_DIR / "results" / "aqua_sla_final" /
    "iquantum_platform_integration" / "csv" / "aqua_sla_iquantum_workload.csv"
)
IQUANTUM_POLICIES = [
    "random", "least_loaded", "compatibility_aware", "risk_aware"
]
IQUANTUM_SEEDS = [101, 202, 303]
IQUANTUM_WORKLOAD_SIZES = [30, 60]


@dataclass
class Paths:
    root: Path
    manifest: Path
    target: Path
    prediction: Path
    models: Path
    kernels: Path
    scheduling: Path
    qaoa: Path
    stats: Path
    iquantum: Path
    qasm: Path
    figures: Path
    logs: Path

    @classmethod
    def create(cls, run_name: str) -> "Paths":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = V2_ROOT / f"{stamp}_{run_name}"
        obj = cls(
            root=root,
            manifest=root / "00_manifest",
            target=root / "01_future_target",
            prediction=root / "02_prediction_hcqkl",
            models=root / "03_models",
            kernels=root / "04_kernel_artifacts",
            scheduling=root / "05_capacity_scheduling",
            qaoa=root / "06_qaoa",
            stats=root / "07_statistics",
            iquantum=root / "08_iquantum_policies",
            qasm=root / "08_iquantum_policies" / "qasm",
            figures=root / "09_figures",
            logs=root / "10_logs",
        )
        for p in asdict(obj).values():
            Path(p).mkdir(parents=True, exist_ok=True)
        return obj


def save_json(obj: Any, path: Path) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, Path): return str(x)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, dict): return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [conv(v) for v in x]
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def numeric(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    x = frame.loc[:, cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def natural_subsample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    """Subsample without forcing 50/50 prevalence."""
    if len(frame) <= maximum:
        return frame.copy().reset_index(drop=True)
    frac = maximum / len(frame)
    chunks = []
    for label, g in frame.groupby("future_sla_event", sort=False):
        n = max(1, int(round(len(g) * frac)))
        chunks.append(g.sample(n=min(n, len(g)), random_state=seed + int(label)))
    out = pd.concat(chunks).sort_values(TIME_COLUMN, kind="mergesort")
    if len(out) > maximum:
        out = out.iloc[np.linspace(0, len(out)-1, maximum, dtype=int)]
    return out.reset_index(drop=True)


def enriched_kernel_train_sample(frame: pd.DataFrame, maximum: int,
                                 positive_max: int, seed: int) -> pd.DataFrame:
    """Class-enrich training only; validation and test remain natural prevalence.

    The same returned rows are used for RBF, quantum-fidelity, and HCQKL so the
    matched-kernel comparison differs only by kernel construction.
    """
    pos = frame[frame["future_sla_event"] == 1]
    neg = frame[frame["future_sla_event"] == 0]
    if pos.empty or neg.empty:
        raise ValueError("Kernel training requires both future-event classes.")

    n_pos = min(len(pos), positive_max, max(1, maximum // 2))
    n_neg = min(len(neg), maximum - n_pos)
    if n_neg <= 0:
        n_neg = min(len(neg), n_pos)

    # Draw across the full training interval, then restore temporal order.
    pos_s = pos.sample(n=n_pos, random_state=seed)
    neg_s = neg.sample(n=n_neg, random_state=seed + 1)
    out = pd.concat([pos_s, neg_s], ignore_index=False)
    return out.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ans = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p <= edges[i+1]) if i == bins-1 else (p < edges[i+1]))
        if mask.any():
            ans += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ans)


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_t, best = 0.5, -1e9
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        mcc = matthews_corrcoef(y, pred)
        rec = recall_score(y, pred, zero_division=0)
        score = 0.45*f1 + 0.30*((mcc+1)/2) + 0.25*rec
        if score > best:
            best, best_t = score, float(t)
    return best_t


def metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, Any]:
    p = np.clip(np.asarray(p, float), 1e-8, 1-1e-8)
    y = np.asarray(y, int)
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "n": len(y), "prevalence": float(y.mean()), "threshold": t,
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn/(tn+fp) if tn+fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0,1]),
        "ece10": ece(y, p, 10),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    n = len(df)
    a = int(n*TRAIN_FRACTION)
    b = int(n*(TRAIN_FRACTION + VALIDATION_FRACTION))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def split_val(val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = val.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    cut = len(val)//2
    return val.iloc[:cut].copy(), val.iloc[cut:].copy()


# -----------------------------------------------------------------------------
# 1. Future target
# -----------------------------------------------------------------------------
def build_future_target(raw: pd.DataFrame, horizon: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    req = TASK_KEYS + [TIME_COLUMN, EVENT_COLUMN]
    missing = [c for c in req if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing target-construction columns: {missing}")

    df = raw.copy()
    df[TIME_COLUMN] = pd.to_numeric(df[TIME_COLUMN], errors="coerce")
    df[EVENT_COLUMN] = df[EVENT_COLUMN].astype(str).str.upper().str.strip()
    df = df.dropna(subset=[TIME_COLUMN]).copy()
    for c in TASK_KEYS:
        df[c] = df[c].astype(str)
    df = df.sort_values(TASK_KEYS + [TIME_COLUMN], kind="mergesort").reset_index(drop=True)

    nxt = np.full(len(df), np.nan)
    prior = np.zeros(len(df), dtype=bool)
    groups = df.groupby(TASK_KEYS, sort=False).indices

    for _, ids in groups.items():
        ids = np.asarray(ids, int)
        times = df.loc[ids, TIME_COLUMN].to_numpy(float)
        events = df.loc[ids, EVENT_COLUMN].to_numpy(str)
        next_t = np.nan
        for pos in range(len(ids)-1, -1, -1):
            gid = ids[pos]
            nxt[gid] = next_t
            if events[pos] in DISRUPTIVE_EVENTS:
                next_t = times[pos]
        seen = False
        for pos in range(len(ids)):
            gid = ids[pos]
            prior[gid] = seen
            if events[pos] in DISRUPTIVE_EVENTS:
                seen = True

    df["next_disruption_time"] = nxt
    df["prior_disruption"] = prior
    df["current_disruption"] = df[EVENT_COLUMN].isin(DISRUPTIVE_EVENTS)
    end = float(df[TIME_COLUMN].max())
    df["right_censored"] = df[TIME_COLUMN] > end - horizon
    delta = df["next_disruption_time"] - df[TIME_COLUMN]
    df["future_sla_event"] = (
        df["next_disruption_time"].notna() & (delta > 0) & (delta <= horizon)
    ).astype(int)

    eligible = df[
        ~df["current_disruption"] & ~df["prior_disruption"] & ~df["right_censored"]
    ].copy().reset_index(drop=True)

    meta = {
        "horizon_seconds": horizon,
        "decision_time": TIME_COLUMN,
        "target": "future_sla_event",
        "positive_definition": "later FAIL/LOST/EVICT/KILL for same task in (Ti, Ti+H]",
        "censoring": "drop current/post-disruption rows and rows within H of trace end",
        "source_rows": len(raw), "eligible_rows": len(eligible),
        "positives": int(eligible["future_sla_event"].sum()),
        "prevalence": float(eligible["future_sla_event"].mean()),
    }
    return eligible, meta


def horizon_sensitivity(raw: pd.DataFrame, out: Paths, fast: bool) -> pd.DataFrame:
    rows = []
    for H in ([PRIMARY_HORIZON_SECONDS] if fast else HORIZON_SENSITIVITY):
        data, meta = build_future_target(raw, H)
        tr, va, te = temporal_split(data)
        feats = [c for c in EARLY_FEATURES if c in data.columns]
        tr_use = natural_subsample(tr, min(10_000 if fast else CLASSICAL_TRAIN_MAX, len(tr)), SEED)
        Xtr, ytr = numeric(tr_use, feats), tr_use["future_sla_event"].to_numpy(int)
        Xva, yva = numeric(va, feats), va["future_sla_event"].to_numpy(int)
        Xte, yte = numeric(te, feats), te["future_sla_event"].to_numpy(int)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
                min_samples_leaf=20, l2_regularization=1.0, random_state=SEED,
            )),
        ])
        sw = compute_sample_weight("balanced", ytr)
        model.fit(Xtr, ytr, clf__sample_weight=sw)
        pv = model.predict_proba(Xva)[:,1]
        t = choose_threshold(yva, pv)
        pt = model.predict_proba(Xte)[:,1]
        m = metrics(yte, pt, t)
        rows.append({"horizon_seconds": H, "eligible_rows": meta["eligible_rows"], **m})
    ans = pd.DataFrame(rows)
    ans.to_csv(out.target / "horizon_sensitivity.csv", index=False)
    return ans


# -----------------------------------------------------------------------------
# 2. Matched kernels + HCQKL
# -----------------------------------------------------------------------------
def statevectors(Z: np.ndarray) -> np.ndarray:
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.quantum_info import Statevector
    except Exception as exc:
        raise RuntimeError("Qiskit is required for the V2 quantum kernel.") from exc
    fmap = ZZFeatureMap(feature_dimension=Z.shape[1], reps=2, entanglement="linear")
    states = []
    for row in Z:
        qc = fmap.assign_parameters(row, inplace=False)
        states.append(np.asarray(Statevector.from_instruction(qc).data, complex))
    return np.vstack(states)


def fidelity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs(A @ B.conj().T)**2


def fit_calibrator(scores: np.ndarray, y: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(solver="lbfgs", random_state=SEED)
    lr.fit(np.asarray(scores).reshape(-1,1), y)
    return lr


def eval_kernel_model(name: str, Ktr: np.ndarray, Kvc: np.ndarray, Kte: np.ndarray,
                      ytr: np.ndarray, yvc: np.ndarray, yte: np.ndarray):
    model = SVC(kernel="precomputed", C=10, class_weight="balanced", probability=False, random_state=SEED)
    start = time.perf_counter(); model.fit(Ktr, ytr); train_sec = time.perf_counter()-start
    dvc = model.decision_function(Kvc)
    cal = fit_calibrator(dvc, yvc)
    pvc = cal.predict_proba(dvc.reshape(-1,1))[:,1]
    threshold = choose_threshold(yvc, pvc)
    start = time.perf_counter(); dte = model.decision_function(Kte); infer_sec = time.perf_counter()-start
    pte = cal.predict_proba(dte.reshape(-1,1))[:,1]
    m = metrics(yte, pte, threshold)
    m.update({"model": name, "train_seconds": train_sec,
              "infer_ms_per_sample": 1000*infer_sec/max(len(yte),1)})
    return m, model, cal, pte


def run_hcqkl(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
              out: Paths, fast: bool):
    feats = [c for c in EARLY_FEATURES if c in train.columns]
    vs, vc = split_val(val)

    # Fast mode is still large enough to contain useful positive support.
    tr_max = 600 if fast else KERNEL_TRAIN_MAX
    pos_max = 150 if fast else KERNEL_TRAIN_POSITIVE_MAX
    vs_max = 600 if fast else KERNEL_VAL_SELECT_MAX
    vc_max = 600 if fast else KERNEL_VAL_CAL_MAX
    te_max = 1500 if fast else KERNEL_TEST_MAX

    tr = enriched_kernel_train_sample(train, tr_max, pos_max, SEED)
    # Validation/test preserve natural prevalence; no class rebalancing here.
    vs = natural_subsample(vs, vs_max, SEED + 1)
    vc = natural_subsample(vc, vc_max, SEED + 2)
    te = natural_subsample(test, te_max, SEED + 3)

    for split_name, split_df in [("train_enriched", tr), ("val_select", vs),
                                 ("val_cal", vc), ("test_natural", te)]:
        pd.DataFrame([{
            "split": split_name,
            "rows": len(split_df),
            "positives": int(split_df["future_sla_event"].sum()),
            "prevalence": float(split_df["future_sla_event"].mean()),
            "time_min": float(split_df[TIME_COLUMN].min()),
            "time_max": float(split_df[TIME_COLUMN].max()),
        }]).to_csv(out.prediction / f"kernel_{split_name}_summary.csv", index=False)

    imp, sc = SimpleImputer(strategy="median"), RobustScaler()
    Xtr = sc.fit_transform(imp.fit_transform(numeric(tr, feats)))
    Xvs = sc.transform(imp.transform(numeric(vs, feats)))
    Xvc = sc.transform(imp.transform(numeric(vc, feats)))
    Xte = sc.transform(imp.transform(numeric(te, feats)))

    ncomp = min(PCA_COMPONENTS, Xtr.shape[1], Xtr.shape[0] - 1)
    pca = PCA(n_components=ncomp, whiten=True, random_state=SEED)
    Ztr = pca.fit_transform(Xtr)
    Zvs = pca.transform(Xvs)
    Zvc = pca.transform(Xvc)
    Zte = pca.transform(Xte)

    ang = MinMaxScaler(feature_range=(0, np.pi))
    Ztr = ang.fit_transform(Ztr)
    Zvs = ang.transform(Zvs)
    Zvc = ang.transform(Zvc)
    Zte = ang.transform(Zte)

    gamma = 1 / max(1, Ztr.shape[1])
    Kr_tr = rbf_kernel(Ztr, Ztr, gamma=gamma)
    Kr_vs = rbf_kernel(Zvs, Ztr, gamma=gamma)
    Kr_vc = rbf_kernel(Zvc, Ztr, gamma=gamma)
    Kr_te = rbf_kernel(Zte, Ztr, gamma=gamma)

    # Evaluate the ideal fidelity kernel through statevector simulation.
    Str = statevectors(Ztr)
    Svs = statevectors(Zvs)
    Svc = statevectors(Zvc)
    Ste = statevectors(Zte)
    Kq_tr = fidelity(Str, Str)
    Kq_vs = fidelity(Svs, Str)
    Kq_vc = fidelity(Svc, Str)
    Kq_te = fidelity(Ste, Str)

    for name, arr in {
        "rbf_train": Kr_tr, "rbf_val_select": Kr_vs,
        "rbf_val_cal": Kr_vc, "rbf_test": Kr_te,
        "quantum_train": Kq_tr, "quantum_val_select": Kq_vs,
        "quantum_val_cal": Kq_vc, "quantum_test": Kq_te,
    }.items():
        np.save(out.kernels / f"{name}.npy", arr)

    ytr = tr["future_sla_event"].to_numpy(int)
    yvs = vs["future_sla_event"].to_numpy(int)
    yvc = vc["future_sla_event"].to_numpy(int)
    yte = te["future_sla_event"].to_numpy(int)

    if len(np.unique(yvs)) < 2 or len(np.unique(yvc)) < 2 or len(np.unique(yte)) < 2:
        raise RuntimeError(
            "Natural validation/test kernel subsets contain too few positive events. "
            "Increase KERNEL_VAL_SELECT_MAX/KERNEL_VAL_CAL_MAX/KERNEL_TEST_MAX."
        )

    alpha_rows = []
    for alpha in ALPHA_GRID:
        Khtr = alpha * Kr_tr + (1 - alpha) * Kq_tr
        Khvs = alpha * Kr_vs + (1 - alpha) * Kq_vs
        mod = SVC(kernel="precomputed", C=10, class_weight="balanced",
                  random_state=SEED)
        mod.fit(Khtr, ytr)
        d = mod.decision_function(Khvs)
        ra = roc_auc_score(yvs, d)
        pa = average_precision_score(yvs, d)
        alpha_rows.append({
            "alpha_classical_rbf": float(alpha),
            "alpha_quantum": float(1 - alpha),
            "validation_roc_auc": float(ra),
            "validation_pr_auc": float(pa),
            "selection_score": float(0.5 * ra + 0.5 * pa),
        })

    alpha_df = pd.DataFrame(alpha_rows).sort_values(
        ["selection_score", "validation_roc_auc", "validation_pr_auc"],
        ascending=False,
    ).reset_index(drop=True)
    alpha_df.to_csv(out.prediction / "hcqkl_alpha_sweep.csv", index=False)
    best_alpha = float(alpha_df.iloc[0]["alpha_classical_rbf"])

    rows = []
    pred = pd.DataFrame({
        "record_id": te["record_id"].to_numpy() if "record_id" in te else np.arange(len(te)),
        TIME_COLUMN: te[TIME_COLUMN].to_numpy(),
        "y_true": yte,
    })
    for col in ["requested_cpu", "requested_memory", "duration_seconds",
                "scheduling_class", "collection_id", "instance_index"]:
        if col in te:
            pred[col] = te[col].to_numpy()

    m_r, mod_r, cal_r, p_r = eval_kernel_model(
        "RBF_kernel", Kr_tr, Kr_vc, Kr_te, ytr, yvc, yte
    )
    m_q, mod_q, cal_q, p_q = eval_kernel_model(
        "Quantum_fidelity_kernel", Kq_tr, Kq_vc, Kq_te, ytr, yvc, yte
    )
    Kh_tr = best_alpha * Kr_tr + (1 - best_alpha) * Kq_tr
    Kh_vc = best_alpha * Kr_vc + (1 - best_alpha) * Kq_vc
    Kh_te = best_alpha * Kr_te + (1 - best_alpha) * Kq_te
    m_h, mod_h, cal_h, p_h = eval_kernel_model(
        "HCQKL", Kh_tr, Kh_vc, Kh_te, ytr, yvc, yte
    )
    m_h["alpha_classical_rbf"] = best_alpha
    m_h["alpha_quantum"] = 1 - best_alpha
    rows += [m_r, m_q, m_h]

    pred["risk_rbf"] = p_r
    pred["risk_quantum"] = p_q
    pred["risk_hcqkl"] = p_h

    res = pd.DataFrame(rows)
    res.to_csv(out.prediction / "matched_kernel_test_metrics.csv", index=False)
    pred.to_csv(out.prediction / "matched_kernel_test_predictions.csv", index=False)

    bundle = {
        "features": feats,
        "imputer": imp,
        "scaler": sc,
        "pca": pca,
        "angle_scaler": ang,
        "rbf_gamma": gamma,
        "best_alpha": best_alpha,
        "rbf_svc": mod_r,
        "rbf_calibrator": cal_r,
        "quantum_svc": mod_q,
        "quantum_calibrator": cal_q,
        "hcqkl_svc": mod_h,
        "hcqkl_calibrator": cal_h,
        "z_train": Ztr,
        "statevectors_train": Str,
        "kernel_train_record_ids": (
            tr["record_id"].to_numpy() if "record_id" in tr else np.arange(len(tr))
        ),
        "quantum_feature_map": {
            "type": "ZZFeatureMap", "reps": 2, "entanglement": "linear",
            "execution": "ideal Qiskit Statevector simulation",
        },
    }
    joblib.dump(bundle, out.models / "hcqkl_v2_bundle.joblib", compress=3)
    save_json({
        "features": feats,
        "best_alpha": best_alpha,
        "train_sampling": "class-enriched training only; matched across all three kernels",
        "train_rows": len(tr),
        "train_positives": int(ytr.sum()),
        "validation_selection_rows": len(vs),
        "validation_selection_positives": int(yvs.sum()),
        "validation_calibration_rows": len(vc),
        "validation_calibration_positives": int(yvc.sum()),
        "test_rows": len(te),
        "test_positives": int(yte.sum()),
        "test_prevalence": float(yte.mean()),
        "note": "validation/test preserve natural prevalence; no 50/50 balancing",
    }, out.prediction / "hcqkl_metadata.json")
    return res, alpha_df, pred, bundle


def score_hcqkl(frame: pd.DataFrame, bundle: dict[str, Any], chunk_size: int = 512) -> np.ndarray:
    """Score arbitrary decision-time rows with the fitted HCQKL model.

    This is used for contiguous scheduling windows so scheduling no longer uses
    the sparsely thinned kernel-test sample.
    """
    feats = bundle["features"]
    if frame.empty:
        return np.array([], dtype=float)

    probabilities = []
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start:start + chunk_size]
        X = bundle["scaler"].transform(
            bundle["imputer"].transform(numeric(chunk, feats))
        )
        Z = bundle["pca"].transform(X)
        Z = bundle["angle_scaler"].transform(Z)

        Kr = rbf_kernel(Z, bundle["z_train"], gamma=bundle["rbf_gamma"])
        S = statevectors(Z)
        Kq = fidelity(S, bundle["statevectors_train"])
        alpha = float(bundle["best_alpha"])
        Kh = alpha * Kr + (1 - alpha) * Kq
        d = bundle["hcqkl_svc"].decision_function(Kh)
        p = bundle["hcqkl_calibrator"].predict_proba(
            np.asarray(d).reshape(-1, 1)
        )[:, 1]
        probabilities.append(p)
    return np.concatenate(probabilities)


# -----------------------------------------------------------------------------
# 3. Classical baselines
# -----------------------------------------------------------------------------
def run_baselines(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                  out: Paths, fast: bool) -> pd.DataFrame:
    feats=[c for c in EARLY_FEATURES if c in train.columns]
    _, vc=split_val(val)
    tr=natural_subsample(train, min(10_000 if fast else CLASSICAL_TRAIN_MAX,len(train)), SEED)
    Xtr,ytr=numeric(tr,feats),tr["future_sla_event"].to_numpy(int)
    Xvc,yvc=numeric(vc,feats),vc["future_sla_event"].to_numpy(int)
    Xte,yte=numeric(test,feats),test["future_sla_event"].to_numpy(int)
    models={
        "HistGradientBoosting":Pipeline([("imp",SimpleImputer(strategy="median")),("sc",RobustScaler()),
            ("clf",HistGradientBoostingClassifier(learning_rate=.05,max_iter=250,max_leaf_nodes=31,
             min_samples_leaf=20,l2_regularization=1.0,random_state=SEED))]),
        "RandomForest":Pipeline([("imp",SimpleImputer(strategy="median")),
            ("clf",RandomForestClassifier(n_estimators=100 if fast else 300,min_samples_leaf=2,
             class_weight="balanced_subsample",n_jobs=-1,random_state=SEED))]),
        "MLP":Pipeline([("imp",SimpleImputer(strategy="median")),("sc",RobustScaler()),
            ("clf",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=60 if fast else 150,
             early_stopping=True,random_state=SEED))]),
    }
    try:
        from xgboost import XGBClassifier
        ratio=(ytr==0).sum()/max((ytr==1).sum(),1)
        models["XGBoost"]=Pipeline([("imp",SimpleImputer(strategy="median")),
            ("clf",XGBClassifier(n_estimators=120 if fast else 350,max_depth=6,learning_rate=.05,
             subsample=.9,colsample_bytree=.9,eval_metric="logloss",scale_pos_weight=ratio,
             random_state=SEED,n_jobs=-1))])
    except Exception: pass

    rows=[]
    for name,model in models.items():
        start=time.perf_counter()
        if name=="HistGradientBoosting":
            model.fit(Xtr,ytr,clf__sample_weight=compute_sample_weight("balanced",ytr))
        else: model.fit(Xtr,ytr)
        trsec=time.perf_counter()-start
        pvc=model.predict_proba(Xvc)[:,1]; t=choose_threshold(yvc,pvc)
        start=time.perf_counter(); pte=model.predict_proba(Xte)[:,1]; inf=time.perf_counter()-start
        m=metrics(yte,pte,t); m.update({"model":name,"train_seconds":trsec,
                                       "infer_ms_per_sample":1000*inf/max(len(yte),1)})
        rows.append(m); joblib.dump({"model":model,"features":feats,"threshold":t},
                                    out.models/f"baseline_{name.lower()}.joblib",compress=3)
    ans=pd.DataFrame(rows); ans.to_csv(out.prediction/"classical_baseline_metrics.csv",index=False)
    return ans


# -----------------------------------------------------------------------------
# 4. Capacity-aware scheduling + external SLO
# -----------------------------------------------------------------------------
@dataclass
class Resource:
    resource_id:int; cpu_capacity:float; memory_capacity:float
    speed:float; energy_rate:float; cost_rate:float


def resources_for(batch: pd.DataFrame, m:int) -> list[Resource]:
    cpu=np.clip(pd.to_numeric(batch["requested_cpu"],errors="coerce").fillna(0).to_numpy(float),1e-6,None)
    mem=np.clip(pd.to_numeric(batch["requested_memory"],errors="coerce").fillna(0).to_numpy(float),1e-6,None)
    ac,am=cpu.sum()/m,mem.sum()/m
    cm=np.linspace(.95,1.55,m); mm=np.linspace(1.10,1.45,m)
    sp=np.linspace(.85,1.45,m); en=np.linspace(.85,1.20,m); co=np.linspace(.75,1.25,m)
    return [Resource(j,max(ac*cm[j]*1.35,cpu.max()),max(am*mm[j]*1.35,mem.max()),sp[j],en[j],co[j]) for j in range(m)]


def train_duration_tables(train: pd.DataFrame):
    d=pd.to_numeric(train["duration_seconds"],errors="coerce"); d=d[(d>0)&np.isfinite(d)]
    med=float(np.median(d)) if len(d) else 60.; slo=float(np.quantile(d,.90)) if len(d) else 300.
    meds={}; slos={}
    if "scheduling_class" in train:
        temp=train.copy(); temp["_d"]=pd.to_numeric(temp["duration_seconds"],errors="coerce")
        for cls,g in temp.groupby("scheduling_class",dropna=False):
            v=g["_d"]; v=v[(v>0)&np.isfinite(v)]
            if len(v)>=50: meds[str(cls)]=float(np.median(v)); slos[str(cls)]=float(np.quantile(v,.90))
    return meds,med,slos,slo


def cost_matrix(batch:pd.DataFrame,res:list[Resource],meds:dict[str,float],global_med:float):
    risk=np.clip(batch["risk_hcqkl"].to_numpy(float),0,1)
    cpu=pd.to_numeric(batch["requested_cpu"],errors="coerce").fillna(0).to_numpy(float)
    mem=pd.to_numeric(batch["requested_memory"],errors="coerce").fillna(0).to_numpy(float)
    cls=batch["scheduling_class"] if "scheduling_class" in batch else pd.Series(["global"]*len(batch))
    dur=np.array([meds.get(str(c),global_med) for c in cls],float)
    n,m=len(batch),len(res); L=np.zeros((n,m));E=np.zeros((n,m));M=np.zeros((n,m));R=np.zeros((n,m))
    for i in range(n):
        for j,r in enumerate(res):
            ex=dur[i]/r.speed; pressure=max(cpu[i]/r.cpu_capacity,mem[i]/r.memory_capacity)
            L[i,j]=ex*(1+.35*pressure); E[i,j]=ex*r.energy_rate; M[i,j]=ex*r.cost_rate; R[i,j]=risk[i]*L[i,j]
    def norm(x):
        lo,hi=x.min(),x.max(); return np.zeros_like(x) if np.isclose(lo,hi) else (x-lo)/(hi-lo)
    return .50*norm(R)+.25*norm(L)+.15*norm(E)+.10*norm(M)


def feasible(a:Sequence[int],batch:pd.DataFrame,res:list[Resource]):
    if len(a)!=len(batch) or any(j<0 or j>=len(res) for j in a): return False
    cpu=pd.to_numeric(batch["requested_cpu"],errors="coerce").fillna(0).to_numpy(float)
    mem=pd.to_numeric(batch["requested_memory"],errors="coerce").fillna(0).to_numpy(float)
    for j,r in enumerate(res):
        ids=[i for i,x in enumerate(a) if x==j]
        if ids and (cpu[ids].sum()>r.cpu_capacity+1e-9 or mem[ids].sum()>r.memory_capacity+1e-9): return False
    return True


def solve_exact(batch:pd.DataFrame,res:list[Resource],C:np.ndarray):
    n,m=C.shape; c=C.ravel(); A=[];lb=[];ub=[]
    for i in range(n):
        row=np.zeros(n*m); row[i*m:(i+1)*m]=1; A.append(row);lb.append(1);ub.append(1)
    cpu=pd.to_numeric(batch["requested_cpu"],errors="coerce").fillna(0).to_numpy(float)
    mem=pd.to_numeric(batch["requested_memory"],errors="coerce").fillna(0).to_numpy(float)
    for j,r in enumerate(res):
        rc=np.zeros(n*m); rm=np.zeros(n*m)
        for i in range(n): rc[i*m+j]=cpu[i]; rm[i*m+j]=mem[i]
        A += [rc,rm]; lb += [-np.inf,-np.inf]; ub += [r.cpu_capacity,r.memory_capacity]
    ans=milp(c=c,integrality=np.ones(n*m,int),bounds=Bounds(np.zeros(n*m),np.ones(n*m)),
             constraints=LinearConstraint(np.vstack(A),np.array(lb),np.array(ub)),options={"time_limit":60})
    if ans.x is None: return None,np.nan,str(ans.message)
    a=np.argmax(ans.x.reshape(n,m),axis=1).astype(int).tolist(); obj=float(C[np.arange(n),a].sum())
    return a,obj,str(ans.message)


def greedy(batch,res,C):
    order=np.argsort(-batch["risk_hcqkl"].to_numpy(float)); cpu=pd.to_numeric(batch["requested_cpu"],errors="coerce").fillna(0).to_numpy(float)
    mem=pd.to_numeric(batch["requested_memory"],errors="coerce").fillna(0).to_numpy(float); uc=np.zeros(len(res));um=np.zeros(len(res));a=[-1]*len(batch)
    for i in order:
        cand=[j for j,r in enumerate(res) if uc[j]+cpu[i]<=r.cpu_capacity+1e-9 and um[j]+mem[i]<=r.memory_capacity+1e-9]
        if not cand:return None
        j=min(cand,key=lambda jj:C[i,jj]);a[i]=j;uc[j]+=cpu[i];um[j]+=mem[i]
    return a


def random_feasible(batch,res,rng):
    for _ in range(500):
        a=rng.integers(0,len(res),size=len(batch)).tolist()
        if feasible(a,batch,res):return a
    return None


def evaluate_queue(batch,res,a,slos,global_slo):
    arr=pd.to_numeric(batch[TIME_COLUMN],errors="coerce").to_numpy(float);arr=arr-np.nanmin(arr)
    dur=pd.to_numeric(batch["duration_seconds"],errors="coerce").to_numpy(float); good=np.isfinite(dur)&(dur>0)
    fill=np.nanmedian(dur[good]) if good.any() else 60.;dur=np.where(good,dur,fill)
    wait=np.zeros(len(batch));finish=np.zeros(len(batch));energy=np.zeros(len(batch));cost=np.zeros(len(batch));miss=np.zeros(len(batch),int)
    for j,r in enumerate(res):
        ids=sorted([i for i,x in enumerate(a) if x==j],key=lambda i:arr[i]);avail=0.
        for i in ids:
            start=max(arr[i],avail);wait[i]=start-arr[i];service=dur[i]/r.speed;finish[i]=start+service;avail=finish[i]
            energy[i]=service*r.energy_rate;cost[i]=service*r.cost_rate
            cls=batch["scheduling_class"].iloc[i] if "scheduling_class" in batch else "global"
            miss[i]=int(finish[i]>arr[i]+slos.get(str(cls),global_slo))
    return {"slo_miss_rate":miss.mean(),"mean_waiting":wait.mean(),"mean_completion":np.mean(finish-arr),
            "makespan":finish.max()-arr.min(),"total_energy":energy.sum(),"total_cost":cost.sum()}


def _qaoa_sampler(seed: int):
    """Support both modern and legacy Qiskit sampler APIs."""
    try:
        from qiskit.primitives import StatevectorSampler
        return StatevectorSampler(default_shots=QAOA_SHOTS, seed=seed), "StatevectorSampler"
    except Exception:
        try:
            from qiskit.primitives import Sampler
            return Sampler(options={"shots": QAOA_SHOTS, "seed": seed}), "Sampler"
        except Exception as exc:
            raise RuntimeError(f"No compatible Qiskit sampler is available: {exc}")


def _quantized_capacity(batch: pd.DataFrame, res: list[Resource], units: int):
    """Small integer capacity encoding for the QAOA constraint converter."""
    cpu = np.clip(pd.to_numeric(batch["requested_cpu"], errors="coerce").fillna(0).to_numpy(float), 1e-12, None)
    mem = np.clip(pd.to_numeric(batch["requested_memory"], errors="coerce").fillna(0).to_numpy(float), 1e-12, None)
    max_cpu = max(r.cpu_capacity for r in res)
    max_mem = max(r.memory_capacity for r in res)

    cpu_q = np.maximum(1, np.ceil(units * cpu / max_cpu)).astype(int)
    mem_q = np.maximum(1, np.ceil(units * mem / max_mem)).astype(int)
    cap_cpu_q = np.maximum(cpu_q.max(), np.floor([units * r.cpu_capacity / max_cpu for r in res])).astype(int)
    cap_mem_q = np.maximum(mem_q.max(), np.floor([units * r.memory_capacity / max_mem for r in res])).astype(int)
    return cpu, mem, cpu_q, mem_q, cap_cpu_q, cap_mem_q


def solve_qaoa(batch, res, C, seed):
    diagnostics = {
        "available": False, "success": False, "seed": int(seed),
        "reps": QAOA_REPS, "shots": QAOA_SHOTS, "maxiter": QAOA_MAXITER,
        "penalty": QAOA_PENALTY, "capacity_units": QAOA_CAPACITY_UNITS,
    }
    try:
        try:
            from qiskit_algorithms.minimum_eigensolvers import QAOA
            from qiskit_algorithms.optimizers import COBYLA
            try:
                from qiskit_algorithms.utils import algorithm_globals
                algorithm_globals.random_seed = seed
            except Exception:
                pass
        except Exception:
            from qiskit.algorithms.minimum_eigensolvers import QAOA
            from qiskit.algorithms.optimizers import COBYLA

        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        sampler, sampler_name = _qaoa_sampler(seed)
        diagnostics.update({"available": True, "sampler": sampler_name})
    except Exception as exc:
        diagnostics["error_stage"] = "imports"
        diagnostics["error"] = repr(exc)
        return diagnostics

    n, m = C.shape
    qp = QuadraticProgram("capacity_assignment_v2_quantized")
    for i in range(n):
        for j in range(m):
            qp.binary_var(name=f"x_{i}_{j}")

    qp.minimize(linear={
        f"x_{i}_{j}": float(C[i, j])
        for i in range(n) for j in range(m)
    })

    for i in range(n):
        qp.linear_constraint(
            linear={f"x_{i}_{j}": 1 for j in range(m)},
            sense="==", rhs=1, name=f"task_{i}",
        )

    cpu, mem, cpu_q, mem_q, cap_cpu_q, cap_mem_q = _quantized_capacity(
        batch, res, QAOA_CAPACITY_UNITS
    )
    diagnostics["quantized_cpu"] = cpu_q.tolist()
    diagnostics["quantized_memory"] = mem_q.tolist()
    diagnostics["quantized_cpu_capacity"] = cap_cpu_q.tolist()
    diagnostics["quantized_memory_capacity"] = cap_mem_q.tolist()

    for j in range(m):
        qp.linear_constraint(
            linear={f"x_{i}_{j}": int(cpu_q[i]) for i in range(n)},
            sense="<=", rhs=int(cap_cpu_q[j]), name=f"cpu_{j}",
        )
        qp.linear_constraint(
            linear={f"x_{i}_{j}": int(mem_q[i]) for i in range(n)},
            sense="<=", rhs=int(cap_mem_q[j]), name=f"mem_{j}",
        )

    try:
        qaoa = QAOA(
            sampler=sampler,
            optimizer=COBYLA(maxiter=QAOA_MAXITER),
            reps=QAOA_REPS,
        )
        opt = MinimumEigenOptimizer(qaoa, penalty=QAOA_PENALTY)
        start = time.perf_counter()
        result = opt.solve(qp)
        elapsed = time.perf_counter() - start
    except Exception as exc:
        diagnostics["error_stage"] = "optimization"
        diagnostics["error"] = repr(exc)
        return diagnostics

    try:
        x = np.asarray(result.x, float).reshape(n, m)
        raw = np.argmax(x, axis=1).astype(int).tolist()
    except Exception as exc:
        diagnostics["error_stage"] = "decode"
        diagnostics["error"] = repr(exc)
        diagnostics["elapsed_seconds"] = elapsed
        return diagnostics

    raw_feasible_actual = feasible(raw, batch, res)
    raw_obj = float(C[np.arange(n), raw].sum())

    # Also report feasibility in the exact integer constraints QAOA received.
    def quantized_feasible(a):
        for j in range(m):
            ids = [i for i, rj in enumerate(a) if rj == j]
            if ids:
                if cpu_q[ids].sum() > cap_cpu_q[j]: return False
                if mem_q[ids].sum() > cap_mem_q[j]: return False
        return True

    raw_feasible_quantized = quantized_feasible(raw)

    # Deterministic repair is explicitly separate from raw QAOA.
    uc = np.zeros(m); um = np.zeros(m); repaired = [-1] * n
    order = np.argsort(-np.max(x, axis=1))
    for i in order:
        j = raw[i]
        if (uc[j] + cpu[i] <= res[j].cpu_capacity + 1e-9 and
                um[j] + mem[i] <= res[j].memory_capacity + 1e-9):
            repaired[i] = j
            uc[j] += cpu[i]; um[j] += mem[i]

    for i in range(n):
        if repaired[i] >= 0:
            continue
        candidates = [
            j for j, rr in enumerate(res)
            if uc[j] + cpu[i] <= rr.cpu_capacity + 1e-9
            and um[j] + mem[i] <= rr.memory_capacity + 1e-9
        ]
        if not candidates:
            diagnostics.update({
                "success": True,
                "elapsed_seconds": elapsed,
                "raw_assignment": raw,
                "raw_feasible_actual": bool(raw_feasible_actual),
                "raw_feasible_quantized": bool(raw_feasible_quantized),
                "raw_objective": raw_obj,
                "repaired_assignment": None,
                "repaired_feasible": False,
                "status": str(result.status),
            })
            return diagnostics
        j = min(candidates, key=lambda jj: C[i, jj])
        repaired[i] = int(j)
        uc[j] += cpu[i]; um[j] += mem[i]

    diagnostics.update({
        "success": True,
        "elapsed_seconds": elapsed,
        "raw_assignment": raw,
        "raw_feasible_actual": bool(raw_feasible_actual),
        "raw_feasible_quantized": bool(raw_feasible_quantized),
        "raw_objective": raw_obj,
        "repaired_assignment": repaired,
        "repaired_feasible": bool(feasible(repaired, batch, res)),
        "repaired_objective": float(C[np.arange(n), repaired].sum()),
        "status": str(result.status),
    })
    return diagnostics


def dense_window_starts(pool: pd.DataFrame, batch_size: int, blocks: int) -> list[int]:
    """Pick short contiguous temporal windows distributed across the test period."""
    n = len(pool)
    if n < batch_size:
        return []
    max_start = n - batch_size
    segment_edges = np.linspace(0, max_start + 1, blocks + 1, dtype=int)
    times = pool[TIME_COLUMN].to_numpy(float)
    starts = []
    for b in range(blocks):
        lo = int(segment_edges[b])
        hi = int(max(lo + 1, segment_edges[b + 1]))
        hi = min(hi, max_start + 1)
        candidates = np.arange(lo, hi, dtype=int)
        if len(candidates) == 0:
            continue
        spans = times[candidates + batch_size - 1] - times[candidates]
        starts.append(int(candidates[int(np.argmin(spans))]))
    return starts


def _append_solver_result(rows, assign, batch, res, C, solver, assignment,
                          batch_size, block, milp_o, milp_status,
                          slos, gslo, extra=None):
    extra = extra or {}
    if assignment is None:
        rows.append({
            "batch_size": batch_size, "block": block, "solver": solver,
            "feasible": False, "objective": np.nan,
            "milp_objective": milp_o, "objective_gap": np.nan,
            "milp_status": milp_status, **extra,
        })
        return

    actual_feasible = feasible(assignment, batch, res)
    obj = float(C[np.arange(len(batch)), assignment].sum())
    ev = evaluate_queue(batch, res, assignment, slos, gslo) if actual_feasible else {
        k: np.nan for k in ["slo_miss_rate", "mean_waiting", "mean_completion",
                            "makespan", "total_energy", "total_cost"]
    }
    rows.append({
        "batch_size": batch_size, "block": block, "solver": solver,
        "feasible": bool(actual_feasible), "objective": obj,
        "milp_objective": milp_o,
        "objective_gap": obj - milp_o if np.isfinite(milp_o) else np.nan,
        "milp_status": milp_status,
        "window_span_seconds": float(batch[TIME_COLUMN].max() - batch[TIME_COLUMN].min()),
        **ev, **extra,
    })
    for i, j in enumerate(assignment):
        assign.append({
            "batch_size": batch_size, "block": block, "solver": solver,
            "task_index": i,
            "record_id": batch["record_id"].iloc[i] if "record_id" in batch else i,
            "risk_hcqkl": batch["risk_hcqkl"].iloc[i],
            "resource_id": j,
            "requested_cpu": batch["requested_cpu"].iloc[i],
            "requested_memory": batch["requested_memory"].iloc[i],
            "duration_seconds": batch["duration_seconds"].iloc[i],
            TIME_COLUMN: batch[TIME_COLUMN].iloc[i],
        })


def run_scheduling(train, test, hcqkl_bundle, out, fast):
    # Scheduling uses contiguous untouched-test rows, not the thinned kernel-test sample.
    req = [TIME_COLUMN, "requested_cpu", "requested_memory", "duration_seconds"]
    missing = [c for c in req if c not in test]
    if missing:
        raise ValueError(f"Scheduling fields missing from temporal test set: {missing}")

    pool = test.copy()
    for c in req:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=req)
    pool = pool[
        (pool.requested_cpu > 0) &
        (pool.requested_memory > 0) &
        (pool.duration_seconds > 0)
    ].sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)

    meds, gmed, slos, gslo = train_duration_tables(train)
    save_json({
        "external_slo": "training-period 90th percentile duration by scheduling class",
        "global_slo_seconds": gslo,
        "class_slo_seconds": slos,
        "test_label_used_in_assignment": False,
        "batch_sampling": "dense contiguous temporal windows from untouched test period",
    }, out.scheduling / "external_slo_definition.json")

    rows = []; assign = []; rng = np.random.default_rng(SEED)
    blocks = 5 if fast else SCHED_BLOCKS
    batch_sizes = [4] if fast else SCHED_BATCH_SIZES

    # Classical capacity-aware experiment on 4/6/8 task instances.
    for bs in batch_sizes:
        for block, start in enumerate(dense_window_starts(pool, bs, blocks)):
            batch = pool.iloc[start:start + bs].copy().reset_index(drop=True)
            batch["risk_hcqkl"] = score_hcqkl(batch, hcqkl_bundle)
            res = resources_for(batch, SCHED_RESOURCES)
            C = cost_matrix(batch, res, meds, gmed)
            milp_a, milp_o, milp_status = solve_exact(batch, res, C)

            candidates = {
                "MILP": milp_a,
                "RiskGreedy": greedy(batch, res, C),
                "RandomFeasible": random_feasible(batch, res, rng),
            }
            save_json([asdict(r) for r in res], out.scheduling / "resource_profiles" / f"bs{bs}_block{block:02d}.json")
            np.save(out.scheduling / f"cost_bs{bs}_block{block:02d}.npy", C)

            for solver, assignment in candidates.items():
                _append_solver_result(
                    rows, assign, batch, res, C, solver, assignment,
                    bs, block, milp_o, milp_status, slos, gslo,
                )

    # Separate QAOA benchmark: 3 tasks over 2 resources permits multiple tasks
    # per resource while keeping the statevector optimization tractable.
    qaoa_rows = []
    q_blocks = 3 if fast else QAOA_BLOCKS
    for block, start in enumerate(dense_window_starts(pool, QAOA_BATCH_SIZE, q_blocks)):
        batch = pool.iloc[start:start + QAOA_BATCH_SIZE].copy().reset_index(drop=True)
        batch["risk_hcqkl"] = score_hcqkl(batch, hcqkl_bundle)
        res = resources_for(batch, QAOA_RESOURCES)
        C = cost_matrix(batch, res, meds, gmed)
        milp_a, milp_o, milp_status = solve_exact(batch, res, C)

        _append_solver_result(
            rows, assign, batch, res, C, "MILP", milp_a,
            QAOA_BATCH_SIZE, block, milp_o, milp_status, slos, gslo,
            extra={"experiment": "qaoa_small_capacity"},
        )

        qi = solve_qaoa(batch, res, C, SEED + 1000 + block)
        save_json(qi, out.qaoa / f"qaoa_bs{QAOA_BATCH_SIZE}_block{block:02d}.json")
        qaoa_rows.append({
            "block": block,
            "available": qi.get("available", False),
            "success": qi.get("success", False),
            "sampler": qi.get("sampler"),
            "error_stage": qi.get("error_stage"),
            "error": qi.get("error"),
            "elapsed_seconds": qi.get("elapsed_seconds"),
            "raw_feasible_actual": qi.get("raw_feasible_actual"),
            "raw_feasible_quantized": qi.get("raw_feasible_quantized"),
            "repaired_feasible": qi.get("repaired_feasible"),
        })

        if qi.get("success"):
            _append_solver_result(
                rows, assign, batch, res, C, "QAOA_raw",
                qi.get("raw_assignment"), QAOA_BATCH_SIZE, block,
                milp_o, milp_status, slos, gslo,
                extra={
                    "experiment": "qaoa_small_capacity",
                    "qaoa_quantized_feasible": qi.get("raw_feasible_quantized"),
                    "qaoa_elapsed_seconds": qi.get("elapsed_seconds"),
                },
            )
            _append_solver_result(
                rows, assign, batch, res, C, "QAOA_repaired",
                qi.get("repaired_assignment"), QAOA_BATCH_SIZE, block,
                milp_o, milp_status, slos, gslo,
                extra={
                    "experiment": "qaoa_small_capacity",
                    "qaoa_elapsed_seconds": qi.get("elapsed_seconds"),
                },
            )
        else:
            # Failure is retained as evidence; it is never silently omitted.
            for solver in ["QAOA_raw", "QAOA_repaired"]:
                _append_solver_result(
                    rows, assign, batch, res, C, solver, None,
                    QAOA_BATCH_SIZE, block, milp_o, milp_status, slos, gslo,
                    extra={
                        "experiment": "qaoa_small_capacity",
                        "qaoa_error": qi.get("error"),
                        "qaoa_error_stage": qi.get("error_stage"),
                    },
                )

    pd.DataFrame(qaoa_rows).to_csv(out.qaoa / "qaoa_execution_summary.csv", index=False)

    rdf = pd.DataFrame(rows)
    adf = pd.DataFrame(assign)
    rdf.to_csv(out.scheduling / "capacity_scheduling_results.csv", index=False)
    adf.to_csv(out.scheduling / "capacity_scheduling_assignments.csv", index=False)
    if not rdf.empty:
        summ = rdf.groupby(["batch_size", "solver"], as_index=False).agg(
            runs=("block", "count"),
            feasibility_rate=("feasible", "mean"),
            mean_window_span_seconds=("window_span_seconds", "mean"),
            mean_objective=("objective", "mean"),
            mean_gap=("objective_gap", "mean"),
            mean_slo_miss=("slo_miss_rate", "mean"),
            mean_waiting=("mean_waiting", "mean"),
            mean_completion=("mean_completion", "mean"),
            mean_makespan=("makespan", "mean"),
            mean_energy=("total_energy", "mean"),
            mean_cost=("total_cost", "mean"),
        )
        summ.to_csv(out.scheduling / "capacity_scheduling_summary.csv", index=False)
    return rdf, adf


# -----------------------------------------------------------------------------
# 5. Statistics
# -----------------------------------------------------------------------------
def bootstrap_ci(d,resamples,seed):
    d=np.asarray(d,float);d=d[np.isfinite(d)]
    if len(d)==0:return np.nan,np.nan
    rng=np.random.default_rng(seed);means=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(resamples)])
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def holm(p):
    p=np.asarray(p,float);out=np.full(len(p),np.nan);valid=np.where(np.isfinite(p))[0];order=valid[np.argsort(p[valid])];running=0
    for rank,idx in enumerate(order):running=max(running,min(1,(len(order)-rank)*p[idx]));out[idx]=running
    return out


def run_stats(rdf,out,fast):
    if rdf.empty:return pd.DataFrame()
    metrics_list=["objective","slo_miss_rate","mean_waiting","mean_completion","makespan","total_energy","total_cost"]
    comps=[("RiskGreedy","MILP"),("RandomFeasible","MILP"),("QAOA_raw","MILP"),("QAOA_repaired","MILP")];rows=[]
    for bs in sorted(rdf.batch_size.unique()):
        s=rdf[rdf.batch_size==bs]
        for a,b in comps:
            for met in metrics_list:
                aa=s[s.solver==a][["block",met]].rename(columns={met:"a"});bb=s[s.solver==b][["block",met]].rename(columns={met:"b"});p=aa.merge(bb,on="block").dropna()
                if p.empty:continue
                d=p.a.to_numpy(float)-p.b.to_numpy(float);lo,hi=bootstrap_ci(d,1000 if fast else BOOTSTRAP_RESAMPLES,SEED+int(bs));nz=d[~np.isclose(d,0)]
                pv=st=np.nan
                if len(nz)>=6:
                    try:w=wilcoxon(nz,alternative="two-sided",zero_method="wilcox");st=float(w.statistic);pv=float(w.pvalue)
                    except Exception:pass
                sd=np.std(d,ddof=1) if len(d)>1 else np.nan;dz=0 if np.isfinite(sd) and np.isclose(sd,0) else (np.mean(d)/sd if np.isfinite(sd) else np.nan)
                rows.append({"batch_size":bs,"solver_a":a,"solver_b":b,"metric":met,"pairs":len(p),"mean_a":p.a.mean(),"mean_b":p.b.mean(),"mean_diff":d.mean(),"median_diff":np.median(d),"bootstrap95_low":lo,"bootstrap95_high":hi,"paired_cohens_dz":dz,"wilcoxon_stat":st,"p_raw":pv})
    ans=pd.DataFrame(rows)
    if not ans.empty:ans["p_holm"]=holm(ans.p_raw.to_numpy(float));ans["significant_holm_0_05"]=ans.p_holm<.05;ans.to_csv(out.stats/"paired_block_statistics.csv",index=False)
    return ans


# -----------------------------------------------------------------------------
# 6. Independent QASM + iQuantum policy comparison
# -----------------------------------------------------------------------------
def generate_qasm(risks,out,fast):
    from qiskit import QuantumCircuit, qasm2
    rng=np.random.default_rng(SEED);n=12 if fast else min(60,max(20,len(risks)));rv=np.resize(np.asarray(risks,float) if len(risks) else np.linspace(.05,.95,n),n)
    qcycle=[4,6,8,10,12];dcycle=[16,24,32,40,48];scycle=[256,512,1024,512,256];rows=[]
    for tid in range(n):
        q=qcycle[tid%5];depth=dcycle[tid%5];shots=scycle[tid%5];qc=QuantumCircuit(q,q)
        for layer in range(depth):
            for qb in range(q):
                choice=(tid+layer+qb)%3
                if choice==0:qc.sx(qb)
                elif choice==1:qc.rz(float(rng.uniform(-np.pi,np.pi)),qb)
                else:qc.x(qb)
            for qb in range(layer%2,q-1,2):qc.cx(qb,qb+1)
        qc.measure(range(q),range(q));qp=out.qasm/f"task_{tid:03d}.qasm";qasm2.dump(qc,str(qp))
        rows.append({"task_id":tid,"source_risk":float(rv[tid]),"qasm_file":str(qp),"num_qubits":q,"num_layers":int(qc.depth()),"num_shots":shots,"gate_set":"CX|ID|RZ|SX|X","application":"AQUA-SLA-V2-independent-QASM"})
    df=pd.DataFrame(rows);df.to_csv(out.iquantum/"independent_qasm_manifest.csv",index=False);return df


def qnode_policy(df, policy, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    # Same heterogeneous nodes for every policy. Circuit complexity is already
    # fixed and remains independent of SLA risk.
    caps = {0: 7, 1: 27}
    speed = {0: 1.0, 1: 2.0}
    load = {0: 0.0, 1: 0.0}
    assigned = {}

    order = (
        out.sort_values("source_risk", ascending=False).index
        if policy == "risk_aware" else out.index
    )
    for idx in order:
        q = int(out.loc[idx, "num_qubits"])
        work = float(out.loc[idx, "num_layers"] * out.loc[idx, "num_shots"] * q)
        candidates = [n for n, c in caps.items() if q <= c]
        if not candidates:
            assigned[idx] = -1
            continue

        if policy == "random":
            node = int(rng.choice(candidates))
        elif policy == "least_loaded":
            node = min(candidates, key=lambda n: load[n] / speed[n])
        elif policy == "compatibility_aware":
            # Prefer the smallest compatible QNode.
            node = min(candidates, key=lambda n: caps[n])
        elif policy == "risk_aware":
            risk = float(out.loc[idx, "source_risk"])
            # Risk changes placement only, never qubits/depth/shots. High-risk
            # tasks prefer the faster compatible node; lower-risk tasks use the
            # projected least-loaded compatible node.
            if risk >= 0.50 and 1 in candidates:
                node = 1
            else:
                node = min(candidates, key=lambda n: load[n] / speed[n])
        else:
            raise ValueError(f"Unknown iQuantum policy: {policy}")

        assigned[idx] = node
        load[node] += work / speed[node]

    out["policy"] = policy
    out["preferred_qnode"] = [assigned[i] for i in out.index]
    return out


def find_col(cols,all_tokens,any_tokens=()):
    for c in cols:
        low=c.lower()
        if all(t in low for t in all_tokens) and (not any_tokens or any(t in low for t in any_tokens)):return c
    return None


def bridge_csv(policy_df,template_path,out_csv):
    if not template_path.exists():policy_df.to_csv(out_csv,index=False);return False,"existing iQuantum workload template missing"
    temp=pd.read_csv(template_path)
    if temp.empty:policy_df.to_csv(out_csv,index=False);return False,"existing iQuantum workload template empty"
    reps=math.ceil(len(policy_df)/len(temp));a=pd.concat([temp]*reps,ignore_index=True).iloc[:len(policy_df)].copy();cols=list(a.columns)
    mapping={
        "task_id":semantic(cols,["task","id"]) or semantic(cols,["id"]),
        "num_qubits":semantic(cols,["qubit"]),
        "num_layers":semantic(cols,["layer"]) or semantic(cols,["depth"]),
        "num_shots":semantic(cols,["shot"]),
        "gate_set":semantic(cols,["gate"]),"application":semantic(cols,["app"]),
        "preferred_qnode":semantic(cols,["preferred"],["node","qnode"]) or semantic(cols,["qnode"],["id"]),
        "source_risk":semantic(cols,["risk"]) or semantic(cols,["prob"]),
    }
    if any(mapping[k] is None for k in ["num_qubits","num_layers","num_shots","preferred_qnode"]):
        policy_df.to_csv(out_csv,index=False);return False,f"could not infer bridge schema; mapping={mapping}; columns={cols}"
    for logical,col in mapping.items():
        if col and logical in policy_df:a[col]=policy_df[logical].to_numpy()
    a.to_csv(out_csv,index=False);save_json(mapping,out_csv.with_suffix(".schema_mapping.json"));return True,"ok"


def locate_maven():
    for x in [shutil.which("mvn.cmd"),shutil.which("mvn"),r"C:\Program Files\Apache\maven\bin\mvn.cmd"]:
        if x and Path(x).exists():return str(x)
    return None


def run_bridge(inp,out_csv,log):
    mvn=locate_maven();pom=IQUANTUM_HOME/"modules"/"iquantum-examples"/"pom.xml";status={"input":str(inp),"output":str(out_csv),"executed":False,"success":False,"maven":mvn,"pom":str(pom)}
    if not mvn or not pom.exists():status["reason"]="Maven or iQuantum examples pom unavailable";return status
    cmd=[mvn,"-q","-f",str(pom),"compile","exec:java",f"-Dexec.mainClass={IQUANTUM_BRIDGE_CLASS}",f'-Dexec.args="{inp}" "{out_csv}"']
    start=time.perf_counter();p=subprocess.run(cmd,cwd=str(IQUANTUM_HOME),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace");elapsed=time.perf_counter()-start
    log.write_text(p.stdout or "",encoding="utf-8",errors="replace");status.update({"executed":True,"returncode":p.returncode,"elapsed_seconds":elapsed,"command":cmd,"result_exists":out_csv.exists(),"success":p.returncode==0 and out_csv.exists()});return status


def summarize_iq(policy,csv,status):
    row={"policy":policy,"executed":status.get("executed",False),"success":status.get("success",False)}
    if not row["success"] or not csv.exists():return row
    d=pd.read_csv(csv);cols=list(d.columns);wait=semantic(cols,["waiting"]) or semantic(cols,["wait"]);qpu=semantic(cols,["actual","qpu"]) or semantic(cols,["qpu","time"]);finish=semantic(cols,["finish"]);cost=semantic(cols,["cost"]);node=semantic(cols,["qnode"]) or semantic(cols,["resource","id"])
    row["tasks_returned"]=len(d)
    if wait:row["mean_waiting_time"]=pd.to_numeric(d[wait],errors="coerce").mean()
    if qpu:row["mean_qpu_time"]=pd.to_numeric(d[qpu],errors="coerce").mean()
    if finish:row["makespan"]=pd.to_numeric(d[finish],errors="coerce").max()
    if cost:row["total_cost"]=pd.to_numeric(d[cost],errors="coerce").sum()
    if node:row["qnode_distribution"]=json.dumps(d[node].value_counts(dropna=False).to_dict())
    return row


def run_iquantum(risks, out, fast):
    rows = []
    seeds = [IQUANTUM_SEEDS[0]] if fast else IQUANTUM_SEEDS
    workload_sizes = [12] if fast else IQUANTUM_WORKLOAD_SIZES

    for workload_size in workload_sizes:
        for workload_seed in seeds:
            # Generate one independent workload and reuse it for all policies.
            # generate_qasm currently chooses n from the risk vector, so resize
            # the risk vector to the requested workload size and use a seed-local
            # subdirectory for artifacts.
            workload_dir = out.iquantum / f"workload_n{workload_size}_seed{workload_seed}"
            workload_dir.mkdir(parents=True, exist_ok=True)
            qasm_dir = workload_dir / "qasm"
            qasm_dir.mkdir(parents=True, exist_ok=True)

            # Local independent-QASM generator, seeded by workload_seed.
            try:
                from qiskit import QuantumCircuit, qasm2
            except Exception as exc:
                rows.append({
                    "workload_size": workload_size, "workload_seed": workload_seed,
                    "policy": "generation", "executed": False, "success": False,
                    "reason": repr(exc),
                })
                continue

            rng = np.random.default_rng(workload_seed)
            rv = np.resize(
                np.asarray(risks, float) if len(risks) else np.linspace(.05, .95, workload_size),
                workload_size,
            )
            qcycle = [4, 6, 8, 10, 12]
            dcycle = [16, 24, 32, 40, 48]
            scycle = [256, 512, 1024, 512, 256]
            manifest_rows = []
            for tid in range(workload_size):
                q = qcycle[(tid + workload_seed) % len(qcycle)]
                target_depth = dcycle[(tid + workload_seed) % len(dcycle)]
                shots = scycle[(tid + workload_seed) % len(scycle)]
                qc = QuantumCircuit(q, q)
                for layer in range(target_depth):
                    for qb in range(q):
                        choice = (tid + layer + qb + workload_seed) % 3
                        if choice == 0: qc.sx(qb)
                        elif choice == 1: qc.rz(float(rng.uniform(-np.pi, np.pi)), qb)
                        else: qc.x(qb)
                    for qb in range(layer % 2, q - 1, 2):
                        qc.cx(qb, qb + 1)
                qc.measure(range(q), range(q))
                qpath = qasm_dir / f"task_{tid:03d}.qasm"
                qasm2.dump(qc, str(qpath))
                manifest_rows.append({
                    "task_id": tid,
                    "source_risk": float(rv[tid]),
                    "qasm_file": str(qpath),
                    "num_qubits": q,
                    "num_layers": int(qc.depth()),
                    "num_shots": shots,
                    "gate_set": "CX|ID|RZ|SX|X",
                    "application": "AQUA-SLA-V2-independent-QASM",
                })
            manifest = pd.DataFrame(manifest_rows)
            manifest.to_csv(workload_dir / "independent_qasm_manifest.csv", index=False)

            for k, policy in enumerate(IQUANTUM_POLICIES):
                d = workload_dir / policy
                d.mkdir(parents=True, exist_ok=True)
                pf = qnode_policy(manifest, policy, workload_seed + 100 * k)
                pf.to_csv(d / "policy_manifest.csv", index=False)
                inp = d / "iquantum_workload.csv"
                ok, msg = bridge_csv(pf, OLD_IQUANTUM_WORKLOAD, inp)
                res = d / "iquantum_task_results.csv"
                status = run_bridge(inp, res, d / "maven_execution.log") if ok else {
                    "executed": False, "success": False, "reason": msg
                }
                save_json(status, d / "execution_status.json")
                summary = summarize_iq(policy, res, status)
                summary["workload_size"] = workload_size
                summary["workload_seed"] = workload_seed
                rows.append(summary)

    ans = pd.DataFrame(rows)
    ans.to_csv(out.iquantum / "iquantum_policy_summary.csv", index=False)
    if not ans.empty and "makespan" in ans.columns:
        numeric_cols = [c for c in ["mean_waiting_time", "mean_qpu_time", "makespan", "total_cost"] if c in ans.columns]
        agg = ans[ans.get("success", False) == True].groupby("policy", as_index=False)[numeric_cols].agg(["mean", "std"]) if numeric_cols else pd.DataFrame()
        if not agg.empty:
            agg.to_csv(out.iquantum / "iquantum_policy_aggregate.csv")
    save_json({
        "workload_complexity": "independent of SLA risk",
        "policies": IQUANTUM_POLICIES,
        "workload_sizes": workload_sizes,
        "workload_seeds": seeds,
        "qnodes": {"0_qubits": 7, "1_qubits": 27},
        "risk_aware_rule": "risk affects placement only; high-risk compatible tasks prefer faster QNode",
        "claim_boundary": "iQuantum is discrete-event simulation; preferred-node match is not a quality metric",
    }, out.iquantum / "protocol.json")
    return ans


# -----------------------------------------------------------------------------
# 7. Figures
# -----------------------------------------------------------------------------
def figures(alpha,kres,bres,sres,iqres,out):
    try:import matplotlib.pyplot as plt
    except Exception:return
    if not alpha.empty:
        x=alpha.sort_values("alpha_classical_rbf");fig,ax=plt.subplots(figsize=(7,4));ax.plot(x.alpha_classical_rbf,x.selection_score,marker="o");ax.set(xlabel=r"$\alpha$ (RBF weight)",ylabel="Validation selection score",title="HCQKL blending sensitivity");ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out.figures/"hcqkl_alpha_sensitivity.png",dpi=300);plt.close(fig)
    comb=pd.concat([kres[["model","roc_auc"]] if not kres.empty else pd.DataFrame(),bres[["model","roc_auc"]] if not bres.empty else pd.DataFrame()],ignore_index=True)
    if not comb.empty:
        fig,ax=plt.subplots(figsize=(8,4.5));ax.barh(comb.model,comb.roc_auc);ax.set_xlim(0,1);ax.set_xlabel("ROC-AUC");ax.set_title("Future-horizon prediction");fig.tight_layout();fig.savefig(out.figures/"prediction_comparison.png",dpi=300);plt.close(fig)
    if not sres.empty:
        z=sres.dropna(subset=["slo_miss_rate"]).groupby(["batch_size","solver"],as_index=False).slo_miss_rate.mean();fig,ax=plt.subplots(figsize=(8,4.5))
        for sol,g in z.groupby("solver"):ax.plot(g.batch_size,g.slo_miss_rate,marker="o",label=sol)
        ax.set(xlabel="Tasks per instance",ylabel="External SLO miss rate",title="Capacity-aware scheduling");ax.legend(fontsize=8);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out.figures/"scheduling_slo_miss.png",dpi=300);plt.close(fig)
    if not iqres.empty and "makespan" in iqres:
        z=iqres.dropna(subset=["makespan"]).groupby("policy",as_index=False).makespan.mean()
        if not z.empty:
            fig,ax=plt.subplots(figsize=(7,4));ax.bar(z.policy,z.makespan);ax.set_ylabel("Mean simulated makespan");ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(out.figures/"iquantum_policy_makespan.png",dpi=300);plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["full","prediction","scheduling","iquantum"],default="full");ap.add_argument("--run-name",default="review_v2");ap.add_argument("--fast",action="store_true");args=ap.parse_args()
    out=Paths.create(args.run_name);print("AQUA-SLA V2 output:",out.root);print("V1 folders are not modified.")
    if not PREPARED_DATA.exists():raise FileNotFoundError(f"Prepared dataset not found: {PREPARED_DATA}")
    raw=pd.read_csv(PREPARED_DATA,low_memory=False);primary,meta=build_future_target(raw,PRIMARY_HORIZON_SECONDS);primary.to_csv(out.target/"future_target_dataset.csv",index=False);save_json(meta,out.target/"future_target_metadata.json");horizon_sensitivity(raw,out,args.fast)
    train,val,test=temporal_split(primary);pd.DataFrame([{"split":n,"rows":len(d),"positives":int(d.future_sla_event.sum()),"prevalence":float(d.future_sla_event.mean()),"time_min":d[TIME_COLUMN].min(),"time_max":d[TIME_COLUMN].max()} for n,d in [("train",train),("validation",val),("test",test)]]).to_csv(out.target/"temporal_split_summary.csv",index=False)
    save_json({"project":"AQUA-SLA","version":"review-revision-v2","created":datetime.now().isoformat(),"python":sys.version,"platform":platform.platform(),"output":str(out.root),"mode":args.mode,"fast":args.fast,"no_overwrite":True,"primary_horizon_seconds":PRIMARY_HORIZON_SECONDS,"target":meta},out.manifest/"run_manifest.json")

    kres=alpha=pred=bres=sres=assign=stats=iq=pd.DataFrame(); hcqkl_bundle=None
    if args.mode in {"full","prediction","scheduling","iquantum"}:
        kres,alpha,pred,hcqkl_bundle=run_hcqkl(train,val,test,out,args.fast);bres=run_baselines(train,val,test,out,args.fast)
    if args.mode in {"full","scheduling"}:
        sres,assign=run_scheduling(train,test,hcqkl_bundle,out,args.fast);stats=run_stats(sres,out,args.fast)
    if args.mode in {"full","iquantum"}:
        risks=pred.risk_hcqkl.to_numpy(float) if "risk_hcqkl" in pred else np.array([]);iq=run_iquantum(risks,out,args.fast)
    figures(alpha,kres,bres,sres,iq,out)
    qaoa_summary_path=out.qaoa/"qaoa_execution_summary.csv"
    qaoa_summary=pd.read_csv(qaoa_summary_path) if qaoa_summary_path.exists() else pd.DataFrame()
    iq_successes=int(iq["success"].fillna(False).astype(bool).sum()) if (not iq.empty and "success" in iq.columns) else 0
    save_json({
        "completed":True,"output":str(out.root),"kernel_models":len(kres),
        "scheduling_rows":len(sres),"statistics_rows":len(stats),
        "iquantum_rows":len(iq),"iquantum_successful_runs":iq_successes,
        "qaoa_attempts":len(qaoa_summary),
        "qaoa_successes":int(qaoa_summary["success"].fillna(False).astype(bool).sum()) if (not qaoa_summary.empty and "success" in qaoa_summary) else 0,
        "notes":[
            "Check each iQuantum execution_status.json before making platform claims.",
            "Raw and repaired QAOA are separate rows and failed QAOA attempts are retained.",
            "Scheduling batches use dense contiguous untouched-test windows.",
            "Matched kernels use class-enriched training but natural-prevalence validation/test.",
            "All quantum optimization/kernel results are simulator based unless separately executed on hardware."
        ]},out.manifest/"final_status.json")
    (out.manifest/"README_V2_RESULTS.txt").write_text("AQUA-SLA reviewer revision V2. Use matched_kernel_test_metrics.csv, capacity_scheduling_results.csv, paired_block_statistics.csv, and iquantum_policy_summary.csv as the primary V2 evidence.\n",encoding="utf-8")
    print("Finished. Inspect:",out.manifest/"final_status.json")


# =============================================================================
# FINAL COMPLETION / RESUME PATCH
# =============================================================================
# This section intentionally overrides selected V2 functions above.  It keeps
# all completed prediction logic unchanged, fixes the iQuantum schema helper,
# checkpoints classical scheduling BEFORE QAOA, bounds every QAOA call with a
# hard wall-clock timeout, and adds a --resume-run / --mode remaining workflow.
#
# The QAOA benchmark remains capacity-aware but is deliberately small:
# two tasks, two resources, cumulative CPU/memory inequalities, p=1.  Unlike
# the original V1 matching formulation, both tasks may share one resource when
# capacity permits.  The larger 4/6/8-task capacity-aware study remains the
# principal classical scheduling experiment.

import multiprocessing as _mp
import queue as _queue
import traceback as _traceback

# Fix two late-stage helper calls in the supplied V2 file.  The implemented
# helper is find_col(); bridge_csv() and summarize_iq() referred to semantic().
semantic = find_col

# Bounded QAOA configuration.  These values are chosen to keep the simulator
# experiment tractable while retaining a genuine constrained QAOA solve.
QAOA_BATCH_SIZE = 2
QAOA_RESOURCES = 2
QAOA_BLOCKS = 10
QAOA_CAPACITY_UNITS = 2
QAOA_REPS = 1
QAOA_SHOTS = 128
QAOA_MAXITER = 10
QAOA_PENALTY = 20.0
QAOA_TIMEOUT_SECONDS = 120

# Keep a handle to the fully implemented solver defined earlier in this file.
_ORIGINAL_SOLVE_QAOA = solve_qaoa


def existing_paths(root: str | Path) -> Paths:
    """Return the standard V2 path layout for an existing timestamped run."""
    root = Path(root)
    obj = Paths(
        root=root,
        manifest=root / "00_manifest",
        target=root / "01_future_target",
        prediction=root / "02_prediction_hcqkl",
        models=root / "03_models",
        kernels=root / "04_kernel_artifacts",
        scheduling=root / "05_capacity_scheduling",
        qaoa=root / "06_qaoa",
        stats=root / "07_statistics",
        iquantum=root / "08_iquantum_policies",
        qasm=root / "08_iquantum_policies" / "qasm",
        figures=root / "09_figures",
        logs=root / "10_logs",
    )
    for p in asdict(obj).values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return obj


def _stage_status_path(out: Paths) -> Path:
    return out.manifest / "stage_status.json"


def update_stage_status(out: Paths, stage: str, state: str, **details) -> None:
    path = _stage_status_path(out)
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    else:
        current = {}
    current[stage] = {
        "state": state,
        "time": datetime.now().isoformat(),
        **details,
    }
    save_json(current, path)


def load_completed_prediction(out: Paths):
    """Load prediction artifacts from a prior partial V2 run."""
    target_csv = out.target / "future_target_dataset.csv"
    bundle_path = out.models / "hcqkl_v2_bundle.joblib"
    if not target_csv.exists():
        raise FileNotFoundError(f"Missing future target dataset: {target_csv}")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing HCQKL bundle: {bundle_path}")

    primary = pd.read_csv(target_csv, low_memory=False)
    train, val, test = temporal_split(primary)
    bundle = joblib.load(bundle_path)

    def csv_or_empty(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    kres = csv_or_empty(out.prediction / "matched_kernel_test_metrics.csv")
    alpha = csv_or_empty(out.prediction / "hcqkl_alpha_sweep.csv")
    pred = csv_or_empty(out.prediction / "matched_kernel_test_predictions.csv")
    bres = csv_or_empty(out.prediction / "classical_baseline_metrics.csv")
    return primary, train, val, test, kres, alpha, pred, bres, bundle


def _qaoa_worker(result_queue, batch, resource_dicts, C, seed):
    """Child-process worker used so a slow QAOA call cannot block the run."""
    try:
        resources = [Resource(**d) for d in resource_dicts]
        answer = _ORIGINAL_SOLVE_QAOA(batch, resources, C, seed)
        result_queue.put({"ok": True, "answer": answer})
    except BaseException as exc:
        result_queue.put({
            "ok": False,
            "error": repr(exc),
            "traceback": _traceback.format_exc(),
        })


def solve_qaoa_bounded(batch, res, C, seed: int, timeout_seconds: int):
    """Run one QAOA instance in a spawned process with a hard timeout."""
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_qaoa_worker,
        args=(q, batch, [asdict(r) for r in res], C, seed),
        daemon=False,
    )
    started = time.perf_counter()
    proc.start()
    proc.join(timeout_seconds)
    elapsed = time.perf_counter() - started

    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        return {
            "available": True,
            "success": False,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": elapsed,
            "seed": int(seed),
            "reps": QAOA_REPS,
            "shots": QAOA_SHOTS,
            "maxiter": QAOA_MAXITER,
            "penalty": QAOA_PENALTY,
            "capacity_units": QAOA_CAPACITY_UNITS,
            "error_stage": "wall_clock_timeout",
            "error": f"QAOA exceeded {timeout_seconds} s and was terminated.",
        }

    try:
        payload = q.get(timeout=5)
    except _queue.Empty:
        return {
            "available": True,
            "success": False,
            "timed_out": False,
            "elapsed_seconds": elapsed,
            "seed": int(seed),
            "error_stage": "worker_no_result",
            "error": f"QAOA worker exited with code {proc.exitcode} without returning a result.",
        }

    if not payload.get("ok"):
        return {
            "available": True,
            "success": False,
            "timed_out": False,
            "elapsed_seconds": elapsed,
            "seed": int(seed),
            "error_stage": "worker_exception",
            "error": payload.get("error"),
            "traceback": payload.get("traceback"),
        }

    answer = payload["answer"]
    answer["timed_out"] = False
    answer["wall_clock_elapsed_seconds"] = elapsed
    answer["timeout_seconds"] = timeout_seconds
    return answer


def _scheduling_pool(test: pd.DataFrame) -> pd.DataFrame:
    req = [TIME_COLUMN, "requested_cpu", "requested_memory", "duration_seconds"]
    missing = [c for c in req if c not in test]
    if missing:
        raise ValueError(f"Scheduling fields missing from temporal test set: {missing}")
    pool = test.copy()
    for c in req:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")
    pool = pool.dropna(subset=req)
    pool = pool[
        (pool.requested_cpu > 0)
        & (pool.requested_memory > 0)
        & (pool.duration_seconds > 0)
    ].sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    return pool


def run_classical_scheduling_checkpointed(train, test, hcqkl_bundle, out, fast):
    """Run and save all 4/6/8-task classical capacity experiments first."""
    pool = _scheduling_pool(test)
    meds, gmed, slos, gslo = train_duration_tables(train)
    save_json({
        "external_slo": "training-period 90th percentile duration by scheduling class",
        "global_slo_seconds": gslo,
        "class_slo_seconds": slos,
        "test_label_used_in_assignment": False,
        "batch_sampling": "dense contiguous temporal windows from untouched test period",
        "note": "Classical results are checkpointed before any QAOA execution.",
    }, out.scheduling / "external_slo_definition.json")

    rows, assign = [], []
    rng = np.random.default_rng(SEED)
    blocks = 5 if fast else SCHED_BLOCKS
    batch_sizes = [4] if fast else SCHED_BATCH_SIZES

    for bs in batch_sizes:
        starts = dense_window_starts(pool, bs, blocks)
        for block, start in enumerate(starts):
            batch = pool.iloc[start:start + bs].copy().reset_index(drop=True)
            batch["risk_hcqkl"] = score_hcqkl(batch, hcqkl_bundle)
            res = resources_for(batch, SCHED_RESOURCES)
            C = cost_matrix(batch, res, meds, gmed)
            milp_a, milp_o, milp_status = solve_exact(batch, res, C)

            candidates = {
                "MILP": milp_a,
                "RiskGreedy": greedy(batch, res, C),
                "RandomFeasible": random_feasible(batch, res, rng),
            }
            save_json(
                [asdict(r) for r in res],
                out.scheduling / "resource_profiles" / f"bs{bs}_block{block:02d}.json",
            )
            np.save(out.scheduling / f"cost_bs{bs}_block{block:02d}.npy", C)

            for solver, assignment in candidates.items():
                _append_solver_result(
                    rows, assign, batch, res, C, solver, assignment,
                    bs, block, milp_o, milp_status, slos, gslo,
                    extra={"experiment": "classical_capacity"},
                )

            # Per-block checkpoint.  A later quantum failure can no longer erase
            # the completed classical scheduling evidence.
            pd.DataFrame(rows).to_csv(
                out.scheduling / "capacity_scheduling_results_classical.csv",
                index=False,
            )
            pd.DataFrame(assign).to_csv(
                out.scheduling / "capacity_scheduling_assignments_classical.csv",
                index=False,
            )

    rdf = pd.DataFrame(rows)
    adf = pd.DataFrame(assign)
    if not rdf.empty:
        summary = rdf.groupby(["batch_size", "solver"], as_index=False).agg(
            runs=("block", "count"),
            feasibility_rate=("feasible", "mean"),
            mean_window_span_seconds=("window_span_seconds", "mean"),
            mean_objective=("objective", "mean"),
            mean_gap=("objective_gap", "mean"),
            mean_slo_miss=("slo_miss_rate", "mean"),
            mean_waiting=("mean_waiting", "mean"),
            mean_completion=("mean_completion", "mean"),
            mean_makespan=("makespan", "mean"),
            mean_energy=("total_energy", "mean"),
            mean_cost=("total_cost", "mean"),
        )
        summary.to_csv(
            out.scheduling / "capacity_scheduling_summary_classical.csv",
            index=False,
        )
    return rdf, adf, pool, (meds, gmed, slos, gslo)


def run_qaoa_benchmark_checkpointed(pool, duration_tables, hcqkl_bundle, out, fast):
    """Run bounded 2-task/2-resource capacity-aware QAOA instances."""
    meds, gmed, slos, gslo = duration_tables
    rows, assign, qaoa_rows = [], [], []
    q_blocks = 2 if fast else QAOA_BLOCKS
    timeout = 60 if fast else QAOA_TIMEOUT_SECONDS
    starts = dense_window_starts(pool, QAOA_BATCH_SIZE, q_blocks)

    for block, start in enumerate(starts):
        batch = pool.iloc[start:start + QAOA_BATCH_SIZE].copy().reset_index(drop=True)
        batch["risk_hcqkl"] = score_hcqkl(batch, hcqkl_bundle)
        res = resources_for(batch, QAOA_RESOURCES)
        C = cost_matrix(batch, res, meds, gmed)
        milp_a, milp_o, milp_status = solve_exact(batch, res, C)

        save_json(
            [asdict(r) for r in res],
            out.qaoa / f"resource_bs{QAOA_BATCH_SIZE}_block{block:02d}.json",
        )
        np.save(out.qaoa / f"cost_bs{QAOA_BATCH_SIZE}_block{block:02d}.npy", C)

        _append_solver_result(
            rows, assign, batch, res, C, "MILP", milp_a,
            QAOA_BATCH_SIZE, block, milp_o, milp_status, slos, gslo,
            extra={"experiment": "qaoa_small_capacity"},
        )

        qi = solve_qaoa_bounded(
            batch, res, C,
            seed=SEED + 1000 + block,
            timeout_seconds=timeout,
        )
        save_json(qi, out.qaoa / f"qaoa_bs{QAOA_BATCH_SIZE}_block{block:02d}.json")
        qaoa_rows.append({
            "block": block,
            "available": qi.get("available", False),
            "success": qi.get("success", False),
            "timed_out": qi.get("timed_out", False),
            "sampler": qi.get("sampler"),
            "elapsed_seconds": qi.get("elapsed_seconds"),
            "wall_clock_elapsed_seconds": qi.get("wall_clock_elapsed_seconds"),
            "raw_feasible_actual": qi.get("raw_feasible_actual"),
            "raw_feasible_quantized": qi.get("raw_feasible_quantized"),
            "repaired_feasible": qi.get("repaired_feasible"),
            "status": qi.get("status"),
            "error_stage": qi.get("error_stage"),
            "error": qi.get("error"),
        })

        if qi.get("success"):
            _append_solver_result(
                rows, assign, batch, res, C, "QAOA_raw",
                qi.get("raw_assignment"), QAOA_BATCH_SIZE, block,
                milp_o, milp_status, slos, gslo,
                extra={
                    "experiment": "qaoa_small_capacity",
                    "qaoa_quantized_feasible": qi.get("raw_feasible_quantized"),
                    "qaoa_elapsed_seconds": qi.get("elapsed_seconds"),
                    "qaoa_timed_out": False,
                },
            )
            _append_solver_result(
                rows, assign, batch, res, C, "QAOA_repaired",
                qi.get("repaired_assignment"), QAOA_BATCH_SIZE, block,
                milp_o, milp_status, slos, gslo,
                extra={
                    "experiment": "qaoa_small_capacity",
                    "qaoa_elapsed_seconds": qi.get("elapsed_seconds"),
                    "qaoa_timed_out": False,
                },
            )
        else:
            for solver in ["QAOA_raw", "QAOA_repaired"]:
                _append_solver_result(
                    rows, assign, batch, res, C, solver, None,
                    QAOA_BATCH_SIZE, block, milp_o, milp_status, slos, gslo,
                    extra={
                        "experiment": "qaoa_small_capacity",
                        "qaoa_timed_out": qi.get("timed_out", False),
                        "qaoa_error": qi.get("error"),
                        "qaoa_error_stage": qi.get("error_stage"),
                    },
                )

        # Save after every individual QAOA attempt.
        pd.DataFrame(qaoa_rows).to_csv(out.qaoa / "qaoa_execution_summary.csv", index=False)
        pd.DataFrame(rows).to_csv(out.qaoa / "qaoa_scheduling_results.csv", index=False)
        pd.DataFrame(assign).to_csv(out.qaoa / "qaoa_assignments.csv", index=False)

    return pd.DataFrame(rows), pd.DataFrame(assign), pd.DataFrame(qaoa_rows)


def run_scheduling(train, test, hcqkl_bundle, out, fast):
    """Checkpointed classical scheduling followed by bounded QAOA."""
    update_stage_status(out, "scheduling_classical", "running")
    classical_rdf, classical_adf, pool, duration_tables = run_classical_scheduling_checkpointed(
        train, test, hcqkl_bundle, out, fast
    )
    update_stage_status(
        out, "scheduling_classical", "completed", rows=len(classical_rdf)
    )

    update_stage_status(out, "qaoa", "running")
    qdf, qadf, qsummary = run_qaoa_benchmark_checkpointed(
        pool, duration_tables, hcqkl_bundle, out, fast
    )
    update_stage_status(
        out,
        "qaoa",
        "completed",
        attempts=len(qsummary),
        successes=int(qsummary.get("success", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not qsummary.empty else 0,
        timeouts=int(qsummary.get("timed_out", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not qsummary.empty else 0,
    )

    rdf = pd.concat([classical_rdf, qdf], ignore_index=True, sort=False)
    adf = pd.concat([classical_adf, qadf], ignore_index=True, sort=False)
    rdf.to_csv(out.scheduling / "capacity_scheduling_results.csv", index=False)
    adf.to_csv(out.scheduling / "capacity_scheduling_assignments.csv", index=False)

    if not rdf.empty:
        summ = rdf.groupby(["experiment", "batch_size", "solver"], dropna=False, as_index=False).agg(
            runs=("block", "count"),
            feasibility_rate=("feasible", "mean"),
            mean_window_span_seconds=("window_span_seconds", "mean"),
            mean_objective=("objective", "mean"),
            mean_gap=("objective_gap", "mean"),
            mean_slo_miss=("slo_miss_rate", "mean"),
            mean_waiting=("mean_waiting", "mean"),
            mean_completion=("mean_completion", "mean"),
            mean_makespan=("makespan", "mean"),
            mean_energy=("total_energy", "mean"),
            mean_cost=("total_cost", "mean"),
        )
        summ.to_csv(out.scheduling / "capacity_scheduling_summary.csv", index=False)
    return rdf, adf


def run_iquantum_policy_statistics(iq: pd.DataFrame, out: Paths, fast: bool) -> pd.DataFrame:
    """Paired workload-level comparisons of risk-aware iQuantum placement."""
    if iq.empty:
        return pd.DataFrame()
    required = {"policy", "workload_size", "workload_seed", "success"}
    if not required.issubset(iq.columns):
        return pd.DataFrame()

    successful = iq[iq["success"].fillna(False).astype(bool)].copy()
    metrics_iq = [
        c for c in ["mean_waiting_time", "mean_qpu_time", "makespan", "total_cost"]
        if c in successful.columns
    ]
    baselines = ["random", "least_loaded", "compatibility_aware"]
    rows = []
    for baseline in baselines:
        for metric in metrics_iq:
            a = successful[successful.policy == "risk_aware"][
                ["workload_size", "workload_seed", metric]
            ].rename(columns={metric: "risk_aware"})
            b = successful[successful.policy == baseline][
                ["workload_size", "workload_seed", metric]
            ].rename(columns={metric: "baseline"})
            paired = a.merge(b, on=["workload_size", "workload_seed"], how="inner").dropna()
            if paired.empty:
                continue
            d = paired.risk_aware.to_numpy(float) - paired.baseline.to_numpy(float)
            lo, hi = bootstrap_ci(d, 1000 if fast else BOOTSTRAP_RESAMPLES, SEED + 900)
            nz = d[~np.isclose(d, 0)]
            stat = pval = np.nan
            if len(nz) >= 6:
                try:
                    w = wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
                    stat, pval = float(w.statistic), float(w.pvalue)
                except Exception:
                    pass
            sd = np.std(d, ddof=1) if len(d) > 1 else np.nan
            dz = 0.0 if np.isfinite(sd) and np.isclose(sd, 0) else (
                float(np.mean(d) / sd) if np.isfinite(sd) else np.nan
            )
            rows.append({
                "comparison": f"risk_aware_vs_{baseline}",
                "metric": metric,
                "pairs": len(paired),
                "mean_risk_aware": float(paired.risk_aware.mean()),
                "mean_baseline": float(paired.baseline.mean()),
                "mean_difference": float(np.mean(d)),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
                "paired_cohens_dz": dz,
                "wilcoxon_stat": stat,
                "p_raw": pval,
            })
    ans = pd.DataFrame(rows)
    if not ans.empty:
        ans["p_holm"] = holm(ans.p_raw.to_numpy(float))
        ans["significant_holm_0_05"] = ans.p_holm < 0.05
        ans.to_csv(out.iquantum / "iquantum_policy_paired_statistics.csv", index=False)
    return ans


def extra_figures(sres: pd.DataFrame, out: Paths) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    qsummary_path = out.qaoa / "qaoa_execution_summary.csv"
    if qsummary_path.exists():
        q = pd.read_csv(qsummary_path)
        if not q.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(len(q))
            vals = q.get("wall_clock_elapsed_seconds", q.get("elapsed_seconds", pd.Series(np.zeros(len(q)))))
            ax.bar(x, pd.to_numeric(vals, errors="coerce"))
            ax.set_xlabel("QAOA temporal block")
            ax.set_ylabel("Wall-clock seconds")
            ax.set_title("Bounded QAOA execution time")
            fig.tight_layout()
            fig.savefig(out.figures / "qaoa_bounded_runtime.png", dpi=300)
            plt.close(fig)

    if not sres.empty and {"solver", "objective_gap"}.issubset(sres.columns):
        z = sres[sres.solver.isin(["QAOA_raw", "QAOA_repaired"])].dropna(subset=["objective_gap"])
        if not z.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            for solver, g in z.groupby("solver"):
                ax.plot(g.block, g.objective_gap, marker="o", label=solver)
            ax.set_xlabel("QAOA temporal block")
            ax.set_ylabel("Objective gap to MILP")
            ax.set_title("Raw and repaired QAOA objective gap")
            ax.legend()
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(out.figures / "qaoa_objective_gap.png", dpi=300)
            plt.close(fig)


def _fresh_prediction_run(args, out: Paths):
    if not PREPARED_DATA.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {PREPARED_DATA}")
    raw = pd.read_csv(PREPARED_DATA, low_memory=False)
    primary, meta = build_future_target(raw, PRIMARY_HORIZON_SECONDS)
    primary.to_csv(out.target / "future_target_dataset.csv", index=False)
    save_json(meta, out.target / "future_target_metadata.json")
    horizon_sensitivity(raw, out, args.fast)
    train, val, test = temporal_split(primary)
    pd.DataFrame([
        {
            "split": n,
            "rows": len(d),
            "positives": int(d.future_sla_event.sum()),
            "prevalence": float(d.future_sla_event.mean()),
            "time_min": d[TIME_COLUMN].min(),
            "time_max": d[TIME_COLUMN].max(),
        }
        for n, d in [("train", train), ("validation", val), ("test", test)]
    ]).to_csv(out.target / "temporal_split_summary.csv", index=False)
    save_json({
        "project": "AQUA-SLA",
        "version": "review-revision-v2-final-completion",
        "created": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "output": str(out.root),
        "mode": args.mode,
        "fast": args.fast,
        "no_overwrite_v1": True,
        "primary_horizon_seconds": PRIMARY_HORIZON_SECONDS,
        "target": meta,
        "qaoa": {
            "batch_size": QAOA_BATCH_SIZE,
            "resources": QAOA_RESOURCES,
            "blocks": 2 if args.fast else QAOA_BLOCKS,
            "shots": QAOA_SHOTS,
            "maxiter": QAOA_MAXITER,
            "timeout_seconds": 60 if args.fast else QAOA_TIMEOUT_SECONDS,
        },
    }, out.manifest / "run_manifest.json")
    update_stage_status(out, "prediction", "running")
    kres, alpha, pred, bundle = run_hcqkl(train, val, test, out, args.fast)
    bres = run_baselines(train, val, test, out, args.fast)
    update_stage_status(out, "prediction", "completed", kernel_models=len(kres), baselines=len(bres))
    return primary, train, val, test, kres, alpha, pred, bres, bundle


def main():
    _mp.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["full", "prediction", "scheduling", "iquantum", "remaining"],
        default="full",
    )
    ap.add_argument("--run-name", default="review_v2_final")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument(
        "--resume-run",
        default=None,
        help=(
            "Existing timestamped V2 folder. With --mode remaining the script "
            "reuses the completed target/prediction artifacts and completes "
            "scheduling, bounded QAOA, statistics, iQuantum, and figures."
        ),
    )
    args = ap.parse_args()

    if args.resume_run:
        out = existing_paths(args.resume_run)
        print("AQUA-SLA V2 RESUME:", out.root)
    else:
        out = Paths.create(args.run_name)
        print("AQUA-SLA V2 output:", out.root)
    print("V1 folders are not modified.")

    completed = False
    error_text = None
    kres = alpha = pred = bres = sres = assign = stats = iq = iqstats = pd.DataFrame()

    try:
        if args.resume_run:
            primary, train, val, test, kres, alpha, pred, bres, bundle = load_completed_prediction(out)
            update_stage_status(out, "prediction", "reused", source="existing run artifacts")
        else:
            primary, train, val, test, kres, alpha, pred, bres, bundle = _fresh_prediction_run(args, out)

        if args.mode == "prediction":
            completed = True
        else:
            if args.mode in {"full", "remaining", "scheduling"}:
                update_stage_status(out, "scheduling", "running")
                sres, assign = run_scheduling(train, test, bundle, out, args.fast)
                update_stage_status(out, "scheduling", "completed", rows=len(sres))

                update_stage_status(out, "statistics", "running")
                stats = run_stats(sres, out, args.fast)
                update_stage_status(out, "statistics", "completed", rows=len(stats))
            else:
                sched_path = out.scheduling / "capacity_scheduling_results.csv"
                if sched_path.exists():
                    sres = pd.read_csv(sched_path)

            if args.mode in {"full", "remaining", "iquantum"}:
                update_stage_status(out, "iquantum", "running")
                risks = pred.risk_hcqkl.to_numpy(float) if "risk_hcqkl" in pred else np.array([])
                iq = run_iquantum(risks, out, args.fast)
                iqstats = run_iquantum_policy_statistics(iq, out, args.fast)
                update_stage_status(
                    out,
                    "iquantum",
                    "completed",
                    rows=len(iq),
                    successful_runs=int(iq.get("success", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not iq.empty else 0,
                    paired_statistics_rows=len(iqstats),
                )
            else:
                iq_path = out.iquantum / "iquantum_policy_summary.csv"
                if iq_path.exists():
                    iq = pd.read_csv(iq_path)

            update_stage_status(out, "figures", "running")
            figures(alpha, kres, bres, sres, iq, out)
            extra_figures(sres, out)
            update_stage_status(out, "figures", "completed")
            completed = True

    except BaseException as exc:
        error_text = repr(exc)
        update_stage_status(out, "fatal", "failed", error=error_text, traceback=_traceback.format_exc())
        print("ERROR:", error_text)
        print(_traceback.format_exc())

    qaoa_summary_path = out.qaoa / "qaoa_execution_summary.csv"
    qaoa_summary = pd.read_csv(qaoa_summary_path) if qaoa_summary_path.exists() else pd.DataFrame()
    iq_path = out.iquantum / "iquantum_policy_summary.csv"
    if iq.empty and iq_path.exists():
        iq = pd.read_csv(iq_path)
    stats_path = out.stats / "paired_block_statistics.csv"
    if stats.empty and stats_path.exists():
        stats = pd.read_csv(stats_path)
    sched_path = out.scheduling / "capacity_scheduling_results.csv"
    if sres.empty and sched_path.exists():
        sres = pd.read_csv(sched_path)

    final = {
        "completed": bool(completed),
        "output": str(out.root),
        "mode": args.mode,
        "resumed": bool(args.resume_run),
        "error": error_text,
        "kernel_models": len(kres),
        "scheduling_rows": len(sres),
        "statistics_rows": len(stats),
        "iquantum_rows": len(iq),
        "iquantum_successful_runs": int(iq.get("success", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not iq.empty else 0,
        "qaoa_attempts": len(qaoa_summary),
        "qaoa_successes": int(qaoa_summary.get("success", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not qaoa_summary.empty else 0,
        "qaoa_timeouts": int(qaoa_summary.get("timed_out", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not qaoa_summary.empty else 0,
        "qaoa_raw_feasible": int(qaoa_summary.get("raw_feasible_actual", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not qaoa_summary.empty else 0,
        "qaoa_repaired_feasible": int(qaoa_summary.get("repaired_feasible", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not qaoa_summary.empty else 0,
        "qaoa_configuration": {
            "tasks": QAOA_BATCH_SIZE,
            "resources": QAOA_RESOURCES,
            "blocks": 2 if args.fast else QAOA_BLOCKS,
            "reps": QAOA_REPS,
            "shots": QAOA_SHOTS,
            "maxiter": QAOA_MAXITER,
            "capacity_units": QAOA_CAPACITY_UNITS,
            "timeout_seconds": 60 if args.fast else QAOA_TIMEOUT_SECONDS,
            "note": "Both tasks may share one resource when cumulative capacity permits; this is not one-to-one matching.",
        },
        "notes": [
            "Classical 4/6/8-task capacity scheduling is checkpointed before QAOA.",
            "Every QAOA instance has a hard wall-clock timeout; timeout/failure is retained as a result and does not stop later stages.",
            "Raw and repaired QAOA remain separate.",
            "Independent-QASM iQuantum policies are compared on identical workloads across workload sizes and seeds.",
            "iQuantum is discrete-event simulation; Qiskit quantum-kernel/QAOA results are simulator based unless separately executed on hardware.",
        ],
    }
    save_json(final, out.manifest / "final_status.json")
    (out.manifest / "README_V2_RESULTS.txt").write_text(
        "AQUA-SLA reviewer revision V2 final-completion runner. Primary evidence: "
        "matched_kernel_test_metrics.csv; capacity_scheduling_results.csv; "
        "qaoa_execution_summary.csv; paired_block_statistics.csv; "
        "iquantum_policy_summary.csv; iquantum_policy_paired_statistics.csv.\n",
        encoding="utf-8",
    )

    print("Finished. Inspect:", out.manifest / "final_status.json")
    if not completed:
        raise SystemExit(1)


if __name__=="__main__":main()
