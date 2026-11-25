
#!/usr/bin/env python3
"""
AQUA-SLA SCRIPT 55
Workload-history-aware HCQKL prediction.

Preserves:
    300-second future SLA violation target
    Original six-qubit ZZ quantum feature map
    Chronological evaluation
    Multiple predetermined random seeds

Adds:
    Causal historical workload features
    Matched classical and quantum comparisons
    Representative landmark selection
    Regularized classification
    Temporal and class weighting
    Robust validation selection
    Train / Validation-A / Validation-B / Test reporting
    Probability and rank-based fusion
    Saved models and preprocessing
    CSV, LaTeX and publication figures

IMPORTANT:
    Historical trace records must genuinely be available
    at their recorded timestamps before enabling history.

    Results on the repeatedly examined original trace
    remain exploratory.

    No test-driven seed or configuration selection.
    No guaranteed performance target.
"""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    RobustScaler,
    QuantileTransformer
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DATA = (
    ROOT / "Dataset" / "processed"
    / "aqua_sla_prepared.csv"
)

OUTPUT = (
    ROOT / "results"
    / "55_final_optimized"
)

SEEDS = [5101, 5201, 5301, 5401, 5501]

EARLY_FEATURES = [
    "scheduling_class",
    "collection_type",
    "priority",
    "vertical_scaling",
    "scheduler",
    "requested_cpu",
    "requested_memory",
    "requested_cpu_missing",
    "requested_memory_missing"
]

FAILURE_EVENTS = [
    "EVICT", "FAIL", "KILL", "LOST"
]

HORIZON = 300


# =====================================================
# UTILITIES
# =====================================================

def log(message):
    print(f"[55] {message}", flush=True)


def digest(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def dump(path, obj):

    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            default=lambda x: (
                x.item()
                if isinstance(x, np.generic)
                else (
                    x.tolist()
                    if isinstance(x, np.ndarray)
                    else str(x)
                )
            )
        ),
        encoding="utf-8"
    )


def save_csv(df, path):

    df.to_csv(
        path,
        index=False,
        float_format="%.17g"
    )


def export(df, name, out):

    save_csv(
        df,
        out / "tables" / f"{name}.csv"
    )

    text = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.4f}"
    )

    (
        out / "latex" / f"{name}.tex"
    ).write_text(
        "\\begingroup\n"
        "\\color{blue}\n"
        "\\scriptsize\n"
        + text
        + "\n\\endgroup\n",
        encoding="utf-8"
    )

    log(f"EXPORTED: {name}")


def load_quantum_module():

    path = (
        HERE /
        "47_hcqkl_bandwidth_ablation.py"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "aqua55_quantum",
        path
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    required = [
        "read_data",
        "lifecycle_300",
        "original_zz_states",
        "fit_trainable",
        "FEATURES"
    ]

    for name in required:

        if not hasattr(module, name):

            raise AttributeError(
                f"Quantum companion missing: {name}"
            )

    return module


# =====================================================
# CAUSAL HISTORICAL FEATURES
# =====================================================

def history_group(
    raw,
    query,
    group_columns,
    prefix,
    include_events
):

    windows = [
        300,
        1800,
        3600
    ]

    output = {}

    for w in windows:

        output[f"{prefix}_count_{w}"] = (
            np.zeros(len(query), dtype=np.float32)
        )

        if include_events:

            output[f"{prefix}_failure_rate_{w}"] = (
                np.zeros(len(query), dtype=np.float32)
            )

    if group_columns:

        raw_groups = raw.groupby(
            group_columns,
            sort=False,
            dropna=False
        ).indices

        query_groups = query.groupby(
            group_columns,
            sort=False,
            dropna=False
        ).indices

    else:

        raw_groups = {
            "all": np.arange(len(raw))
        }

        query_groups = {
            "all": np.arange(len(query))
        }

    for key, target_idx in query_groups.items():

        history_idx = raw_groups.get(key)

        if history_idx is None:
            continue

        r = raw.iloc[history_idx]

        event_time = r["time_seconds"].to_numpy(
            dtype=float
        )

        order = np.argsort(
            event_time,
            kind="mergesort"
        )

        event_time = event_time[order]

        current_time = query.iloc[
            target_idx
        ]["time_seconds"].to_numpy(dtype=float)

        # Strictly earlier observations only.
        # Events at the prediction timestamp are excluded.

        right = np.searchsorted(
            event_time,
            current_time,
            side="left"
        )

        if include_events:

            failures = (
                r["event"]
                .isin(FAILURE_EVENTS)
                .to_numpy(dtype=int)[order]
            )

            cumulative = np.r_[
                0,
                np.cumsum(failures)
            ]

        for window in windows:

            left = np.searchsorted(
                event_time,
                current_time - window,
                side="left"
            )

            count = right - left

            output[
                f"{prefix}_count_{window}"
            ][target_idx] = np.log1p(count)

            if include_events:

                failure_count = (
                    cumulative[right]
                    - cumulative[left]
                )

                # Shrink sparse historical rates.
                rate = (
                    failure_count + 0.5
                ) / (
                    count + 5.0
                )

                output[
                    f"{prefix}_failure_rate_{window}"
                ][target_idx] = rate

    return pd.DataFrame(output)


def add_history(df, source, args):

    if args.history == "none":
        return df.copy(), []

    if not args.confirm_observed_rows:

        raise ValueError(
            "History requires confirmation that original "
            "trace records were observable at time_seconds. "
            "Use --confirm-observed-rows only after "
            "verifying this property."
        )

    include_events = (
        args.history == "events"
    )

    if (
        include_events
        and not args.confirm_event_time
    ):

        raise ValueError(
            "Event history requires confirmation that "
            "time_seconds is the actual event availability "
            "timestamp. Use --confirm-event-time only "
            "if that assumption is verified."
        )

    columns = [
        "time_seconds",
        "collection_id",
        "cluster"
    ]

    if include_events:
        columns.append("event")

    dtype = {
        "collection_id": "string",
        "cluster": "string"
    }

    if include_events:
        dtype["event"] = "string"

    log("Loading raw trace history")

    raw = pd.read_csv(
        source,
        usecols=columns,
        dtype=dtype
    )

    if "_source_row" not in df.columns:

        raise ValueError(
            "Lifecycle data missing _source_row. "
            "Cannot establish source-record alignment."
        )

    source_index = df[
        "_source_row"
    ].to_numpy(dtype=int)

    if (
        source_index.min() < 0
        or source_index.max() >= len(raw)
    ):

        raise ValueError(
            "Lifecycle-to-raw index mismatch."
        )

    query = raw.iloc[
        source_index
    ].copy().reset_index(drop=True)

    if not np.allclose(
        query["time_seconds"].to_numpy(),
        df["time_seconds"].to_numpy(),
        atol=1e-6,
        rtol=0
    ):

        raise ValueError(
            "Source-row timestamps do not match. "
            "History disabled to prevent invalid alignment."
        )

    raw = raw[
        np.isfinite(raw["time_seconds"])
    ].copy().reset_index(drop=True)

    for key in (
        "collection_id",
        "cluster"
    ):

        raw[key] = (
            raw[key]
            .fillna("__MISSING__")
            .astype(str)
        )

        query[key] = (
            query[key]
            .fillna("__MISSING__")
            .astype(str)
        )

    if include_events:

        raw["event"] = (
            raw["event"]
            .fillna("")
            .str.upper()
            .str.strip()
        )

    context = []

    for keys, prefix in [
        ([], "global"),
        (["cluster"], "cluster"),
        (
            ["cluster", "collection_id"],
            "collection"
        )
    ]:

        log(
            f"Computing causal {prefix} history"
        )

        part = history_group(
            raw,
            query,
            keys,
            prefix,
            include_events
        )

        context.append(part)

    history = pd.concat(
        context,
        axis=1
    )

    result = pd.concat(
        [
            df.reset_index(drop=True),
            history
        ],
        axis=1
    )

    log(
        f"Added {history.shape[1]} "
        "historical features"
    )

    return result, list(history.columns)


# =====================================================
# DATA AND TEMPORAL PARTITIONS
# =====================================================

def load_lifecycle(module, path, args):

    frame, meta = module.lifecycle_300(
        module.read_data(str(path))
    )

    frame = frame.sort_values(
        "time_seconds",
        kind="mergesort"
    ).reset_index(drop=True)

    frame, history_features = add_history(
        frame,
        path,
        args
    )

    features = list(module.FEATURES)

    if not set(features).issubset(
        set(EARLY_FEATURES)
    ):

        raise ValueError(
            "Unexpected feature list in Script 47. "
            "Review availability before training: "
            + str(features)
        )

    features += history_features

    missing = [
        name for name in features
        if name not in frame.columns
    ]

    if missing:

        raise ValueError(
            f"Missing features: {missing}"
        )

    return frame, meta, features


def prepare_partitions(module, args, out):

    df, meta, features = load_lifecycle(
        module,
        args.data,
        args
    )

    n = len(df)

    c1, c2, c3 = [
        int(n * x)
        for x in (0.60, 0.70, 0.80)
    ]

    train = df.iloc[:c1].copy()
    va = df.iloc[c1:c2].copy()
    vb = df.iloc[c2:c3].copy()
    test = df.iloc[c3:].copy()

    train = train[
        train.time_seconds + HORIZON
        < va.time_seconds.min()
    ]

    va = va[
        va.time_seconds + HORIZON
        < vb.time_seconds.min()
    ]

    vb = vb[
        vb.time_seconds + HORIZON
        < test.time_seconds.min()
    ]

    if args.external_test:

        test, external_meta, ext_features = (
            load_lifecycle(
                module,
                args.external_test,
                args
            )
        )

        if features != ext_features:

            raise ValueError(
                "External feature mismatch."
            )

        meta["external"] = external_meta

    # Independent sampling generators prevent
    # changing training size from changing test rows.

    def sample(frame, count, offset):

        if count <= 0 or count >= len(frame):
            return frame.copy()

        rng = np.random.default_rng(
            args.split_seed + offset
        )

        idx = np.sort(
            rng.choice(
                len(frame),
                count,
                replace=False
            )
        )

        return frame.iloc[idx].copy()

    frames = {
        "train": sample(train, args.train_n, 1),
        "valA": sample(va, args.val_n, 2),
        "valB": sample(vb, args.val_n, 3),
        "test": sample(test, args.test_n, 4)
    }

    records = []

    for name, frame in frames.items():

        y = frame[
            "future_sla_event"
        ].to_numpy(dtype=int)

        if set(np.unique(y)) != {0, 1}:

            raise ValueError(
                f"{name}: both classes required"
            )

        records.append({
            "partition": name,
            "n": len(y),
            "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()),
            "prevalence": float(y.mean()),
            "majority_accuracy": float(
                max(y.mean(), 1 - y.mean())
            )
        })

        save_csv(
            frame[
                [
                    "_source_row",
                    "time_seconds",
                    "future_sla_event"
                ]
            ],
            out / f"partition_{name}.csv"
        )

    table = pd.DataFrame(records)

    export(
        table,
        "dataset_partitions",
        out
    )

    dump(
        out / "dataset_metadata.json",
        meta
    )

    dump(
        out / "feature_manifest.json",
        {
            "original_features": list(
                module.FEATURES
            ),
            "all_features": features,
            "history_mode": args.history,
            "target": "future_sla_event",
            "horizon_seconds": HORIZON
        }
    )

    print(
        table.to_string(index=False),
        flush=True
    )

    return frames, features


# =====================================================
# PREPROCESSING
# =====================================================

def fit_preprocessing(
    frames,
    features,
    original_count,
    seed
):

    imp = SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )

    X = imp.fit_transform(
        frames["train"][features]
    )

    # Preserve original quantum input specification.
    original = X[:, :original_count]

    scaler = RobustScaler().fit(original)

    scaled = scaler.transform(original)

    pca = PCA(
        n_components=6,
        random_state=seed
    ).fit(scaled)

    reduced = pca.transform(scaled)

    quantile = QuantileTransformer(
        n_quantiles=min(500, len(X)),
        output_distribution="uniform",
        subsample=min(100000, len(X)),
        random_state=seed
    ).fit(reduced)

    return {
        "features": features,
        "original_count": original_count,
        "imputer": imp,
        "scaler": scaler,
        "pca": pca,
        "quantile": quantile
    }


def transform(frame, prep):

    X = prep["imputer"].transform(
        frame[prep["features"]]
    )

    original = X[
        :, :prep["original_count"]
    ]

    Z = (
        np.pi
        * prep["quantile"].transform(
            prep["pca"].transform(
                prep["scaler"].transform(
                    original
                )
            )
        )
    )

    return (
        X.astype(np.float32),
        np.clip(Z, 0, np.pi)
    )


# =====================================================
# TRAINING WEIGHTS
# =====================================================

def weight_options(y, times):

    ratio = (
        (1 - y.mean())
        / max(y.mean(), 1e-8)
    )

    balance = np.where(
        y == 1,
        np.sqrt(ratio),
        1.0
    )

    relative_time = (
        times - times.min()
    ) / max(
        np.ptp(times),
        1.0
    )

    recency = np.exp(
        1.2 * relative_time
    )

    combined = balance * recency

    choices = {
        "uniform": np.ones(len(y)),
        "balanced": balance,
        "recent": recency,
        "balanced_recent": combined
    }

    return {
        name: value / value.mean()
        for name, value in choices.items()
    }


# =====================================================
# VALIDATION OBJECTIVE
# =====================================================

def objective(y, p):

    auc = roc_auc_score(y, p)

    ap = average_precision_score(y, p)

    blocks = []

    for idx in np.array_split(
        np.arange(len(y)), 4
    ):

        if len(np.unique(y[idx])) == 2:

            blocks.append(
                roc_auc_score(
                    y[idx],
                    p[idx]
                )
            )

    if not blocks:
        blocks = [auc]

    return {
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "block_auc_mean": float(
            np.mean(blocks)
        ),
        "block_auc_std": float(
            np.std(blocks)
        ),
        "objective": float(
            0.65 * auc
            + 0.35 * np.mean(blocks)
            + 0.20 * ap
            - 0.15 * np.std(blocks)
        )
    }


# =====================================================
# MODEL CONFIGURATIONS
# =====================================================

def hgb_models(seed, smoke):

    iterations = 55 if smoke else 180

    return {
        "HGB15": HistGradientBoostingClassifier(
            max_iter=iterations,
            max_leaf_nodes=15,
            max_depth=4,
            learning_rate=0.045,
            min_samples_leaf=40,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=seed
        ),

        "HGB31": HistGradientBoostingClassifier(
            max_iter=iterations,
            max_leaf_nodes=31,
            max_depth=6,
            learning_rate=0.035,
            min_samples_leaf=50,
            l2_regularization=8.0,
            early_stopping=False,
            random_state=seed
        )
    }


def other_models(seed, args):

    trees = 60 if args.smoke else 300

    models = {
        "RF": RandomForestClassifier(
            n_estimators=trees,
            max_depth=18,
            min_samples_leaf=12,
            n_jobs=args.workers,
            random_state=seed
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=trees,
            max_depth=20,
            min_samples_leaf=10,
            n_jobs=args.workers,
            random_state=seed
        )
    }

    try:

        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=trees,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            reg_alpha=0.5,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=args.workers,
            random_state=seed
        )

    except ImportError:

        log("XGBoost not installed")

    return models


def fit_select(
    X,
    A,
    y,
    ya,
    models,
    weights
):

    best = None
    trials = []

    for name, template in models.items():

        for mode, sample_weight in weights.items():

            model = clone(template)

            started = time.perf_counter()

            model.fit(
                X,
                y,
                sample_weight=sample_weight
            )

            pa = model.predict_proba(
                A
            )[:, 1].astype(np.float64)

            row = {
                "estimator": name,
                "weighting": mode,
                "fit_seconds": (
                    time.perf_counter()
                    - started
                ),
                **objective(ya, pa)
            }

            trials.append(row)

            if (
                best is None
                or row["objective"]
                > best["record"]["objective"]
            ):

                best = {
                    "model": model,
                    "record": row
                }

    return best, trials


# =====================================================
# CLASS-STRATIFIED LANDMARKS
# =====================================================

def landmarks(
    Z,
    y,
    count,
    seed,
    strategy
):

    rng = np.random.default_rng(seed)

    count = min(count, len(Z))

    if strategy == "random":

        return rng.choice(
            len(Z),
            count,
            replace=False
        )

    selected = []

    for cls, requested in [
        (1, count // 2),
        (0, count - count // 2)
    ]:

        ids = np.flatnonzero(y == cls)

        k = min(requested, len(ids))

        if k == 0:
            continue

        if len(ids) > 12000:

            ids = rng.choice(
                ids,
                12000,
                replace=False
            )

        km = MiniBatchKMeans(
            n_clusters=k,
            batch_size=1024,
            n_init=3,
            random_state=seed
        ).fit(Z[ids])

        used = set()

        for center in km.cluster_centers_:

            nearest = np.argsort(
                np.sum(
                    (Z[ids] - center) ** 2,
                    axis=1
                )
            )

            for j in nearest:

                candidate = int(ids[j])

                if candidate not in used:

                    used.add(candidate)
                    selected.append(candidate)
                    break

    ids = np.asarray(
        selected,
        dtype=int
    )

    if len(ids) < count:

        remainder = np.setdiff1d(
            np.arange(len(Z)),
            ids
        )

        extra = rng.choice(
            remainder,
            count - len(ids),
            replace=False
        )

        ids = np.r_[ids, extra]

    rng.shuffle(ids)

    return ids


# =====================================================
# KERNEL COMPUTATION
# =====================================================

def kernel_features(
    Z,
    states,
    config,
    batch=2048
):

    refs = config["refs"]

    alpha = config["alpha"]

    gamma = config["gamma"]

    result = []

    for start in range(
        0, len(Z), batch
    ):

        stop = min(
            start + batch,
            len(Z)
        )

        z = Z[start:stop]

        kc = 0.0
        kq = 0.0

        if alpha > 0:

            if config["kernel"] == "Laplacian":

                d = np.sum(
                    np.abs(
                        z[:, None, :]
                        - refs[None, :, :]
                    ),
                    axis=2
                )

                kc = np.exp(
                    -np.sqrt(gamma) * d
                )

            elif config["kernel"] == "Polynomial":

                kc = (
                    (
                        1.0
                        + z @ refs.T
                        / (6 * np.pi ** 2)
                    ) / 2.0
                ) ** 2

            else:

                kc = rbf_kernel(
                    z,
                    refs,
                    gamma=gamma
                )

        if alpha < 1:

            state_batch = states[
                start:stop
            ]

            kq = np.abs(
                state_batch
                @ config[
                    "ref_states"
                ].conj().T
            ) ** 2

        result.append(
            np.asarray(
                alpha * kc
                + (1 - alpha) * kq,
                dtype=np.float32
            )
        )

    return np.concatenate(
        result,
        axis=0
    )


def candidate_configs():

    result = []

    for kind in (
        "original",
        "trained"
    ):

        result.append({
            "name": f"Quantum_{kind}",
            "family": "PureQuantum",
            "encoding": kind,
            "kernel": "RBF",
            "alpha": 0.0,
            "gamma_multiplier": 1.0
        })

        for alpha in (
            0.25, 0.50, 0.75
        ):

            result.append({
                "name": (
                    f"HCQKL_{kind}_"
                    f"a{int(alpha * 100)}"
                ),
                "family": "HCQKL",
                "encoding": kind,
                "kernel": "RBF",
                "alpha": alpha,
                "gamma_multiplier": 1.0
            })

    for kernel in (
        "RBF",
        "Laplacian"
    ):

        for mult in (
            0.5, 1.0, 2.0
        ):

            result.append({
                "name": f"{kernel}_g{mult}",
                "family": "ClassicalKernel",
                "encoding": "original",
                "kernel": kernel,
                "alpha": 1.0,
                "gamma_multiplier": mult
            })

    result.append({
        "name": "Polynomial",
        "family": "ClassicalKernel",
        "encoding": "original",
        "kernel": "Polynomial",
        "alpha": 1.0,
        "gamma_multiplier": 1.0
    })

    return result


# =====================================================
# FIT ONE COMPLETE SEED
# =====================================================

def train_seed(
    module,
    frames,
    features,
    args,
    seed,
    out
):

    log("=" * 65)
    log(f"TRAINING SEED {seed}")

    directory = (
        out / "seeds" / f"seed_{seed}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    original_count = len(module.FEATURES)

    prep = fit_preprocessing(
        frames,
        features,
        original_count,
        seed
    )

    joblib.dump(
        prep,
        directory / "preprocessing.joblib"
    )

    development = {}

    for split in (
        "train", "valA", "valB"
    ):

        development[split] = transform(
            frames[split],
            prep
        )

    X, Z = development["train"]
    A, ZA = development["valA"]
    B, ZB = development["valB"]

    y = frames["train"][
        "future_sla_event"
    ].to_numpy(int)

    ya = frames["valA"][
        "future_sla_event"
    ].to_numpy(int)

    weights = weight_options(
        y,
        frames["train"][
            "time_seconds"
        ].to_numpy(dtype=float)
    )

    if args.smoke:

        weights = {
            "uniform": weights["uniform"],
            "balanced": weights["balanced"]
        }

    bundles = {}
    predictions = {
        "train": {},
        "valA": {},
        "valB": {}
    }

    trial_records = []
    kernel_records = []

    def register(
        name,
        bundle,
        matrices
    ):

        path = (
            directory / f"{name}.joblib"
        )

        joblib.dump(
            bundle,
            path
        )

        bundles[name] = str(path)

        model = bundle["model"]

        for split, matrix in matrices.items():

            predictions[split][name] = (
                model.predict_proba(
                    matrix
                )[:, 1].astype(np.float64)
            )

    # Original-feature comparator.

    best, trials = fit_select(
        X[:, :original_count],
        A[:, :original_count],
        y,
        ya,
        hgb_models(seed, args.smoke),
        weights
    )

    register(
        "HGB_original",
        {
            "model": best["model"],
            "type": "original",
            "selection": best["record"]
        },
        {
            "train": X[:, :original_count],
            "valA": A[:, :original_count],
            "valB": B[:, :original_count]
        }
    )

    trial_records.extend([
        {
            "seed": seed,
            "configuration": "HGB_original",
            **row
        }
        for row in trials
    ])

    # Classical models using identical contextual
    # features available to the kernel models.

    baseline_groups = {
        "HGB_context": hgb_models(
            seed, args.smoke
        )
    }

    for name, model in other_models(
        seed, args
    ).items():

        baseline_groups[
            name + "_context"
        ] = {
            name: model
        }

    for name, models in baseline_groups.items():

        best, trials = fit_select(
            X,
            A,
            y,
            ya,
            models,
            weights
        )

        register(
            name,
            {
                "model": best["model"],
                "type": "context",
                "selection": best["record"]
            },
            {
                "train": X,
                "valA": A,
                "valB": B
            }
        )

        trial_records.extend([
            {
                "seed": seed,
                "configuration": name,
                **row
            }
            for row in trials
        ])

        log(
            f"{name}: "
            f"Validation-A ROC-AUC="
            f"{best['record']['roc_auc']:.4f}"
        )

    # Quantum alignment uses training labels only.

    theta, alignment = module.fit_trainable(
        Z,
        y,
        seed,
        args.align_evals
    )

    theta = np.asarray(theta)

    dump(
        directory / "quantum_alignment.json",
        alignment
    )

    np.save(
        directory / "quantum_bandwidth.npy",
        theta
    )

    state_cache = {}

    def get_states(kind, split):

        key = (kind, split)

        if key not in state_cache:

            z = development[split][1]

            multiplier = (
                theta
                if kind == "trained"
                else 1.0
            )

            state_cache[key] = (
                module.original_zz_states(
                    z * multiplier
                )
            )

        return state_cache[key]

    best_families = {}

    configs = candidate_configs()

    for strategy in args.strategies:

        for count in args.landmarks:

            ids = landmarks(
                Z,
                y,
                count,
                seed,
                strategy
            )

            refs = Z[ids]

            distance = np.sum(
                (
                    refs[:, None, :]
                    - refs[None, :, :]
                ) ** 2,
                axis=2
            )

            positive = distance[
                distance > 1e-12
            ]

            gamma_base = (
                1.0 / np.median(positive)
                if len(positive)
                else 1.0
            )

            for template in configs:

                kind = template["encoding"]

                alpha = template["alpha"]

                family = template["family"]

                tag = (
                    f"{template['name']}_"
                    f"{strategy}_L{len(ids)}"
                )

                config = {
                    **template,
                    "refs": refs,
                    "theta": theta,
                    "gamma": float(
                        gamma_base
                        * template[
                            "gamma_multiplier"
                        ]
                    ),
                    "ref_states": (
                        get_states(
                            kind,
                            "train"
                        )[ids]
                        if alpha < 1
                        else None
                    ),
                    "landmark_indices": ids
                }

                kt = kernel_features(
                    Z,
                    (
                        get_states(
                            kind,
                            "train"
                        )
                        if alpha < 1
                        else None
                    ),
                    config
                )

                ka = kernel_features(
                    ZA,
                    (
                        get_states(
                            kind,
                            "valA"
                        )
                        if alpha < 1
                        else None
                    ),
                    config
                )

                xtr = np.column_stack(
                    [X, kt]
                )

                xva = np.column_stack(
                    [A, ka]
                )

                best, trials = fit_select(
                    xtr,
                    xva,
                    y,
                    ya,
                    hgb_models(
                        seed,
                        args.smoke
                    ),
                    weights
                )

                record = {
                    "seed": seed,
                    "configuration": tag,
                    "family": family,
                    "encoding": kind,
                    "kernel": template["kernel"],
                    "alpha": alpha,
                    "landmarks": len(ids),
                    "strategy": strategy,
                    **best["record"]
                }

                kernel_records.append(record)

                trial_records.extend([
                    {
                        "seed": seed,
                        "configuration": tag,
                        **row
                    }
                    for row in trials
                ])

                log(
                    f"{tag}: "
                    f"VA ROC={record['roc_auc']:.4f}, "
                    f"PR={record['pr_auc']:.4f}"
                )

                previous = best_families.get(
                    family
                )

                if (
                    previous is None
                    or record["objective"]
                    > previous["selection"]["objective"]
                ):

                    best_families[family] = {
                        "type": "kernel",
                        "model": best["model"],
                        "config": config,
                        "selection": record
                    }

                del kt, ka, xtr, xva

    # Score only the validation-selected model
    # from each kernel family.

    for family, bundle in best_families.items():

        config = bundle["config"]

        matrices = {}

        for split, (x, z) in (
            development.items()
        ):

            states = (
                get_states(
                    config["encoding"],
                    split
                )
                if config["alpha"] < 1
                else None
            )

            k = kernel_features(
                z,
                states,
                config
            )

            matrices[split] = (
                np.column_stack([x, k])
            )

        register(
            family,
            bundle,
            matrices
        )

        log(
            f"SELECTED {family}: "
            + bundle["selection"][
                "configuration"
            ]
        )

    del state_cache

    save_csv(
        pd.DataFrame(trial_records),
        directory / "hyperparameter_trials.csv"
    )

    save_csv(
        pd.DataFrame(kernel_records),
        directory / "kernel_ablation.csv"
    )

    for split, values in predictions.items():

        frame = frames[split]

        save_csv(
            pd.DataFrame({
                "source_row": frame[
                    "_source_row"
                ].to_numpy(),
                "time_seconds": frame[
                    "time_seconds"
                ].to_numpy(),
                "y_true": frame[
                    "future_sla_event"
                ].to_numpy(),
                **values
            }),
            directory /
            f"{split}_scores.csv"
        )

        np.savez_compressed(
            directory /
            f"{split}_scores.npz",
            y_true=frame[
                "future_sla_event"
            ].to_numpy(),
            **values
        )

    dump(
        directory / "model_manifest.json",
        bundles
    )

    return {
        "seed": seed,
        "directory": directory,
        "prep": prep,
        "models": bundles,
        "predictions": predictions,
        "ablation": pd.DataFrame(
            kernel_records
        )
    }


# =====================================================
# THRESHOLDS AND COMPLETE METRICS
# =====================================================

def choose_threshold(y, p, criterion):

    candidates = np.unique(
        np.r_[
            np.quantile(
                p,
                np.linspace(0, 1, 300)
            ),
            np.nextafter(
                np.max(p),
                np.inf
            )
        ]
    )

    if criterion == "f1":

        def score(t):

            return f1_score(
                y,
                p >= t,
                zero_division=0
            )

    else:

        def score(t):

            return balanced_accuracy_score(
                y,
                p >= t
            )

    return float(
        max(candidates, key=score)
    )


def evaluate(y, p, threshold):

    p = np.asarray(
        p,
        dtype=np.float64
    )

    if not np.isfinite(p).all():

        raise ValueError(
            "Non-finite model predictions"
        )

    yp = (
        p >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            yp,
            labels=[0, 1]
        ).ravel()
    )

    budget = max(
        1,
        int(0.10 * len(y))
    )

    top = np.argsort(
        -p,
        kind="mergesort"
    )[:budget]

    return {
        "n": len(y),
        "prevalence": float(y.mean()),
        "accuracy": accuracy_score(y, yp),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "balanced_accuracy":
            balanced_accuracy_score(y, yp),
        "precision": precision_score(
            y, yp, zero_division=0
        ),
        "recall": recall_score(
            y, yp, zero_division=0
        ),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(
            y, yp, zero_division=0
        ),
        "mcc": matthews_corrcoef(y, yp),
        "top10_recall": (
            y[top].sum()
            / max(y.sum(), 1)
        ),
        "threshold": threshold,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }


# =====================================================
# VALIDATION-REFERENCED RANK FUSION
# =====================================================

def rank_transform(p, reference):

    left = np.searchsorted(
        reference,
        p,
        side="left"
    )

    right = np.searchsorted(
        reference,
        p,
        side="right"
    )

    return (
        left + right
    ) / (2 * len(reference))


def mean_predictions(results, split):

    names = set(
        results[0]["predictions"][split]
    )

    for result in results[1:]:

        names &= set(
            result["predictions"][split]
        )

    return {
        name: np.mean(
            [
                result[
                    "predictions"
                ][split][name]
                for result in results
            ],
            axis=0
        )
        for name in sorted(names)
    }


def fit_fusions(development, yb):

    pa = development["valA"]
    pb = development["valB"]

    baselines = [
        name for name in pb
        if name.endswith("_context")
        or name == "HGB_original"
    ]

    base = max(
        baselines,
        key=lambda name: (
            objective(
                yb,
                pb[name]
            )["objective"]
        )
    )

    references = {
        name: np.sort(p)
        for name, p in pa.items()
    }

    selected = {}
    records = []

    for family in (
        "HCQKL",
        "PureQuantum",
        "ClassicalKernel"
    ):

        trials = []

        for calibration in (
            "probability",
            "rank"
        ):

            a = pb[base]
            b = pb[family]

            if calibration == "rank":

                a = rank_transform(
                    a,
                    references[base]
                )

                b = rank_transform(
                    b,
                    references[family]
                )

            for weight in np.linspace(
                0, 1, 21
            ):

                p = (
                    (1 - weight) * a
                    + weight * b
                )

                record = {
                    "family": family,
                    "baseline": base,
                    "calibration": calibration,
                    "weight": float(weight),
                    **objective(yb, p)
                }

                records.append(record)
                trials.append(record)

        selected[family] = max(
            trials,
            key=lambda row: row["objective"]
        )

    return (
        selected,
        references,
        pd.DataFrame(records),
        base
    )


def apply_fusions(
    predictions,
    fusions,
    references
):

    result = predictions.copy()

    for family, config in fusions.items():

        base = config["baseline"]

        a = predictions[base]
        b = predictions[family]

        if config["calibration"] == "rank":

            a = rank_transform(
                a,
                references[base]
            )

            b = rank_transform(
                b,
                references[family]
            )

        w = config["weight"]

        result[
            f"Fusion_{family}"
        ] = (
            (1 - w) * a
            + w * b
        )

    return result


# =====================================================
# TEST INFERENCE FROM SAVED MODEL FILES
# =====================================================

def predict_test(module, frame, result):

    prep = joblib.load(
        result["directory"]
        / "preprocessing.joblib"
    )

    X, Z = transform(frame, prep)

    output = {}

    for name, path in result[
        "models"
    ].items():

        bundle = joblib.load(path)

        kind = bundle["type"]

        if kind == "original":

            matrix = X[
                :, :prep["original_count"]
            ]

        elif kind == "context":

            matrix = X

        else:

            config = bundle["config"]

            if config["alpha"] < 1:

                multiplier = (
                    config["theta"]
                    if config[
                        "encoding"
                    ] == "trained"
                    else 1.0
                )

                states = (
                    module.original_zz_states(
                        Z * multiplier
                    )
                )

            else:

                states = None

            k = kernel_features(
                Z,
                states,
                config
            )

            matrix = np.column_stack(
                [X, k]
            )

        output[name] = bundle[
            "model"
        ].predict_proba(
            matrix
        )[:, 1].astype(np.float64)

    return output


# =====================================================
# PAIRED TEMPORAL BOOTSTRAP
# =====================================================

def paired_ci(
    y,
    p,
    q,
    method,
    control,
    iterations
):

    rng = np.random.default_rng(550001)

    blocks = np.array_split(
        np.arange(len(y)),
        12
    )

    samples = []

    for _ in range(iterations):

        ids = np.concatenate([
            blocks[j]
            for j in rng.integers(
                0,
                len(blocks),
                len(blocks)
            )
        ])

        if len(
            np.unique(y[ids])
        ) != 2:

            continue

        samples.append([
            roc_auc_score(
                y[ids], p[ids]
            )
            - roc_auc_score(
                y[ids], q[ids]
            ),

            average_precision_score(
                y[ids], p[ids]
            )
            - average_precision_score(
                y[ids], q[ids]
            )
        ])

    if not samples:

        raise ValueError(
            "No valid paired bootstrap samples."
        )

    ci = np.quantile(
        samples,
        [0.025, 0.975],
        axis=0
    )

    return {
        "method": method,
        "control": control,
        "delta_roc": (
            roc_auc_score(y, p)
            - roc_auc_score(y, q)
        ),
        "roc_ci_low": ci[0, 0],
        "roc_ci_high": ci[1, 0],
        "delta_pr": (
            average_precision_score(y, p)
            - average_precision_score(y, q)
        ),
        "pr_ci_low": ci[0, 1],
        "pr_ci_high": ci[1, 1],
        "valid_bootstraps": len(samples)
    }


# =====================================================
# FIGURES
# =====================================================

def figures(all_metrics, ablation, y, scores, out):

    # Comparison across every evaluation partition.

    compact = all_metrics[
        all_metrics["threshold_type"]
        == "balanced_accuracy"
    ]

    pivot = compact.pivot(
        index="model",
        columns="partition",
        values="roc_auc"
    )

    pivot = pivot[
        ["train", "valA", "valB", "test"]
    ]

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.80
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("")
    ax.legend(title="Partition")
    ax.grid(axis="y", alpha=0.2)

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        out / "figures"
        / "train_validation_test.png",
        dpi=600
    )

    plt.close()

    # ROC and PR curves.

    for kind in (
        "ROC", "PR"
    ):

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        for name, p in scores.items():

            if kind == "ROC":

                x, z, _ = roc_curve(y, p)

                value = roc_auc_score(
                    y, p
                )

            else:

                z, x, _ = (
                    precision_recall_curve(
                        y, p
                    )
                )

                value = (
                    average_precision_score(
                        y, p
                    )
                )

            ax.plot(
                x,
                z,
                label=f"{name} ({value:.4f})"
            )

        if kind == "ROC":

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--"
            )

            ax.set_xlabel(
                "False positive rate"
            )

            ax.set_ylabel(
                "True positive rate"
            )

        else:

            ax.axhline(
                y.mean(),
                linestyle="--"
            )

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)

        fig.tight_layout()

        fig.savefig(
            out / "figures"
            / f"test_{kind}.png",
            dpi=600
        )

        plt.close(fig)

    # HCQKL mixing-coefficient sensitivity.

    sub = ablation[
        (ablation["family"] == "HCQKL")
        & (
            ablation["encoding"]
            == "original"
        )
        & (
            ablation["strategy"]
            == "representative"
        )
    ]

    if not sub.empty:

        fig, ax = plt.subplots(
            figsize=(7, 4.5)
        )

        for count, group in sub.groupby(
            "landmarks"
        ):

            summary = group.groupby(
                "alpha"
            )["roc_auc"].agg(
                ["mean", "std"]
            ).reset_index()

            ax.errorbar(
                summary["alpha"],
                summary["mean"],
                yerr=summary[
                    "std"
                ].fillna(0),
                marker="o",
                capsize=3,
                label=f"L{count}"
            )

        ax.set_xlabel(
            "Classical mixing coefficient"
        )

        ax.set_ylabel(
            "Validation-A ROC-AUC"
        )

        ax.legend()
        ax.grid(alpha=0.2)

        fig.tight_layout()

        fig.savefig(
            out / "figures"
            / "kernel_sensitivity.png",
            dpi=600
        )

        plt.close(fig)


# =====================================================
# MAIN WORKFLOW
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        default=str(DATA)
    )

    parser.add_argument(
        "--external-test",
        default=None
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT)
    )

    parser.add_argument(
        "--train-n",
        type=int,
        default=100000
    )

    parser.add_argument(
        "--val-n",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--test-n",
        type=int,
        default=0,
        help="0 uses every eligible test observation"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=5101
    )

    parser.add_argument(
        "--landmarks",
        type=int,
        nargs="+",
        default=[24, 48, 96]
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[
            "representative",
            "random"
        ],
        default=["representative"]
    )

    parser.add_argument(
        "--align-evals",
        type=int,
        default=75
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--history",
        choices=[
            "none",
            "activity",
            "events"
        ],
        default="none"
    )

    parser.add_argument(
        "--confirm-observed-rows",
        action="store_true"
    )

    parser.add_argument(
        "--confirm-event-time",
        action="store_true"
    )

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    if args.smoke:

        args.train_n = 5000
        args.val_n = 1000
        args.test_n = 1500
        args.seeds = args.seeds[:1]
        args.landmarks = [24]
        args.align_evals = 20
        args.bootstrap = 100
        args.strategies = [
            "representative"
        ]

    if len(set(args.seeds)) != len(args.seeds):

        raise ValueError(
            "Duplicate seeds are prohibited."
        )

    if not Path(args.data).is_file():

        raise FileNotFoundError(
            args.data
        )

    if args.external_test:

        if digest(args.external_test) == digest(
            args.data
        ):

            raise ValueError(
                "External test file is identical "
                "to the original dataset."
            )

    out = Path(args.output).resolve()

    if out.exists() and any(out.iterdir()):

        raise FileExistsError(
            "Output is not empty. Please use "
            "a new experiment output directory."
        )

    for folder in (
        "tables",
        "latex",
        "figures",
        "seeds"
    ):

        (out / folder).mkdir(
            parents=True,
            exist_ok=True
        )

    start = time.perf_counter()

    module = load_quantum_module()

    audit = {
        "script": "55_final_optimized",
        "arguments": vars(args),
        "script_sha256": digest(__file__),
        "quantum_source_sha256": digest(
            HERE /
            "47_hcqkl_bandwidth_ablation.py"
        ),
        "dataset_sha256": digest(args.data),
        "external_sha256": (
            digest(args.external_test)
            if args.external_test
            else None
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "target_roc_auc": 0.80,
        "target_guaranteed": False,
        "external_independence_verified": False,
        "training_metrics_in_sample": True
    }

    dump(
        out / "55_configuration.json",
        audit
    )

    frames, features = prepare_partitions(
        module,
        args,
        out
    )

    labels = {
        name: frame[
            "future_sla_event"
        ].to_numpy(int)
        for name, frame in frames.items()
    }

    results = []

    with threadpool_limits(
        limits=args.workers
    ):

        for seed in args.seeds:

            result = train_seed(
                module,
                frames,
                features,
                args,
                seed,
                out
            )

            results.append(result)

        # All development predictions are available
        # before any final test model inference.

        development = {
            split: mean_predictions(
                results,
                split
            )
            for split in (
                "train",
                "valA",
                "valB"
            )
        }

        (
            fusions,
            references,
            fusion_grid,
            best_classical
        ) = fit_fusions(
            development,
            labels["valB"]
        )

        for split in development:

            development[split] = apply_fusions(
                development[split],
                fusions,
                references
            )

        selected_model = max(
            development["valB"],
            key=lambda name: objective(
                labels["valB"],
                development["valB"][name]
            )["objective"]
        )

        thresholds = {}

        for name, p in development[
            "valB"
        ].items():

            thresholds[name] = {
                criterion: choose_threshold(
                    labels["valB"],
                    p,
                    criterion
                )
                for criterion in (
                    "f1",
                    "balanced_accuracy"
                )
            }

        # Freeze all identities, weights and
        # decision thresholds before test scoring.

        dump(
            out / "55_frozen_selection.json",
            {
                "selected_model": selected_model,
                "classical_baseline":
                    best_classical,
                "fusions": fusions,
                "thresholds": thresholds,
                "seeds": args.seeds
            }
        )

        joblib.dump(
            references,
            out / "rank_references.joblib"
        )

        log("=" * 65)
        log("MODEL SELECTION FROZEN")
        log(
            f"Selected model: {selected_model}"
        )

        # Print every development partition.

        display_columns = [
            "model",
            "accuracy",
            "roc_auc",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
            "mcc"
        ]

        all_rows = []

        for split in (
            "train", "valA", "valB"
        ):

            rows = []

            for name, p in development[
                split
            ].items():

                for criterion in (
                    "f1",
                    "balanced_accuracy"
                ):

                    record = {
                        "model": name,
                        "partition": split,
                        "threshold_type": criterion,
                        **evaluate(
                            labels[split],
                            p,
                            thresholds[
                                name
                            ][criterion]
                        )
                    }

                    rows.append(record)

            frame = pd.DataFrame(rows)

            all_rows.extend(rows)

            log(
                f"{split.upper()} RESULTS"
            )

            print(
                frame[
                    frame["threshold_type"]
                    == "balanced_accuracy"
                ][display_columns].to_string(
                    index=False,
                    float_format=lambda x:
                        f"{x:.6f}"
                ),
                flush=True
            )

        # Now perform test inference.

        log("STARTING FINAL TEST INFERENCE")

        per_seed_records = []

        for result in results:

            scores = predict_test(
                module,
                frames["test"],
                result
            )

            result[
                "predictions"
            ]["test"] = scores

            directory = result["directory"]

            for name, p in scores.items():

                for split in (
                    "train",
                    "valA",
                    "valB",
                    "test"
                ):

                    split_scores = result[
                        "predictions"
                    ][split][name]

                    val_scores = result[
                        "predictions"
                    ]["valB"][name]

                    threshold = choose_threshold(
                        labels["valB"],
                        val_scores,
                        "balanced_accuracy"
                    )

                    per_seed_records.append({
                        "seed": result["seed"],
                        "model": name,
                        "partition": split,
                        **evaluate(
                            labels[split],
                            split_scores,
                            threshold
                        )
                    })

            save_csv(
                pd.DataFrame({
                    "source_row": frames[
                        "test"
                    ]["_source_row"].to_numpy(),
                    "time_seconds": frames[
                        "test"
                    ]["time_seconds"].to_numpy(),
                    "y_true": labels["test"],
                    **scores
                }),
                directory / "test_scores.csv"
            )

            np.savez_compressed(
                directory / "test_scores.npz",
                y_true=labels["test"],
                **scores
            )

            seed_table = pd.DataFrame(
                per_seed_records
            )

            log(
                f"SEED {result['seed']} TEST RESULTS"
            )

            print(
                seed_table[
                    (
                        seed_table["seed"]
                        == result["seed"]
                    )
                    & (
                        seed_table["partition"]
                        == "test"
                    )
                ][display_columns].to_string(
                    index=False
                ),
                flush=True
            )

        test_scores = apply_fusions(
            mean_predictions(
                results,
                "test"
            ),
            fusions,
            references
        )

        test_rows = []

        for name, p in test_scores.items():

            for criterion in (
                "f1",
                "balanced_accuracy"
            ):

                record = {
                    "model": name,
                    "partition": "test",
                    "threshold_type": criterion,
                    **evaluate(
                        labels["test"],
                        p,
                        thresholds[
                            name
                        ][criterion]
                    )
                }

                test_rows.append(record)

        all_rows.extend(test_rows)

    # =================================================
    # EXPORT ALL MEASURED RESULTS
    # =================================================

    full = pd.DataFrame(all_rows)

    export(
        full,
        "all_train_validation_test_metrics",
        out
    )

    primary = full[
        full["threshold_type"]
        == "balanced_accuracy"
    ].copy()

    for split in (
        "train",
        "valA",
        "valB",
        "test"
    ):

        split_table = primary[
            primary["partition"] == split
        ]

        export(
            split_table[
                display_columns
            ],
            f"{split}_results",
            out
        )

    compact = primary.pivot(
        index="model",
        columns="partition",
        values="roc_auc"
    ).reset_index()

    compact = compact.rename(
        columns={
            "train": "Train ROC",
            "valA": "Val-A ROC",
            "valB": "Val-B ROC",
            "test": "Test ROC"
        }
    )

    test_table = primary[
        primary["partition"] == "test"
    ]

    compact = compact.merge(
        test_table[
            [
                "model",
                "accuracy",
                "pr_auc",
                "f1",
                "balanced_accuracy"
            ]
        ],
        on="model"
    )

    export(
        compact,
        "table_III_new_prediction_protocol",
        out
    )

    per_seed = pd.DataFrame(
        per_seed_records
    )

    export(
        per_seed,
        "all_seeds_all_partitions",
        out
    )

    seed_summary = per_seed.groupby(
        [
            "model",
            "partition"
        ],
        as_index=False
    ).agg(
        roc_mean=("roc_auc", "mean"),
        roc_std=("roc_auc", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        pr_mean=("pr_auc", "mean"),
        f1_mean=("f1", "mean"),
        balanced_accuracy_mean=(
            "balanced_accuracy",
            "mean"
        )
    )

    export(
        seed_summary,
        "multiseed_summary",
        out
    )

    ablation = pd.concat(
        [
            result["ablation"]
            for result in results
        ],
        ignore_index=True
    )

    export(
        ablation,
        "kernel_ablation_raw",
        out
    )

    ablation_summary = ablation.groupby(
        [
            "configuration",
            "family",
            "landmarks",
            "alpha",
            "strategy"
        ],
        as_index=False
    ).agg(
        roc_mean=("roc_auc", "mean"),
        roc_std=("roc_auc", "std"),
        pr_mean=("pr_auc", "mean"),
        pr_std=("pr_auc", "std"),
        selection_mean=(
            "objective",
            "mean"
        )
    )

    export(
        ablation_summary,
        "table_VIII_A_validationA",
        out
    )

    export(
        fusion_grid,
        "table_VIII_B_validationB",
        out
    )

    export(
        pd.DataFrame(
            list(fusions.values())
        ),
        "selected_fusion_configurations",
        out
    )

    export(
        compact,
        "table_VIII_C_prediction",
        out
    )

    export(
        full[
            full["partition"] == "test"
        ],
        "threshold_sensitivity",
        out
    )

    # Preserve exact ensemble scores.

    ensemble_frame = pd.DataFrame({
        "source_row": frames[
            "test"
        ]["_source_row"].to_numpy(),
        "time_seconds": frames[
            "test"
        ]["time_seconds"].to_numpy(),
        "y_true": labels["test"],
        **test_scores
    })

    save_csv(
        ensemble_frame,
        out / "ensemble_test_scores.csv"
    )

    np.savez_compressed(
        out / "ensemble_test_scores.npz",
        y_true=labels["test"],
        **test_scores
    )

    for split, values in development.items():

        save_csv(
            pd.DataFrame({
                "source_row": frames[
                    split
                ]["_source_row"].to_numpy(),
                "y_true": labels[split],
                **values
            }),
            out / f"ensemble_{split}_scores.csv"
        )

    # Paired quantum-versus-classical comparisons.

    comparisons = []

    for method, control in [
        ("HCQKL", "ClassicalKernel"),
        ("HCQKL", best_classical),
        (
            "Fusion_HCQKL",
            "Fusion_ClassicalKernel"
        )
    ]:

        comparisons.append(
            paired_ci(
                labels["test"],
                test_scores[method],
                test_scores[control],
                method,
                control,
                args.bootstrap
            )
        )

    export(
        pd.DataFrame(comparisons),
        "paired_confidence_intervals",
        out
    )

    figures(
        full,
        ablation,
        labels["test"],
        test_scores,
        out
    )

    # Verify table entries against exact binary scores.

    saved = np.load(
        out / "ensemble_test_scores.npz"
    )

    for _, row in test_table.iterrows():

        name = row["model"]

        measured = evaluate(
            saved["y_true"],
            saved[name],
            float(row["threshold"])
        )

        for metric in (
            "accuracy",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "mcc"
        ):

            if not np.isclose(
                measured[metric],
                row[metric],
                atol=1e-12,
                rtol=1e-10
            ):

                raise ValueError(
                    f"Verification failure: "
                    f"{name} / {metric}"
                )

    selected = test_table[
        test_table["model"]
        == selected_model
    ].iloc[0]

    hcqkl = test_table[
        test_table["model"]
        == "HCQKL"
    ].iloc[0]

    audit.update({
        "elapsed_seconds": (
            time.perf_counter() - start
        ),
        "selected_model": selected_model,
        "hcqkl_test_auc": float(
            hcqkl["roc_auc"]
        ),
        "hcqkl_test_accuracy": float(
            hcqkl["accuracy"]
        ),
        "hcqkl_test_balanced_accuracy": float(
            hcqkl["balanced_accuracy"]
        ),
        "hcqkl_test_f1": float(
            hcqkl["f1"]
        ),
        "selected_test_auc": float(
            selected["roc_auc"]
        ),
        "hcqkl_80_auc_target_achieved": bool(
            hcqkl["roc_auc"] >= 0.80
        ),
        "selected_80_auc_target_achieved": bool(
            selected["roc_auc"] >= 0.80
        ),
        "metrics_verified_from_binary_scores": True,
        "test_based_selection": False,
        "scheduling_updated": False,
        "iquantum_updated": False,
        "exploratory_on_original_trace": (
            args.external_test is None
        )
    })

    dump(
        out / "55_reproducibility_audit.json",
        audit
    )

    log("=" * 65)
    log("FINAL TRAIN / VALIDATION / TEST RESULTS")

    print(
        compact.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        ),
        flush=True
    )

    log("FINAL TEST METRICS")

    print(
        test_table[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        ),
        flush=True
    )

    log("MULTI-SEED TEST SUMMARY")

    print(
        seed_summary[
            seed_summary["partition"]
            == "test"
        ].to_string(index=False),
        flush=True
    )

    log(
        f"HCQKL Test ROC-AUC: "
        f"{hcqkl['roc_auc']:.6f}"
    )

    log(
        f"HCQKL Test Accuracy: "
        f"{hcqkl['accuracy']:.6f}"
    )

    log(
        f"HCQKL Balanced Accuracy: "
        f"{hcqkl['balanced_accuracy']:.6f}"
    )

    log(
        "HCQKL 80% ROC-AUC target achieved: "
        + str(
            audit[
                "hcqkl_80_auc_target_achieved"
            ]
        )
    )

    log("ALL TEST METRICS VERIFIED")

    log(
        f"ALL RESULTS SAVED TO {out}"
    )


if __name__ == "__main__":
    main()
