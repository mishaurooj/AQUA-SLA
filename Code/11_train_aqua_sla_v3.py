from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC

PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
DATA_PATH = PROJECT_DIR / "Dataset" / "processed" / "aqua_sla_runtime_prediction.csv"
RESULT_DIR = PROJECT_DIR / "results" / "quantum" / "aqua_sla_v3"
MODEL_DIR = PROJECT_DIR / "models" / "quantum" / "aqua_sla_v3"

TARGET = "sla_violation"
SEED = 42
TRAIN_SIZE = 4000
VALIDATION_SIZE = 1000
TEST_SIZE = 2000
TOP_FEATURES = 20
QUANTUM_COMPONENTS = 6
BLEND_POINTS = 101
THRESHOLD_POINTS = 600
FALSE_NEGATIVE_COST = 3.0
FALSE_POSITIVE_COST = 1.0

FORBIDDEN = {
    TARGET, "sla_failure", "event", "instance_events_type",
    "collections_events_type", "end_time", "end_time_seconds",
    "duration", "duration_seconds", "failure_time",
    "failure_timestamp", "termination_reason",
}

CANDIDATES = [
    "priority", "scheduling_class", "collection_type", "vertical_scaling",
    "scheduler", "requested_cpu", "requested_memory",
    "requested_cpu_missing", "requested_memory_missing", "average_cpu",
    "average_memory", "maximum_cpu", "maximum_memory", "random_sample_cpu",
    "average_cpu_request_ratio", "maximum_cpu_request_ratio",
    "average_memory_request_ratio", "maximum_memory_request_ratio",
    "average_resource_pressure", "maximum_resource_pressure",
    "resource_pressure", "cpu_request_headroom", "memory_request_headroom",
    "cpu_request_exceeded", "memory_request_exceeded",
    "cpu_peak_to_average_ratio", "memory_peak_to_average_ratio",
    "cpu_distribution_minimum", "cpu_distribution_median",
    "cpu_distribution_maximum", "cpu_distribution_mean",
    "cpu_distribution_std", "cpu_distribution_range",
    "tail_cpu_distribution_minimum", "tail_cpu_distribution_median",
    "tail_cpu_distribution_maximum", "tail_cpu_distribution_mean",
    "tail_cpu_distribution_std", "tail_cpu_distribution_range",
    "cycles_per_instruction", "memory_accesses_per_instruction",
    "cycles_per_instruction_missing", "memory_accesses_per_instruction_missing",
]


def temporal_split(df: pd.DataFrame):
    df = df.copy()
    if "time_seconds" not in df.columns:
        raise ValueError("time_seconds is required")
    df["_order"] = pd.to_numeric(df["time_seconds"], errors="coerce")
    bad = df["_order"].isna() | (df["_order"] < 0) | (df["_order"] > 3_000_000)
    df.loc[bad, "_order"] = np.nan
    sort_cols = ["_order"] + (["record_id"] if "record_id" in df.columns else [])
    df = df.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    a, b = int(len(df) * 0.60), int(len(df) * 0.80)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def balanced_sample(df: pd.DataFrame, n: int, seed: int):
    p = n // 2
    pos, neg = df[df[TARGET] == 1], df[df[TARGET] == 0]
    if len(pos) < p or len(neg) < n - p:
        raise ValueError("Insufficient class samples")
    out = pd.concat([
        pos.sample(p, random_state=seed),
        neg.sample(n - p, random_state=seed + 1),
    ], ignore_index=True)
    return out.sample(frac=1, random_state=seed + 2).reset_index(drop=True)


def choose_features(train: pd.DataFrame):
    available = [c for c in CANDIDATES if c in train.columns and c not in FORBIDDEN]
    if len(available) < 4:
        raise ValueError("Too few valid features")
    raw = train[available].apply(pd.to_numeric, errors="coerce")
    x = SimpleImputer(strategy="median").fit_transform(raw)
    y = train[TARGET].to_numpy(int)
    mi = mutual_info_classif(x, y, random_state=SEED)
    ranking = pd.DataFrame({"feature": available, "mutual_information": mi}).sort_values(
        "mutual_information", ascending=False
    ).reset_index(drop=True)
    return ranking.head(min(TOP_FEATURES, len(ranking)))["feature"].tolist(), ranking


def metric_dict(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_search(y, score):
    best_t, best_u, rows = 0.5, -np.inf, []
    for t in np.linspace(float(score.min()), float(score.max()), THRESHOLD_POINTS):
        pred = (score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        f1 = f1_score(y, pred, zero_division=0)
        mcc = matthews_corrcoef(y, pred)
        rec = recall_score(y, pred, zero_division=0)
        pre = precision_score(y, pred, zero_division=0)
        cost = (FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp) / len(y)
        utility = 0.40*f1 + 0.25*((mcc+1)/2) + 0.25*rec + 0.10*pre - 0.08*cost
        rows.append({"threshold": t, "utility": utility, "f1": f1, "mcc": mcc,
                     "recall": rec, "precision": pre, "fn": fn, "fp": fp})
        if utility > best_u:
            best_t, best_u = float(t), float(utility)
    return best_t, pd.DataFrame(rows)


def optimize_blend(y, classical, quantum):
    best = None
    rows = []
    best_threshold_table = None
    for wc in np.linspace(0, 1, BLEND_POINTS):
        wq = 1 - wc
        score = wc * classical + wq * quantum
        threshold, table = threshold_search(y, score)
        pred = (score >= threshold).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        mcc = matthews_corrcoef(y, pred)
        auc = roc_auc_score(y, score)
        pr = average_precision_score(y, score)
        objective = 0.30*auc + 0.25*pr + 0.25*f1 + 0.20*((mcc+1)/2)
        row = {"classical_weight": wc, "quantum_weight": wq, "threshold": threshold,
               "objective": objective, "f1": f1, "mcc": mcc, "roc_auc": auc, "pr_auc": pr}
        rows.append(row)
        if best is None or objective > best[0]:
            best = (objective, float(wc), float(threshold))
            best_threshold_table = table.copy()
    return best[1], best[2], pd.DataFrame(rows), best_threshold_table


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target {TARGET}")
    print("Forbidden source columns present:", sorted(set(df.columns) & FORBIDDEN))
    print("They are excluded from predictors; target is used only as the label.")

    train_all, val_all, test_all = temporal_split(df)
    train = balanced_sample(train_all, TRAIN_SIZE, SEED)
    val = balanced_sample(val_all, VALIDATION_SIZE, SEED + 100)
    test = balanced_sample(test_all, TEST_SIZE, SEED + 200)

    features, ranking = choose_features(train)
    print("Selected features:")
    for f in features:
        print("-", f)
    ranking.to_csv(RESULT_DIR / "feature_ranking.csv", index=False)

    def numeric(frame):
        return frame[features].apply(pd.to_numeric, errors="coerce")

    y_train = train[TARGET].to_numpy(int)
    y_val = val[TARGET].to_numpy(int)
    y_test = test[TARGET].to_numpy(int)

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(numeric(train)))
    x_val = scaler.transform(imputer.transform(numeric(val)))
    x_test = scaler.transform(imputer.transform(numeric(test)))

    print("Training classical branch")
    classical = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=350, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=30, random_state=SEED,
    )
    t0 = time.perf_counter()
    classical.fit(x_train, y_train)
    classical_seconds = time.perf_counter() - t0
    c_val = classical.predict_proba(x_val)[:, 1]
    c_test = classical.predict_proba(x_test)[:, 1]

    print("Constructing quantum latent representation")
    n_components = min(QUANTUM_COMPONENTS, x_train.shape[1])
    pca = PCA(n_components=n_components, whiten=True, random_state=SEED)
    z_train = pca.fit_transform(x_train)
    z_val = pca.transform(x_val)
    z_test = pca.transform(x_test)
    angle = MinMaxScaler((0.0, np.pi))
    q_train = angle.fit_transform(z_train)
    q_val = angle.transform(z_val)
    q_test = angle.transform(z_test)

    fmap = zz_feature_map(
        feature_dimension=n_components,
        reps=2,
        entanglement="linear",
    )
    qkernel = FidelityStatevectorKernel(feature_map=fmap, enforce_psd=True)

    print("Computing quantum kernels")
    t0 = time.perf_counter()
    k_train = qkernel.evaluate(x_vec=q_train)
    k_val = qkernel.evaluate(x_vec=q_val, y_vec=q_train)
    k_test = qkernel.evaluate(x_vec=q_test, y_vec=q_train)
    quantum_kernel_seconds = time.perf_counter() - t0

    print("Training quantum-kernel SVM")
    qmodel = SVC(kernel="precomputed", C=10.0, class_weight="balanced",
                 probability=True, random_state=SEED)
    t0 = time.perf_counter()
    qmodel.fit(k_train, y_train)
    quantum_training_seconds = time.perf_counter() - t0
    q_val_prob = qmodel.predict_proba(k_val)[:, 1]
    q_test_prob = qmodel.predict_proba(k_test)[:, 1]

    print("Optimizing validation-only blend and SLA threshold")
    wc, threshold, blend_table, threshold_table = optimize_blend(y_val, c_val, q_val_prob)
    wq = 1 - wc
    val_score = wc*c_val + wq*q_val_prob
    test_score = wc*c_test + wq*q_test_prob
    val_pred = (val_score >= threshold).astype(int)
    test_pred = (test_score >= threshold).astype(int)
    val_metrics = metric_dict(y_val, val_pred, val_score)
    test_metrics = metric_dict(y_test, test_pred, test_score)

    ct, _ = threshold_search(y_val, c_val)
    qt, _ = threshold_search(y_val, q_val_prob)
    c_metrics = metric_dict(y_test, (c_test >= ct).astype(int), c_test)
    q_metrics = metric_dict(y_test, (q_test_prob >= qt).astype(int), q_test_prob)

    print(f"\nSelected weights: classical={wc:.4f}, quantum={wq:.4f}")
    print(f"Selected threshold: {threshold:.6f}")
    print("\nAQUA-SLA v3 validation")
    print(f"Accuracy={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, "
          f"MCC={val_metrics['mcc']:.4f}, ROC-AUC={val_metrics['roc_auc']:.4f}, "
          f"PR-AUC={val_metrics['pr_auc']:.4f}")
    print("\nAQUA-SLA v3 test")
    print(f"Accuracy={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, "
          f"MCC={test_metrics['mcc']:.4f}, ROC-AUC={test_metrics['roc_auc']:.4f}, "
          f"PR-AUC={test_metrics['pr_auc']:.4f}")

    comparison = pd.DataFrame([
        {"model": "classical_branch", "classical_weight": 1.0, "quantum_weight": 0.0,
         "decision_threshold": ct, **c_metrics},
        {"model": "quantum_branch", "classical_weight": 0.0, "quantum_weight": 1.0,
         "decision_threshold": qt, **q_metrics},
        {"model": "AQUA-SLA-v3", "classical_weight": wc, "quantum_weight": wq,
         "decision_threshold": threshold, **test_metrics},
    ])
    print("\nFinal test comparison")
    print(comparison[["model", "accuracy", "f1", "mcc", "roc_auc", "pr_auc",
                      "recall", "specificity", "fn", "fp"]].to_string(index=False))

    blend_table.to_csv(RESULT_DIR / "blend_search.csv", index=False)
    threshold_table.to_csv(RESULT_DIR / "threshold_search.csv", index=False)
    comparison.to_csv(RESULT_DIR / "test_model_comparison.csv", index=False)
    pd.DataFrame({
        "record_id": test["record_id"].to_numpy() if "record_id" in test.columns else np.arange(len(test)),
        "actual": y_test,
        "classical_probability": c_test,
        "quantum_probability": q_test_prob,
        "hybrid_probability": test_score,
        "prediction": test_pred,
        "decision_threshold": threshold,
    }).to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    joblib.dump({
        "imputer": imputer, "scaler": scaler, "classical_model": classical,
        "pca": pca, "angle_scaler": angle, "quantum_model": qmodel,
        "selected_features": features, "classical_weight": wc,
        "quantum_weight": wq, "decision_threshold": threshold,
    }, MODEL_DIR / "aqua_sla_v3.joblib")

    metadata = {
        "model": "AQUA-SLA-v3", "dataset": str(DATA_PATH), "target": TARGET,
        "seed": SEED, "train_size": TRAIN_SIZE, "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE, "selected_features": features,
        "quantum_components": n_components, "feature_map": "zz_feature_map",
        "classical_weight": wc, "quantum_weight": wq,
        "decision_threshold": threshold,
        "classical_training_seconds": classical_seconds,
        "quantum_kernel_seconds": quantum_kernel_seconds,
        "quantum_training_seconds": quantum_training_seconds,
        "validation_metrics": val_metrics, "test_metrics": test_metrics,
        "classical_test_metrics": c_metrics, "quantum_test_metrics": q_metrics,
        "leakage_controls": {
            "forbidden_columns": sorted(FORBIDDEN),
            "feature_selection_training_only": True,
            "pca_training_only": True,
            "blend_validation_only": True,
            "threshold_validation_only": True,
            "test_evaluated_once": True,
        },
    }
    with (RESULT_DIR / "aqua_sla_v3_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to: {RESULT_DIR}")
    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
