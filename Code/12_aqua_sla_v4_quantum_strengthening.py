from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from qiskit.circuit.library import pauli_feature_map, zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC

PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
DATA_PATH = PROJECT_DIR / "Dataset" / "processed" / "aqua_sla_runtime_prediction.csv"
RESULT_DIR = PROJECT_DIR / "results" / "quantum" / "aqua_sla_v4"
MODEL_DIR = PROJECT_DIR / "models" / "quantum" / "aqua_sla_v4"
TARGET = "sla_violation"

SEEDS = [11, 22, 33, 44, 55]
TRAIN_SIZE = 1200
VALIDATION_SIZE = 500
TEST_SIZE = 800
TOP_FEATURES = 16
QUANTUM_DIMS = [4, 6, 8]
C_VALUES = [0.1, 1.0, 10.0, 100.0]
THRESHOLD_POINTS = 300
BLEND_POINTS = 41
FN_COST = 3.0
FP_COST = 1.0

FORBIDDEN = {
    TARGET, "sla_failure", "event", "instance_events_type",
    "collections_events_type", "end_time", "end_time_seconds",
    "duration", "duration_seconds", "failure_time", "failure_timestamp",
    "termination_reason",
}

CANDIDATES = [
    "priority", "scheduling_class", "collection_type", "vertical_scaling",
    "scheduler", "requested_cpu", "requested_memory", "requested_cpu_missing",
    "requested_memory_missing", "average_cpu", "average_memory", "maximum_cpu",
    "maximum_memory", "random_sample_cpu", "average_cpu_request_ratio",
    "maximum_cpu_request_ratio", "average_memory_request_ratio",
    "maximum_memory_request_ratio", "average_resource_pressure",
    "maximum_resource_pressure", "resource_pressure", "cpu_request_headroom",
    "memory_request_headroom", "cpu_request_exceeded", "memory_request_exceeded",
    "cpu_peak_to_average_ratio", "memory_peak_to_average_ratio",
    "cpu_distribution_minimum", "cpu_distribution_median",
    "cpu_distribution_maximum", "cpu_distribution_mean", "cpu_distribution_std",
    "cpu_distribution_range", "tail_cpu_distribution_minimum",
    "tail_cpu_distribution_median", "tail_cpu_distribution_maximum",
    "tail_cpu_distribution_mean", "tail_cpu_distribution_std",
    "tail_cpu_distribution_range", "cycles_per_instruction",
    "memory_accesses_per_instruction", "cycles_per_instruction_missing",
    "memory_accesses_per_instruction_missing",
]


def temporal_split(df: pd.DataFrame):
    if "time_seconds" not in df.columns:
        raise ValueError("time_seconds is required for temporal splitting")
    out = df.copy()
    out["_order"] = pd.to_numeric(out["time_seconds"], errors="coerce")
    bad = out["_order"].isna() | (out["_order"] < 0) | (out["_order"] > 3_000_000)
    out.loc[bad, "_order"] = np.nan
    sort_cols = ["_order"] + (["record_id"] if "record_id" in out.columns else [])
    out = out.sort_values(sort_cols, kind="mergesort", na_position="last").reset_index(drop=True)
    a, b = int(len(out) * 0.60), int(len(out) * 0.80)
    return out.iloc[:a].copy(), out.iloc[a:b].copy(), out.iloc[b:].copy()


def balanced_sample(df: pd.DataFrame, n: int, seed: int):
    n1, n0 = n // 2, n - n // 2
    pos, neg = df[df[TARGET] == 1], df[df[TARGET] == 0]
    if len(pos) < n1 or len(neg) < n0:
        raise ValueError("Insufficient class samples")
    out = pd.concat([
        pos.sample(n1, random_state=seed),
        neg.sample(n0, random_state=seed + 1),
    ], ignore_index=True)
    return out.sample(frac=1, random_state=seed + 2).reset_index(drop=True)


def metric_dict(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp) if tn + fp else 0.0),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_search(y, score):
    best_t, best_u = 0.5, -np.inf
    for t in np.linspace(float(score.min()), float(score.max()), THRESHOLD_POINTS):
        pred = (score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        f1 = f1_score(y, pred, zero_division=0)
        mcc = matthews_corrcoef(y, pred)
        rec = recall_score(y, pred, zero_division=0)
        pre = precision_score(y, pred, zero_division=0)
        cost = (FN_COST * fn + FP_COST * fp) / len(y)
        util = 0.35 * f1 + 0.25 * ((mcc + 1) / 2) + 0.25 * rec + 0.15 * pre - 0.08 * cost
        if util > best_u:
            best_t, best_u = float(t), float(util)
    return best_t


def selection_score(y, score, threshold):
    pred = (score >= threshold).astype(int)
    return (
        0.30 * roc_auc_score(y, score)
        + 0.25 * average_precision_score(y, score)
        + 0.25 * f1_score(y, pred, zero_division=0)
        + 0.20 * ((matthews_corrcoef(y, pred) + 1) / 2)
    )


def maps(dim: int):
    return {
        "zz_linear_r1": zz_feature_map(dim, reps=1, entanglement="linear"),
        "zz_linear_r2": zz_feature_map(dim, reps=2, entanglement="linear"),
        "zz_full_r1": zz_feature_map(dim, reps=1, entanglement="full"),
        "pauli_xyz_r1": pauli_feature_map(
            dim, reps=1, paulis=["X", "Y", "ZZ"], entanglement="linear"
        ),
    }


def run_seed(data: pd.DataFrame, seed: int):
    print(f"\n===== Seed {seed} =====")
    random.seed(seed)
    np.random.seed(seed)
    train_full, val_full, test_full = temporal_split(data)
    train = balanced_sample(train_full, TRAIN_SIZE, seed)
    val = balanced_sample(val_full, VALIDATION_SIZE, seed + 100)
    test = balanced_sample(test_full, TEST_SIZE, seed + 200)

    available = [c for c in CANDIDATES if c in train.columns and c not in FORBIDDEN]
    sel_imp = SimpleImputer(strategy="median")
    x_sel = sel_imp.fit_transform(train[available].apply(pd.to_numeric, errors="coerce"))
    mi = mutual_info_classif(x_sel, train[TARGET].to_numpy(int), random_state=seed)
    ranking = pd.DataFrame({"feature": available, "mutual_information": mi}).sort_values(
        "mutual_information", ascending=False
    )
    features = ranking.head(TOP_FEATURES)["feature"].tolist()
    print("Selected:", ", ".join(features))

    val = val.sample(frac=1, random_state=seed + 500).reset_index(drop=True)
    cut = len(val) // 2
    cal, select = val.iloc[:cut], val.iloc[cut:]

    imp, scaler = SimpleImputer(strategy="median"), RobustScaler()
    def num(frame): return frame[features].apply(pd.to_numeric, errors="coerce")
    x_train = scaler.fit_transform(imp.fit_transform(num(train)))
    x_cal = scaler.transform(imp.transform(num(cal)))
    x_select = scaler.transform(imp.transform(num(select)))
    x_test = scaler.transform(imp.transform(num(test)))
    y_train = train[TARGET].to_numpy(int)
    y_cal = cal[TARGET].to_numpy(int)
    y_select = select[TARGET].to_numpy(int)
    y_test = test[TARGET].to_numpy(int)

    classical = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=350, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=seed,
    ).fit(x_train, y_train)
    c_select = classical.predict_proba(x_select)[:, 1]
    c_test = classical.predict_proba(x_test)[:, 1]
    c_t = threshold_search(y_select, c_select)
    c_metrics = metric_dict(y_test, (c_test >= c_t).astype(int), c_test)

    best = None
    search_rows = []
    for dim in QUANTUM_DIMS:
        dim = min(dim, x_train.shape[1])
        pca = PCA(n_components=dim, whiten=True, random_state=seed)
        tr_lat = pca.fit_transform(x_train)
        ca_lat, se_lat, te_lat = pca.transform(x_cal), pca.transform(x_select), pca.transform(x_test)
        angle = MinMaxScaler((0, np.pi))
        q_train = angle.fit_transform(tr_lat)
        q_cal, q_select, q_test = angle.transform(ca_lat), angle.transform(se_lat), angle.transform(te_lat)

        for map_name, fmap in maps(dim).items():
            print(f"Quantum search dim={dim}, map={map_name}")
            kernel = FidelityStatevectorKernel(feature_map=fmap, enforce_psd=True)
            start = time.perf_counter()
            k_train = kernel.evaluate(x_vec=q_train)
            k_cal = kernel.evaluate(x_vec=q_cal, y_vec=q_train)
            k_select = kernel.evaluate(x_vec=q_select, y_vec=q_train)
            k_test = kernel.evaluate(x_vec=q_test, y_vec=q_train)
            seconds = time.perf_counter() - start

            for c_value in C_VALUES:
                q_model = SVC(kernel="precomputed", C=c_value, class_weight="balanced", random_state=seed)
                q_model.fit(k_train, y_train)
                cal_raw = q_model.decision_function(k_cal)
                sel_raw = q_model.decision_function(k_select)
                test_raw = q_model.decision_function(k_test)
                calibrator = LogisticRegression(random_state=seed).fit(cal_raw.reshape(-1, 1), y_cal)
                q_select_prob = calibrator.predict_proba(sel_raw.reshape(-1, 1))[:, 1]
                q_test_prob = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
                q_t = threshold_search(y_select, q_select_prob)
                s = selection_score(y_select, q_select_prob, q_t)
                search_rows.append({
                    "seed": seed, "dimension": dim, "feature_map": map_name,
                    "svm_c": c_value, "selection_score": s,
                    "selection_roc_auc": roc_auc_score(y_select, q_select_prob),
                    "selection_pr_auc": average_precision_score(y_select, q_select_prob),
                    "threshold": q_t, "kernel_seconds": seconds,
                })
                if best is None or s > best["score"]:
                    best = {
                        "score": s, "dimension": dim, "map": map_name, "c": c_value,
                        "threshold": q_t, "select_prob": q_select_prob,
                        "test_prob": q_test_prob, "model": q_model,
                        "calibrator": calibrator, "pca": pca, "angle": angle,
                    }

    q_pred = (best["test_prob"] >= best["threshold"]).astype(int)
    q_metrics = metric_dict(y_test, q_pred, best["test_prob"])

    best_blend = None
    blend_rows = []
    for cw in np.linspace(0, 1, BLEND_POINTS):
        blend_select = cw * c_select + (1 - cw) * best["select_prob"]
        t = threshold_search(y_select, blend_select)
        s = selection_score(y_select, blend_select, t)
        blend_rows.append({"classical_weight": cw, "quantum_weight": 1-cw, "threshold": t, "score": s})
        if best_blend is None or s > best_blend["score"]:
            best_blend = {"cw": float(cw), "threshold": t, "score": s}

    h_test = best_blend["cw"] * c_test + (1 - best_blend["cw"]) * best["test_prob"]
    h_pred = (h_test >= best_blend["threshold"]).astype(int)
    h_metrics = metric_dict(y_test, h_pred, h_test)

    seed_dir = RESULT_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(seed_dir / "feature_ranking.csv", index=False)
    pd.DataFrame(search_rows).to_csv(seed_dir / "quantum_configuration_search.csv", index=False)
    pd.DataFrame(blend_rows).to_csv(seed_dir / "blend_search.csv", index=False)
    pd.DataFrame({
        "record_id": test["record_id"].to_numpy() if "record_id" in test.columns else np.arange(len(test)),
        "actual": y_test,
        "classical_probability": c_test,
        "quantum_probability": best["test_prob"],
        "hybrid_probability": h_test,
        "predicted_sla_violation": h_pred,
        "sla_risk_probability": h_test,
        "scheduling_priority_score": FN_COST * h_test,
    }).to_csv(seed_dir / "iquantum_risk_export.csv", index=False)

    joblib.dump({
        "features": features, "imputer": imp, "scaler": scaler,
        "classical_model": classical, "quantum_model": best["model"],
        "quantum_calibrator": best["calibrator"], "pca": best["pca"],
        "angle_scaler": best["angle"], "quantum_map": best["map"],
        "quantum_dimension": best["dimension"], "quantum_c": best["c"],
        "quantum_threshold": best["threshold"], "classical_weight": best_blend["cw"],
        "quantum_weight": 1-best_blend["cw"], "hybrid_threshold": best_blend["threshold"],
    }, MODEL_DIR / f"aqua_sla_v4_seed_{seed}.joblib")

    print(f"Best quantum: map={best['map']}, dim={best['dimension']}, C={best['c']}")
    print(f"Quantum: Accuracy={q_metrics['accuracy']:.4f}, F1={q_metrics['f1']:.4f}, MCC={q_metrics['mcc']:.4f}, ROC-AUC={q_metrics['roc_auc']:.4f}")
    print(f"Hybrid:  Accuracy={h_metrics['accuracy']:.4f}, F1={h_metrics['f1']:.4f}, MCC={h_metrics['mcc']:.4f}, ROC-AUC={h_metrics['roc_auc']:.4f}")

    rows = []
    for name, m in [("classical_branch", c_metrics), ("quantum_branch", q_metrics), ("aqua_sla_v4_hybrid", h_metrics)]:
        rows.append({"seed": seed, "model": name, **m})
    return rows


def ci95(series: pd.Series):
    values = series.to_numpy(float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    margin = 1.96 * float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean - margin, mean + margin


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, low_memory=False)
    print("Forbidden source columns present:", sorted(set(data.columns) & FORBIDDEN))

    rows = []
    for seed in SEEDS:
        rows.extend(run_seed(data, seed))
    results = pd.DataFrame(rows)
    results.to_csv(RESULT_DIR / "repeated_model_results.csv", index=False)

    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "pr_auc", "fn", "fp"]
    summary_rows = []
    for model, group in results.groupby("model"):
        row = {"model": model}
        for metric in metrics:
            lo, hi = ci95(group[metric])
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULT_DIR / "repeated_model_summary.csv", index=False)

    classical = results[results.model == "classical_branch"].sort_values("seed")
    sig = []
    for model in ["quantum_branch", "aqua_sla_v4_hybrid"]:
        other = results[results.model == model].sort_values("seed")
        for metric in ["accuracy", "f1", "mcc", "roc_auc", "pr_auc"]:
            try:
                stat, p = wilcoxon(other[metric].to_numpy(), classical[metric].to_numpy())
            except ValueError:
                stat, p = np.nan, np.nan
            sig.append({
                "comparison_model": model, "reference_model": "classical_branch",
                "metric": metric, "wilcoxon_statistic": stat, "p_value": p,
                "mean_difference": float(other[metric].mean() - classical[metric].mean()),
            })
    pd.DataFrame(sig).to_csv(RESULT_DIR / "paired_significance_tests.csv", index=False)

    with (RESULT_DIR / "experiment_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "seeds": SEEDS, "train_size": TRAIN_SIZE, "validation_size": VALIDATION_SIZE,
            "test_size": TEST_SIZE, "quantum_dims": QUANTUM_DIMS,
            "c_values": C_VALUES, "leakage_controls": sorted(FORBIDDEN),
        }, f, indent=2)

    print("\n===== Aggregate summary =====")
    cols = ["model", "accuracy_mean", "f1_mean", "mcc_mean", "roc_auc_mean", "pr_auc_mean", "recall_mean", "specificity_mean"]
    print(summary[cols].to_string(index=False))
    print(f"\nOutputs saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
