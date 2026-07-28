from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from scipy.optimize import differential_evolution
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC


# ============================================================
# AQUA-SLA v2
# Three-kernel multiple-kernel learning:
#
#   K_v2 = w_c K_c + w_q K_q + w_i (K_c ⊙ K_q)
#
# where:
#   w_c, w_q, w_i >= 0
#   w_c + w_q + w_i = 1
#
# The kernel weights, SVM C, RBF gamma, and decision threshold
# are selected using validation data only.
#
# The runtime model is expected to perform better than the
# four-feature early-prediction model because it uses legitimate
# runtime telemetry. It must not include event-derived leakage.
# ============================================================


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")

DATA_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_runtime_prediction.csv"
)

RESULT_DIR = (
    PROJECT_DIR
    / "results"
    / "quantum"
    / "aqua_sla_v2"
)

MODEL_DIR = (
    PROJECT_DIR
    / "models"
    / "quantum"
    / "aqua_sla_v2"
)

TARGET = "sla_violation"

SEED = 42
TRAIN_SIZE = 800
VALIDATION_SIZE = 250
TEST_SIZE = 400

TOP_K_FEATURES = 8

FEATURE_MAP_REPS = 2
FEATURE_MAP_ENTANGLEMENT = "linear"

THRESHOLD_POINTS = 500

# Differential-evolution settings.
OPTIMIZER_MAXITER = 30
OPTIMIZER_POPSIZE = 8

# Multi-objective validation score:
#
# objective =
#   0.35 * ROC-AUC
# + 0.25 * PR-AUC
# + 0.20 * F1
# + 0.20 * normalized MCC
OBJECTIVE_WEIGHTS = {
    "roc_auc": 0.35,
    "pr_auc": 0.25,
    "f1": 0.20,
    "mcc": 0.20,
}

# Candidate features are all legitimate runtime or pre-runtime
# measurements. The script automatically keeps only columns
# present in the dataset.
CANDIDATE_FEATURES = [
    # Pre-execution / scheduler context
    "priority",
    "scheduling_class",
    "collection_type",
    "vertical_scaling",
    "scheduler",
    "requested_cpu",
    "requested_memory",
    "requested_cpu_missing",
    "requested_memory_missing",

    # Runtime resource usage
    "average_cpu",
    "average_memory",
    "maximum_cpu",
    "maximum_memory",
    "random_sample_cpu",

    # Request ratios and resource pressure
    "average_cpu_request_ratio",
    "maximum_cpu_request_ratio",
    "average_memory_request_ratio",
    "maximum_memory_request_ratio",
    "average_resource_pressure",
    "maximum_resource_pressure",

    # Headroom / exceedance indicators
    "cpu_request_headroom",
    "memory_request_headroom",
    "cpu_request_exceeded",
    "memory_request_exceeded",

    # Burst and volatility
    "cpu_peak_to_average_ratio",
    "memory_peak_to_average_ratio",

    # CPU distribution features
    "cpu_distribution_minimum",
    "cpu_distribution_median",
    "cpu_distribution_maximum",
    "cpu_distribution_mean",
    "cpu_distribution_std",
    "cpu_distribution_range",

    # Tail CPU distribution
    "tail_cpu_distribution_minimum",
    "tail_cpu_distribution_median",
    "tail_cpu_distribution_maximum",
    "tail_cpu_distribution_mean",
    "tail_cpu_distribution_std",
    "tail_cpu_distribution_range",

    # Hardware-performance counters
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
    "cycles_per_instruction_missing",
    "memory_accesses_per_instruction_missing",
]

# Explicitly forbidden leakage columns.
FORBIDDEN_COLUMNS = {
    "event",
    "instance_events_type",
    "collections_events_type",
    "sla_failure",
    "sla_violation",
    "end_time",
    "end_time_seconds",
}


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def temporal_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()

    if "time_seconds" not in frame.columns:
        raise ValueError(
            "time_seconds is required for temporal splitting."
        )

    frame["ordering_time"] = pd.to_numeric(
        frame["time_seconds"],
        errors="coerce",
    )

    invalid = (
        frame["ordering_time"].isna()
        | (frame["ordering_time"] < 0)
        | (frame["ordering_time"] > 3_000_000)
    )
    frame.loc[invalid, "ordering_time"] = np.nan

    sort_columns = ["ordering_time"]
    if "record_id" in frame.columns:
        sort_columns.append("record_id")

    frame = frame.sort_values(
        sort_columns,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    train_end = int(len(frame) * 0.60)
    validation_end = int(len(frame) * 0.80)

    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def balanced_sample(
    frame: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    positive_size = sample_size // 2
    negative_size = sample_size - positive_size

    positives = frame[frame[TARGET] == 1]
    negatives = frame[frame[TARGET] == 0]

    if len(positives) < positive_size:
        raise ValueError(
            f"Need {positive_size} positive rows but only "
            f"{len(positives)} are available."
        )

    if len(negatives) < negative_size:
        raise ValueError(
            f"Need {negative_size} negative rows but only "
            f"{len(negatives)} are available."
        )

    sampled = pd.concat(
        [
            positives.sample(
                n=positive_size,
                random_state=seed,
            ),
            negatives.sample(
                n=negative_size,
                random_state=seed + 1,
            ),
        ],
        ignore_index=True,
    )

    return sampled.sample(
        frac=1.0,
        random_state=seed + 2,
    ).reset_index(drop=True)


def select_features(
    train_frame: pd.DataFrame,
    candidate_features: list[str],
    top_k: int,
) -> tuple[list[str], pd.DataFrame]:
    available_features = [
        column
        for column in candidate_features
        if column in train_frame.columns
        and column not in FORBIDDEN_COLUMNS
    ]

    if len(available_features) < 2:
        raise ValueError(
            "Too few valid feature columns are available."
        )

    feature_frame = train_frame[
        available_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(feature_frame)

    y_train = train_frame[TARGET].to_numpy(dtype=int)

    scores = mutual_info_classif(
        imputed,
        y_train,
        discrete_features="auto",
        random_state=SEED,
    )

    ranking = pd.DataFrame(
        {
            "feature": available_features,
            "mutual_information": scores,
        }
    ).sort_values(
        "mutual_information",
        ascending=False,
    ).reset_index(drop=True)

    selected = ranking.head(
        min(top_k, len(ranking))
    )["feature"].tolist()

    return selected, ranking


def normalize_train_kernel(
    kernel: np.ndarray,
) -> np.ndarray:
    diagonal = np.sqrt(
        np.clip(np.diag(kernel), 1e-12, None)
    )

    normalized = kernel / np.outer(
        diagonal,
        diagonal,
    )

    return np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def normalize_cross_kernel(
    kernel: np.ndarray,
    left_diagonal: np.ndarray,
    right_diagonal: np.ndarray,
) -> np.ndarray:
    denominator = np.sqrt(
        np.outer(
            np.clip(left_diagonal, 1e-12, None),
            np.clip(right_diagonal, 1e-12, None),
        )
    )

    normalized = kernel / denominator

    return np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def construct_three_kernel_fusion(
    classical_kernel: np.ndarray,
    quantum_kernel: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    interaction_kernel = (
        classical_kernel * quantum_kernel
    )

    return (
        weights[0] * classical_kernel
        + weights[1] * quantum_kernel
        + weights[2] * interaction_kernel
    )


def calculate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(
            y_true,
            predictions,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            scores,
        ),
        "pr_auc": average_precision_score(
            y_true,
            scores,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_search(
    y_validation: np.ndarray,
    validation_scores: np.ndarray,
    false_negative_cost: float = 2.0,
    false_positive_cost: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    candidates = np.linspace(
        float(validation_scores.min()),
        float(validation_scores.max()),
        THRESHOLD_POINTS,
    )

    rows: list[dict[str, float]] = []

    best_threshold = 0.0
    best_utility = -np.inf

    for threshold in candidates:
        predictions = (
            validation_scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )
        mcc = matthews_corrcoef(
            y_validation,
            predictions,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )
        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        normalized_cost = (
            false_negative_cost * fn
            + false_positive_cost * fp
        ) / len(y_validation)

        utility = (
            0.45 * f1
            + 0.25 * ((mcc + 1.0) / 2.0)
            + 0.20 * recall
            + 0.10 * precision
            - 0.10 * normalized_cost
        )

        rows.append(
            {
                "threshold": float(threshold),
                "utility": float(utility),
                "f1": float(f1),
                "mcc": float(mcc),
                "recall": float(recall),
                "precision": float(precision),
                "false_negative": int(fn),
                "false_positive": int(fp),
            }
        )

        if utility > best_utility:
            best_utility = utility
            best_threshold = float(threshold)

    return best_threshold, pd.DataFrame(rows)


def validation_objective(
    y_validation: np.ndarray,
    validation_scores: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    threshold, _ = threshold_search(
        y_validation,
        validation_scores,
    )

    predictions = (
        validation_scores >= threshold
    ).astype(int)

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )
    mcc = matthews_corrcoef(
        y_validation,
        predictions,
    )
    roc_auc = roc_auc_score(
        y_validation,
        validation_scores,
    )
    pr_auc = average_precision_score(
        y_validation,
        validation_scores,
    )

    normalized_mcc = (mcc + 1.0) / 2.0

    score = (
        OBJECTIVE_WEIGHTS["roc_auc"] * roc_auc
        + OBJECTIVE_WEIGHTS["pr_auc"] * pr_auc
        + OBJECTIVE_WEIGHTS["f1"] * f1
        + OBJECTIVE_WEIGHTS["mcc"] * normalized_mcc
    )

    return score, threshold, {
        "f1": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Runtime dataset not found: {DATA_PATH}"
        )

    print(f"Loading runtime dataset: {DATA_PATH}")

    data = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    if TARGET not in data.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    leakage_found = sorted(
        set(data.columns) & FORBIDDEN_COLUMNS
    )

    print(
        "Leakage-sensitive columns present in source dataset: "
        + ", ".join(leakage_found)
    )
    print(
        "Only the target is used for labels. Other forbidden "
        "columns are excluded from model features."
    )

    train_full, validation_full, test_full = (
        temporal_split(data)
    )

    train = balanced_sample(
        train_full,
        TRAIN_SIZE,
        SEED,
    )
    validation = balanced_sample(
        validation_full,
        VALIDATION_SIZE,
        SEED + 10,
    )
    test = balanced_sample(
        test_full,
        TEST_SIZE,
        SEED + 20,
    )

    print("Selecting features using training data only")

    selected_features, feature_ranking = (
        select_features(
            train,
            CANDIDATE_FEATURES,
            TOP_K_FEATURES,
        )
    )

    print(
        "Selected quantum features:\n- "
        + "\n- ".join(selected_features)
    )

    feature_ranking.to_csv(
        RESULT_DIR / "feature_ranking.csv",
        index=False,
    )

    imputer = SimpleImputer(strategy="median")
    robust_scaler = RobustScaler()
    quantum_scaler = MinMaxScaler(
        feature_range=(0.0, np.pi)
    )

    train_numeric = train[
        selected_features
    ].apply(pd.to_numeric, errors="coerce")

    validation_numeric = validation[
        selected_features
    ].apply(pd.to_numeric, errors="coerce")

    test_numeric = test[
        selected_features
    ].apply(pd.to_numeric, errors="coerce")

    x_train_imputed = imputer.fit_transform(
        train_numeric
    )
    x_validation_imputed = imputer.transform(
        validation_numeric
    )
    x_test_imputed = imputer.transform(
        test_numeric
    )

    # Classical representation
    x_train_classical = robust_scaler.fit_transform(
        x_train_imputed
    )
    x_validation_classical = robust_scaler.transform(
        x_validation_imputed
    )
    x_test_classical = robust_scaler.transform(
        x_test_imputed
    )

    # Quantum angle representation
    x_train_quantum = quantum_scaler.fit_transform(
        x_train_imputed
    )
    x_validation_quantum = quantum_scaler.transform(
        x_validation_imputed
    )
    x_test_quantum = quantum_scaler.transform(
        x_test_imputed
    )

    y_train = train[TARGET].to_numpy(dtype=int)
    y_validation = validation[
        TARGET
    ].to_numpy(dtype=int)
    y_test = test[TARGET].to_numpy(dtype=int)

    print("Constructing quantum feature map")

    feature_map = ZZFeatureMap(
        feature_dimension=len(selected_features),
        reps=FEATURE_MAP_REPS,
        entanglement=FEATURE_MAP_ENTANGLEMENT,
    )

    quantum_kernel = FidelityStatevectorKernel(
        feature_map=feature_map,
        enforce_psd=True,
    )

    print("Computing quantum kernels")
    quantum_start = time.perf_counter()

    kq_train_raw = quantum_kernel.evaluate(
        x_vec=x_train_quantum,
    )
    kq_validation_raw = quantum_kernel.evaluate(
        x_vec=x_validation_quantum,
        y_vec=x_train_quantum,
    )
    kq_test_raw = quantum_kernel.evaluate(
        x_vec=x_test_quantum,
        y_vec=x_train_quantum,
    )

    validation_self_q = quantum_kernel.evaluate(
        x_vec=x_validation_quantum,
    )
    test_self_q = quantum_kernel.evaluate(
        x_vec=x_test_quantum,
    )

    quantum_kernel_seconds = (
        time.perf_counter() - quantum_start
    )

    kq_train = normalize_train_kernel(
        kq_train_raw
    )
    kq_validation = normalize_cross_kernel(
        kq_validation_raw,
        np.diag(validation_self_q),
        np.diag(kq_train_raw),
    )
    kq_test = normalize_cross_kernel(
        kq_test_raw,
        np.diag(test_self_q),
        np.diag(kq_train_raw),
    )

    optimization_history: list[
        dict[str, float]
    ] = []

    evaluation_counter = 0

    def objective(
        parameters: np.ndarray,
    ) -> float:
        nonlocal evaluation_counter
        evaluation_counter += 1

        weights = softmax(parameters[:3])
        gamma = float(np.exp(parameters[3]))
        svm_c = float(np.exp(parameters[4]))

        kc_train = normalize_train_kernel(
            rbf_kernel(
                x_train_classical,
                x_train_classical,
                gamma=gamma,
            )
        )
        kc_validation = rbf_kernel(
            x_validation_classical,
            x_train_classical,
            gamma=gamma,
        )

        fused_train = construct_three_kernel_fusion(
            kc_train,
            kq_train,
            weights,
        )
        fused_validation = construct_three_kernel_fusion(
            kc_validation,
            kq_validation,
            weights,
        )

        model = SVC(
            kernel="precomputed",
            C=svm_c,
            class_weight="balanced",
            random_state=SEED,
        )
        model.fit(
            fused_train,
            y_train,
        )

        validation_scores = model.decision_function(
            fused_validation
        )

        score, threshold, metric_values = (
            validation_objective(
                y_validation,
                validation_scores,
            )
        )

        optimization_history.append(
            {
                "evaluation": evaluation_counter,
                "objective": float(score),
                "classical_weight": float(weights[0]),
                "quantum_weight": float(weights[1]),
                "interaction_weight": float(weights[2]),
                "gamma": gamma,
                "svm_c": svm_c,
                "threshold": threshold,
                **metric_values,
            }
        )

        return -score

    print(
        "Optimizing three-kernel fusion, RBF gamma, "
        "and SVM C"
    )

    optimization_start = time.perf_counter()

    optimization_result = differential_evolution(
        objective,
        bounds=[
            (-4.0, 4.0),   # classical logit
            (-4.0, 4.0),   # quantum logit
            (-4.0, 4.0),   # interaction logit
            (np.log(0.01), np.log(10.0)),  # gamma
            (np.log(0.1), np.log(100.0)),  # C
        ],
        seed=SEED,
        maxiter=OPTIMIZER_MAXITER,
        popsize=OPTIMIZER_POPSIZE,
        workers=1,
        updating="immediate",
        polish=True,
        disp=True,
    )

    optimization_seconds = (
        time.perf_counter() - optimization_start
    )

    best_parameters = optimization_result.x
    best_weights = softmax(
        best_parameters[:3]
    )
    best_gamma = float(
        np.exp(best_parameters[3])
    )
    best_c = float(
        np.exp(best_parameters[4])
    )

    print("\nBest fusion weights")
    print(
        f"Classical:   {best_weights[0]:.4f}\n"
        f"Quantum:     {best_weights[1]:.4f}\n"
        f"Interaction: {best_weights[2]:.4f}"
    )
    print(
        f"Best gamma: {best_gamma:.6f}\n"
        f"Best C:     {best_c:.6f}"
    )

    kc_train = normalize_train_kernel(
        rbf_kernel(
            x_train_classical,
            x_train_classical,
            gamma=best_gamma,
        )
    )
    kc_validation = rbf_kernel(
        x_validation_classical,
        x_train_classical,
        gamma=best_gamma,
    )
    kc_test = rbf_kernel(
        x_test_classical,
        x_train_classical,
        gamma=best_gamma,
    )

    fused_train = construct_three_kernel_fusion(
        kc_train,
        kq_train,
        best_weights,
    )
    fused_validation = construct_three_kernel_fusion(
        kc_validation,
        kq_validation,
        best_weights,
    )
    fused_test = construct_three_kernel_fusion(
        kc_test,
        kq_test,
        best_weights,
    )

    print("Training final AQUA-SLA v2 model")

    training_start = time.perf_counter()

    final_model = SVC(
        kernel="precomputed",
        C=best_c,
        class_weight="balanced",
        random_state=SEED,
    )
    final_model.fit(
        fused_train,
        y_train,
    )

    training_seconds = (
        time.perf_counter() - training_start
    )

    validation_scores = (
        final_model.decision_function(
            fused_validation
        )
    )
    test_scores = (
        final_model.decision_function(
            fused_test
        )
    )

    best_threshold, threshold_results = (
        threshold_search(
            y_validation,
            validation_scores,
            false_negative_cost=2.0,
            false_positive_cost=1.0,
        )
    )

    validation_predictions = (
        validation_scores >= best_threshold
    ).astype(int)
    test_predictions = (
        test_scores >= best_threshold
    ).astype(int)

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
        validation_scores,
    )
    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_scores,
    )

    print("\nAQUA-SLA v2 validation")
    print(
        f"Accuracy={validation_metrics['accuracy']:.4f}, "
        f"F1={validation_metrics['f1']:.4f}, "
        f"MCC={validation_metrics['mcc']:.4f}, "
        f"ROC-AUC={validation_metrics['roc_auc']:.4f}, "
        f"PR-AUC={validation_metrics['pr_auc']:.4f}"
    )

    print("\nAQUA-SLA v2 test")
    print(
        f"Accuracy={test_metrics['accuracy']:.4f}, "
        f"F1={test_metrics['f1']:.4f}, "
        f"MCC={test_metrics['mcc']:.4f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}, "
        f"PR-AUC={test_metrics['pr_auc']:.4f}"
    )

    pd.DataFrame(
        optimization_history
    ).to_csv(
        RESULT_DIR / "optimization_history.csv",
        index=False,
    )

    threshold_results.to_csv(
        RESULT_DIR / "threshold_search.csv",
        index=False,
    )

    prediction_frame = pd.DataFrame(
        {
            "record_id": (
                test["record_id"].to_numpy()
                if "record_id" in test.columns
                else np.arange(len(test))
            ),
            "actual": y_test,
            "decision_score": test_scores,
            "threshold": best_threshold,
            "prediction": test_predictions,
        }
    )
    prediction_frame.to_csv(
        RESULT_DIR / "test_predictions.csv",
        index=False,
    )

    results = pd.DataFrame(
        [
            {
                "model": "AQUA-SLA-v2",
                "split": "validation",
                "threshold": best_threshold,
                "classical_weight": best_weights[0],
                "quantum_weight": best_weights[1],
                "interaction_weight": best_weights[2],
                "rbf_gamma": best_gamma,
                "svm_c": best_c,
                **validation_metrics,
            },
            {
                "model": "AQUA-SLA-v2",
                "split": "test",
                "threshold": best_threshold,
                "classical_weight": best_weights[0],
                "quantum_weight": best_weights[1],
                "interaction_weight": best_weights[2],
                "rbf_gamma": best_gamma,
                "svm_c": best_c,
                **test_metrics,
            },
        ]
    )
    results.to_csv(
        RESULT_DIR / "aqua_sla_v2_results.csv",
        index=False,
    )

    np.save(
        MODEL_DIR / "best_fusion_weights.npy",
        best_weights,
    )
    np.save(
        MODEL_DIR / "best_parameters.npy",
        best_parameters,
    )
    np.save(
        MODEL_DIR / "selected_features.npy",
        np.array(selected_features, dtype=object),
    )

    metadata = {
        "model_name": "AQUA-SLA-v2",
        "dataset": str(DATA_PATH),
        "target": TARGET,
        "selected_features": selected_features,
        "top_k_features": TOP_K_FEATURES,
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": FEATURE_MAP_REPS,
        "feature_map_entanglement": (
            FEATURE_MAP_ENTANGLEMENT
        ),
        "fusion_equation": (
            "K = w_c*K_c + w_q*K_q + "
            "w_i*(K_c elementwise K_q)"
        ),
        "weights": {
            "classical": float(best_weights[0]),
            "quantum": float(best_weights[1]),
            "interaction": float(best_weights[2]),
        },
        "rbf_gamma": best_gamma,
        "svm_c": best_c,
        "decision_threshold": best_threshold,
        "threshold_objective": (
            "SLA-aware utility with higher "
            "false-negative cost"
        ),
        "validation_objective_weights": (
            OBJECTIVE_WEIGHTS
        ),
        "quantum_kernel_seconds": (
            quantum_kernel_seconds
        ),
        "optimization_seconds": (
            optimization_seconds
        ),
        "training_seconds": training_seconds,
        "optimizer_success": bool(
            optimization_result.success
        ),
        "optimizer_message": str(
            optimization_result.message
        ),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "random_seed": SEED,
        "leakage_policy": {
            "forbidden_columns": sorted(
                FORBIDDEN_COLUMNS
            ),
            "feature_selection_uses_training_only": True,
            "threshold_selected_on_validation_only": True,
            "test_used_once_for_final_evaluation": True,
        },
    }

    with (
        RESULT_DIR / "aqua_sla_v2_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(f"\nResults saved to: {RESULT_DIR}")
    print(f"Model parameters saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
