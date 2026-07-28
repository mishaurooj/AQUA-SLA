from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from scipy.optimize import differential_evolution
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
DATA_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_early_prediction.csv"
)
RESULT_DIR = PROJECT_DIR / "results" / "quantum" / "repeated"

TARGET = "sla_violation"
FEATURE_COLUMNS = [
    "priority",
    "scheduling_class",
    "requested_cpu",
    "requested_memory",
]

SEEDS = [11, 22, 33, 44, 55]
TRAIN_SIZE = 1_000
VALIDATION_SIZE = 300
TEST_SIZE = 500

FEATURE_MAP_REPS = 2
ENTANGLEMENT = "linear"
RBF_GAMMA = 1.0
SVM_C = 10.0

GATE_MAXITER = 20
GATE_POPSIZE = 6
THRESHOLD_POINTS = 500


def temporal_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["ordering_time"] = pd.to_numeric(
        frame["time_seconds"],
        errors="coerce",
    )

    invalid_time = (
        frame["ordering_time"].isna()
        | (frame["ordering_time"] < 0)
        | (frame["ordering_time"] > 3_000_000)
    )
    frame.loc[invalid_time, "ordering_time"] = np.nan

    frame = frame.sort_values(
        ["ordering_time", "record_id"],
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
        raise ValueError("Insufficient positive samples.")
    if len(negatives) < negative_size:
        raise ValueError("Insufficient negative samples.")

    sampled = pd.concat(
        [
            positives.sample(
                n=positive_size,
                random_state=seed,
            ),
            negatives.sample(
                n=negative_size,
                random_state=seed,
            ),
        ],
        ignore_index=True,
    )

    return sampled.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (
        1.0 + np.exp(-np.clip(values, -30.0, 30.0))
    )


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
        np.outer(left_diagonal, right_diagonal)
    )
    normalized = kernel / np.clip(
        denominator,
        1e-12,
        None,
    )
    return np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def adaptive_kernel(
    classical_kernel: np.ndarray,
    quantum_kernel: np.ndarray,
    left_alpha: np.ndarray,
    right_alpha: np.ndarray,
) -> np.ndarray:
    pair_alpha = (
        left_alpha[:, None]
        + right_alpha[None, :]
    ) / 2.0

    return (
        (1.0 - pair_alpha) * classical_kernel
        + pair_alpha * quantum_kernel
    )


def metrics(
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


def choose_threshold(
    y_validation: np.ndarray,
    scores: np.ndarray,
) -> float:
    candidates = np.linspace(
        float(scores.min()),
        float(scores.max()),
        THRESHOLD_POINTS,
    )

    best_threshold = 0.0
    best_f1 = -np.inf
    best_mcc = -np.inf

    for threshold in candidates:
        predictions = (
            scores >= threshold
        ).astype(int)

        candidate_f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )
        candidate_mcc = matthews_corrcoef(
            y_validation,
            predictions,
        )

        if (
            candidate_f1 > best_f1
            or (
                np.isclose(candidate_f1, best_f1)
                and candidate_mcc > best_mcc
            )
        ):
            best_f1 = candidate_f1
            best_mcc = candidate_mcc
            best_threshold = float(threshold)

    return best_threshold


def compute_kernels(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> dict[str, np.ndarray]:
    feature_map = zz_feature_map(
        feature_dimension=x_train.shape[1],
        reps=FEATURE_MAP_REPS,
        entanglement=ENTANGLEMENT,
    )

    quantum_kernel = FidelityStatevectorKernel(
        feature_map=feature_map,
        enforce_psd=True,
    )

    kq_train = quantum_kernel.evaluate(
        x_vec=x_train,
    )
    kq_validation = quantum_kernel.evaluate(
        x_vec=x_validation,
        y_vec=x_train,
    )
    kq_test = quantum_kernel.evaluate(
        x_vec=x_test,
        y_vec=x_train,
    )

    kc_train = rbf_kernel(
        x_train,
        x_train,
        gamma=RBF_GAMMA,
    )
    kc_validation = rbf_kernel(
        x_validation,
        x_train,
        gamma=RBF_GAMMA,
    )
    kc_test = rbf_kernel(
        x_test,
        x_train,
        gamma=RBF_GAMMA,
    )

    kq_train = normalize_train_kernel(kq_train)
    kc_train = normalize_train_kernel(kc_train)

    train_diagonal = np.ones(len(x_train))
    validation_q_diagonal = np.diag(
        quantum_kernel.evaluate(
            x_vec=x_validation,
        )
    )
    test_q_diagonal = np.diag(
        quantum_kernel.evaluate(
            x_vec=x_test,
        )
    )

    kq_validation = normalize_cross_kernel(
        kq_validation,
        validation_q_diagonal,
        train_diagonal,
    )
    kq_test = normalize_cross_kernel(
        kq_test,
        test_q_diagonal,
        train_diagonal,
    )

    kc_validation = normalize_cross_kernel(
        kc_validation,
        np.ones(len(x_validation)),
        train_diagonal,
    )
    kc_test = normalize_cross_kernel(
        kc_test,
        np.ones(len(x_test)),
        train_diagonal,
    )

    return {
        "kq_train": kq_train,
        "kq_validation": kq_validation,
        "kq_test": kq_test,
        "kc_train": kc_train,
        "kc_validation": kc_validation,
        "kc_test": kc_test,
    }


def fit_and_score(
    model_name: str,
    train_kernel: np.ndarray,
    validation_kernel: np.ndarray,
    test_kernel: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    model = SVC(
        kernel="precomputed",
        C=SVM_C,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train_kernel, y_train)

    validation_scores = model.decision_function(
        validation_kernel
    )
    test_scores = model.decision_function(
        test_kernel
    )

    threshold = choose_threshold(
        y_validation,
        validation_scores,
    )

    test_predictions = (
        test_scores >= threshold
    ).astype(int)

    return {
        "model": model_name,
        "decision_threshold": threshold,
        **metrics(
            y_test,
            test_predictions,
            test_scores,
        ),
    }


def fit_adaptive(
    kernels: dict[str, np.ndarray],
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    feature_count = x_train.shape[1]

    def get_alpha(
        features: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        return sigmoid(
            features @ parameters[:feature_count]
            + parameters[-1]
        )

    def objective(
        parameters: np.ndarray,
    ) -> float:
        train_alpha = get_alpha(
            x_train,
            parameters,
        )
        validation_alpha = get_alpha(
            x_validation,
            parameters,
        )

        train_kernel = adaptive_kernel(
            kernels["kc_train"],
            kernels["kq_train"],
            train_alpha,
            train_alpha,
        )
        validation_kernel = adaptive_kernel(
            kernels["kc_validation"],
            kernels["kq_validation"],
            validation_alpha,
            train_alpha,
        )

        model = SVC(
            kernel="precomputed",
            C=SVM_C,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(train_kernel, y_train)

        validation_scores = model.decision_function(
            validation_kernel
        )

        return -roc_auc_score(
            y_validation,
            validation_scores,
        )

    result = differential_evolution(
        objective,
        bounds=[
            (-5.0, 5.0)
            for _ in range(feature_count + 1)
        ],
        seed=seed,
        maxiter=GATE_MAXITER,
        popsize=GATE_POPSIZE,
        workers=1,
        updating="immediate",
        polish=True,
        disp=False,
    )

    parameters = result.x
    train_alpha = get_alpha(
        x_train,
        parameters,
    )
    validation_alpha = get_alpha(
        x_validation,
        parameters,
    )
    test_alpha = get_alpha(
        x_test,
        parameters,
    )

    train_kernel = adaptive_kernel(
        kernels["kc_train"],
        kernels["kq_train"],
        train_alpha,
        train_alpha,
    )
    validation_kernel = adaptive_kernel(
        kernels["kc_validation"],
        kernels["kq_validation"],
        validation_alpha,
        train_alpha,
    )
    test_kernel = adaptive_kernel(
        kernels["kc_test"],
        kernels["kq_test"],
        test_alpha,
        train_alpha,
    )

    model = SVC(
        kernel="precomputed",
        C=SVM_C,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train_kernel, y_train)

    validation_scores = model.decision_function(
        validation_kernel
    )
    test_scores = model.decision_function(
        test_kernel
    )

    threshold = choose_threshold(
        y_validation,
        validation_scores,
    )
    test_predictions = (
        test_scores >= threshold
    ).astype(int)

    return {
        "model": "aqua_sla_adaptive",
        "decision_threshold": threshold,
        "mean_alpha": float(test_alpha.mean()),
        "std_alpha": float(test_alpha.std()),
        "gate_success": bool(result.success),
        **metrics(
            y_test,
            test_predictions,
            test_scores,
        ),
    }


def summarize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]

    rows: list[dict[str, object]] = []

    for model_name, group in results.groupby("model"):
        row: dict[str, object] = {
            "model": model_name,
            "runs": len(group),
        }

        for metric_name in metric_columns:
            values = group[
                metric_name
            ].to_numpy(dtype=float)

            mean_value = float(values.mean())
            std_value = float(
                values.std(ddof=1)
                if len(values) > 1
                else 0.0
            )
            ci_half_width = float(
                1.96 * std_value / np.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )

            row[f"{metric_name}_mean"] = mean_value
            row[f"{metric_name}_std"] = std_value
            row[
                f"{metric_name}_ci95_low"
            ] = mean_value - ci_half_width
            row[
                f"{metric_name}_ci95_high"
            ] = mean_value + ci_half_width

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    required_columns = [
        "record_id",
        "time_seconds",
        TARGET,
        *FEATURE_COLUMNS,
    ]

    print(f"Loading dataset: {DATA_PATH}")

    data = pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
        low_memory=False,
    )

    train_full, validation_full, test_full = (
        temporal_split(data)
    )

    all_results: list[dict[str, object]] = []

    for seed in SEEDS:
        print(f"\n===== Seed {seed} =====")

        random.seed(seed)
        np.random.seed(seed)

        train = balanced_sample(
            train_full,
            TRAIN_SIZE,
            seed,
        )
        validation = balanced_sample(
            validation_full,
            VALIDATION_SIZE,
            seed + 1,
        )
        test = balanced_sample(
            test_full,
            TEST_SIZE,
            seed + 2,
        )

        imputer = SimpleImputer(
            strategy="median"
        )
        scaler = MinMaxScaler(
            feature_range=(0.0, np.pi)
        )

        x_train = scaler.fit_transform(
            imputer.fit_transform(
                train[FEATURE_COLUMNS]
            )
        )
        x_validation = scaler.transform(
            imputer.transform(
                validation[FEATURE_COLUMNS]
            )
        )
        x_test = scaler.transform(
            imputer.transform(
                test[FEATURE_COLUMNS]
            )
        )

        y_train = train[TARGET].to_numpy(
            dtype=int
        )
        y_validation = validation[
            TARGET
        ].to_numpy(dtype=int)
        y_test = test[TARGET].to_numpy(
            dtype=int
        )

        kernel_start = time.perf_counter()
        kernels = compute_kernels(
            x_train,
            x_validation,
            x_test,
        )
        kernel_seconds = (
            time.perf_counter() - kernel_start
        )

        pure_quantum = fit_and_score(
            "pure_quantum_kernel_svm",
            kernels["kq_train"],
            kernels["kq_validation"],
            kernels["kq_test"],
            y_train,
            y_validation,
            y_test,
            seed,
        )

        static_hybrid = fit_and_score(
            "static_hcqkl",
            0.5 * kernels["kc_train"]
            + 0.5 * kernels["kq_train"],
            0.5 * kernels["kc_validation"]
            + 0.5 * kernels["kq_validation"],
            0.5 * kernels["kc_test"]
            + 0.5 * kernels["kq_test"],
            y_train,
            y_validation,
            y_test,
            seed,
        )

        adaptive = fit_adaptive(
            kernels,
            x_train,
            x_validation,
            x_test,
            y_train,
            y_validation,
            y_test,
            seed,
        )

        for result in (
            pure_quantum,
            static_hybrid,
            adaptive,
        ):
            result["seed"] = seed
            result[
                "kernel_computation_seconds"
            ] = kernel_seconds
            all_results.append(result)

            print(
                f"{result['model']}: "
                f"F1={result['f1']:.4f}, "
                f"MCC={result['mcc']:.4f}, "
                f"ROC-AUC={result['roc_auc']:.4f}, "
                f"PR-AUC={result['pr_auc']:.4f}"
            )

    results_frame = pd.DataFrame(
        all_results
    )
    summary_frame = summarize_results(
        results_frame
    )

    results_path = (
        RESULT_DIR
        / "repeated_quantum_results.csv"
    )
    summary_path = (
        RESULT_DIR
        / "repeated_quantum_summary.csv"
    )

    results_frame.to_csv(
        results_path,
        index=False,
    )
    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    metadata = {
        "dataset": str(DATA_PATH),
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "seeds": SEEDS,
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "feature_map": "zz_feature_map",
        "feature_map_reps": FEATURE_MAP_REPS,
        "entanglement": ENTANGLEMENT,
        "rbf_gamma": RBF_GAMMA,
        "svm_c": SVM_C,
        "gate_maxiter": GATE_MAXITER,
        "gate_popsize": GATE_POPSIZE,
        "threshold_selection": (
            "validation F1 with MCC tie-break"
        ),
        "confidence_interval": (
            "normal approximation, 95%"
        ),
    }

    with (
        RESULT_DIR
        / "repeated_experiment_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\n===== Aggregate results =====")
    display_columns = [
        "model",
        "f1_mean",
        "f1_std",
        "mcc_mean",
        "mcc_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
    ]
    print(
        summary_frame[
            display_columns
        ].to_string(index=False)
    )

    print(f"\nPer-run results: {results_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()
