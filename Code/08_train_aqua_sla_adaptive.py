from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path(r"D:\other\AQUA-SLA")

MODEL_DIR = PROJECT_DIR / "models" / "quantum"
RESULT_DIR = PROJECT_DIR / "results" / "quantum"

TARGET = "sla_violation"

FEATURE_COLUMNS = [
    "priority",
    "scheduling_class",
    "requested_cpu",
    "requested_memory",
]

TRAIN_SIZE = 1_000
VALIDATION_SIZE = 300
TEST_SIZE = 500

SVM_C = 10.0
THRESHOLD_CANDIDATES = 500


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


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

    bounded_scores = sigmoid(scores)

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
        "f1_score": f1_score(
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
        "brier_score_approx": brier_score_loss(
            y_true,
            bounded_scores,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def validate_required_files() -> None:
    required_files = [
        MODEL_DIR / "quantum_kernel_train.npy",
        MODEL_DIR / "quantum_kernel_validation.npy",
        MODEL_DIR / "quantum_kernel_test.npy",
        MODEL_DIR / "classical_kernel_train.npy",
        MODEL_DIR / "classical_kernel_validation.npy",
        MODEL_DIR / "classical_kernel_test.npy",
        MODEL_DIR / "quantum_x_train.npy",
        MODEL_DIR / "quantum_x_validation.npy",
        MODEL_DIR / "quantum_x_test.npy",
        MODEL_DIR / "quantum_y_train.npy",
        MODEL_DIR / "quantum_y_validation.npy",
        MODEL_DIR / "quantum_y_test.npy",
        RESULT_DIR / "quantum_experiment_samples.csv",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "Required files are missing. Run the updated "
            "07_train_quantum_models.py first.\n"
            f"{formatted}"
        )


def load_arrays() -> dict[str, np.ndarray]:
    arrays = {
        "kq_train": np.load(
            MODEL_DIR / "quantum_kernel_train.npy"
        ),
        "kq_validation": np.load(
            MODEL_DIR / "quantum_kernel_validation.npy"
        ),
        "kq_test": np.load(
            MODEL_DIR / "quantum_kernel_test.npy"
        ),
        "kc_train": np.load(
            MODEL_DIR / "classical_kernel_train.npy"
        ),
        "kc_validation": np.load(
            MODEL_DIR / "classical_kernel_validation.npy"
        ),
        "kc_test": np.load(
            MODEL_DIR / "classical_kernel_test.npy"
        ),
        "x_train": np.load(
            MODEL_DIR / "quantum_x_train.npy"
        ),
        "x_validation": np.load(
            MODEL_DIR / "quantum_x_validation.npy"
        ),
        "x_test": np.load(
            MODEL_DIR / "quantum_x_test.npy"
        ),
        "y_train": np.load(
            MODEL_DIR / "quantum_y_train.npy"
        ).astype(int),
        "y_validation": np.load(
            MODEL_DIR / "quantum_y_validation.npy"
        ).astype(int),
        "y_test": np.load(
            MODEL_DIR / "quantum_y_test.npy"
        ).astype(int),
    }

    expected_shapes = {
        "kq_train": (TRAIN_SIZE, TRAIN_SIZE),
        "kq_validation": (
            VALIDATION_SIZE,
            TRAIN_SIZE,
        ),
        "kq_test": (
            TEST_SIZE,
            TRAIN_SIZE,
        ),
        "kc_train": (TRAIN_SIZE, TRAIN_SIZE),
        "kc_validation": (
            VALIDATION_SIZE,
            TRAIN_SIZE,
        ),
        "kc_test": (
            TEST_SIZE,
            TRAIN_SIZE,
        ),
        "x_train": (
            TRAIN_SIZE,
            len(FEATURE_COLUMNS),
        ),
        "x_validation": (
            VALIDATION_SIZE,
            len(FEATURE_COLUMNS),
        ),
        "x_test": (
            TEST_SIZE,
            len(FEATURE_COLUMNS),
        ),
        "y_train": (TRAIN_SIZE,),
        "y_validation": (VALIDATION_SIZE,),
        "y_test": (TEST_SIZE,),
    }

    for name, expected_shape in expected_shapes.items():
        actual_shape = arrays[name].shape

        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} has shape {actual_shape}; "
                f"expected {expected_shape}"
            )

        if not np.isfinite(arrays[name]).all():
            raise ValueError(
                f"{name} contains non-finite values"
            )

    return arrays


def load_sample_manifest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_path = (
        RESULT_DIR / "quantum_experiment_samples.csv"
    )

    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    required_columns = {
        "record_id",
        TARGET,
        "split",
        *FEATURE_COLUMNS,
    }

    missing_columns = sorted(
        required_columns - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Sample manifest is missing columns: "
            + ", ".join(missing_columns)
        )

    train = manifest[
        manifest["split"] == "train"
    ].reset_index(drop=True)

    validation = manifest[
        manifest["split"] == "validation"
    ].reset_index(drop=True)

    test = manifest[
        manifest["split"] == "test"
    ].reset_index(drop=True)

    expected_sizes = {
        "train": (train, TRAIN_SIZE),
        "validation": (
            validation,
            VALIDATION_SIZE,
        ),
        "test": (test, TEST_SIZE),
    }

    for split_name, (
        split_frame,
        expected_size,
    ) in expected_sizes.items():
        if len(split_frame) != expected_size:
            raise ValueError(
                f"{split_name} manifest contains "
                f"{len(split_frame)} rows; "
                f"expected {expected_size}"
            )

    return train, validation, test


def search_decision_threshold(
    y_validation: np.ndarray,
    validation_scores: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    if np.allclose(
        validation_scores.min(),
        validation_scores.max(),
    ):
        threshold_candidates = np.array(
            [float(validation_scores.min())]
        )
    else:
        threshold_candidates = np.linspace(
            validation_scores.min(),
            validation_scores.max(),
            THRESHOLD_CANDIDATES,
        )

    threshold_rows: list[dict[str, float]] = []

    best_threshold = 0.0
    best_f1 = -np.inf
    best_mcc = -np.inf

    for threshold in threshold_candidates:
        predictions = (
            validation_scores >= threshold
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

        candidate_precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        candidate_recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        threshold_rows.append(
            {
                "threshold": float(threshold),
                "validation_f1": float(
                    candidate_f1
                ),
                "validation_mcc": float(
                    candidate_mcc
                ),
                "validation_precision": float(
                    candidate_precision
                ),
                "validation_recall": float(
                    candidate_recall
                ),
            }
        )

        is_better = (
            candidate_f1 > best_f1
            or (
                np.isclose(candidate_f1, best_f1)
                and candidate_mcc > best_mcc
            )
        )

        if is_better:
            best_f1 = candidate_f1
            best_mcc = candidate_mcc
            best_threshold = float(threshold)

    return (
        best_threshold,
        pd.DataFrame(threshold_rows),
    )


def main() -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_required_files()

    print("Loading saved kernel matrices and experiment arrays")

    arrays = load_arrays()

    kq_train = arrays["kq_train"]
    kq_validation = arrays["kq_validation"]
    kq_test = arrays["kq_test"]

    kc_train = arrays["kc_train"]
    kc_validation = arrays["kc_validation"]
    kc_test = arrays["kc_test"]

    x_train = arrays["x_train"]
    x_validation = arrays["x_validation"]
    x_test = arrays["x_test"]

    y_train = arrays["y_train"]
    y_validation = arrays["y_validation"]
    y_test = arrays["y_test"]

    train_manifest, validation_manifest, test_manifest = (
        load_sample_manifest()
    )

    if not np.array_equal(
        train_manifest[TARGET].to_numpy(dtype=int),
        y_train,
    ):
        raise ValueError(
            "Training labels do not match the saved sample manifest."
        )

    if not np.array_equal(
        validation_manifest[TARGET].to_numpy(dtype=int),
        y_validation,
    ):
        raise ValueError(
            "Validation labels do not match the saved sample manifest."
        )

    if not np.array_equal(
        test_manifest[TARGET].to_numpy(dtype=int),
        y_test,
    ):
        raise ValueError(
            "Test labels do not match the saved sample manifest."
        )

    feature_count = x_train.shape[1]

    def get_alpha(
        features: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        weights = parameters[:feature_count]
        bias = parameters[-1]

        return sigmoid(
            features @ weights + bias
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

        train_alpha = get_alpha(
            x_train,
            parameters,
        )

        validation_alpha = get_alpha(
            x_validation,
            parameters,
        )

        adaptive_train = adaptive_kernel(
            kc_train,
            kq_train,
            train_alpha,
            train_alpha,
        )

        adaptive_validation = adaptive_kernel(
            kc_validation,
            kq_validation,
            validation_alpha,
            train_alpha,
        )

        model = SVC(
            kernel="precomputed",
            C=SVM_C,
            class_weight="balanced",
            random_state=SEED,
        )

        model.fit(
            adaptive_train,
            y_train,
        )

        validation_scores = (
            model.decision_function(
                adaptive_validation
            )
        )

        validation_auc = roc_auc_score(
            y_validation,
            validation_scores,
        )

        optimization_history.append(
            {
                "evaluation": evaluation_counter,
                "validation_roc_auc": float(
                    validation_auc
                ),
                "mean_train_alpha": float(
                    train_alpha.mean()
                ),
                "std_train_alpha": float(
                    train_alpha.std()
                ),
                "mean_validation_alpha": float(
                    validation_alpha.mean()
                ),
                "std_validation_alpha": float(
                    validation_alpha.std()
                ),
            }
        )

        return -validation_auc

    parameter_bounds = [
        (-5.0, 5.0)
        for _ in range(feature_count + 1)
    ]

    print("Optimizing adaptive fusion gate")

    optimization_start = time.perf_counter()

    optimization_result = differential_evolution(
        objective,
        bounds=parameter_bounds,
        seed=SEED,
        maxiter=30,
        popsize=8,
        workers=1,
        updating="immediate",
        polish=True,
        disp=False,
    )

    optimization_time = (
        time.perf_counter()
        - optimization_start
    )

    best_parameters = optimization_result.x

    train_alpha = get_alpha(
        x_train,
        best_parameters,
    )

    validation_alpha = get_alpha(
        x_validation,
        best_parameters,
    )

    test_alpha = get_alpha(
        x_test,
        best_parameters,
    )

    adaptive_train = adaptive_kernel(
        kc_train,
        kq_train,
        train_alpha,
        train_alpha,
    )

    adaptive_validation = adaptive_kernel(
        kc_validation,
        kq_validation,
        validation_alpha,
        train_alpha,
    )

    adaptive_test = adaptive_kernel(
        kc_test,
        kq_test,
        test_alpha,
        train_alpha,
    )

    print("Training final AQUA-SLA-Adaptive model")

    training_start = time.perf_counter()

    adaptive_model = SVC(
        kernel="precomputed",
        C=SVM_C,
        class_weight="balanced",
        random_state=SEED,
    )

    adaptive_model.fit(
        adaptive_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    validation_inference_start = (
        time.perf_counter()
    )

    validation_scores = (
        adaptive_model.decision_function(
            adaptive_validation
        )
    )

    validation_inference_time = (
        time.perf_counter()
        - validation_inference_start
    )

    test_inference_start = time.perf_counter()

    test_scores = (
        adaptive_model.decision_function(
            adaptive_test
        )
    )

    test_inference_time = (
        time.perf_counter()
        - test_inference_start
    )

    (
        best_threshold,
        threshold_results,
    ) = search_decision_threshold(
        y_validation,
        validation_scores,
    )

    threshold_results_path = (
        RESULT_DIR
        / "aqua_sla_adaptive_threshold_search.csv"
    )

    threshold_results.to_csv(
        threshold_results_path,
        index=False,
    )

    validation_predictions_default = (
        validation_scores >= 0.0
    ).astype(int)

    test_predictions_default = (
        test_scores >= 0.0
    ).astype(int)

    validation_predictions_optimized = (
        validation_scores >= best_threshold
    ).astype(int)

    test_predictions_optimized = (
        test_scores >= best_threshold
    ).astype(int)

    validation_default_metrics = calculate_metrics(
        y_validation,
        validation_predictions_default,
        validation_scores,
    )

    test_default_metrics = calculate_metrics(
        y_test,
        test_predictions_default,
        test_scores,
    )

    validation_optimized_metrics = calculate_metrics(
        y_validation,
        validation_predictions_optimized,
        validation_scores,
    )

    test_optimized_metrics = calculate_metrics(
        y_test,
        test_predictions_optimized,
        test_scores,
    )

    print(
        f"\nSelected decision threshold: "
        f"{best_threshold:.6f}"
    )

    print("\nAQUA-SLA-Adaptive validation — default threshold")
    print(
        f"F1={validation_default_metrics['f1_score']:.4f}, "
        f"ROC-AUC={validation_default_metrics['roc_auc']:.4f}, "
        f"PR-AUC={validation_default_metrics['pr_auc']:.4f}, "
        f"MCC={validation_default_metrics['mcc']:.4f}"
    )

    print("\nAQUA-SLA-Adaptive validation — optimized threshold")
    print(
        f"F1={validation_optimized_metrics['f1_score']:.4f}, "
        f"ROC-AUC={validation_optimized_metrics['roc_auc']:.4f}, "
        f"PR-AUC={validation_optimized_metrics['pr_auc']:.4f}, "
        f"MCC={validation_optimized_metrics['mcc']:.4f}"
    )

    print("\nAQUA-SLA-Adaptive test — default threshold")
    print(
        f"F1={test_default_metrics['f1_score']:.4f}, "
        f"ROC-AUC={test_default_metrics['roc_auc']:.4f}, "
        f"PR-AUC={test_default_metrics['pr_auc']:.4f}, "
        f"MCC={test_default_metrics['mcc']:.4f}"
    )

    print("\nAQUA-SLA-Adaptive test — optimized threshold")
    print(
        f"F1={test_optimized_metrics['f1_score']:.4f}, "
        f"ROC-AUC={test_optimized_metrics['roc_auc']:.4f}, "
        f"PR-AUC={test_optimized_metrics['pr_auc']:.4f}, "
        f"MCC={test_optimized_metrics['mcc']:.4f}"
    )

    joblib.dump(
        adaptive_model,
        MODEL_DIR / "aqua_sla_adaptive.joblib",
    )

    np.save(
        MODEL_DIR / "adaptive_gate_parameters.npy",
        best_parameters,
    )

    np.save(
        MODEL_DIR / "adaptive_train_alpha.npy",
        train_alpha,
    )

    np.save(
        MODEL_DIR / "adaptive_validation_alpha.npy",
        validation_alpha,
    )

    np.save(
        MODEL_DIR / "adaptive_test_alpha.npy",
        test_alpha,
    )

    pd.DataFrame(
        optimization_history
    ).to_csv(
        RESULT_DIR
        / "aqua_sla_adaptive_optimization_history.csv",
        index=False,
    )

    prediction_output = pd.DataFrame(
        {
            "record_id": test_manifest[
                "record_id"
            ].to_numpy(),
            "actual": y_test,
            "decision_score": test_scores,
            "default_threshold": 0.0,
            "default_prediction": (
                test_predictions_default
            ),
            "optimized_threshold": (
                best_threshold
            ),
            "optimized_prediction": (
                test_predictions_optimized
            ),
            "alpha": test_alpha,
        }
    )

    prediction_output.to_csv(
        RESULT_DIR
        / "aqua_sla_adaptive_test_predictions.csv",
        index=False,
    )

    result_rows = [
        {
            "model": "aqua_sla_adaptive",
            "split": "validation",
            "threshold_type": "default",
            "decision_threshold": 0.0,
            "optimization_time_seconds": (
                optimization_time
            ),
            "training_time_seconds": (
                training_time
            ),
            "inference_time_seconds": (
                validation_inference_time
            ),
            "inference_ms_per_record": (
                validation_inference_time
                / len(y_validation)
                * 1_000
            ),
            "mean_alpha": float(
                validation_alpha.mean()
            ),
            "std_alpha": float(
                validation_alpha.std()
            ),
            "minimum_alpha": float(
                validation_alpha.min()
            ),
            "maximum_alpha": float(
                validation_alpha.max()
            ),
            **validation_default_metrics,
        },
        {
            "model": "aqua_sla_adaptive",
            "split": "validation",
            "threshold_type": "optimized",
            "decision_threshold": (
                best_threshold
            ),
            "optimization_time_seconds": (
                optimization_time
            ),
            "training_time_seconds": (
                training_time
            ),
            "inference_time_seconds": (
                validation_inference_time
            ),
            "inference_ms_per_record": (
                validation_inference_time
                / len(y_validation)
                * 1_000
            ),
            "mean_alpha": float(
                validation_alpha.mean()
            ),
            "std_alpha": float(
                validation_alpha.std()
            ),
            "minimum_alpha": float(
                validation_alpha.min()
            ),
            "maximum_alpha": float(
                validation_alpha.max()
            ),
            **validation_optimized_metrics,
        },
        {
            "model": "aqua_sla_adaptive",
            "split": "test",
            "threshold_type": "default",
            "decision_threshold": 0.0,
            "optimization_time_seconds": (
                optimization_time
            ),
            "training_time_seconds": (
                training_time
            ),
            "inference_time_seconds": (
                test_inference_time
            ),
            "inference_ms_per_record": (
                test_inference_time
                / len(y_test)
                * 1_000
            ),
            "mean_alpha": float(
                test_alpha.mean()
            ),
            "std_alpha": float(
                test_alpha.std()
            ),
            "minimum_alpha": float(
                test_alpha.min()
            ),
            "maximum_alpha": float(
                test_alpha.max()
            ),
            **test_default_metrics,
        },
        {
            "model": "aqua_sla_adaptive",
            "split": "test",
            "threshold_type": "optimized",
            "decision_threshold": (
                best_threshold
            ),
            "optimization_time_seconds": (
                optimization_time
            ),
            "training_time_seconds": (
                training_time
            ),
            "inference_time_seconds": (
                test_inference_time
            ),
            "inference_ms_per_record": (
                test_inference_time
                / len(y_test)
                * 1_000
            ),
            "mean_alpha": float(
                test_alpha.mean()
            ),
            "std_alpha": float(
                test_alpha.std()
            ),
            "minimum_alpha": float(
                test_alpha.min()
            ),
            "maximum_alpha": float(
                test_alpha.max()
            ),
            **test_optimized_metrics,
        },
    ]

    results_frame = pd.DataFrame(
        result_rows
    )

    results_frame.to_csv(
        RESULT_DIR
        / "aqua_sla_adaptive_results.csv",
        index=False,
    )

    metadata = {
        "model_name": "AQUA-SLA-Adaptive",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "fusion": (
            "sample-dependent logistic gate "
            "with pairwise averaged alpha"
        ),
        "gate_equation": (
            "alpha_i = sigmoid(w^T x_i + b)"
        ),
        "pairwise_alpha": (
            "alpha_ij = (alpha_i + alpha_j) / 2"
        ),
        "optimization_objective": (
            "validation ROC-AUC"
        ),
        "threshold_selection_objective": (
            "validation F1 with MCC tie-break"
        ),
        "decision_threshold": (
            best_threshold
        ),
        "optimizer": (
            "scipy differential_evolution"
        ),
        "optimizer_success": bool(
            optimization_result.success
        ),
        "optimizer_message": str(
            optimization_result.message
        ),
        "optimizer_iterations": int(
            optimization_result.nit
        ),
        "optimizer_function_evaluations": int(
            optimization_result.nfev
        ),
        "svm_c": SVM_C,
        "training_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "optimization_time_seconds": (
            optimization_time
        ),
        "training_time_seconds": (
            training_time
        ),
        "best_parameters": (
            best_parameters.tolist()
        ),
        "gate_weights": (
            best_parameters[
                :feature_count
            ].tolist()
        ),
        "gate_bias": float(
            best_parameters[-1]
        ),
        "train_alpha_statistics": {
            "mean": float(
                train_alpha.mean()
            ),
            "std": float(
                train_alpha.std()
            ),
            "minimum": float(
                train_alpha.min()
            ),
            "maximum": float(
                train_alpha.max()
            ),
        },
        "validation_alpha_statistics": {
            "mean": float(
                validation_alpha.mean()
            ),
            "std": float(
                validation_alpha.std()
            ),
            "minimum": float(
                validation_alpha.min()
            ),
            "maximum": float(
                validation_alpha.max()
            ),
        },
        "test_alpha_statistics": {
            "mean": float(
                test_alpha.mean()
            ),
            "std": float(
                test_alpha.std()
            ),
            "minimum": float(
                test_alpha.min()
            ),
            "maximum": float(
                test_alpha.max()
            ),
        },
        "validation_default_metrics": (
            validation_default_metrics
        ),
        "validation_optimized_metrics": (
            validation_optimized_metrics
        ),
        "test_default_metrics": (
            test_default_metrics
        ),
        "test_optimized_metrics": (
            test_optimized_metrics
        ),
        "random_seed": SEED,
        "files": {
            "model": str(
                MODEL_DIR
                / "aqua_sla_adaptive.joblib"
            ),
            "gate_parameters": str(
                MODEL_DIR
                / "adaptive_gate_parameters.npy"
            ),
            "results": str(
                RESULT_DIR
                / "aqua_sla_adaptive_results.csv"
            ),
            "predictions": str(
                RESULT_DIR
                / "aqua_sla_adaptive_test_predictions.csv"
            ),
            "threshold_search": str(
                threshold_results_path
            ),
            "optimization_history": str(
                RESULT_DIR
                / "aqua_sla_adaptive_optimization_history.csv"
            ),
        },
    }

    with (
        MODEL_DIR
        / "aqua_sla_adaptive_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nFinal AQUA-SLA-Adaptive test comparison")

    print(
        results_frame[
            results_frame["split"] == "test"
        ][
            [
                "threshold_type",
                "decision_threshold",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
                "f1_score",
                "mcc",
                "roc_auc",
                "pr_auc",
                "false_negative",
            ]
        ].to_string(index=False)
    )

    print(f"\nModel saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
