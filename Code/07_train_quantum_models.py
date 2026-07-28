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
from sklearn.impute import SimpleImputer
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
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
DATA_PATH = PROJECT_DIR / "Dataset" / "processed" / "aqua_sla_early_prediction.csv"
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
ALPHA_VALUES = np.round(np.arange(0.0, 1.01, 0.1), 2)
RBF_GAMMA = 1.0
SVM_C = 10.0


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["ordering_time"] = pd.to_numeric(frame["time_seconds"], errors="coerce")

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


def balanced_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    positive = frame[frame[TARGET] == 1]
    negative = frame[frame[TARGET] == 0]

    positive_size = sample_size // 2
    negative_size = sample_size - positive_size

    if len(positive) < positive_size:
        raise ValueError(
            f"Insufficient positive records: requested {positive_size}, available {len(positive)}"
        )
    if len(negative) < negative_size:
        raise ValueError(
            f"Insufficient negative records: requested {negative_size}, available {len(negative)}"
        )

    sampled = pd.concat(
        [
            positive.sample(n=positive_size, random_state=seed),
            negative.sample(n=negative_size, random_state=seed),
        ],
        ignore_index=True,
    )

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    diagonal = np.sqrt(np.clip(np.diag(kernel), 1e-12, None))
    normalized = kernel / np.outer(diagonal, diagonal)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_cross_kernel(
    cross_kernel: np.ndarray,
    left_diagonal: np.ndarray,
    right_diagonal: np.ndarray,
) -> np.ndarray:
    denominator = np.sqrt(np.outer(left_diagonal, right_diagonal))
    normalized = cross_kernel / np.clip(denominator, 1e-12, None)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def calculate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    decision_scores: np.ndarray,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bounded_scores = 1.0 / (1.0 + np.exp(-np.clip(decision_scores, -30.0, 30.0)))

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "mcc": matthews_corrcoef(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, decision_scores),
        "pr_auc": average_precision_score(y_true, decision_scores),
        "brier_score_approx": brier_score_loss(y_true, bounded_scores),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def evaluate_model(
    model_name: str,
    model: SVC,
    kernel_matrix: np.ndarray,
    labels: np.ndarray,
    split_name: str,
) -> dict[str, object]:
    start = time.perf_counter()
    predictions = model.predict(kernel_matrix)
    decision_scores = model.decision_function(kernel_matrix)
    inference_time = time.perf_counter() - start

    return {
        "model": model_name,
        "split": split_name,
        "inference_time_seconds": inference_time,
        "inference_ms_per_record": inference_time / len(labels) * 1_000,
        **calculate_metrics(labels, predictions, decision_scores),
    }


def verify_kernels(kernels: dict[str, np.ndarray]) -> None:
    expected_shapes = {
        "kq_train": (TRAIN_SIZE, TRAIN_SIZE),
        "kq_validation": (VALIDATION_SIZE, TRAIN_SIZE),
        "kq_test": (TEST_SIZE, TRAIN_SIZE),
        "kc_train": (TRAIN_SIZE, TRAIN_SIZE),
        "kc_validation": (VALIDATION_SIZE, TRAIN_SIZE),
        "kc_test": (TEST_SIZE, TRAIN_SIZE),
    }

    for kernel_name, expected_shape in expected_shapes.items():
        kernel_matrix = kernels[kernel_name]
        if kernel_matrix.shape != expected_shape:
            raise ValueError(
                f"{kernel_name} has shape {kernel_matrix.shape}; expected {expected_shape}"
            )
        if not np.isfinite(kernel_matrix).all():
            raise ValueError(f"{kernel_name} contains non-finite values")

    print("\nKernel verification completed")
    for kernel_name, kernel_matrix in kernels.items():
        print(
            f"{kernel_name}: shape={kernel_matrix.shape}, "
            f"min={kernel_matrix.min():.6f}, max={kernel_matrix.max():.6f}"
        )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    print(f"Loading dataset: {DATA_PATH}")

    required_columns = ["record_id", "time_seconds", TARGET, *FEATURE_COLUMNS]
    data = pd.read_csv(DATA_PATH, usecols=required_columns, low_memory=False)

    train_full, validation_full, test_full = temporal_split(data)
    train = balanced_sample(train_full, TRAIN_SIZE, SEED)
    validation = balanced_sample(validation_full, VALIDATION_SIZE, SEED + 1)
    test = balanced_sample(test_full, TEST_SIZE, SEED + 2)

    print("\nQuantum experiment sample sizes")
    print(f"Train: {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test: {len(test):,}")

    sample_manifest = pd.concat(
        [
            train[["record_id", "time_seconds", TARGET, *FEATURE_COLUMNS]].assign(split="train"),
            validation[["record_id", "time_seconds", TARGET, *FEATURE_COLUMNS]].assign(split="validation"),
            test[["record_id", "time_seconds", TARGET, *FEATURE_COLUMNS]].assign(split="test"),
        ],
        ignore_index=True,
    )
    sample_manifest_path = RESULT_DIR / "quantum_experiment_samples.csv"
    sample_manifest.to_csv(sample_manifest_path, index=False)

    x_train_raw = train[FEATURE_COLUMNS]
    x_validation_raw = validation[FEATURE_COLUMNS]
    x_test_raw = test[FEATURE_COLUMNS]

    y_train = train[TARGET].to_numpy(dtype=int)
    y_validation = validation[TARGET].to_numpy(dtype=int)
    y_test = test[TARGET].to_numpy(dtype=int)

    imputer = SimpleImputer(strategy="median")
    x_train_imputed = imputer.fit_transform(x_train_raw)
    x_validation_imputed = imputer.transform(x_validation_raw)
    x_test_imputed = imputer.transform(x_test_raw)

    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    x_train = scaler.fit_transform(x_train_imputed)
    x_validation = scaler.transform(x_validation_imputed)
    x_test = scaler.transform(x_test_imputed)

    joblib.dump(imputer, MODEL_DIR / "quantum_imputer.joblib")
    joblib.dump(scaler, MODEL_DIR / "quantum_scaler.joblib")

    np.save(MODEL_DIR / "quantum_x_train.npy", x_train)
    np.save(MODEL_DIR / "quantum_x_validation.npy", x_validation)
    np.save(MODEL_DIR / "quantum_x_test.npy", x_test)
    np.save(MODEL_DIR / "quantum_y_train.npy", y_train)
    np.save(MODEL_DIR / "quantum_y_validation.npy", y_validation)
    np.save(MODEL_DIR / "quantum_y_test.npy", y_test)

    print("\nCreating four-qubit feature map")
    feature_map = zz_feature_map(
        feature_dimension=len(FEATURE_COLUMNS),
        reps=2,
        entanglement="linear",
    )
    quantum_kernel = FidelityStatevectorKernel(
        feature_map=feature_map,
        enforce_psd=True,
    )

    print("\nComputing quantum training kernel")
    quantum_start = time.perf_counter()
    kq_train = quantum_kernel.evaluate(x_vec=x_train)
    print(
        "Quantum training kernel completed in "
        f"{time.perf_counter() - quantum_start:.2f} seconds"
    )

    print("Computing quantum validation kernel")
    kq_validation = quantum_kernel.evaluate(x_vec=x_validation, y_vec=x_train)

    print("Computing quantum test kernel")
    kq_test = quantum_kernel.evaluate(x_vec=x_test, y_vec=x_train)

    print("Computing classical RBF kernels")
    kc_train = rbf_kernel(x_train, x_train, gamma=RBF_GAMMA)
    kc_validation = rbf_kernel(x_validation, x_train, gamma=RBF_GAMMA)
    kc_test = rbf_kernel(x_test, x_train, gamma=RBF_GAMMA)

    kq_train = normalize_kernel(kq_train)
    kc_train = normalize_kernel(kc_train)

    train_diagonal = np.ones(len(x_train))
    validation_q_diagonal = np.diag(quantum_kernel.evaluate(x_vec=x_validation))
    test_q_diagonal = np.diag(quantum_kernel.evaluate(x_vec=x_test))
    validation_c_diagonal = np.ones(len(x_validation))
    test_c_diagonal = np.ones(len(x_test))

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
        validation_c_diagonal,
        train_diagonal,
    )
    kc_test = normalize_cross_kernel(
        kc_test,
        test_c_diagonal,
        train_diagonal,
    )

    kernels = {
        "kq_train": kq_train,
        "kq_validation": kq_validation,
        "kq_test": kq_test,
        "kc_train": kc_train,
        "kc_validation": kc_validation,
        "kc_test": kc_test,
    }
    verify_kernels(kernels)

    np.save(MODEL_DIR / "quantum_kernel_train.npy", kq_train)
    np.save(MODEL_DIR / "quantum_kernel_validation.npy", kq_validation)
    np.save(MODEL_DIR / "quantum_kernel_test.npy", kq_test)
    np.save(MODEL_DIR / "classical_kernel_train.npy", kc_train)
    np.save(MODEL_DIR / "classical_kernel_validation.npy", kc_validation)
    np.save(MODEL_DIR / "classical_kernel_test.npy", kc_test)

    results: list[dict[str, object]] = []

    print("\nTraining pure Quantum Kernel SVM")
    qksvm_start = time.perf_counter()
    qksvm = SVC(
        kernel="precomputed",
        C=SVM_C,
        class_weight="balanced",
        random_state=SEED,
    )
    qksvm.fit(kq_train, y_train)
    qksvm_training_time = time.perf_counter() - qksvm_start
    joblib.dump(qksvm, MODEL_DIR / "pure_quantum_kernel_svm.joblib")

    validation_result = evaluate_model(
        "pure_quantum_kernel_svm",
        qksvm,
        kq_validation,
        y_validation,
        "validation",
    )
    validation_result["training_time_seconds"] = qksvm_training_time

    test_result = evaluate_model(
        "pure_quantum_kernel_svm",
        qksvm,
        kq_test,
        y_test,
        "test",
    )
    test_result["training_time_seconds"] = qksvm_training_time
    results.extend([validation_result, test_result])

    print(
        "Pure quantum test: "
        f"F1={test_result['f1_score']:.4f}, "
        f"ROC-AUC={test_result['roc_auc']:.4f}"
    )

    static_alpha = 0.50
    print(f"\nTraining static HCQKL, alpha={static_alpha}")

    kh_static_train = (1.0 - static_alpha) * kc_train + static_alpha * kq_train
    kh_static_validation = (
        (1.0 - static_alpha) * kc_validation + static_alpha * kq_validation
    )
    kh_static_test = (1.0 - static_alpha) * kc_test + static_alpha * kq_test

    static_start = time.perf_counter()
    static_model = SVC(
        kernel="precomputed",
        C=SVM_C,
        class_weight="balanced",
        random_state=SEED,
    )
    static_model.fit(kh_static_train, y_train)
    static_training_time = time.perf_counter() - static_start
    joblib.dump(static_model, MODEL_DIR / "static_hcqkl.joblib")

    static_validation_result = evaluate_model(
        "static_hcqkl",
        static_model,
        kh_static_validation,
        y_validation,
        "validation",
    )
    static_validation_result["training_time_seconds"] = static_training_time

    static_test_result = evaluate_model(
        "static_hcqkl",
        static_model,
        kh_static_test,
        y_test,
        "test",
    )
    static_test_result["training_time_seconds"] = static_training_time
    results.extend([static_validation_result, static_test_result])

    print(
        "Static HCQKL test: "
        f"F1={static_test_result['f1_score']:.4f}, "
        f"ROC-AUC={static_test_result['roc_auc']:.4f}"
    )

    print("\nSearching AQUA-SLA alpha")
    alpha_search_results = []
    best_alpha: float | None = None
    best_validation_f1 = -np.inf
    best_model: SVC | None = None
    best_training_time: float | None = None

    for alpha in ALPHA_VALUES:
        kh_train = (1.0 - alpha) * kc_train + alpha * kq_train
        kh_validation = (1.0 - alpha) * kc_validation + alpha * kq_validation

        training_start = time.perf_counter()
        candidate_model = SVC(
            kernel="precomputed",
            C=SVM_C,
            class_weight="balanced",
            random_state=SEED,
        )
        candidate_model.fit(kh_train, y_train)
        training_time = time.perf_counter() - training_start

        validation_predictions = candidate_model.predict(kh_validation)
        validation_f1 = f1_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )

        alpha_search_results.append(
            {
                "alpha": float(alpha),
                "validation_f1": float(validation_f1),
                "training_time_seconds": training_time,
            }
        )

        print(f"alpha={alpha:.1f}: validation F1={validation_f1:.4f}")

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_alpha = float(alpha)
            best_model = candidate_model
            best_training_time = training_time

    if best_model is None or best_alpha is None or best_training_time is None:
        raise RuntimeError("AQUA-SLA alpha optimization failed.")

    print(f"\nBest AQUA-SLA alpha: {best_alpha:.2f}")

    best_validation_kernel = (
        (1.0 - best_alpha) * kc_validation + best_alpha * kq_validation
    )
    best_test_kernel = (1.0 - best_alpha) * kc_test + best_alpha * kq_test

    joblib.dump(best_model, MODEL_DIR / "aqua_sla_lite.joblib")

    aqua_validation_result = evaluate_model(
        "aqua_sla_lite",
        best_model,
        best_validation_kernel,
        y_validation,
        "validation",
    )
    aqua_validation_result["training_time_seconds"] = best_training_time
    aqua_validation_result["alpha"] = best_alpha

    aqua_test_result = evaluate_model(
        "aqua_sla_lite",
        best_model,
        best_test_kernel,
        y_test,
        "test",
    )
    aqua_test_result["training_time_seconds"] = best_training_time
    aqua_test_result["alpha"] = best_alpha
    results.extend([aqua_validation_result, aqua_test_result])

    print(
        "AQUA-SLA-Lite test: "
        f"F1={aqua_test_result['f1_score']:.4f}, "
        f"ROC-AUC={aqua_test_result['roc_auc']:.4f}, "
        f"PR-AUC={aqua_test_result['pr_auc']:.4f}"
    )

    pd.DataFrame(alpha_search_results).to_csv(
        RESULT_DIR / "alpha_search_results.csv",
        index=False,
    )

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(
        RESULT_DIR / "quantum_model_results.csv",
        index=False,
    )

    test_predictions = best_model.predict(best_test_kernel)
    test_scores = best_model.decision_function(best_test_kernel)

    pd.DataFrame(
        {
            "record_id": test["record_id"].to_numpy(),
            "actual": y_test,
            "prediction": test_predictions,
            "decision_score": test_scores,
        }
    ).to_csv(
        RESULT_DIR / "aqua_sla_lite_test_predictions.csv",
        index=False,
    )

    metadata = {
        "model_name": "AQUA-SLA-Lite",
        "target": TARGET,
        "features": FEATURE_COLUMNS,
        "qubits": len(FEATURE_COLUMNS),
        "feature_map": "zz_feature_map",
        "feature_map_repetitions": 2,
        "entanglement": "linear",
        "quantum_backend": "statevector",
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "best_alpha": best_alpha,
        "rbf_gamma": RBF_GAMMA,
        "svm_c": SVM_C,
        "random_seed": SEED,
        "sample_manifest": str(sample_manifest_path),
        "saved_kernel_files": {
            "quantum_train": str(MODEL_DIR / "quantum_kernel_train.npy"),
            "quantum_validation": str(MODEL_DIR / "quantum_kernel_validation.npy"),
            "quantum_test": str(MODEL_DIR / "quantum_kernel_test.npy"),
            "classical_train": str(MODEL_DIR / "classical_kernel_train.npy"),
            "classical_validation": str(MODEL_DIR / "classical_kernel_validation.npy"),
            "classical_test": str(MODEL_DIR / "classical_kernel_test.npy"),
        },
        "saved_feature_files": {
            "x_train": str(MODEL_DIR / "quantum_x_train.npy"),
            "x_validation": str(MODEL_DIR / "quantum_x_validation.npy"),
            "x_test": str(MODEL_DIR / "quantum_x_test.npy"),
            "y_train": str(MODEL_DIR / "quantum_y_train.npy"),
            "y_validation": str(MODEL_DIR / "quantum_y_validation.npy"),
            "y_test": str(MODEL_DIR / "quantum_y_test.npy"),
        },
    }

    with (MODEL_DIR / "aqua_sla_lite_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nFinal quantum-based test results")
    test_results = results_frame[results_frame["split"] == "test"]
    print(
        test_results[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
                "f1_score",
                "mcc",
                "roc_auc",
                "pr_auc",
                "training_time_seconds",
            ]
        ]
        .sort_values("f1_score", ascending=False)
        .to_string(index=False)
    )

    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
