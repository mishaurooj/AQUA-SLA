from __future__ import annotations

"""
AQUA-SLA Sample-Size Learning-Curve Study
=========================================

Purpose
-------
Evaluate whether increasing the classical predictor's training sample size
across 2k, 5k, 10k, 25k, 50k, 100k, 150k, 200k, and 248k rows.

This is a classical prediction-branch scalability/robustness study.
It is NOT a quantum-kernel experiment and should not be presented as one.

Protocol
--------
1. Strict temporal split.
2. Balanced training subsets from 2k through 248k.
3. The same validation and test samples are reused across sample sizes
   within each seed for paired comparison.
4. Feature selection is fitted on the training subset only.
5. Threshold calibration uses validation data only.
6. Test data are used once for final evaluation.
7. Five seeds provide repeated estimates and paired Wilcoxon tests.

Outputs
-------
D:\other\AQUA-SLA\results\prediction\sample_size_study\
"""

import json
import math
import time
import warnings
import joblib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(r"D:\other\AQUA-SLA")

DATA_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_runtime_prediction.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "prediction"
    / "sample_size_study"
)

MODEL_DIR = OUTPUT_DIR / "saved_models"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"

TARGET_COLUMN = "sla_violation"

TRAIN_SIZES = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000, 248_000]
SEEDS = [11, 22, 33, 44, 55]

VALIDATION_SIZE = 2_000
TEST_SIZE = 2_000
TOP_K_FEATURES = 20

TEMPORAL_TRAIN_FRACTION = 0.70
TEMPORAL_VALIDATION_FRACTION = 0.15

THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)

# Fixed model settings. Do not tune separately for each sample size.
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 250,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
}

# Columns that can leak outcomes, are identifiers, or are unsuitable as
# direct predictor inputs.
EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
    "sla_failure",
    "instance_events_type",
    "collections_events_type",
    "record_id",
    "machine_id",
    "collection_id",
    "instance_index",
}

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Helpers
# ============================================================

def confidence_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(np.mean(array))

    if len(array) < 2:
        return mean, mean

    margin = 1.96 * float(np.std(array, ddof=1) / math.sqrt(len(array)))
    return mean - margin, mean + margin


def balanced_sample(
    frame: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Draw an exactly balanced sample without replacement.

    For large requested sizes, the function fails with a detailed message
    rather than silently oversampling or changing the class ratio.
    """
    if size % 2 != 0:
        raise ValueError("Balanced sample size must be even.")

    per_class = size // 2
    class_counts = frame[TARGET_COLUMN].value_counts().to_dict()

    missing = {
        label: per_class - int(class_counts.get(label, 0))
        for label in [0, 1]
        if int(class_counts.get(label, 0)) < per_class
    }

    if missing:
        maximum_balanced_size = 2 * min(
            int(class_counts.get(0, 0)),
            int(class_counts.get(1, 0)),
        )
        raise ValueError(
            f"Cannot draw balanced sample of {size:,} rows from this "
            f"temporal partition. Class counts are {class_counts}; "
            f"maximum balanced size is {maximum_balanced_size:,}. "
            "Reduce TRAIN_SIZES or change the protocol explicitly."
        )

    groups = []

    for label in [0, 1]:
        subset = frame[frame[TARGET_COLUMN] == label]
        groups.append(
            subset.sample(
                n=per_class,
                random_state=seed + label,
                replace=False,
            )
        )

    return (
        pd.concat(groups, axis=0)
        .sample(frac=1.0, random_state=seed + 100)
        .reset_index(drop=True)
    )


def detect_time_column(frame: pd.DataFrame) -> str:
    candidates = [
        "time_seconds",
        "start_time",
        "timestamp",
        "time",
        "collection_start_time",
    ]

    for column in candidates:
        if column in frame.columns:
            return column

    raise ValueError(
        "No temporal ordering column was found. "
        f"Tried: {candidates}"
    )


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATA_PATH}")

    frame = pd.read_csv(DATA_PATH, low_memory=False)

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing from the dataset."
        )

    frame[TARGET_COLUMN] = (
        pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    time_column = detect_time_column(frame)
    frame[time_column] = pd.to_numeric(frame[time_column], errors="coerce")
    frame = frame.dropna(subset=[time_column]).sort_values(
        time_column,
        kind="mergesort",
    ).reset_index(drop=True)

    numeric_columns = frame.select_dtypes(include=[np.number, "bool"]).columns
    feature_columns = [
        column
        for column in numeric_columns
        if column not in EXCLUDED_COLUMNS
        and column != time_column
        and not column.lower().endswith("_label")
        and "event_type" not in column.lower()
        and "end_time" not in column.lower()
    ]

    if not feature_columns:
        raise ValueError("No usable numeric features remain after exclusions.")

    n = len(frame)
    train_end = int(n * TEMPORAL_TRAIN_FRACTION)
    validation_end = int(
        n
        * (
            TEMPORAL_TRAIN_FRACTION
            + TEMPORAL_VALIDATION_FRACTION
        )
    )

    train_pool = frame.iloc[:train_end].copy()
    validation_pool = frame.iloc[train_end:validation_end].copy()
    test_pool = frame.iloc[validation_end:].copy()

    print(f"Rows: {n:,}")
    print(f"Temporal train pool: {len(train_pool):,}")
    print(f"Temporal validation pool: {len(validation_pool):,}")
    print(f"Temporal test pool: {len(test_pool):,}")
    print(f"Candidate numeric features: {len(feature_columns)}")

    train_counts = train_pool[TARGET_COLUMN].value_counts().to_dict()
    maximum_balanced_train_size = 2 * min(
        int(train_counts.get(0, 0)),
        int(train_counts.get(1, 0)),
    )

    print(f"Train class counts: {train_counts}")
    print(
        "Maximum feasible balanced training size: "
        f"{maximum_balanced_train_size:,}"
    )

    infeasible = [
        size
        for size in TRAIN_SIZES
        if size > maximum_balanced_train_size
    ]

    if infeasible:
        raise ValueError(
            "The following requested training sizes are not feasible under "
            f"the balanced protocol: {infeasible}. Maximum feasible size is "
            f"{maximum_balanced_train_size:,}."
        )

    return train_pool, validation_pool, test_pool, feature_columns


def select_features(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> list[str]:
    X = train_frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = train_frame[TARGET_COLUMN].to_numpy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scores = mutual_info_classif(
        X_imputed,
        y,
        random_state=seed,
        discrete_features=False,
    )

    ranking = np.argsort(scores)[::-1]
    top_indices = ranking[: min(TOP_K_FEATURES, len(feature_columns))]
    return [feature_columns[index] for index in top_indices]


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in THRESHOLD_GRID:
        predictions = (probabilities >= threshold).astype(int)

        f1 = f1_score(y_true, predictions, zero_division=0)
        mcc = matthews_corrcoef(y_true, predictions)

        # Equal emphasis on violation detection balance and correlation.
        score = 0.5 * f1 + 0.5 * mcc

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=seed,
                    **MODEL_PARAMS,
                ),
            ),
        ]
    )


# ============================================================
# Main experiment
# ============================================================

def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_pool, validation_pool, test_pool, feature_columns = prepare_data()

    rows = []
    selected_feature_rows = []

    for seed in SEEDS:
        print(f"\n===== Seed {seed} =====")

        # Fixed validation/test samples for all training sizes in this seed.
        validation_frame = balanced_sample(
            validation_pool,
            VALIDATION_SIZE,
            seed + 1_000,
        )
        test_frame = balanced_sample(
            test_pool,
            TEST_SIZE,
            seed + 2_000,
        )

        for train_size in TRAIN_SIZES:
            print(f"Training size: {train_size:,}")

            train_frame = balanced_sample(
                train_pool,
                train_size,
                seed + train_size,
            )

            selected_features = select_features(
                train_frame,
                feature_columns,
                seed + train_size,
            )

            for rank, feature in enumerate(selected_features, start=1):
                selected_feature_rows.append(
                    {
                        "seed": seed,
                        "train_size": train_size,
                        "rank": rank,
                        "feature": feature,
                    }
                )

            X_train = train_frame[selected_features]
            y_train = train_frame[TARGET_COLUMN].to_numpy()

            X_validation = validation_frame[selected_features]
            y_validation = validation_frame[TARGET_COLUMN].to_numpy()

            X_test = test_frame[selected_features]
            y_test = test_frame[TARGET_COLUMN].to_numpy()

            model = build_model(seed)

            start = time.perf_counter()
            model.fit(X_train, y_train)
            training_seconds = time.perf_counter() - start

            validation_probabilities = model.predict_proba(X_validation)[:, 1]
            threshold, validation_score = optimize_threshold(
                y_validation,
                validation_probabilities,
            )

            start = time.perf_counter()
            test_probabilities = model.predict_proba(X_test)[:, 1]
            inference_seconds = time.perf_counter() - start

            metrics = evaluate(
                y_test,
                test_probabilities,
                threshold,
            )

            run_name = f"seed_{seed}_train_{train_size}"
            model_path = MODEL_DIR / f"{run_name}.joblib"
            prediction_path = PREDICTION_DIR / f"{run_name}.csv"
            metadata_path = MODEL_DIR / f"{run_name}_metadata.json"

            # Save the complete fitted sklearn pipeline. For tree ensembles,
            # this is the reproducible equivalent of saving neural weights.
            joblib.dump(
                {
                    "pipeline": model,
                    "selected_features": selected_features,
                    "threshold": threshold,
                    "target_column": TARGET_COLUMN,
                    "train_size": train_size,
                    "seed": seed,
                    "model_params": MODEL_PARAMS,
                },
                model_path,
                compress=3,
            )

            pd.DataFrame(
                {
                    "y_true": y_test,
                    "probability": test_probabilities,
                    "prediction": (
                        test_probabilities >= threshold
                    ).astype(int),
                }
            ).to_csv(prediction_path, index=False)

            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "run_name": run_name,
                        "seed": seed,
                        "train_size": train_size,
                        "validation_size": VALIDATION_SIZE,
                        "test_size": TEST_SIZE,
                        "selected_features": selected_features,
                        "threshold": threshold,
                        "validation_objective": validation_score,
                        "training_seconds": training_seconds,
                        "inference_seconds": inference_seconds,
                        "metrics": metrics,
                    },
                    file,
                    indent=2,
                )

            rows.append(
                {
                    "seed": seed,
                    "train_size": train_size,
                    "validation_size": VALIDATION_SIZE,
                    "test_size": TEST_SIZE,
                    "selected_feature_count": len(selected_features),
                    "threshold": threshold,
                    "validation_objective": validation_score,
                    "training_seconds": float(training_seconds),
                    "inference_seconds": float(inference_seconds),
                    "inference_ms_per_sample": float(
                        1_000.0 * inference_seconds / len(X_test)
                    ),
                    "model_path": str(model_path),
                    "prediction_path": str(prediction_path),
                    **metrics,
                }
            )

            print(
                f"  ROC-AUC={metrics['roc_auc']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"MCC={metrics['mcc']:.4f}, "
                f"Recall={metrics['recall']:.4f}"
            )

    return pd.DataFrame(rows), pd.DataFrame(selected_feature_rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "training_seconds",
        "inference_ms_per_sample",
    ]

    rows = []

    for train_size, group in results.groupby("train_size"):
        row = {
            "train_size": int(train_size),
            "runs": int(len(group)),
        }

        for metric in metrics:
            values = group[metric].astype(float)
            ci_low, ci_high = confidence_interval(values)

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high

        rows.append(row)

    return pd.DataFrame(rows).sort_values("train_size").reset_index(drop=True)


def paired_tests(results: pd.DataFrame) -> pd.DataFrame:
    comparisons = list(zip(TRAIN_SIZES[:-1], TRAIN_SIZES[1:]))

    if len(TRAIN_SIZES) > 2:
        comparisons.append((TRAIN_SIZES[0], TRAIN_SIZES[-1]))

    metrics = [
        "accuracy",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "training_seconds",
    ]

    rows = []

    for size_a, size_b in comparisons:
        left = results[results["train_size"] == size_a][
            ["seed", *metrics]
        ].copy()
        right = results[results["train_size"] == size_b][
            ["seed", *metrics]
        ].copy()

        paired = left.merge(
            right,
            on="seed",
            suffixes=("_a", "_b"),
            how="inner",
            validate="one_to_one",
        )

        for metric in metrics:
            a = paired[f"{metric}_a"].astype(float).to_numpy()
            b = paired[f"{metric}_b"].astype(float).to_numpy()
            differences = b - a

            if np.allclose(differences, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(
                    b,
                    a,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            rows.append(
                {
                    "train_size_a": size_a,
                    "train_size_b": size_b,
                    "metric": metric,
                    "paired_seeds": int(len(paired)),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "mean_change_b_minus_a": float(np.mean(differences)),
                    "relative_change": (
                        float(np.mean(b) - np.mean(a)) / abs(float(np.mean(a)))
                        if not math.isclose(float(np.mean(a)), 0.0)
                        else float("nan")
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "significant_at_0_05": bool(p_value < 0.05),
                }
            )

    return pd.DataFrame(rows)


def feature_stability(selected_features: pd.DataFrame) -> pd.DataFrame:
    counts = (
        selected_features
        .groupby(["train_size", "feature"])
        .agg(
            selection_count=("seed", "nunique"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )

    counts["selection_frequency"] = (
        counts["selection_count"] / len(SEEDS)
    )

    return counts.sort_values(
        ["train_size", "selection_frequency", "mean_rank"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    results, selected_features = run_experiment()
    summary = summarize_results(results)
    tests = paired_tests(results)
    stability = feature_stability(selected_features)

    results.to_csv(
        OUTPUT_DIR / "sample_size_runs.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "sample_size_summary.csv",
        index=False,
    )
    tests.to_csv(
        OUTPUT_DIR / "sample_size_paired_tests.csv",
        index=False,
    )
    selected_features.to_csv(
        OUTPUT_DIR / "selected_features_by_run.csv",
        index=False,
    )
    stability.to_csv(
        OUTPUT_DIR / "feature_stability.csv",
        index=False,
    )

    metadata = {
        "project": "AQUA-SLA",
        "study": "classical prediction branch extended sample-size ablation",
        "dataset": str(DATA_PATH),
        "target": TARGET_COLUMN,
        "train_sizes": TRAIN_SIZES,
        "seeds": SEEDS,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "top_k_features": TOP_K_FEATURES,
        "split": {
            "train_fraction": TEMPORAL_TRAIN_FRACTION,
            "validation_fraction": TEMPORAL_VALIDATION_FRACTION,
            "test_fraction": (
                1.0
                - TEMPORAL_TRAIN_FRACTION
                - TEMPORAL_VALIDATION_FRACTION
            ),
        },
        "model": "HistGradientBoostingClassifier",
        "model_params": MODEL_PARAMS,
        "scope_note": (
            "This experiment evaluates the classical prediction branch only. "
            "It is not a quantum-kernel or QAOA sample-size experiment."
        ),
    }

    with (OUTPUT_DIR / "experiment_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\n===== Sample-size summary =====")
    columns = [
        "train_size",
        "accuracy_mean",
        "recall_mean",
        "f1_mean",
        "mcc_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "training_seconds_mean",
    ]
    print(summary[columns].to_string(index=False))

    print("\n===== Paired tests =====")
    print(tests.to_string(index=False))

    print(f"\nOutputs saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
