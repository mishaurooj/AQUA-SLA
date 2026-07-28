from __future__ import annotations

import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


SEED = 42

random.seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path(r"D:\other\AQUA-SLA")

DATA_PATH = (
    PROJECT_DIR
    / "Dataset"
    / "processed"
    / "aqua_sla_early_prediction.csv"
)

MODEL_DIR = PROJECT_DIR / "models" / "classical"
RESULT_DIR = PROJECT_DIR / "results" / "metrics"

TARGET = "sla_violation"

IDENTIFIER_COLUMNS = [
    "record_id",
    "collection_id",
    "instance_index",
    "machine_id",
    "cluster",
    "time",
    "time_seconds",
]

FEATURE_COLUMNS = [
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


def temporal_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values(
        by=["time_seconds", "record_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    n_rows = len(frame)

    train_end = int(n_rows * 0.60)
    validation_end = int(n_rows * 0.80)

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()

    return train, validation, test


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
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
        "accuracy": accuracy_score(y_true, predictions),
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
        "mcc": matthews_corrcoef(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def create_pipeline(model: object) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print(f"Rows: {len(df):,}")
    print(f"Target: {TARGET}")
    print(
        df[TARGET]
        .value_counts(normalize=True)
        .sort_index()
    )

    train, validation, test = temporal_split(df)

    print("\nTemporal partitions")
    print(f"Train: {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test: {len(test):,}")

    split_report = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train),
                "violation_rate": train[TARGET].mean(),
                "minimum_time": train["time_seconds"].min(),
                "maximum_time": train["time_seconds"].max(),
            },
            {
                "split": "validation",
                "rows": len(validation),
                "violation_rate": validation[TARGET].mean(),
                "minimum_time": validation["time_seconds"].min(),
                "maximum_time": validation["time_seconds"].max(),
            },
            {
                "split": "test",
                "rows": len(test),
                "violation_rate": test[TARGET].mean(),
                "minimum_time": test["time_seconds"].min(),
                "maximum_time": test["time_seconds"].max(),
            },
        ]
    )

    split_report.to_csv(
        RESULT_DIR / "temporal_split_report.csv",
        index=False,
    )

    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET]

    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET]

    x_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET]

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=SEED,
            cache_size=4096,
        ),
    }

    all_results: list[dict[str, object]] = []

    for model_name, estimator in models.items():
        print("\n" + "=" * 80)
        print(f"Training: {model_name}")
        print("=" * 80)

        pipeline = create_pipeline(estimator)

        training_start = time.perf_counter()
        pipeline.fit(x_train, y_train)
        training_time = time.perf_counter() - training_start

        model_path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)

        for split_name, x_split, y_split in [
            ("validation", x_validation, y_validation),
            ("test", x_test, y_test),
        ]:
            inference_start = time.perf_counter()

            probabilities = pipeline.predict_proba(x_split)[:, 1]
            predictions = (probabilities >= 0.50).astype(int)

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            metrics = calculate_metrics(
                y_split,
                predictions,
                probabilities,
            )

            result = {
                "model": model_name,
                "split": split_name,
                "training_time_seconds": training_time,
                "inference_time_seconds": inference_time,
                "inference_ms_per_record": (
                    inference_time
                    / len(x_split)
                    * 1_000
                ),
                **metrics,
            }

            all_results.append(result)

            prediction_output = pd.DataFrame(
                {
                    "record_id": (
                        validation["record_id"].values
                        if split_name == "validation"
                        else test["record_id"].values
                    ),
                    "actual": y_split.values,
                    "prediction": predictions,
                    "probability": probabilities,
                }
            )

            prediction_output.to_csv(
                RESULT_DIR
                / f"{model_name}_{split_name}_predictions.csv",
                index=False,
            )

            print(
                f"{split_name}: "
                f"F1={metrics['f1_score']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}, "
                f"MCC={metrics['mcc']:.4f}"
            )

        metadata = {
            "model_name": model_name,
            "target": TARGET,
            "dataset": str(DATA_PATH),
            "features": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
            "training_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "random_seed": SEED,
            "threshold": 0.50,
            "model_path": str(model_path),
        }

        with (
            MODEL_DIR / f"{model_name}_metadata.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    results = pd.DataFrame(all_results)

    results.to_csv(
        RESULT_DIR / "baseline_results.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("Final test results")
    print("=" * 80)

    test_results = results[
        results["split"] == "test"
    ].copy()

    columns_to_print = [
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
        "brier_score",
        "training_time_seconds",
    ]

    print(
        test_results[columns_to_print]
        .sort_values("f1_score", ascending=False)
        .to_string(index=False)
    )

    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()