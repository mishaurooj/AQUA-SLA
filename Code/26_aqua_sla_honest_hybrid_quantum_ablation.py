from __future__ import annotations

r"""
AQUA-SLA Honest Hybrid-Fusion + Quantum Ablation Extension
==========================================================

Purpose
-------
Preserve the original AQUA-SLA purpose:
1. Predict SLA violations using classical and quantum probability signals.
2. Use the predicted SLA risk in quantum-aware scheduling.
3. Retain QAOA depth/shot/optimizer ablations and optional IBM/IQM execution.

Scientific safeguards
---------------------
- This script does not force metrics to a target value.
- Hybrid fusion is evaluated with nested repeated stratified cross-validation.
- Model selection and threshold selection occur only inside each outer fold.
- Classical-only, quantum-only, simple blend, logistic fusion, random-forest
  fusion, and nonlinear HistGradientBoosting fusion are compared.
- Quantum-only and quantum-included ablations are always saved.
- The original locked test result is preserved and never overwritten.
- The final deployable stacker is marked as development-trained because the
  locked export is reused for this secondary fusion study.

Required companion script
-------------------------
Place this file beside:
    25_aqua_sla_complete_quantum_ablation.py

The companion script is reused for the QAOA, circuit, figure, and optional
hardware experiments. Set AQUA_SKIP_QAOA=1 to run only the fusion study.

Output
------
D:\other\AQUA-SLA\results\aqua_sla_final\
honest_hybrid_quantum_ablation
"""

import importlib.util
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(r"D:\other\AQUA-SLA")

LOCKED_EXPORT_CANDIDATES = [
    PROJECT_DIR
    / "results"
    / "quantum"
    / "aqua_sla_v3"
    / "locked_prediction_export.csv",
    PROJECT_DIR
    / "results"
    / "aqua_sla_final"
    / "locked_prediction_export.csv",
]

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "aqua_sla_final"
    / "honest_hybrid_quantum_ablation"
)

CSV_DIR = OUTPUT_DIR / "csv"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures_jpeg"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

OUTER_SPLITS = 5
OUTER_REPEATS = 3
INNER_SPLITS = 4
RANDOM_SEED = 2026

THRESHOLD_GRID = np.linspace(0.10, 0.90, 161)

FIGURE_WIDTH_PX = 1920
FIGURE_HEIGHT_PX = 1080
FIGURE_RENDER_DPI = 160
JPEG_DPI_METADATA = 900

RUN_QAOA_EXTENSION = os.getenv("AQUA_SKIP_QAOA", "0") != "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    for directory in [
        OUTPUT_DIR,
        CSV_DIR,
        MODEL_DIR,
        FIGURE_DIR,
        REPORT_DIR,
        LOG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def locate_locked_export() -> Path:
    for candidate in LOCKED_EXPORT_CANDIDATES:
        if candidate.exists():
            return candidate

    matches = list((PROJECT_DIR / "results").rglob("locked_prediction_export.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "locked_prediction_export.csv was not found under the project results."
    )


def save_figure_jpeg(fig: plt.Figure, filename: str) -> Path:
    temp = FIGURE_DIR / f"_{filename}.png"
    output = FIGURE_DIR / f"{filename}.jpeg"

    fig.set_size_inches(
        FIGURE_WIDTH_PX / FIGURE_RENDER_DPI,
        FIGURE_HEIGHT_PX / FIGURE_RENDER_DPI,
    )
    fig.savefig(
        temp,
        dpi=FIGURE_RENDER_DPI,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
    )
    plt.close(fig)

    image = Image.open(temp).convert("RGB")
    image = image.resize(
        (FIGURE_WIDTH_PX, FIGURE_HEIGHT_PX),
        Image.Resampling.LANCZOS,
    )
    image.save(
        output,
        "JPEG",
        quality=100,
        subsampling=0,
        optimize=True,
        dpi=(JPEG_DPI_METADATA, JPEG_DPI_METADATA),
    )
    temp.unlink(missing_ok=True)
    return output


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))

    if len(values) < 2:
        return mean, mean

    margin = 1.96 * float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return mean - margin, mean + margin


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0

    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        count = int(mask.sum())
        if count == 0:
            continue

        confidence = float(probabilities[mask].mean())
        observed = float(y_true[mask].mean())
        value += (count / len(y_true)) * abs(confidence - observed)

    return float(value)


def metric_bundle(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-8,
        1.0 - 1e-8,
    )
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    recall = recall_score(y_true, predictions, zero_division=0)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "ece_10_bins": expected_calibration_error(
            y_true,
            probabilities,
            bins=10,
        ),
        "g_mean": float(math.sqrt(max(0.0, recall * specificity))),
        "false_alarm_rate": float(1.0 - specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_objective(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> float:
    predictions = (probabilities >= threshold).astype(int)
    f1 = f1_score(y_true, predictions, zero_division=0)
    mcc = matthews_corrcoef(y_true, predictions)
    balanced = balanced_accuracy_score(y_true, predictions)

    # Balanced operational objective; no metric is allowed to dominate alone.
    return float(0.40 * f1 + 0.35 * mcc + 0.25 * balanced)


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in THRESHOLD_GRID:
        score = threshold_objective(y_true, probabilities, float(threshold))

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


# ---------------------------------------------------------------------------
# Fusion feature sets and candidate models
# ---------------------------------------------------------------------------

def build_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    classical = frame["classical_probability"].astype(float)
    quantum = frame["quantum_probability"].astype(float)

    if feature_set == "classical_only":
        return pd.DataFrame({"classical_probability": classical})

    if feature_set == "quantum_only":
        return pd.DataFrame({"quantum_probability": quantum})

    if feature_set == "classical_quantum":
        return pd.DataFrame(
            {
                "classical_probability": classical,
                "quantum_probability": quantum,
            }
        )

    if feature_set == "interaction_fusion":
        return pd.DataFrame(
            {
                "classical_probability": classical,
                "quantum_probability": quantum,
                "mean_probability": 0.5 * (classical + quantum),
                "probability_difference": classical - quantum,
                "absolute_difference": np.abs(classical - quantum),
                "probability_product": classical * quantum,
                "probability_maximum": np.maximum(classical, quantum),
                "probability_minimum": np.minimum(classical, quantum),
                "classical_squared": classical ** 2,
                "quantum_squared": quantum ** 2,
            }
        )

    raise ValueError(f"Unknown feature set: {feature_set}")


def candidate_estimators(seed: int) -> dict[str, BaseEstimator]:
    return {
        "logistic_fusion": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest_fusion": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_fusion": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=seed,
        ),
    }


@dataclass
class SelectedConfiguration:
    model_name: str
    feature_set: str
    threshold: float
    inner_score: float


def inner_oof_probabilities(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    splitter: StratifiedKFold,
) -> np.ndarray:
    probabilities = np.zeros(len(y), dtype=float)

    for train_index, validation_index in splitter.split(X, y):
        model = clone(estimator)
        model.fit(X.iloc[train_index], y[train_index])
        probabilities[validation_index] = model.predict_proba(
            X.iloc[validation_index]
        )[:, 1]

    return probabilities


def choose_configuration(
    development_frame: pd.DataFrame,
    y_development: np.ndarray,
    seed: int,
) -> tuple[SelectedConfiguration, pd.DataFrame]:
    splitter = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    feature_sets = [
        "classical_only",
        "quantum_only",
        "classical_quantum",
        "interaction_fusion",
    ]

    candidates = candidate_estimators(seed)
    rows: list[dict[str, Any]] = []

    # Locked simple blends are included as transparent quantum ablations.
    for alpha in np.linspace(0.0, 1.0, 21):
        probabilities = (
            alpha
            * development_frame["classical_probability"].to_numpy(dtype=float)
            + (1.0 - alpha)
            * development_frame["quantum_probability"].to_numpy(dtype=float)
        )
        threshold, score = optimize_threshold(y_development, probabilities)
        rows.append(
            {
                "model_name": f"weighted_blend_alpha_{alpha:.2f}",
                "feature_set": "classical_quantum",
                "threshold": threshold,
                "inner_score": score,
            }
        )

    for feature_set in feature_sets:
        X = build_features(development_frame, feature_set)

        for model_name, estimator in candidates.items():
            probabilities = inner_oof_probabilities(
                estimator,
                X,
                y_development,
                splitter,
            )
            threshold, score = optimize_threshold(
                y_development,
                probabilities,
            )
            rows.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "threshold": threshold,
                    "inner_score": score,
                }
            )

    table = pd.DataFrame(rows).sort_values(
        ["inner_score", "model_name"],
        ascending=[False, True],
    ).reset_index(drop=True)

    best = table.iloc[0]

    return (
        SelectedConfiguration(
            model_name=str(best["model_name"]),
            feature_set=str(best["feature_set"]),
            threshold=float(best["threshold"]),
            inner_score=float(best["inner_score"]),
        ),
        table,
    )


def fit_selected_configuration(
    configuration: SelectedConfiguration,
    development_frame: pd.DataFrame,
    y_development: np.ndarray,
    evaluation_frame: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, Any]:
    if configuration.model_name.startswith("weighted_blend_alpha_"):
        alpha = float(configuration.model_name.rsplit("_", 1)[-1])
        probabilities = (
            alpha
            * evaluation_frame["classical_probability"].to_numpy(dtype=float)
            + (1.0 - alpha)
            * evaluation_frame["quantum_probability"].to_numpy(dtype=float)
        )
        artifact = {
            "type": "weighted_blend",
            "alpha_classical": alpha,
            "alpha_quantum": 1.0 - alpha,
        }
        return probabilities, artifact

    estimators = candidate_estimators(seed)
    estimator = clone(estimators[configuration.model_name])
    X_development = build_features(
        development_frame,
        configuration.feature_set,
    )
    X_evaluation = build_features(
        evaluation_frame,
        configuration.feature_set,
    )

    estimator.fit(X_development, y_development)
    probabilities = estimator.predict_proba(X_evaluation)[:, 1]
    return probabilities, estimator


# ---------------------------------------------------------------------------
# Nested repeated CV
# ---------------------------------------------------------------------------

def run_nested_fusion_study(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_column = (
        "locked_actual"
        if "locked_actual" in frame.columns
        else "actual"
    )
    y = frame[target_column].astype(int).to_numpy()

    required = {
        "classical_probability",
        "quantum_probability",
        target_column,
    }
    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            "Locked export is missing required columns: "
            + ", ".join(missing)
        )

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_SEED,
    )

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for fold_id, (development_index, evaluation_index) in enumerate(
        outer.split(frame, y),
        start=1,
    ):
        fold_seed = RANDOM_SEED + fold_id

        development = frame.iloc[development_index].reset_index(drop=True)
        evaluation = frame.iloc[evaluation_index].reset_index(drop=True)
        y_development = y[development_index]
        y_evaluation = y[evaluation_index]

        selected, candidate_table = choose_configuration(
            development,
            y_development,
            seed=fold_seed,
        )

        candidate_table.insert(0, "outer_fold", fold_id)
        selection_rows.extend(candidate_table.to_dict(orient="records"))

        probabilities, fitted_artifact = fit_selected_configuration(
            selected,
            development,
            y_development,
            evaluation,
            seed=fold_seed,
        )

        metrics = metric_bundle(
            y_evaluation,
            probabilities,
            selected.threshold,
        )

        fold_rows.append(
            {
                "outer_fold": fold_id,
                "development_samples": len(development_index),
                "evaluation_samples": len(evaluation_index),
                "selected_model": selected.model_name,
                "selected_feature_set": selected.feature_set,
                "inner_threshold": selected.threshold,
                "inner_selection_score": selected.inner_score,
                **metrics,
            }
        )

        model_path = MODEL_DIR / f"outer_fold_{fold_id:02d}.joblib"
        joblib.dump(
            {
                "selected_configuration": selected.__dict__,
                "model_or_blend": fitted_artifact,
                "development_indices": development_index,
                "evaluation_indices": evaluation_index,
                "target_column": target_column,
            },
            model_path,
            compress=3,
        )

        record_ids = evaluation.get(
            "record_id",
            pd.Series(evaluation_index),
        ).to_numpy()

        predictions = (probabilities >= selected.threshold).astype(int)

        for local_index in range(len(evaluation_index)):
            prediction_rows.append(
                {
                    "outer_fold": fold_id,
                    "source_index": int(evaluation_index[local_index]),
                    "record_id": record_ids[local_index],
                    "actual": int(y_evaluation[local_index]),
                    "probability": float(probabilities[local_index]),
                    "prediction": int(predictions[local_index]),
                    "selected_model": selected.model_name,
                    "selected_feature_set": selected.feature_set,
                    "threshold": selected.threshold,
                }
            )

        print(
            f"Fold {fold_id:02d}: "
            f"{selected.model_name} / {selected.feature_set} | "
            f"F1={metrics['f1']:.4f}, "
            f"MCC={metrics['mcc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)

    folds.to_csv(CSV_DIR / "nested_fusion_fold_metrics.csv", index=False)
    predictions.to_csv(
        CSV_DIR / "nested_fusion_oof_predictions.csv",
        index=False,
    )
    selections.to_csv(
        CSV_DIR / "nested_fusion_inner_model_selection.csv",
        index=False,
    )

    return folds, predictions, selections


def summarize_nested_fusion(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "cohen_kappa",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "ece_10_bins",
        "g_mean",
        "false_alarm_rate",
    ]

    row: dict[str, Any] = {
        "outer_folds": int(len(folds)),
        "outer_splits": OUTER_SPLITS,
        "outer_repeats": OUTER_REPEATS,
        "inner_splits": INNER_SPLITS,
    }

    for metric in metrics:
        values = folds[metric].to_numpy(dtype=float)
        low, high = confidence_interval(values)
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values, ddof=1))
        row[f"{metric}_ci95_low"] = low
        row[f"{metric}_ci95_high"] = high

    summary = pd.DataFrame([row])
    summary.to_csv(CSV_DIR / "nested_fusion_summary.csv", index=False)
    return summary


def run_transparent_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    target_column = (
        "locked_actual"
        if "locked_actual" in frame.columns
        else "actual"
    )
    y = frame[target_column].astype(int).to_numpy()
    threshold = float(
        frame["decision_threshold"].iloc[0]
        if "decision_threshold" in frame.columns
        else 0.5
    )

    variants = {
        "classical_only_locked": frame["classical_probability"].to_numpy(
            dtype=float
        ),
        "quantum_only_locked": frame["quantum_probability"].to_numpy(
            dtype=float
        ),
        "simple_equal_hybrid": (
            0.5
            * frame["classical_probability"].to_numpy(dtype=float)
            + 0.5
            * frame["quantum_probability"].to_numpy(dtype=float)
        ),
    }

    if "hybrid_probability" in frame.columns:
        variants["original_hybrid_locked"] = frame[
            "hybrid_probability"
        ].to_numpy(dtype=float)

    if "locked_probability" in frame.columns:
        variants["operational_locked"] = frame[
            "locked_probability"
        ].to_numpy(dtype=float)

    rows = []

    for name, probabilities in variants.items():
        rows.append(
            {
                "variant": name,
                "quantum_included": "quantum" in name
                or "hybrid" in name
                or "operational" in name,
                **metric_bundle(y, probabilities, threshold),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        CSV_DIR / "transparent_locked_component_ablation.csv",
        index=False,
    )
    return results


def fit_development_final_model(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    target_column = (
        "locked_actual"
        if "locked_actual" in frame.columns
        else "actual"
    )
    y = frame[target_column].astype(int).to_numpy()

    selected, selection_table = choose_configuration(
        frame.reset_index(drop=True),
        y,
        seed=RANDOM_SEED + 99_999,
    )

    probabilities, fitted_artifact = fit_selected_configuration(
        selected,
        frame.reset_index(drop=True),
        y,
        frame.reset_index(drop=True),
        seed=RANDOM_SEED + 99_999,
    )

    artifact = {
        "warning": (
            "Development-trained model. Obtain a new untouched temporal test "
            "set before reporting this artifact as a final external result."
        ),
        "selected_configuration": selected.__dict__,
        "model_or_blend": fitted_artifact,
        "target_column": target_column,
        "training_samples": len(frame),
        "apparent_training_metrics": metric_bundle(
            y,
            probabilities,
            selected.threshold,
        ),
    }

    joblib.dump(
        artifact,
        MODEL_DIR / "development_hybrid_fusion_model.joblib",
        compress=3,
    )
    selection_table.to_csv(
        CSV_DIR / "development_final_model_selection.csv",
        index=False,
    )

    with (
        REPORT_DIR / "development_final_model_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                key: value
                for key, value in artifact.items()
                if key != "model_or_blend"
            },
            file,
            indent=2,
        )

    return artifact


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------

def create_figures(
    transparent: pd.DataFrame,
    folds: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    score_columns = ["f1", "mcc", "roc_auc", "pr_auc"]

    fig, ax = plt.subplots()
    transparent.set_index("variant")[score_columns].plot(
        kind="bar",
        ax=ax,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Locked prediction variant")
    ax.set_ylabel("Score")
    ax.set_title("AQUA-SLA classical–quantum component ablation")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "01_locked_component_ablation")

    fig, ax = plt.subplots()
    folds[["f1", "mcc", "roc_auc", "pr_auc"]].boxplot(ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Outer-fold score")
    ax.set_title("Nested repeated-CV hybrid-fusion performance")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "02_nested_fusion_score_distribution")

    model_counts = folds["selected_model"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(model_counts.index.astype(str), model_counts.values)
    ax.set_xlabel("Selected fusion model")
    ax.set_ylabel("Number of outer folds")
    ax.set_title("Fusion-model selection frequency")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "03_fusion_model_selection_frequency")

    feature_counts = folds["selected_feature_set"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(feature_counts.index.astype(str), feature_counts.values)
    ax.set_xlabel("Feature ablation")
    ax.set_ylabel("Number of outer folds")
    ax.set_title("Classical/quantum feature-set selection frequency")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "04_quantum_feature_ablation_frequency")

    top = (
        selections.groupby(["model_name", "feature_set"], as_index=False)[
            "inner_score"
        ]
        .mean()
        .sort_values("inner_score", ascending=False)
        .head(12)
    )
    labels = (
        top["model_name"].astype(str)
        + "\n"
        + top["feature_set"].astype(str)
    )

    fig, ax = plt.subplots()
    ax.bar(labels, top["inner_score"])
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Mean inner selection score")
    ax.set_title("Fusion candidates under nested validation")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "05_nested_candidate_comparison")


def write_report(
    source_path: Path,
    transparent: pd.DataFrame,
    folds: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    selected_models = folds["selected_model"].value_counts().to_dict()
    selected_features = (
        folds["selected_feature_set"].value_counts().to_dict()
    )

    lines = [
        "# AQUA-SLA Honest Hybrid-Fusion and Quantum Ablation",
        "",
        "## Data source",
        "",
        f"`{source_path}`",
        "",
        "## Purpose retained",
        "",
        (
            "The experiment retains SLA-violation prediction and QAOA-based "
            "scheduling. Classical and quantum probability branches are "
            "evaluated independently and jointly."
        ),
        "",
        "## Locked transparent ablation",
        "",
        transparent.to_markdown(index=False),
        "",
        "## Nested repeated-CV fusion result",
        "",
        summary.to_markdown(index=False),
        "",
        "## Model-selection frequency",
        "",
        f"`{selected_models}`",
        "",
        "## Feature-ablation frequency",
        "",
        f"`{selected_features}`",
        "",
        "## Interpretation constraint",
        "",
        (
            "The nested repeated-CV result is a secondary fusion-development "
            "study performed on the locked prediction export. It must not be "
            "described as the original untouched test result. The original "
            "locked result remains unchanged. A new temporally later dataset "
            "is required for final external confirmation."
        ),
        "",
        "## Quantum requirement",
        "",
        (
            "Quantum-only, classical-only, equal hybrid, original hybrid, and "
            "learned quantum-inclusive fusion variants are saved. The QAOA "
            "companion experiment remains available and is executed unless "
            "AQUA_SKIP_QAOA=1."
        ),
    ]

    (REPORT_DIR / "honest_hybrid_quantum_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_companion_qaoa() -> None:
    if not RUN_QAOA_EXTENSION:
        (
            LOG_DIR / "qaoa_status.txt"
        ).write_text(
            "Skipped because AQUA_SKIP_QAOA=1.",
            encoding="utf-8",
        )
        return

    script_path = Path(__file__).with_name(
        "25_aqua_sla_complete_quantum_ablation.py"
    )

    if not script_path.exists():
        (
            LOG_DIR / "qaoa_status.txt"
        ).write_text(
            (
                "Companion QAOA script not found. Place "
                "25_aqua_sla_complete_quantum_ablation.py beside this file."
            ),
            encoding="utf-8",
        )
        return

    specification = importlib.util.spec_from_file_location(
        "aqua_qaoa_companion",
        script_path,
    )

    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the companion QAOA script.")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    start = time.perf_counter()
    module.main()
    elapsed = time.perf_counter() - start

    (
        LOG_DIR / "qaoa_status.txt"
    ).write_text(
        f"Companion QAOA experiment completed in {elapsed:.3f} seconds.",
        encoding="utf-8",
    )


def save_metadata(source_path: Path) -> None:
    metadata = {
        "project": "AQUA-SLA",
        "study": "honest hybrid-fusion and quantum ablation extension",
        "source_locked_export": str(source_path),
        "outer_splits": OUTER_SPLITS,
        "outer_repeats": OUTER_REPEATS,
        "inner_splits": INNER_SPLITS,
        "threshold_grid_points": len(THRESHOLD_GRID),
        "random_seed": RANDOM_SEED,
        "qaoa_extension_enabled": RUN_QAOA_EXTENSION,
        "python": sys.version,
        "platform": platform.platform(),
        "claim_note": (
            "Nested repeated-CV development result; original locked test "
            "result remains unchanged."
        ),
    }

    with (REPORT_DIR / "experiment_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    ensure_directories()
    source_path = locate_locked_export()
    save_metadata(source_path)

    frame = pd.read_csv(source_path)

    transparent = run_transparent_ablation(frame)
    folds, predictions, selections = run_nested_fusion_study(frame)
    summary = summarize_nested_fusion(folds)
    final_artifact = fit_development_final_model(frame)

    create_figures(transparent, folds, selections)
    write_report(source_path, transparent, folds, summary)

    try:
        with pd.ExcelWriter(
            OUTPUT_DIR / "aqua_sla_honest_hybrid_quantum_ablation.xlsx",
            engine="openpyxl",
        ) as writer:
            transparent.to_excel(
                writer,
                sheet_name="Locked Ablation",
                index=False,
            )
            folds.to_excel(
                writer,
                sheet_name="Nested Fold Metrics",
                index=False,
            )
            predictions.to_excel(
                writer,
                sheet_name="Nested Predictions",
                index=False,
            )
            selections.to_excel(
                writer,
                sheet_name="Inner Selection",
                index=False,
            )
            summary.to_excel(
                writer,
                sheet_name="Nested Summary",
                index=False,
            )
    except Exception as exc:
        (
            LOG_DIR / "excel_export_error.txt"
        ).write_text(str(exc), encoding="utf-8")

    run_companion_qaoa()

    print("\nCompleted.")
    print(f"Output directory:\n{OUTPUT_DIR}")
    print("\nNested fusion summary:")
    print(summary.to_string(index=False))
    print("\nDevelopment model:")
    print(
        json.dumps(
            {
                key: value
                for key, value in final_artifact.items()
                if key != "model_or_blend"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
