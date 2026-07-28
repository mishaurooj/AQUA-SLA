from __future__ import annotations

"""
AQUA-SLA Combined Evaluation Pipeline
=====================================

This script combines:

1. The locked AQUA-SLA-v3 prediction results.
2. A repeated 5-seed QAOA SLA-aware scheduling evaluation.
3. A unified hybrid summary for paper reporting.

It does NOT retrain the predictor and does NOT reuse the test set for tuning.

Expected inputs
---------------
D:\other\AQUA-SLA\results\quantum\aqua_sla_v3\test_predictions.csv

Optional prediction summary files
---------------------------------
D:\other\AQUA-SLA\results\quantum\aqua_sla_v3\model_comparison.csv
D:\other\AQUA-SLA\results\quantum\aqua_sla_v3\metrics.csv
D:\other\AQUA-SLA\results\quantum\aqua_sla_v3\test_metrics.json

Outputs
-------
D:\other\AQUA-SLA\results\combined\aqua_sla_final\
"""

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
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

try:
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
except ImportError as exc:
    raise SystemExit(
        "\nMissing Qiskit packages.\n"
        "Install them in the aqua-sla environment:\n\n"
        "  pip install -U qiskit qiskit-algorithms qiskit-optimization\n"
    ) from exc


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

V3_RESULT_DIR = (
    PROJECT_DIR
    / "results"
    / "quantum"
    / "aqua_sla_v3"
)

PREDICTION_PATH = V3_RESULT_DIR / "test_predictions.csv"

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "combined"
    / "aqua_sla_final"
)

# Repeated and reproducible proof-of-concept QAOA configuration.
SEEDS = [11, 22, 33, 44, 55]
TASKS_PER_BATCH = 2
BATCHES_PER_SEED = 10

QAOA_REPS = 1
QAOA_MAX_ITER = 30
QAOA_SHOTS = 512
CONSTRAINT_PENALTY = 20.0

# Scheduling objective.
WEIGHT_SLA_RISK = 0.45
WEIGHT_RESPONSE_TIME = 0.25
WEIGHT_ENERGY = 0.15
WEIGHT_COST = 0.15

# Operational SLA threshold.
SLA_SUCCESS_THRESHOLD = 0.62

# Suppress sparse matrix efficiency warnings that do not affect correctness.
warnings.filterwarnings(
    "ignore",
    message=".*SparseEfficiencyWarning.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="scipy.sparse",
)


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class Resource:
    resource_id: int
    cpu_capacity: float
    memory_capacity: float
    speed_factor: float
    energy_rate: float
    cost_rate: float


@dataclass(frozen=True)
class Task:
    task_id: int
    record_id: int
    risk_probability: float
    actual_violation: int
    cpu_demand: float
    memory_demand: float
    runtime_proxy: float
    deadline_proxy: float


# ============================================================
# Utility functions
# ============================================================

def normalize_series(
    values: pd.Series,
    lower: float = 0.05,
    upper: float = 1.0,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.fillna(numeric.median())

    q_low = numeric.quantile(0.01)
    q_high = numeric.quantile(0.99)
    numeric = numeric.clip(q_low, q_high)

    minimum = float(numeric.min())
    maximum = float(numeric.max())

    if math.isclose(minimum, maximum):
        return pd.Series(
            np.full(len(numeric), (lower + upper) / 2.0),
            index=numeric.index,
        )

    scaled = (numeric - minimum) / (maximum - minimum)
    return lower + scaled * (upper - lower)


def confidence_interval(
    values: Iterable[float],
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(np.mean(array))

    if len(array) < 2:
        return mean, mean

    margin = 1.96 * float(
        np.std(array, ddof=1) / math.sqrt(len(array))
    )
    return mean - margin, mean + margin


def safe_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


# ============================================================
# Prediction evaluation
# ============================================================

def detect_prediction_columns(
    predictions: pd.DataFrame,
) -> tuple[str, str, str | None]:
    probability_column = next(
        (
            column
            for column in [
                "hybrid_probability",
                "sla_risk_probability",
                "classical_probability",
                "probability",
                "y_probability",
            ]
            if column in predictions.columns
        ),
        None,
    )

    actual_column = next(
        (
            column
            for column in [
                "actual",
                "sla_violation",
                "y_true",
                "target",
            ]
            if column in predictions.columns
        ),
        None,
    )

    prediction_column = next(
        (
            column
            for column in [
                "hybrid_prediction",
                "prediction",
                "y_pred",
                "predicted_label",
            ]
            if column in predictions.columns
        ),
        None,
    )

    if probability_column is None:
        raise ValueError(
            "Could not identify a probability column in test_predictions.csv."
        )

    if actual_column is None:
        raise ValueError(
            "Could not identify the actual target column in test_predictions.csv."
        )

    return probability_column, actual_column, prediction_column


def evaluate_locked_predictor(
    predictions: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    probability_column, actual_column, prediction_column = (
        detect_prediction_columns(predictions)
    )

    y_true = pd.to_numeric(
        predictions[actual_column],
        errors="coerce",
    ).fillna(0).astype(int).to_numpy()

    probabilities = pd.to_numeric(
        predictions[probability_column],
        errors="coerce",
    ).fillna(0.5).clip(0.0, 1.0).to_numpy()

    if prediction_column is not None:
        y_pred = pd.to_numeric(
            predictions[prediction_column],
            errors="coerce",
        ).fillna(0).astype(int).to_numpy()

        # Infer the effective threshold only for reporting where possible.
        positive_probabilities = probabilities[y_pred == 1]
        negative_probabilities = probabilities[y_pred == 0]
        if len(positive_probabilities) and len(negative_probabilities):
            threshold = float(
                (
                    np.min(positive_probabilities)
                    + np.max(negative_probabilities)
                )
                / 2.0
            )
        else:
            threshold = 0.5
    else:
        # Locked threshold from the completed v3 experiment.
        threshold = 0.274514
        y_pred = (probabilities >= threshold).astype(int)

    metrics = {
        "model": "aqua_sla_v3_locked_predictor",
        "samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": safe_specificity(y_true, y_pred),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
    }

    export = predictions.copy()
    export["locked_probability"] = probabilities
    export["locked_prediction"] = y_pred
    export["locked_actual"] = y_true

    return metrics, export


# ============================================================
# Scheduling data preparation
# ============================================================

def load_scheduling_frame(
    prediction_export: pd.DataFrame,
) -> pd.DataFrame:
    source = pd.read_csv(DATA_PATH, low_memory=False)

    if "record_id" not in prediction_export.columns:
        raise ValueError(
            "test_predictions.csv must contain record_id "
            "to link predictions to runtime telemetry."
        )

    keep_columns = [
        column
        for column in [
            "record_id",
            "requested_cpu",
            "requested_memory",
            "average_cpu",
            "average_memory",
            "maximum_cpu",
            "maximum_memory",
            "cpu_peak_to_average_ratio",
            "time_seconds",
        ]
        if column in source.columns
    ]

    merged = prediction_export.merge(
        source[keep_columns].drop_duplicates("record_id"),
        on="record_id",
        how="left",
        validate="many_to_one",
    )

    merged["risk_probability"] = merged[
        "locked_probability"
    ].clip(0.0, 1.0)

    merged["actual_violation"] = merged[
        "locked_actual"
    ].astype(int)

    cpu_source = next(
        (
            column
            for column in [
                "maximum_cpu",
                "average_cpu",
                "requested_cpu",
            ]
            if column in merged.columns
        ),
        None,
    )

    memory_source = next(
        (
            column
            for column in [
                "maximum_memory",
                "average_memory",
                "requested_memory",
            ]
            if column in merged.columns
        ),
        None,
    )

    if cpu_source is None or memory_source is None:
        raise ValueError(
            "The runtime dataset does not contain usable CPU/memory columns."
        )

    merged["cpu_demand"] = normalize_series(
        merged[cpu_source]
    )
    merged["memory_demand"] = normalize_series(
        merged[memory_source]
    )

    if "cpu_peak_to_average_ratio" in merged.columns:
        runtime_raw = pd.to_numeric(
            merged["cpu_peak_to_average_ratio"],
            errors="coerce",
        )
    else:
        runtime_raw = (
            merged["cpu_demand"]
            + merged["memory_demand"]
        )

    merged["runtime_proxy"] = normalize_series(
        runtime_raw,
        lower=0.2,
        upper=1.0,
    )

    merged["deadline_proxy"] = (
        1.15
        - 0.45 * merged["risk_probability"]
        + 0.15 * (1.0 - merged["runtime_proxy"])
    ).clip(0.45, 1.25)

    return merged.dropna(
        subset=[
            "risk_probability",
            "cpu_demand",
            "memory_demand",
            "runtime_proxy",
            "deadline_proxy",
        ]
    ).reset_index(drop=True)


def build_resources(
    number_of_resources: int,
) -> list[Resource]:
    templates = [
        (1.00, 1.00, 1.30, 1.15, 1.30),
        (0.90, 1.10, 1.10, 0.90, 1.05),
        (1.15, 0.90, 1.00, 0.80, 0.85),
        (1.05, 1.20, 0.90, 0.70, 0.75),
    ]

    return [
        Resource(
            resource_id=index,
            cpu_capacity=templates[index % len(templates)][0],
            memory_capacity=templates[index % len(templates)][1],
            speed_factor=templates[index % len(templates)][2],
            energy_rate=templates[index % len(templates)][3],
            cost_rate=templates[index % len(templates)][4],
        )
        for index in range(number_of_resources)
    ]


def rows_to_tasks(
    batch: pd.DataFrame,
) -> list[Task]:
    tasks = []

    for task_id, row in batch.reset_index(drop=True).iterrows():
        tasks.append(
            Task(
                task_id=int(task_id),
                record_id=int(row["record_id"]),
                risk_probability=float(
                    row["risk_probability"]
                ),
                actual_violation=int(
                    row["actual_violation"]
                ),
                cpu_demand=float(row["cpu_demand"]),
                memory_demand=float(
                    row["memory_demand"]
                ),
                runtime_proxy=float(
                    row["runtime_proxy"]
                ),
                deadline_proxy=float(
                    row["deadline_proxy"]
                ),
            )
        )

    return tasks


# ============================================================
# Scheduling objective
# ============================================================

def assignment_components(
    task: Task,
    resource: Resource,
) -> dict[str, float]:
    cpu_pressure = (
        task.cpu_demand / resource.cpu_capacity
    )
    memory_pressure = (
        task.memory_demand / resource.memory_capacity
    )
    capacity_pressure = max(
        cpu_pressure,
        memory_pressure,
    )

    response_time = (
        task.runtime_proxy
        * (1.0 + 0.70 * capacity_pressure)
        / resource.speed_factor
    )

    overflow = max(
        0.0,
        capacity_pressure - 1.0,
    )

    risk_adjusted_violation = (
        task.risk_probability
        * (0.55 + 0.45 * capacity_pressure)
        + 0.70 * overflow
        + 0.20
        * max(
            0.0,
            response_time - task.deadline_proxy,
        )
    )

    energy = (
        response_time
        * resource.energy_rate
        * (
            0.55
            + 0.45
            * min(capacity_pressure, 1.5)
        )
    )

    cost = response_time * resource.cost_rate

    return {
        "response_time": float(response_time),
        "risk_adjusted_violation": float(
            risk_adjusted_violation
        ),
        "energy": float(energy),
        "cost": float(cost),
        "capacity_pressure": float(
            capacity_pressure
        ),
        "overflow": float(overflow),
    }


def build_cost_matrix(
    tasks: list[Task],
    resources: list[Resource],
) -> np.ndarray:
    matrix = np.zeros(
        (len(tasks), len(resources)),
        dtype=float,
    )

    raw = [
        [
            assignment_components(task, resource)
            for resource in resources
        ]
        for task in tasks
    ]

    for component, weight in [
        (
            "risk_adjusted_violation",
            WEIGHT_SLA_RISK,
        ),
        (
            "response_time",
            WEIGHT_RESPONSE_TIME,
        ),
        (
            "energy",
            WEIGHT_ENERGY,
        ),
        (
            "cost",
            WEIGHT_COST,
        ),
    ]:
        values = np.array(
            [
                [
                    raw[i][j][component]
                    for j in range(len(resources))
                ]
                for i in range(len(tasks))
            ],
            dtype=float,
        )

        minimum = float(values.min())
        maximum = float(values.max())

        if math.isclose(
            maximum - minimum,
            0.0,
        ):
            normalized = np.zeros_like(values)
        else:
            normalized = (
                values - minimum
            ) / (maximum - minimum)

        matrix += weight * normalized

    for i, task in enumerate(tasks):
        for j, resource in enumerate(resources):
            overflow = assignment_components(
                task,
                resource,
            )["overflow"]
            matrix[i, j] += 2.5 * overflow

    return matrix


# ============================================================
# Scheduling algorithms
# ============================================================

def round_robin_assignment(
    tasks: list[Task],
    resources: list[Resource],
) -> dict[int, int]:
    return {
        task.task_id:
        task.task_id % len(resources)
        for task in tasks
    }


def greedy_assignment(
    tasks: list[Task],
    resources: list[Resource],
    cost_matrix: np.ndarray,
) -> dict[int, int]:
    available = set(range(len(resources)))
    assignment: dict[int, int] = {}

    task_order = sorted(
        range(len(tasks)),
        key=lambda index:
        tasks[index].risk_probability,
        reverse=True,
    )

    for task_index in task_order:
        selected = min(
            available,
            key=lambda resource_index:
            cost_matrix[
                task_index,
                resource_index,
            ],
        )

        assignment[task_index] = selected
        available.remove(selected)

    return assignment


def hungarian_assignment(
    cost_matrix: np.ndarray,
) -> dict[int, int]:
    task_indices, resource_indices = (
        linear_sum_assignment(cost_matrix)
    )

    return {
        int(task_index): int(resource_index)
        for task_index, resource_index in zip(
            task_indices,
            resource_indices,
        )
    }


def build_quadratic_program(
    cost_matrix: np.ndarray,
) -> QuadraticProgram:
    task_count, resource_count = (
        cost_matrix.shape
    )

    problem = QuadraticProgram(
        "aqua_sla_qaoa_assignment"
    )

    variable_names: dict[
        tuple[int, int],
        str,
    ] = {}

    for task_index in range(task_count):
        for resource_index in range(
            resource_count
        ):
            name = (
                f"x_{task_index}_"
                f"{resource_index}"
            )
            variable_names[
                (
                    task_index,
                    resource_index,
                )
            ] = name
            problem.binary_var(name=name)

    problem.minimize(
        linear={
            variable_names[
                (
                    task_index,
                    resource_index,
                )
            ]:
            float(
                cost_matrix[
                    task_index,
                    resource_index,
                ]
            )
            for task_index in range(
                task_count
            )
            for resource_index in range(
                resource_count
            )
        }
    )

    for task_index in range(task_count):
        problem.linear_constraint(
            linear={
                variable_names[
                    (
                        task_index,
                        resource_index,
                    )
                ]: 1.0
                for resource_index in range(
                    resource_count
                )
            },
            sense="==",
            rhs=1.0,
            name=(
                f"task_{task_index}_"
                "one_resource"
            ),
        )

    for resource_index in range(
        resource_count
    ):
        problem.linear_constraint(
            linear={
                variable_names[
                    (
                        task_index,
                        resource_index,
                    )
                ]: 1.0
                for task_index in range(
                    task_count
                )
            },
            sense="<=",
            rhs=1.0,
            name=(
                f"resource_{resource_index}_"
                "capacity"
            ),
        )

    return problem


def qaoa_assignment(
    cost_matrix: np.ndarray,
    seed: int,
) -> tuple[
    dict[int, int],
    dict[str, float | str],
]:
    problem = build_quadratic_program(
        cost_matrix
    )

    sampler = StatevectorSampler(
        default_shots=QAOA_SHOTS,
        seed=seed,
    )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(
            maxiter=QAOA_MAX_ITER
        ),
        reps=QAOA_REPS,
    )

    solver = MinimumEigenOptimizer(
        min_eigen_solver=qaoa,
        penalty=CONSTRAINT_PENALTY,
    )

    start = time.perf_counter()
    result = solver.solve(problem)
    elapsed = time.perf_counter() - start

    task_count, resource_count = (
        cost_matrix.shape
    )

    assignment: dict[int, int] = {}

    for task_index in range(task_count):
        candidates = []

        for resource_index in range(
            resource_count
        ):
            variable_index = (
                task_index
                * resource_count
                + resource_index
            )

            candidates.append(
                (
                    float(
                        result.x[
                            variable_index
                        ]
                    ),
                    resource_index,
                )
            )

        assignment[task_index] = int(
            max(candidates)[1]
        )

    # Repair duplicate resource usage.
    used: set[int] = set()
    duplicate_tasks: list[int] = []

    for task_index in sorted(assignment):
        resource_index = assignment[
            task_index
        ]

        if resource_index in used:
            duplicate_tasks.append(
                task_index
            )
        else:
            used.add(resource_index)

    unused = [
        resource_index
        for resource_index in range(
            resource_count
        )
        if resource_index not in used
    ]

    for task_index in duplicate_tasks:
        selected = min(
            unused,
            key=lambda resource_index:
            cost_matrix[
                task_index,
                resource_index,
            ],
        )

        assignment[task_index] = selected
        unused.remove(selected)

    metadata: dict[str, float | str] = {
        "qaoa_seconds": float(elapsed),
        "qaoa_objective": float(
            result.fval
        ),
        "qaoa_status": str(
            result.status
        ),
    }

    return assignment, metadata


# ============================================================
# Scheduling evaluation
# ============================================================

def evaluate_assignment(
    scheduler: str,
    tasks: list[Task],
    resources: list[Resource],
    assignment: dict[int, int],
    qaoa_metadata: (
        dict[str, float | str] | None
    ) = None,
) -> dict[str, float | str]:
    rows = []

    for task in tasks:
        resource = resources[
            assignment[task.task_id]
        ]

        components = assignment_components(
            task,
            resource,
        )

        observed_risk = (
            0.65
            * components[
                "risk_adjusted_violation"
            ]
            + 0.35
            * task.actual_violation
        )

        sla_success = float(
            observed_risk
            <= SLA_SUCCESS_THRESHOLD
            and components[
                "response_time"
            ]
            <= task.deadline_proxy
            and components["overflow"]
            <= 0.0
        )

        rows.append(
            {
                **components,
                "sla_success": sla_success,
                "observed_risk":
                observed_risk,
                "cpu_utilization":
                min(
                    task.cpu_demand
                    / resource.cpu_capacity,
                    1.5,
                ),
                "memory_utilization":
                min(
                    task.memory_demand
                    / resource.memory_capacity,
                    1.5,
                ),
            }
        )

    result: dict[str, float | str] = {
        "scheduler": scheduler,
        "sla_satisfaction_rate": float(
            np.mean(
                [
                    row["sla_success"]
                    for row in rows
                ]
            )
        ),
        "sla_violation_rate": float(
            1.0
            - np.mean(
                [
                    row["sla_success"]
                    for row in rows
                ]
            )
        ),
        "mean_observed_risk": float(
            np.mean(
                [
                    row["observed_risk"]
                    for row in rows
                ]
            )
        ),
        "mean_response_time": float(
            np.mean(
                [
                    row["response_time"]
                    for row in rows
                ]
            )
        ),
        "makespan": float(
            np.max(
                [
                    row["response_time"]
                    for row in rows
                ]
            )
        ),
        "total_energy": float(
            np.sum(
                [
                    row["energy"]
                    for row in rows
                ]
            )
        ),
        "total_cost": float(
            np.sum(
                [
                    row["cost"]
                    for row in rows
                ]
            )
        ),
        "mean_cpu_utilization": float(
            np.mean(
                [
                    row[
                        "cpu_utilization"
                    ]
                    for row in rows
                ]
            )
        ),
        "mean_memory_utilization": float(
            np.mean(
                [
                    row[
                        "memory_utilization"
                    ]
                    for row in rows
                ]
            )
        ),
        "capacity_overflow_count": int(
            np.sum(
                [
                    row["overflow"] > 0
                    for row in rows
                ]
            )
        ),
    }

    if qaoa_metadata is None:
        result.update(
            {
                "qaoa_seconds":
                float("nan"),
                "qaoa_objective":
                float("nan"),
                "qaoa_status": "",
            }
        )
    else:
        result.update(qaoa_metadata)

    return result


def run_scheduling_evaluation(
    scheduling_frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    resources = build_resources(
        TASKS_PER_BATCH
    )

    all_results: list[dict] = []
    assignment_rows: list[dict] = []

    for seed in SEEDS:
        print(
            f"\n===== Scheduling seed {seed} ====="
        )

        rng = np.random.default_rng(seed)
        sample_size = (
            TASKS_PER_BATCH
            * BATCHES_PER_SEED
        )

        if len(scheduling_frame) < sample_size:
            raise ValueError(
                f"Need at least {sample_size} "
                "prediction rows."
            )

        sampled = scheduling_frame.iloc[
            rng.choice(
                len(scheduling_frame),
                size=sample_size,
                replace=False,
            )
        ].reset_index(drop=True)

        for batch_index in range(
            BATCHES_PER_SEED
        ):
            start = (
                batch_index
                * TASKS_PER_BATCH
            )
            end = (
                start
                + TASKS_PER_BATCH
            )

            batch = sampled.iloc[
                start:end
            ].copy()

            tasks = rows_to_tasks(batch)
            cost_matrix = build_cost_matrix(
                tasks,
                resources,
            )

            schedules = {
                "round_robin": (
                    round_robin_assignment(
                        tasks,
                        resources,
                    ),
                    None,
                ),
                "classical_risk_greedy": (
                    greedy_assignment(
                        tasks,
                        resources,
                        cost_matrix,
                    ),
                    None,
                ),
                "hungarian_optimal": (
                    hungarian_assignment(
                        cost_matrix
                    ),
                    None,
                ),
            }

            try:
                q_assignment, q_metadata = (
                    qaoa_assignment(
                        cost_matrix,
                        seed=(
                            seed * 1000
                            + batch_index
                        ),
                    )
                )

                schedules[
                    "qaoa_sla_aware"
                ] = (
                    q_assignment,
                    q_metadata,
                )

            except Exception as exc:
                print(
                    "QAOA failed for "
                    f"batch {batch_index}: "
                    f"{exc}"
                )

            for scheduler, (
                assignment,
                metadata,
            ) in schedules.items():
                result = evaluate_assignment(
                    scheduler=scheduler,
                    tasks=tasks,
                    resources=resources,
                    assignment=assignment,
                    qaoa_metadata=metadata,
                )

                result["seed"] = seed
                result["batch"] = (
                    batch_index
                )

                all_results.append(result)

                for task in tasks:
                    assignment_rows.append(
                        {
                            "seed": seed,
                            "batch":
                            batch_index,
                            "scheduler":
                            scheduler,
                            "task_id":
                            task.task_id,
                            "record_id":
                            task.record_id,
                            "risk_probability":
                            task.risk_probability,
                            "actual_violation":
                            task.actual_violation,
                            "assigned_resource":
                            assignment[
                                task.task_id
                            ],
                        }
                    )

            print(
                "Completed batch "
                f"{batch_index + 1}/"
                f"{BATCHES_PER_SEED}"
            )

    results = pd.DataFrame(
        all_results
    )

    assignments = pd.DataFrame(
        assignment_rows
    )

    metrics = [
        "sla_satisfaction_rate",
        "sla_violation_rate",
        "mean_observed_risk",
        "mean_response_time",
        "makespan",
        "total_energy",
        "total_cost",
        "mean_cpu_utilization",
        "mean_memory_utilization",
        "capacity_overflow_count",
    ]

    summary_rows = []

    for scheduler, group in results.groupby(
        "scheduler"
    ):
        row: dict[str, float | str] = {
            "scheduler": scheduler,
            "runs": int(len(group)),
        }

        for metric in metrics:
            values = group[
                metric
            ].astype(float)

            ci_low, ci_high = (
                confidence_interval(values)
            )

            row[
                f"{metric}_mean"
            ] = float(values.mean())

            row[
                f"{metric}_std"
            ] = float(
                values.std(ddof=1)
            )

            row[
                f"{metric}_ci95_low"
            ] = ci_low

            row[
                f"{metric}_ci95_high"
            ] = ci_high

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    return results, assignments, summary



def build_statistical_tests(
    scheduling_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Paired non-parametric comparisons using identical seed/batch instances.
    The Wilcoxon signed-rank test is applied to QAOA versus Round Robin and
    QAOA versus Hungarian for the main operational metrics.
    """
    metrics = [
        "sla_violation_rate",
        "mean_response_time",
        "makespan",
        "total_energy",
        "total_cost",
    ]

    rows = []

    def paired_frame(
        scheduler_a: str,
        scheduler_b: str,
    ) -> pd.DataFrame:
        left = scheduling_results[
            scheduling_results["scheduler"] == scheduler_a
        ][["seed", "batch", *metrics]].copy()

        right = scheduling_results[
            scheduling_results["scheduler"] == scheduler_b
        ][["seed", "batch", *metrics]].copy()

        return left.merge(
            right,
            on=["seed", "batch"],
            suffixes=("_a", "_b"),
            how="inner",
            validate="one_to_one",
        )

    for scheduler_a, scheduler_b in [
        ("qaoa_sla_aware", "round_robin"),
        ("qaoa_sla_aware", "hungarian_optimal"),
    ]:
        paired = paired_frame(
            scheduler_a,
            scheduler_b,
        )

        for metric in metrics:
            a = paired[f"{metric}_a"].astype(float).to_numpy()
            b = paired[f"{metric}_b"].astype(float).to_numpy()
            differences = a - b

            if len(differences) == 0:
                statistic = float("nan")
                p_value = float("nan")
            elif np.allclose(differences, 0.0):
                statistic = 0.0
                p_value = 1.0
            else:
                result = wilcoxon(
                    a,
                    b,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            rows.append(
                {
                    "scheduler_a": scheduler_a,
                    "scheduler_b": scheduler_b,
                    "metric": metric,
                    "paired_instances": int(len(paired)),
                    "mean_a": float(np.mean(a)) if len(a) else float("nan"),
                    "mean_b": float(np.mean(b)) if len(b) else float("nan"),
                    "mean_difference_a_minus_b": (
                        float(np.mean(differences))
                        if len(differences)
                        else float("nan")
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "significant_at_0_05": (
                        bool(p_value < 0.05)
                        if not math.isnan(p_value)
                        else False
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_exact_match_summary(
    scheduling_results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "sla_satisfaction_rate",
        "sla_violation_rate",
        "mean_response_time",
        "makespan",
        "total_energy",
        "total_cost",
    ]

    qaoa = scheduling_results[
        scheduling_results["scheduler"] == "qaoa_sla_aware"
    ][["seed", "batch", *metrics]].copy()

    hungarian = scheduling_results[
        scheduling_results["scheduler"] == "hungarian_optimal"
    ][["seed", "batch", *metrics]].copy()

    paired = qaoa.merge(
        hungarian,
        on=["seed", "batch"],
        suffixes=("_qaoa", "_hungarian"),
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        return pd.DataFrame(
            [
                {
                    "paired_instances": 0,
                    "exact_match_count": 0,
                    "exact_match_rate": float("nan"),
                }
            ]
        )

    exact_flags = []

    for _, row in paired.iterrows():
        exact_flags.append(
            all(
                math.isclose(
                    float(row[f"{metric}_qaoa"]),
                    float(row[f"{metric}_hungarian"]),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for metric in metrics
            )
        )

    return pd.DataFrame(
        [
            {
                "paired_instances": int(len(paired)),
                "exact_match_count": int(sum(exact_flags)),
                "exact_match_rate": float(np.mean(exact_flags)),
            }
        ]
    )


# ============================================================
# Combined reporting
# ============================================================

def build_improvement_table(
    scheduling_summary: pd.DataFrame,
) -> pd.DataFrame:
    baseline = scheduling_summary[
        scheduling_summary[
            "scheduler"
        ] == "round_robin"
    ].iloc[0]

    rows = []

    for _, row in (
        scheduling_summary.iterrows()
    ):
        baseline_violation = baseline[
            "sla_violation_rate_mean"
        ]

        current_violation = row[
            "sla_violation_rate_mean"
        ]

        if math.isclose(
            baseline_violation,
            0.0,
        ):
            violation_reduction = 0.0
        else:
            violation_reduction = (
                baseline_violation
                - current_violation
            ) / baseline_violation

        rows.append(
            {
                "scheduler":
                row["scheduler"],
                "sla_violation_reduction_vs_round_robin":
                float(
                    violation_reduction
                ),
                "sla_satisfaction_improvement_points":
                float(
                    row[
                        "sla_satisfaction_rate_mean"
                    ]
                    - baseline[
                        "sla_satisfaction_rate_mean"
                    ]
                ),
                "response_time_reduction_vs_round_robin":
                float(
                    (
                        baseline[
                            "mean_response_time_mean"
                        ]
                        - row[
                            "mean_response_time_mean"
                        ]
                    )
                    / baseline[
                        "mean_response_time_mean"
                    ]
                ),
                "makespan_reduction_vs_round_robin":
                float(
                    (
                        baseline[
                            "makespan_mean"
                        ]
                        - row[
                            "makespan_mean"
                        ]
                    )
                    / baseline[
                        "makespan_mean"
                    ]
                ),
                "energy_reduction_vs_round_robin":
                float(
                    (
                        baseline[
                            "total_energy_mean"
                        ]
                        - row[
                            "total_energy_mean"
                        ]
                    )
                    / baseline[
                        "total_energy_mean"
                    ]
                ),
                "cost_reduction_vs_round_robin":
                float(
                    (
                        baseline[
                            "total_cost_mean"
                        ]
                        - row[
                            "total_cost_mean"
                        ]
                    )
                    / baseline[
                        "total_cost_mean"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def build_combined_summary(
    prediction_metrics: dict[str, float],
    scheduling_summary: pd.DataFrame,
    improvements: pd.DataFrame,
) -> pd.DataFrame:
    qaoa_summary = scheduling_summary[
        scheduling_summary[
            "scheduler"
        ] == "qaoa_sla_aware"
    ]

    hungarian_summary = scheduling_summary[
        scheduling_summary[
            "scheduler"
        ] == "hungarian_optimal"
    ]

    qaoa_improvements = improvements[
        improvements[
            "scheduler"
        ] == "qaoa_sla_aware"
    ]

    rows = [
        {
            "component":
            "SLA prediction",
            "method":
            "AQUA-SLA-v3 locked hybrid predictor",
            "metric":
            "ROC-AUC",
            "value":
            prediction_metrics["roc_auc"],
            "interpretation":
            "Ranking quality on the locked temporal test set.",
        },
        {
            "component":
            "SLA prediction",
            "method":
            "AQUA-SLA-v3 locked hybrid predictor",
            "metric":
            "PR-AUC",
            "value":
            prediction_metrics["pr_auc"],
            "interpretation":
            "Violation-focused precision-recall performance.",
        },
        {
            "component":
            "SLA prediction",
            "method":
            "AQUA-SLA-v3 locked hybrid predictor",
            "metric":
            "Recall",
            "value":
            prediction_metrics["recall"],
            "interpretation":
            "Fraction of SLA violations detected.",
        },
        {
            "component":
            "SLA prediction",
            "method":
            "AQUA-SLA-v3 locked hybrid predictor",
            "metric":
            "F1",
            "value":
            prediction_metrics["f1"],
            "interpretation":
            "Balanced violation-class performance.",
        },
    ]

    if not qaoa_summary.empty:
        qaoa = qaoa_summary.iloc[0]

        rows.extend(
            [
                {
                    "component":
                    "Quantum scheduling",
                    "method":
                    "QAOA SLA-aware scheduler",
                    "metric":
                    "SLA satisfaction rate",
                    "value":
                    qaoa[
                        "sla_satisfaction_rate_mean"
                    ],
                    "interpretation":
                    "Operational SLA success in the small-scale QAOA study.",
                },
                {
                    "component":
                    "Quantum scheduling",
                    "method":
                    "QAOA SLA-aware scheduler",
                    "metric":
                    "Mean response time",
                    "value":
                    qaoa[
                        "mean_response_time_mean"
                    ],
                    "interpretation":
                    "Normalized response-time proxy.",
                },
                {
                    "component":
                    "Quantum scheduling",
                    "method":
                    "QAOA SLA-aware scheduler",
                    "metric":
                    "Makespan",
                    "value":
                    qaoa[
                        "makespan_mean"
                    ],
                    "interpretation":
                    "Normalized batch completion time.",
                },
            ]
        )

    if not qaoa_improvements.empty:
        q_imp = qaoa_improvements.iloc[0]

        rows.extend(
            [
                {
                    "component":
                    "Quantum scheduling",
                    "method":
                    "QAOA vs Round Robin",
                    "metric":
                    "Response-time reduction",
                    "value":
                    q_imp[
                        "response_time_reduction_vs_round_robin"
                    ],
                    "interpretation":
                    "Relative reduction versus the Round Robin baseline.",
                },
                {
                    "component":
                    "Quantum scheduling",
                    "method":
                    "QAOA vs Round Robin",
                    "metric":
                    "Makespan reduction",
                    "value":
                    q_imp[
                        "makespan_reduction_vs_round_robin"
                    ],
                    "interpretation":
                    "Relative reduction versus the Round Robin baseline.",
                },
                {
                    "component":
                    "Quantum scheduling",
                    "method":
                    "QAOA vs Round Robin",
                    "metric":
                    "Energy reduction",
                    "value":
                    q_imp[
                        "energy_reduction_vs_round_robin"
                    ],
                    "interpretation":
                    "Relative reduction versus the Round Robin baseline.",
                },
                {
                    "component":
                    "Quantum scheduling",
                    "method":
                    "QAOA vs Round Robin",
                    "metric":
                    "Cost reduction",
                    "value":
                    q_imp[
                        "cost_reduction_vs_round_robin"
                    ],
                    "interpretation":
                    "Relative reduction versus the Round Robin baseline.",
                },
            ]
        )

    if (
        not qaoa_summary.empty
        and not hungarian_summary.empty
    ):
        qaoa = qaoa_summary.iloc[0]
        hungarian = (
            hungarian_summary.iloc[0]
        )

        same_objective = (
            math.isclose(
                qaoa[
                    "mean_response_time_mean"
                ],
                hungarian[
                    "mean_response_time_mean"
                ],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                qaoa["makespan_mean"],
                hungarian["makespan_mean"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                qaoa[
                    "total_energy_mean"
                ],
                hungarian[
                    "total_energy_mean"
                ],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                qaoa[
                    "total_cost_mean"
                ],
                hungarian[
                    "total_cost_mean"
                ],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )

        rows.append(
            {
                "component":
                "Quantum scheduling",
                "method":
                "QAOA vs Hungarian",
                "metric":
                "Matched exact optimum",
                "value":
                1.0 if same_objective else 0.0,
                "interpretation":
                (
                    "QAOA matched the exact assignment metrics."
                    if same_objective
                    else
                    "QAOA did not exactly match the Hungarian metrics."
                ),
            }
        )

    return pd.DataFrame(rows)


def write_paper_summary(
    prediction_metrics: dict[str, float],
    scheduling_summary: pd.DataFrame,
    improvements: pd.DataFrame,
) -> None:
    qaoa = scheduling_summary[
        scheduling_summary[
            "scheduler"
        ] == "qaoa_sla_aware"
    ].iloc[0]

    qaoa_imp = improvements[
        improvements[
            "scheduler"
        ] == "qaoa_sla_aware"
    ].iloc[0]

    hungarian = scheduling_summary[
        scheduling_summary[
            "scheduler"
        ] == "hungarian_optimal"
    ].iloc[0]

    matched = (
        math.isclose(
            qaoa[
                "mean_response_time_mean"
            ],
            hungarian[
                "mean_response_time_mean"
            ],
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        and math.isclose(
            qaoa["makespan_mean"],
            hungarian["makespan_mean"],
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )

    text = f"""AQUA-SLA Combined Experimental Summary

Prediction stage
----------------
The locked AQUA-SLA-v3 predictor achieved:

- Accuracy: {prediction_metrics['accuracy']:.4f}
- Precision: {prediction_metrics['precision']:.4f}
- Recall: {prediction_metrics['recall']:.4f}
- Specificity: {prediction_metrics['specificity']:.4f}
- F1-score: {prediction_metrics['f1']:.4f}
- MCC: {prediction_metrics['mcc']:.4f}
- ROC-AUC: {prediction_metrics['roc_auc']:.4f}
- PR-AUC: {prediction_metrics['pr_auc']:.4f}

Quantum scheduling stage
------------------------
The QAOA SLA-aware scheduler achieved:

- SLA satisfaction rate: {qaoa['sla_satisfaction_rate_mean']:.4f}
- SLA violation rate: {qaoa['sla_violation_rate_mean']:.4f}
- Mean response time: {qaoa['mean_response_time_mean']:.6f}
- Makespan: {qaoa['makespan_mean']:.6f}
- Total energy: {qaoa['total_energy_mean']:.6f}
- Total cost: {qaoa['total_cost_mean']:.6f}

Relative to Round Robin:

- Response-time reduction: {100.0 * qaoa_imp['response_time_reduction_vs_round_robin']:.2f}%
- Makespan reduction: {100.0 * qaoa_imp['makespan_reduction_vs_round_robin']:.2f}%
- Energy reduction: {100.0 * qaoa_imp['energy_reduction_vs_round_robin']:.2f}%
- Cost reduction: {100.0 * qaoa_imp['cost_reduction_vs_round_robin']:.2f}%
- SLA-violation reduction: {100.0 * qaoa_imp['sla_violation_reduction_vs_round_robin']:.2f}%

Exact-optimizer comparison
--------------------------
QAOA matched the Hungarian response-time and makespan metrics: {matched}

Recommended claim
-----------------
AQUA-SLA combines a high-recall SLA-risk predictor with a QAOA-based
resource-assignment module. The locked predictor provides the operational
risk signal, while QAOA solves the small-scale discrete scheduling problem.
The quantum scheduling result should be described as a proof-of-concept,
not as evidence of quantum advantage.
"""

    (
        OUTPUT_DIR
        / "paper_ready_summary.txt"
    ).write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{PREDICTION_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Runtime dataset not found:\n"
            f"{DATA_PATH}"
        )

    print(
        "Loading locked v3 predictions:"
    )
    print(PREDICTION_PATH)

    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    prediction_metrics, prediction_export = (
        evaluate_locked_predictor(
            predictions
        )
    )

    print(
        "\n===== Locked prediction metrics ====="
    )
    for key, value in (
        prediction_metrics.items()
    ):
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    prediction_export.to_csv(
        OUTPUT_DIR
        / "locked_prediction_export.csv",
        index=False,
    )

    pd.DataFrame(
        [prediction_metrics]
    ).to_csv(
        OUTPUT_DIR
        / "prediction_metrics.csv",
        index=False,
    )

    scheduling_frame = (
        load_scheduling_frame(
            prediction_export
        )
    )

    (
        scheduling_results,
        assignments,
        scheduling_summary,
    ) = run_scheduling_evaluation(
        scheduling_frame
    )

    improvements = (
        build_improvement_table(
            scheduling_summary
        )
    )

    statistical_tests = build_statistical_tests(
        scheduling_results
    )

    exact_match_summary = build_exact_match_summary(
        scheduling_results
    )

    combined_summary = (
        build_combined_summary(
            prediction_metrics,
            scheduling_summary,
            improvements,
        )
    )

    scheduling_results.to_csv(
        OUTPUT_DIR
        / "scheduling_batch_results.csv",
        index=False,
    )

    assignments.to_csv(
        OUTPUT_DIR
        / "scheduling_assignments.csv",
        index=False,
    )

    scheduling_summary.to_csv(
        OUTPUT_DIR
        / "scheduling_summary.csv",
        index=False,
    )

    improvements.to_csv(
        OUTPUT_DIR
        / "scheduler_improvements.csv",
        index=False,
    )

    statistical_tests.to_csv(
        OUTPUT_DIR
        / "statistical_tests.csv",
        index=False,
    )

    exact_match_summary.to_csv(
        OUTPUT_DIR
        / "qaoa_exact_match_summary.csv",
        index=False,
    )

    combined_summary.to_csv(
        OUTPUT_DIR
        / "combined_hybrid_summary.csv",
        index=False,
    )

    write_paper_summary(
        prediction_metrics,
        scheduling_summary,
        improvements,
    )

    metadata = {
        "project": "AQUA-SLA",
        "prediction_model":
        "AQUA-SLA-v3 locked predictor",
        "prediction_path":
        str(PREDICTION_PATH),
        "dataset_path":
        str(DATA_PATH),
        "seeds": SEEDS,
        "tasks_per_batch":
        TASKS_PER_BATCH,
        "batches_per_seed":
        BATCHES_PER_SEED,
        "qaoa_reps":
        QAOA_REPS,
        "qaoa_max_iter":
        QAOA_MAX_ITER,
        "qaoa_shots":
        QAOA_SHOTS,
        "constraint_penalty":
        CONSTRAINT_PENALTY,
        "objective_weights": {
            "sla_risk":
            WEIGHT_SLA_RISK,
            "response_time":
            WEIGHT_RESPONSE_TIME,
            "energy":
            WEIGHT_ENERGY,
            "cost":
            WEIGHT_COST,
        },
        "scientific_scope":
        (
            "Locked prediction evaluation plus "
            "small-scale QAOA scheduling proof-of-concept."
        ),
    }

    with (
        OUTPUT_DIR
        / "experiment_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\n===== Scheduling summary ====="
    )

    print(
        scheduling_summary[
            [
                "scheduler",
                "sla_satisfaction_rate_mean",
                "sla_violation_rate_mean",
                "mean_response_time_mean",
                "makespan_mean",
                "total_energy_mean",
                "total_cost_mean",
            ]
        ].to_string(index=False)
    )

    print(
        "\n===== Improvements over Round Robin ====="
    )
    print(
        improvements.to_string(
            index=False
        )
    )

    print(
        "\n===== Statistical tests ====="
    )
    print(
        statistical_tests.to_string(
            index=False
        )
    )

    print(
        "\n===== QAOA exact-match summary ====="
    )
    print(
        exact_match_summary.to_string(
            index=False
        )
    )

    print(
        "\n===== Combined AQUA-SLA summary ====="
    )
    print(
        combined_summary.to_string(
            index=False
        )
    )

    print(
        f"\nOutputs saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
