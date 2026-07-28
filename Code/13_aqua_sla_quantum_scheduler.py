from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
except ImportError as exc:
    raise SystemExit(
        "\nMissing Qiskit optimization packages.\n"
        "Install them inside the aqua-sla environment:\n\n"
        "  pip install -U qiskit qiskit-algorithms qiskit-optimization\n"
    ) from exc


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
DATA_PATH = PROJECT_DIR / "Dataset" / "processed" / "aqua_sla_runtime_prediction.csv"

PREDICTION_CANDIDATES = [
    PROJECT_DIR / "results" / "quantum" / "aqua_sla_v3" / "test_predictions.csv",
    PROJECT_DIR / "results" / "quantum" / "aqua_sla_v4" / "seed_11" / "iquantum_risk_export.csv",
]

RESULT_DIR = PROJECT_DIR / "results" / "orchestration" / "aqua_sla_qaoa"

SEEDS = [11, 22, 33, 44, 55]
TASKS_PER_BATCH = 4
NUMBER_OF_BATCHES_PER_SEED = 20

QAOA_REPS = 2
QAOA_MAX_ITER = 150
QAOA_SHOTS = 4096

WEIGHT_SLA_RISK = 0.45
WEIGHT_RESPONSE_TIME = 0.25
WEIGHT_ENERGY = 0.15
WEIGHT_COST = 0.15

CONSTRAINT_PENALTY = 20.0
SLA_SUCCESS_THRESHOLD = 0.62


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


def locate_prediction_file() -> Path:
    for path in PREDICTION_CANDIDATES:
        if path.exists():
            return path
    attempted = "\n".join(f"  - {p}" for p in PREDICTION_CANDIDATES)
    raise FileNotFoundError("No prediction file found. Attempted:\n" + attempted)


def normalize_series(values: pd.Series, lower: float = 0.05, upper: float = 1.0) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.fillna(numeric.median())
    q_low = numeric.quantile(0.01)
    q_high = numeric.quantile(0.99)
    numeric = numeric.clip(q_low, q_high)
    minimum = float(numeric.min())
    maximum = float(numeric.max())

    if math.isclose(minimum, maximum):
        return pd.Series(np.full(len(numeric), (lower + upper) / 2.0), index=numeric.index)

    scaled = (numeric - minimum) / (maximum - minimum)
    return lower + scaled * (upper - lower)


def load_task_frame() -> pd.DataFrame:
    prediction_path = locate_prediction_file()
    print(f"Using prediction file: {prediction_path}")

    predictions = pd.read_csv(prediction_path)
    source = pd.read_csv(DATA_PATH, low_memory=False)

    probability_column = next(
        (column for column in ["hybrid_probability", "sla_risk_probability", "classical_probability"]
         if column in predictions.columns),
        None,
    )
    if probability_column is None:
        raise ValueError("Prediction file requires a probability column.")

    actual_column = next(
        (column for column in ["actual", "sla_violation"] if column in predictions.columns),
        None,
    )
    if actual_column is None:
        raise ValueError("Prediction file requires actual or sla_violation.")

    if "record_id" not in predictions.columns:
        raise ValueError("Prediction file requires record_id.")

    keep_columns = [
        column for column in [
            "record_id", "requested_cpu", "requested_memory", "average_cpu",
            "average_memory", "maximum_cpu", "maximum_memory",
            "cpu_peak_to_average_ratio", "time_seconds"
        ] if column in source.columns
    ]

    merged = predictions.merge(
        source[keep_columns].drop_duplicates("record_id"),
        on="record_id",
        how="left",
        validate="many_to_one",
    )

    merged["risk_probability"] = pd.to_numeric(
        merged[probability_column], errors="coerce"
    ).clip(0.0, 1.0)

    merged["actual_violation"] = pd.to_numeric(
        merged[actual_column], errors="coerce"
    ).fillna(0).astype(int)

    cpu_source = next(
        (column for column in ["maximum_cpu", "average_cpu", "requested_cpu"]
         if column in merged.columns),
        None,
    )
    memory_source = next(
        (column for column in ["maximum_memory", "average_memory", "requested_memory"]
         if column in merged.columns),
        None,
    )
    if cpu_source is None or memory_source is None:
        raise ValueError("Runtime dataset requires CPU and memory columns.")

    merged["cpu_demand"] = normalize_series(merged[cpu_source])
    merged["memory_demand"] = normalize_series(merged[memory_source])

    runtime_raw = (
        pd.to_numeric(merged["cpu_peak_to_average_ratio"], errors="coerce")
        if "cpu_peak_to_average_ratio" in merged.columns
        else merged["cpu_demand"] + merged["memory_demand"]
    )

    merged["runtime_proxy"] = normalize_series(runtime_raw, lower=0.2, upper=1.0)
    merged["deadline_proxy"] = (
        1.15
        - 0.45 * merged["risk_probability"]
        + 0.15 * (1.0 - merged["runtime_proxy"])
    ).clip(0.45, 1.25)

    return merged.dropna(
        subset=[
            "risk_probability", "cpu_demand", "memory_demand",
            "runtime_proxy", "deadline_proxy"
        ]
    ).reset_index(drop=True)


def build_resources(number_of_resources: int) -> list[Resource]:
    templates = [
        (1.00, 1.00, 1.30, 1.15, 1.30),
        (0.90, 1.10, 1.10, 0.90, 1.05),
        (1.15, 0.90, 1.00, 0.80, 0.85),
        (1.05, 1.20, 0.90, 0.70, 0.75),
        (1.25, 1.25, 1.40, 1.25, 1.45),
        (0.85, 0.95, 0.80, 0.60, 0.65),
    ]
    return [
        Resource(i, *templates[i % len(templates)])
        for i in range(number_of_resources)
    ]


def rows_to_tasks(batch: pd.DataFrame) -> list[Task]:
    tasks = []
    for task_id, row in batch.reset_index(drop=True).iterrows():
        tasks.append(
            Task(
                task_id=task_id,
                record_id=int(row["record_id"]),
                risk_probability=float(row["risk_probability"]),
                actual_violation=int(row["actual_violation"]),
                cpu_demand=float(row["cpu_demand"]),
                memory_demand=float(row["memory_demand"]),
                runtime_proxy=float(row["runtime_proxy"]),
                deadline_proxy=float(row["deadline_proxy"]),
            )
        )
    return tasks


def assignment_components(task: Task, resource: Resource) -> dict[str, float]:
    cpu_pressure = task.cpu_demand / resource.cpu_capacity
    memory_pressure = task.memory_demand / resource.memory_capacity
    capacity_pressure = max(cpu_pressure, memory_pressure)

    response_time = (
        task.runtime_proxy
        * (1.0 + 0.70 * capacity_pressure)
        / resource.speed_factor
    )
    overflow = max(0.0, capacity_pressure - 1.0)

    risk_adjusted_violation = (
        task.risk_probability * (0.55 + 0.45 * capacity_pressure)
        + 0.70 * overflow
        + 0.20 * max(0.0, response_time - task.deadline_proxy)
    )
    energy = response_time * resource.energy_rate * (
        0.55 + 0.45 * min(capacity_pressure, 1.5)
    )
    cost = response_time * resource.cost_rate

    return {
        "response_time": response_time,
        "risk_adjusted_violation": risk_adjusted_violation,
        "energy": energy,
        "cost": cost,
        "capacity_pressure": capacity_pressure,
        "overflow": overflow,
    }


def build_cost_matrix(tasks: list[Task], resources: list[Resource]) -> np.ndarray:
    matrix = np.zeros((len(tasks), len(resources)))
    raw = [
        [assignment_components(task, resource) for resource in resources]
        for task in tasks
    ]

    for component, weight in [
        ("risk_adjusted_violation", WEIGHT_SLA_RISK),
        ("response_time", WEIGHT_RESPONSE_TIME),
        ("energy", WEIGHT_ENERGY),
        ("cost", WEIGHT_COST),
    ]:
        values = np.array(
            [[raw[i][j][component] for j in range(len(resources))]
             for i in range(len(tasks))],
            dtype=float,
        )
        minimum = float(values.min())
        maximum = float(values.max())
        normalized = (
            np.zeros_like(values)
            if math.isclose(maximum - minimum, 0.0)
            else (values - minimum) / (maximum - minimum)
        )
        matrix += weight * normalized

    for i, task in enumerate(tasks):
        for j, resource in enumerate(resources):
            matrix[i, j] += 2.5 * assignment_components(task, resource)["overflow"]

    return matrix


def round_robin_assignment(tasks: list[Task], resources: list[Resource]) -> dict[int, int]:
    return {task.task_id: task.task_id % len(resources) for task in tasks}


def greedy_risk_aware_assignment(
    tasks: list[Task],
    resources: list[Resource],
    cost_matrix: np.ndarray,
) -> dict[int, int]:
    available = set(range(len(resources)))
    assignment: dict[int, int] = {}
    task_order = sorted(
        range(len(tasks)),
        key=lambda index: tasks[index].risk_probability,
        reverse=True,
    )

    for task_index in task_order:
        selected = min(
            available,
            key=lambda resource_index: cost_matrix[task_index, resource_index],
        )
        assignment[task_index] = selected
        available.remove(selected)

    return assignment


def hungarian_assignment(cost_matrix: np.ndarray) -> dict[int, int]:
    task_indices, resource_indices = linear_sum_assignment(cost_matrix)
    return {
        int(task_index): int(resource_index)
        for task_index, resource_index in zip(task_indices, resource_indices)
    }


def build_quadratic_program(cost_matrix: np.ndarray) -> QuadraticProgram:
    task_count, resource_count = cost_matrix.shape
    problem = QuadraticProgram("aqua_sla_qaoa_assignment")
    names = {}

    for i in range(task_count):
        for j in range(resource_count):
            name = f"x_{i}_{j}"
            names[(i, j)] = name
            problem.binary_var(name=name)

    problem.minimize(
        linear={
            names[(i, j)]: float(cost_matrix[i, j])
            for i in range(task_count)
            for j in range(resource_count)
        }
    )

    for i in range(task_count):
        problem.linear_constraint(
            linear={names[(i, j)]: 1.0 for j in range(resource_count)},
            sense="==",
            rhs=1.0,
            name=f"task_{i}_one_resource",
        )

    for j in range(resource_count):
        problem.linear_constraint(
            linear={names[(i, j)]: 1.0 for i in range(task_count)},
            sense="<=",
            rhs=1.0,
            name=f"resource_{j}_capacity",
        )

    return problem


def qaoa_assignment(
    cost_matrix: np.ndarray,
    seed: int,
) -> tuple[dict[int, int], dict[str, float]]:
    problem = build_quadratic_program(cost_matrix)
    sampler = StatevectorSampler(default_shots=QAOA_SHOTS, seed=seed)
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=QAOA_MAX_ITER),
        reps=QAOA_REPS,
    )
    solver = MinimumEigenOptimizer(
        min_eigen_solver=qaoa,
        penalty=CONSTRAINT_PENALTY,
    )

    start = time.perf_counter()
    result = solver.solve(problem)
    elapsed = time.perf_counter() - start

    task_count, resource_count = cost_matrix.shape
    assignment = {}

    for i in range(task_count):
        candidates = []
        for j in range(resource_count):
            variable_index = i * resource_count + j
            candidates.append((float(result.x[variable_index]), j))
        assignment[i] = int(max(candidates)[1])

    used = set()
    duplicates = []
    for task_index in sorted(assignment):
        resource_index = assignment[task_index]
        if resource_index in used:
            duplicates.append(task_index)
        else:
            used.add(resource_index)

    unused = [j for j in range(resource_count) if j not in used]
    for task_index in duplicates:
        selected = min(unused, key=lambda j: cost_matrix[task_index, j])
        assignment[task_index] = selected
        unused.remove(selected)

    return assignment, {
        "qaoa_seconds": elapsed,
        "qaoa_objective": float(result.fval),
        "qaoa_status": str(result.status),
    }


def evaluate_assignment(
    scheduler: str,
    tasks: list[Task],
    resources: list[Resource],
    assignment: dict[int, int],
    qaoa_metadata: dict[str, float] | None = None,
) -> dict[str, float]:
    rows = []

    for task in tasks:
        resource = resources[assignment[task.task_id]]
        components = assignment_components(task, resource)
        observed_risk = (
            0.65 * components["risk_adjusted_violation"]
            + 0.35 * task.actual_violation
        )
        sla_success = float(
            observed_risk <= SLA_SUCCESS_THRESHOLD
            and components["response_time"] <= task.deadline_proxy
            and components["overflow"] <= 0.0
        )
        rows.append(
            {
                **components,
                "sla_success": sla_success,
                "observed_risk": observed_risk,
                "cpu_utilization": min(task.cpu_demand / resource.cpu_capacity, 1.5),
                "memory_utilization": min(task.memory_demand / resource.memory_capacity, 1.5),
            }
        )

    result = {
        "scheduler": scheduler,
        "sla_satisfaction_rate": float(np.mean([r["sla_success"] for r in rows])),
        "sla_violation_rate": float(1.0 - np.mean([r["sla_success"] for r in rows])),
        "mean_observed_risk": float(np.mean([r["observed_risk"] for r in rows])),
        "mean_response_time": float(np.mean([r["response_time"] for r in rows])),
        "makespan": float(np.max([r["response_time"] for r in rows])),
        "total_energy": float(np.sum([r["energy"] for r in rows])),
        "total_cost": float(np.sum([r["cost"] for r in rows])),
        "mean_cpu_utilization": float(np.mean([r["cpu_utilization"] for r in rows])),
        "mean_memory_utilization": float(np.mean([r["memory_utilization"] for r in rows])),
        "capacity_overflow_count": int(np.sum([r["overflow"] > 0.0 for r in rows])),
    }

    result.update(
        qaoa_metadata
        if qaoa_metadata
        else {"qaoa_seconds": np.nan, "qaoa_objective": np.nan, "qaoa_status": ""}
    )
    return result


def confidence_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean
    margin = 1.96 * float(np.std(array, ddof=1) / math.sqrt(len(array)))
    return mean - margin, mean + margin


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    task_frame = load_task_frame()
    resources = build_resources(TASKS_PER_BATCH)

    all_results = []
    assignment_rows = []

    for seed in SEEDS:
        print(f"\n===== Scheduling seed {seed} =====")
        rng = np.random.default_rng(seed)
        sample_size = TASKS_PER_BATCH * NUMBER_OF_BATCHES_PER_SEED

        if len(task_frame) < sample_size:
            raise ValueError(
                f"Need {sample_size} rows; only {len(task_frame)} are available."
            )

        sampled = task_frame.iloc[
            rng.choice(len(task_frame), size=sample_size, replace=False)
        ].reset_index(drop=True)

        for batch_index in range(NUMBER_OF_BATCHES_PER_SEED):
            batch = sampled.iloc[
                batch_index * TASKS_PER_BATCH:
                (batch_index + 1) * TASKS_PER_BATCH
            ].copy()

            tasks = rows_to_tasks(batch)
            cost_matrix = build_cost_matrix(tasks, resources)

            schedules = {
                "round_robin": (
                    round_robin_assignment(tasks, resources),
                    None,
                ),
                "classical_risk_greedy": (
                    greedy_risk_aware_assignment(tasks, resources, cost_matrix),
                    None,
                ),
                "hungarian_optimal": (
                    hungarian_assignment(cost_matrix),
                    None,
                ),
            }

            try:
                q_assignment, q_metadata = qaoa_assignment(
                    cost_matrix,
                    seed=seed * 1000 + batch_index,
                )
                schedules["qaoa_sla_aware"] = (q_assignment, q_metadata)
            except Exception as exc:
                print(f"QAOA failed at seed={seed}, batch={batch_index}: {exc}")
                continue

            for scheduler, (assignment, metadata) in schedules.items():
                result = evaluate_assignment(
                    scheduler, tasks, resources, assignment, metadata
                )
                result["seed"] = seed
                result["batch"] = batch_index
                all_results.append(result)

                for task in tasks:
                    assignment_rows.append(
                        {
                            "seed": seed,
                            "batch": batch_index,
                            "scheduler": scheduler,
                            "task_id": task.task_id,
                            "record_id": task.record_id,
                            "risk_probability": task.risk_probability,
                            "actual_violation": task.actual_violation,
                            "assigned_resource": assignment[task.task_id],
                        }
                    )

            print(
                f"Completed batch {batch_index + 1}/"
                f"{NUMBER_OF_BATCHES_PER_SEED}"
            )

    results = pd.DataFrame(all_results)
    if results.empty:
        raise RuntimeError("No scheduling runs completed.")

    results.to_csv(RESULT_DIR / "scheduling_batch_results.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(
        RESULT_DIR / "scheduling_assignments.csv", index=False
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
    for scheduler, group in results.groupby("scheduler"):
        row = {"scheduler": scheduler, "runs": len(group)}
        for metric in metrics:
            values = group[metric].astype(float)
            ci_low, ci_high = confidence_interval(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULT_DIR / "scheduling_summary.csv", index=False)

    baseline = summary[summary["scheduler"] == "round_robin"].iloc[0]
    improvement_rows = []

    for _, row in summary.iterrows():
        baseline_violation = baseline["sla_violation_rate_mean"]
        current_violation = row["sla_violation_rate_mean"]

        violation_reduction = (
            0.0
            if math.isclose(baseline_violation, 0.0)
            else (baseline_violation - current_violation) / baseline_violation
        )

        improvement_rows.append(
            {
                "scheduler": row["scheduler"],
                "sla_violation_reduction_vs_round_robin": violation_reduction,
                "sla_satisfaction_improvement_points": (
                    row["sla_satisfaction_rate_mean"]
                    - baseline["sla_satisfaction_rate_mean"]
                ),
                "response_time_reduction_vs_round_robin": (
                    baseline["mean_response_time_mean"]
                    - row["mean_response_time_mean"]
                ) / baseline["mean_response_time_mean"],
                "energy_reduction_vs_round_robin": (
                    baseline["total_energy_mean"]
                    - row["total_energy_mean"]
                ) / baseline["total_energy_mean"],
                "cost_reduction_vs_round_robin": (
                    baseline["total_cost_mean"]
                    - row["total_cost_mean"]
                ) / baseline["total_cost_mean"],
            }
        )

    improvements = pd.DataFrame(improvement_rows)
    improvements.to_csv(
        RESULT_DIR / "scheduler_improvements.csv", index=False
    )

    summary[
        [
            "scheduler",
            "runs",
            "sla_satisfaction_rate_mean",
            "sla_violation_rate_mean",
            "mean_response_time_mean",
            "makespan_mean",
            "total_energy_mean",
            "total_cost_mean",
            "mean_cpu_utilization_mean",
            "mean_memory_utilization_mean",
        ]
    ].to_csv(
        RESULT_DIR / "iquantum_scheduler_metrics.csv",
        index=False,
    )

    metadata = {
        "prediction_file": str(locate_prediction_file()),
        "dataset": str(DATA_PATH),
        "seeds": SEEDS,
        "tasks_per_batch": TASKS_PER_BATCH,
        "batches_per_seed": NUMBER_OF_BATCHES_PER_SEED,
        "qaoa_reps": QAOA_REPS,
        "qaoa_max_iter": QAOA_MAX_ITER,
        "qaoa_shots": QAOA_SHOTS,
        "constraint_penalty": CONSTRAINT_PENALTY,
        "objective_weights": {
            "sla_risk": WEIGHT_SLA_RISK,
            "response_time": WEIGHT_RESPONSE_TIME,
            "energy": WEIGHT_ENERGY,
            "cost": WEIGHT_COST,
        },
    }

    with (RESULT_DIR / "experiment_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    display_columns = [
        "scheduler",
        "sla_satisfaction_rate_mean",
        "sla_violation_rate_mean",
        "mean_response_time_mean",
        "makespan_mean",
        "total_energy_mean",
        "total_cost_mean",
    ]

    print("\n===== Scheduling summary =====")
    print(summary[display_columns].to_string(index=False))
    print("\n===== Improvements over round robin =====")
    print(improvements.to_string(index=False))
    print(f"\nOutputs saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
