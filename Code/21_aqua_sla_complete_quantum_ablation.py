from __future__ import annotations

r"""
AQUA-SLA Complete Quantum-Centric Ablation Suite
================================================

This script does NOT invent results. It:
1. Loads the locked AQUA-SLA prediction export and existing scheduling results.
2. Recomputes extended prediction metrics for classical, quantum, and hybrid outputs.
3. Runs a real, small-scale QAOA sensitivity benchmark on 2-task x 2-resource
   assignment QUBOs reconstructed from the saved scheduling assignments.
4. Saves all ablation CSV files, fitted/optimized parameter artifacts, circuits,
   backend status, execution logs, and publication JPEG figures.
5. Optionally runs one optimized QAOA circuit on IBM Quantum or IQM hardware.
   Hardware failures are recorded and do not stop the local experiment.

The QAOA benchmark is a controlled quantum-solver sensitivity experiment.
It supplements, but does not replace, the locked scheduling results.

Recommended packages:
    pip install numpy pandas scipy scikit-learn matplotlib pillow joblib
    pip install qiskit qiskit-aer
Optional IBM:
    pip install qiskit-ibm-runtime
Optional IQM:
    pip install qiskit-iqm

Environment variables for optional hardware:
    AQUA_RUN_IBM_HARDWARE=0 or 1
    IBM_BACKEND_NAME=<optional backend name>

    AQUA_RUN_IQM_HARDWARE=0 or 1
    IQM_SERVER_URL=<required for IQM>
    IQM_TOKEN=<configured as recommended by IQM>

Output:
    D:\other\AQUA-SLA\results\aqua_sla_final\quantum_ablation_complete
"""

import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import wilcoxon
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(r"D:\other\AQUA-SLA")

LOCKED_PREDICTION_PATH = (
    PROJECT_DIR
    / "results"
    / "quantum"
    / "aqua_sla_v3"
    / "locked_prediction_export.csv"
)

SCHEDULING_ROOT = (
    PROJECT_DIR
    / "results"
    / "aqua_sla_final"
)

# The script also searches recursively if the exact locations differ.
EXPECTED_FILES = {
    "locked_prediction_export": "locked_prediction_export.csv",
    "prediction_metrics": "prediction_metrics.csv",
    "scheduling_assignments": "scheduling_assignments.csv",
    "scheduling_batch_results": "scheduling_batch_results.csv",
    "scheduling_summary": "scheduling_summary.csv",
    "scheduler_improvements": "scheduler_improvements.csv",
    "statistical_tests": "statistical_tests.csv",
    "qaoa_exact_match_summary": "qaoa_exact_match_summary.csv",
    "experiment_metadata": "experiment_metadata.json",
}

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "aqua_sla_final"
    / "quantum_ablation_complete"
)

CSV_DIR = OUTPUT_DIR / "csv"
MODEL_DIR = OUTPUT_DIR / "models"
CIRCUIT_DIR = OUTPUT_DIR / "circuits"
FIGURE_DIR = OUTPUT_DIR / "figures_jpeg"
BACKEND_DIR = OUTPUT_DIR / "hardware"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

SEEDS = [11, 22, 33, 44, 55]
QAOA_DEPTHS = [1, 2, 3]
SHOT_COUNTS = [128, 256, 512, 1024]
OPTIMIZERS = ["COBYLA", "NELDER_MEAD", "POWELL"]

# Keep runtime reasonable. Increase for the final paper if desired.
MAX_QAOA_INSTANCES = 10
MAX_OPT_ITER = 80
ASSIGNMENT_PENALTY = 20.0

# Resource profiles used only for the controlled 2x2 QAOA sensitivity QUBO.
RESOURCE_SPEED = np.array([1.00, 1.15], dtype=float)
RESOURCE_ENERGY = np.array([1.00, 0.88], dtype=float)
RESOURCE_COST = np.array([0.92, 1.08], dtype=float)

OBJECTIVE_WEIGHTS = {
    "sla_risk": 0.45,
    "response_time": 0.25,
    "energy": 0.15,
    "cost": 0.15,
}

# Exact 1080p raster, JPEG only, quality 100, 900-DPI metadata.
FIGURE_WIDTH_PX = 1920
FIGURE_HEIGHT_PX = 1080
FIGURE_RENDER_DPI = 160
JPEG_DPI_METADATA = 900

RUN_IBM_HARDWARE = os.getenv("AQUA_RUN_IBM_HARDWARE", "0") == "1"
RUN_IQM_HARDWARE = os.getenv("AQUA_RUN_IQM_HARDWARE", "0") == "1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QUBO:
    linear: list[float]
    quadratic: dict[str, float]
    constant: float
    task_costs: list[list[float]]
    penalty: float

    @property
    def n_variables(self) -> int:
        return len(self.linear)


@dataclass
class QAOARun:
    seed: int
    instance_id: int
    depth: int
    shots: int
    optimizer: str
    success: bool
    status: str
    objective: float
    exact_objective: float
    objective_gap: float
    exact_match: bool
    feasible: bool
    best_bitstring: str
    exact_bitstring: str
    runtime_seconds: float
    function_evaluations: int
    gamma: list[float]
    beta: list[float]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    for directory in [
        OUTPUT_DIR,
        CSV_DIR,
        MODEL_DIR,
        CIRCUIT_DIR,
        FIGURE_DIR,
        BACKEND_DIR,
        REPORT_DIR,
        LOG_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def locate_file(filename: str) -> Path | None:
    candidates = [
        PROJECT_DIR / filename,
        PROJECT_DIR / "results" / filename,
        SCHEDULING_ROOT / filename,
        PROJECT_DIR / "results" / "quantum" / "aqua_sla_v3" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list((PROJECT_DIR / "results").rglob(filename))
    return matches[0] if matches else None


def resolve_inputs() -> dict[str, Path]:
    resolved: dict[str, Path] = {}

    for key, filename in EXPECTED_FILES.items():
        path = locate_file(filename)
        if path is not None:
            resolved[key] = path

    required = [
        "locked_prediction_export",
        "scheduling_assignments",
        "scheduling_batch_results",
        "scheduling_summary",
    ]
    missing = [key for key in required if key not in resolved]

    if missing:
        raise FileNotFoundError(
            "Required AQUA-SLA result files were not found: "
            + ", ".join(missing)
        )

    with (REPORT_DIR / "resolved_input_paths.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {key: str(value) for key, value in resolved.items()},
            file,
            indent=2,
        )

    return resolved


def confidence_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(np.mean(array))

    if len(array) < 2:
        return mean, mean

    margin = 1.96 * float(np.std(array, ddof=1) / math.sqrt(len(array)))
    return mean - margin, mean + margin


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0

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
        result += (count / len(y_true)) * abs(confidence - observed)

    return float(result)


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


# ---------------------------------------------------------------------------
# Prediction ablation from locked probabilities
# ---------------------------------------------------------------------------

def evaluate_probability_model(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    probabilities = np.clip(probabilities.astype(float), 1e-8, 1 - 1e-8)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if tn + fp else float("nan")
    false_alarm_rate = fp / (fp + tn) if fp + tn else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    gmean = math.sqrt(
        max(0.0, recall_score(y_true, predictions, zero_division=0))
        * max(0.0, specificity)
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "ece_10_bins": float(
            expected_calibration_error(y_true, probabilities, bins=10)
        ),
        "g_mean": float(gmean),
        "false_alarm_rate": float(false_alarm_rate),
        "negative_predictive_value": float(npv),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_prediction_ablation(
    locked: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_column = (
        "locked_actual"
        if "locked_actual" in locked.columns
        else "actual"
    )
    y_true = locked[actual_column].astype(int).to_numpy()

    threshold = float(
        locked["decision_threshold"].iloc[0]
        if "decision_threshold" in locked.columns
        else 0.5
    )

    probability_columns = {
        "classical_only": "classical_probability",
        "quantum_only": "quantum_probability",
        "hybrid_fusion": "hybrid_probability",
        "locked_operational": "locked_probability",
    }

    rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for model_name, column in probability_columns.items():
        if column not in locked.columns:
            continue

        probabilities = locked[column].astype(float).to_numpy()
        metrics = evaluate_probability_model(y_true, probabilities, threshold)
        rows.append(
            {
                "model": model_name,
                "probability_column": column,
                "samples": len(y_true),
                **metrics,
            }
        )

        prediction_rows.append(
            pd.DataFrame(
                {
                    "record_id": locked.get(
                        "record_id",
                        pd.Series(np.arange(len(locked))),
                    ),
                    "actual": y_true,
                    "model": model_name,
                    "probability": probabilities,
                    "prediction": (probabilities >= threshold).astype(int),
                }
            )
        )

    metrics_df = pd.DataFrame(rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    metrics_df.to_csv(
        CSV_DIR / "prediction_component_ablation.csv",
        index=False,
    )
    predictions_df.to_csv(
        CSV_DIR / "prediction_component_predictions.csv",
        index=False,
    )

    joblib.dump(
        {
            "source": "locked_prediction_export",
            "threshold": threshold,
            "available_probability_columns": probability_columns,
            "metrics": metrics_df.to_dict(orient="records"),
        },
        MODEL_DIR / "locked_prediction_component_ablation.joblib",
        compress=3,
    )

    return metrics_df, predictions_df


# ---------------------------------------------------------------------------
# QUBO construction and exact reference solver
# ---------------------------------------------------------------------------

def variable_index(task: int, resource: int) -> int:
    return task * 2 + resource


def build_assignment_qubo(risks: np.ndarray) -> QUBO:
    if len(risks) != 2:
        raise ValueError("The controlled QAOA benchmark requires two tasks.")

    linear = np.zeros(4, dtype=float)
    quadratic: dict[tuple[int, int], float] = {}
    task_costs = np.zeros((2, 2), dtype=float)

    for task in range(2):
        risk = float(risks[task])

        for resource in range(2):
            response = 1.0 / RESOURCE_SPEED[resource]
            cost = (
                OBJECTIVE_WEIGHTS["sla_risk"] * risk * response
                + OBJECTIVE_WEIGHTS["response_time"] * response
                + OBJECTIVE_WEIGHTS["energy"] * RESOURCE_ENERGY[resource]
                + OBJECTIVE_WEIGHTS["cost"] * RESOURCE_COST[resource]
            )
            task_costs[task, resource] = cost
            linear[variable_index(task, resource)] += cost

    constant = 0.0

    # Penalty * (x_t0 + x_t1 - 1)^2.
    for task in range(2):
        i = variable_index(task, 0)
        j = variable_index(task, 1)

        constant += ASSIGNMENT_PENALTY
        linear[i] += -ASSIGNMENT_PENALTY
        linear[j] += -ASSIGNMENT_PENALTY
        quadratic[(i, j)] = (
            quadratic.get((i, j), 0.0)
            + 2.0 * ASSIGNMENT_PENALTY
        )

    return QUBO(
        linear=linear.tolist(),
        quadratic={
            f"{i},{j}": float(value)
            for (i, j), value in quadratic.items()
        },
        constant=float(constant),
        task_costs=task_costs.tolist(),
        penalty=float(ASSIGNMENT_PENALTY),
    )


def qubo_value(qubo: QUBO, bits: np.ndarray) -> float:
    value = float(qubo.constant)
    value += float(np.dot(np.asarray(qubo.linear), bits))

    for key, coefficient in qubo.quadratic.items():
        i, j = (int(part) for part in key.split(","))
        value += coefficient * bits[i] * bits[j]

    return float(value)


def feasible_assignment(bits: np.ndarray) -> bool:
    return bool(bits[0] + bits[1] == 1 and bits[2] + bits[3] == 1)


def exact_solve_qubo(qubo: QUBO) -> tuple[str, float]:
    best_bitstring = ""
    best_value = float("inf")

    for integer in range(2 ** qubo.n_variables):
        bitstring = format(integer, f"0{qubo.n_variables}b")
        # Qiskit bitstrings are displayed q_(n-1)...q_0.
        bits = np.array(
            [int(character) for character in bitstring[::-1]],
            dtype=int,
        )
        value = qubo_value(qubo, bits)

        if value < best_value:
            best_value = value
            best_bitstring = bitstring

    return best_bitstring, float(best_value)


# ---------------------------------------------------------------------------
# QAOA simulator implemented directly with Qiskit Statevector
# ---------------------------------------------------------------------------

def require_qiskit() -> dict[str, Any]:
    try:
        import qiskit
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Statevector
    except Exception as exc:
        raise RuntimeError(
            "Qiskit is required for the quantum ablation. "
            "Install with: pip install qiskit qiskit-aer"
        ) from exc

    return {
        "qiskit": qiskit,
        "QuantumCircuit": QuantumCircuit,
        "Statevector": Statevector,
        "transpile": transpile,
    }


def append_cost_layer(
    circuit: Any,
    qubo: QUBO,
    gamma: float,
) -> None:
    # Convert QUBO linear/quadratic coefficients to Ising rotations.
    linear = np.asarray(qubo.linear, dtype=float)
    h = -0.5 * linear.copy()

    for key, coefficient in qubo.quadratic.items():
        i, j = (int(part) for part in key.split(","))
        h[i] += -coefficient / 4.0
        h[j] += -coefficient / 4.0
        circuit.rzz(gamma * coefficient / 2.0, i, j)

    for qubit, coefficient in enumerate(h):
        circuit.rz(2.0 * gamma * coefficient, qubit)


def build_qaoa_circuit(
    qubo: QUBO,
    gamma: np.ndarray,
    beta: np.ndarray,
    measure: bool,
) -> Any:
    modules = require_qiskit()
    QuantumCircuit = modules["QuantumCircuit"]

    if len(gamma) != len(beta):
        raise ValueError("gamma and beta must have equal length.")

    circuit = QuantumCircuit(
        qubo.n_variables,
        qubo.n_variables if measure else 0,
    )

    circuit.h(range(qubo.n_variables))

    for layer in range(len(gamma)):
        append_cost_layer(circuit, qubo, float(gamma[layer]))

        for qubit in range(qubo.n_variables):
            circuit.rx(2.0 * float(beta[layer]), qubit)

    if measure:
        circuit.measure(
            range(qubo.n_variables),
            range(qubo.n_variables),
        )

    return circuit


def statevector_probabilities(
    qubo: QUBO,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    modules = require_qiskit()
    Statevector = modules["Statevector"]

    circuit = build_qaoa_circuit(
        qubo,
        gamma=gamma,
        beta=beta,
        measure=False,
    )
    state = Statevector.from_instruction(circuit)
    return np.abs(np.asarray(state.data, dtype=complex)) ** 2


def sampled_probabilities(
    probabilities: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = rng.multinomial(shots, probabilities)
    return counts.astype(float) / float(shots)


def expectation_from_probabilities(
    qubo: QUBO,
    probabilities: np.ndarray,
) -> float:
    values = np.empty_like(probabilities, dtype=float)

    for integer in range(len(probabilities)):
        bits = np.array(
            [
                (integer >> qubit) & 1
                for qubit in range(qubo.n_variables)
            ],
            dtype=int,
        )
        values[integer] = qubo_value(qubo, bits)

    return float(np.dot(probabilities, values))


def decode_best_bitstring(
    probabilities: np.ndarray,
    n_variables: int,
) -> str:
    integer = int(np.argmax(probabilities))
    return format(integer, f"0{n_variables}b")


def optimize_qaoa(
    qubo: QUBO,
    depth: int,
    shots: int,
    optimizer_name: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    x0 = np.concatenate(
        [
            rng.uniform(0.0, np.pi, size=depth),
            rng.uniform(0.0, np.pi / 2.0, size=depth),
        ]
    )

    history: list[dict[str, float]] = []
    evaluations = 0

    def objective(parameters: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1

        gamma = parameters[:depth]
        beta = parameters[depth:]
        exact_probabilities = statevector_probabilities(
            qubo,
            gamma,
            beta,
        )
        probabilities = sampled_probabilities(
            exact_probabilities,
            shots,
            rng,
        )
        value = expectation_from_probabilities(qubo, probabilities)
        history.append(
            {
                "evaluation": evaluations,
                "objective": value,
            }
        )
        return value

    method_map = {
        "COBYLA": "COBYLA",
        "NELDER_MEAD": "Nelder-Mead",
        "POWELL": "Powell",
    }
    method = method_map[optimizer_name]

    start = time.perf_counter()
    result = minimize(
        objective,
        x0=x0,
        method=method,
        options={"maxiter": MAX_OPT_ITER},
    )
    runtime = time.perf_counter() - start

    gamma = np.asarray(result.x[:depth], dtype=float)
    beta = np.asarray(result.x[depth:], dtype=float)

    final_exact_probabilities = statevector_probabilities(
        qubo,
        gamma,
        beta,
    )
    final_sampled_probabilities = sampled_probabilities(
        final_exact_probabilities,
        shots,
        np.random.default_rng(seed + 999_983),
    )

    best_bitstring = decode_best_bitstring(
        final_sampled_probabilities,
        qubo.n_variables,
    )
    bits = np.array(
        [int(character) for character in best_bitstring[::-1]],
        dtype=int,
    )

    return (
        {
            "success": bool(result.success),
            "status": str(result.message),
            "objective": float(qubo_value(qubo, bits)),
            "runtime_seconds": float(runtime),
            "function_evaluations": int(evaluations),
            "gamma": gamma.tolist(),
            "beta": beta.tolist(),
            "best_bitstring": best_bitstring,
            "best_probabilities": final_sampled_probabilities.tolist(),
            "exact_probabilities": final_exact_probabilities.tolist(),
            "feasible": feasible_assignment(bits),
        },
        history,
    )


def select_qaoa_instances(assignments: pd.DataFrame) -> pd.DataFrame:
    qaoa_rows = assignments[
        assignments["scheduler"].astype(str).str.contains(
            "qaoa",
            case=False,
            na=False,
        )
    ].copy()

    if qaoa_rows.empty:
        qaoa_rows = assignments.copy()

    grouped = (
        qaoa_rows
        .sort_values(["seed", "batch", "task_id"])
        .groupby(["seed", "batch"], as_index=False)
        .filter(lambda group: group["task_id"].nunique() == 2)
    )

    unique_instances = (
        grouped[["seed", "batch"]]
        .drop_duplicates()
        .sort_values(["seed", "batch"])
        .head(MAX_QAOA_INSTANCES)
    )

    return grouped.merge(
        unique_instances,
        on=["seed", "batch"],
        how="inner",
    )


def run_qaoa_ablation(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[QUBO]]:
    selected = select_qaoa_instances(assignments)

    if selected.empty:
        raise ValueError(
            "No valid 2-task scheduling instances were found."
        )

    result_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    qubos: list[QUBO] = []

    instances = list(selected.groupby(["seed", "batch"]))

    for instance_id, ((source_seed, batch), group) in enumerate(instances):
        risks = (
            group
            .drop_duplicates("task_id")
            .sort_values("task_id")["risk_probability"]
            .astype(float)
            .to_numpy()
        )

        qubo = build_assignment_qubo(risks)
        qubos.append(qubo)
        exact_bitstring, exact_objective = exact_solve_qubo(qubo)

        with (
            MODEL_DIR / f"instance_{instance_id:03d}_qubo.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "instance_id": instance_id,
                    "source_seed": int(source_seed),
                    "source_batch": int(batch),
                    "risks": risks.tolist(),
                    "qubo": asdict(qubo),
                    "exact_bitstring": exact_bitstring,
                    "exact_objective": exact_objective,
                },
                file,
                indent=2,
            )

        for depth in QAOA_DEPTHS:
            for shots in SHOT_COUNTS:
                for optimizer_name in OPTIMIZERS:
                    for experiment_seed in SEEDS:
                        print(
                            "QAOA",
                            f"instance={instance_id}",
                            f"p={depth}",
                            f"shots={shots}",
                            optimizer_name,
                            f"seed={experiment_seed}",
                        )

                        run_seed = (
                            experiment_seed
                            + 10_000 * instance_id
                            + 100 * depth
                            + shots
                        )

                        try:
                            optimized, history = optimize_qaoa(
                                qubo,
                                depth=depth,
                                shots=shots,
                                optimizer_name=optimizer_name,
                                seed=run_seed,
                            )

                            objective_gap = (
                                optimized["objective"]
                                - exact_objective
                            )
                            exact_match = (
                                optimized["best_bitstring"]
                                == exact_bitstring
                            )

                            record = QAOARun(
                                seed=experiment_seed,
                                instance_id=instance_id,
                                depth=depth,
                                shots=shots,
                                optimizer=optimizer_name,
                                success=optimized["success"],
                                status=optimized["status"],
                                objective=optimized["objective"],
                                exact_objective=exact_objective,
                                objective_gap=float(objective_gap),
                                exact_match=bool(exact_match),
                                feasible=optimized["feasible"],
                                best_bitstring=optimized["best_bitstring"],
                                exact_bitstring=exact_bitstring,
                                runtime_seconds=optimized[
                                    "runtime_seconds"
                                ],
                                function_evaluations=optimized[
                                    "function_evaluations"
                                ],
                                gamma=optimized["gamma"],
                                beta=optimized["beta"],
                            )
                            result_rows.append(asdict(record))

                            run_name = (
                                f"instance_{instance_id:03d}"
                                f"_p_{depth}_shots_{shots}"
                                f"_{optimizer_name}_seed_{experiment_seed}"
                            )

                            joblib.dump(
                                {
                                    "run": asdict(record),
                                    "qubo": asdict(qubo),
                                    "optimized": optimized,
                                },
                                MODEL_DIR / f"{run_name}.joblib",
                                compress=3,
                            )

                            measured_circuit = build_qaoa_circuit(
                                qubo,
                                gamma=np.asarray(optimized["gamma"]),
                                beta=np.asarray(optimized["beta"]),
                                measure=True,
                            )
                            (
                                CIRCUIT_DIR / f"{run_name}.txt"
                            ).write_text(
                                str(measured_circuit.draw(output="text")),
                                encoding="utf-8",
                            )

                            try:
                                qasm = measured_circuit.qasm()
                                (
                                    CIRCUIT_DIR / f"{run_name}.qasm"
                                ).write_text(qasm, encoding="utf-8")
                            except Exception:
                                pass

                            for point in history:
                                history_rows.append(
                                    {
                                        "instance_id": instance_id,
                                        "depth": depth,
                                        "shots": shots,
                                        "optimizer": optimizer_name,
                                        "seed": experiment_seed,
                                        **point,
                                    }
                                )

                        except Exception as exc:
                            result_rows.append(
                                {
                                    "seed": experiment_seed,
                                    "instance_id": instance_id,
                                    "depth": depth,
                                    "shots": shots,
                                    "optimizer": optimizer_name,
                                    "success": False,
                                    "status": (
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                    "objective": float("nan"),
                                    "exact_objective": exact_objective,
                                    "objective_gap": float("nan"),
                                    "exact_match": False,
                                    "feasible": False,
                                    "best_bitstring": "",
                                    "exact_bitstring": exact_bitstring,
                                    "runtime_seconds": float("nan"),
                                    "function_evaluations": 0,
                                    "gamma": [],
                                    "beta": [],
                                }
                            )

    results = pd.DataFrame(result_rows)
    history = pd.DataFrame(history_rows)

    results.to_csv(CSV_DIR / "qaoa_ablation_runs.csv", index=False)
    history.to_csv(CSV_DIR / "qaoa_convergence_history.csv", index=False)

    return results, history, qubos


def summarize_qaoa_ablation(results: pd.DataFrame) -> pd.DataFrame:
    valid = results[results["success"] == True].copy()

    if valid.empty:
        summary = pd.DataFrame()
        summary.to_csv(CSV_DIR / "qaoa_ablation_summary.csv", index=False)
        return summary

    grouped = valid.groupby(
        ["depth", "shots", "optimizer"],
        as_index=False,
    )

    rows: list[dict[str, Any]] = []

    for keys, group in grouped:
        depth, shots, optimizer = keys
        gap_low, gap_high = confidence_interval(group["objective_gap"])
        time_low, time_high = confidence_interval(
            group["runtime_seconds"]
        )

        rows.append(
            {
                "depth": int(depth),
                "shots": int(shots),
                "optimizer": str(optimizer),
                "runs": int(len(group)),
                "success_rate": float(group["success"].mean()),
                "feasibility_rate": float(group["feasible"].mean()),
                "exact_match_rate": float(group["exact_match"].mean()),
                "objective_gap_mean": float(
                    group["objective_gap"].mean()
                ),
                "objective_gap_std": float(
                    group["objective_gap"].std(ddof=1)
                ),
                "objective_gap_ci95_low": gap_low,
                "objective_gap_ci95_high": gap_high,
                "runtime_seconds_mean": float(
                    group["runtime_seconds"].mean()
                ),
                "runtime_seconds_std": float(
                    group["runtime_seconds"].std(ddof=1)
                ),
                "runtime_seconds_ci95_low": time_low,
                "runtime_seconds_ci95_high": time_high,
                "function_evaluations_mean": float(
                    group["function_evaluations"].mean()
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["depth", "shots", "optimizer"]
    )
    summary.to_csv(CSV_DIR / "qaoa_ablation_summary.csv", index=False)
    return summary


def qaoa_paired_tests(results: pd.DataFrame) -> pd.DataFrame:
    valid = results[results["success"] == True].copy()
    rows: list[dict[str, Any]] = []

    # Depth comparisons while holding shots and optimizer fixed.
    for shots in SHOT_COUNTS:
        for optimizer in OPTIMIZERS:
            subset = valid[
                (valid["shots"] == shots)
                & (valid["optimizer"] == optimizer)
            ]

            for depth_a, depth_b in zip(
                QAOA_DEPTHS[:-1],
                QAOA_DEPTHS[1:],
            ):
                left = subset[subset["depth"] == depth_a][
                    ["instance_id", "seed", "objective_gap"]
                ]
                right = subset[subset["depth"] == depth_b][
                    ["instance_id", "seed", "objective_gap"]
                ]
                paired = left.merge(
                    right,
                    on=["instance_id", "seed"],
                    suffixes=("_a", "_b"),
                )

                if paired.empty:
                    continue

                a = paired["objective_gap_a"].to_numpy(dtype=float)
                b = paired["objective_gap_b"].to_numpy(dtype=float)
                differences = b - a

                if np.allclose(differences, 0.0):
                    statistic, p_value = 0.0, 1.0
                else:
                    test = wilcoxon(b, a, zero_method="wilcox")
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)

                rows.append(
                    {
                        "comparison_type": "depth",
                        "configuration": (
                            f"shots={shots};optimizer={optimizer}"
                        ),
                        "level_a": depth_a,
                        "level_b": depth_b,
                        "paired_runs": len(paired),
                        "mean_a": float(np.mean(a)),
                        "mean_b": float(np.mean(b)),
                        "mean_change_b_minus_a": float(
                            np.mean(differences)
                        ),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                        "significant_at_0_05": bool(p_value < 0.05),
                    }
                )

    tests = pd.DataFrame(rows)
    tests.to_csv(CSV_DIR / "qaoa_ablation_paired_tests.csv", index=False)
    return tests


# ---------------------------------------------------------------------------
# Optional hardware smoke tests
# ---------------------------------------------------------------------------

def best_qaoa_configuration(
    results: pd.DataFrame,
) -> pd.Series | None:
    valid = results[
        (results["success"] == True)
        & (results["feasible"] == True)
    ].copy()

    if valid.empty:
        return None

    return valid.sort_values(
        ["objective_gap", "runtime_seconds"],
        ascending=[True, True],
    ).iloc[0]


def counts_to_probabilities(
    counts: dict[str, int],
    n_qubits: int,
) -> np.ndarray:
    probabilities = np.zeros(2 ** n_qubits, dtype=float)
    total = float(sum(counts.values()))

    if total <= 0:
        return probabilities

    for bitstring, count in counts.items():
        clean = str(bitstring).replace(" ", "")
        integer = int(clean, 2)
        probabilities[integer] += float(count) / total

    return probabilities


def run_ibm_hardware_smoke_test(
    circuit: Any,
    qubo: QUBO,
    shots: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provider": "IBM Quantum",
        "attempted": RUN_IBM_HARDWARE,
        "success": False,
        "status": "disabled",
    }

    if not RUN_IBM_HARDWARE:
        return result

    try:
        from qiskit_ibm_runtime import (
            QiskitRuntimeService,
            SamplerV2,
        )
        from qiskit.transpiler import generate_preset_pass_manager

        service = QiskitRuntimeService()
        backend_name = os.getenv("IBM_BACKEND_NAME", "").strip()

        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=circuit.num_qubits,
            )

        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=1,
        )
        isa_circuit = pass_manager.run(circuit)

        sampler = SamplerV2(mode=backend)
        job = sampler.run([isa_circuit], shots=shots)
        pub_result = job.result()[0]

        counts = pub_result.data.meas.get_counts()
        probabilities = counts_to_probabilities(
            counts,
            circuit.num_qubits,
        )
        bitstring = decode_best_bitstring(
            probabilities,
            circuit.num_qubits,
        )
        bits = np.array(
            [int(character) for character in bitstring[::-1]],
            dtype=int,
        )

        result.update(
            {
                "success": True,
                "status": "completed",
                "backend": str(backend.name),
                "job_id": str(job.job_id()),
                "shots": shots,
                "counts": counts,
                "best_bitstring": bitstring,
                "objective": qubo_value(qubo, bits),
                "feasible": feasible_assignment(bits),
                "transpiled_depth": int(isa_circuit.depth()),
                "transpiled_size": int(isa_circuit.size()),
            }
        )

    except Exception as exc:
        result.update(
            {
                "status": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )

    return result


def run_iqm_hardware_smoke_test(
    circuit: Any,
    qubo: QUBO,
    shots: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provider": "IQM",
        "attempted": RUN_IQM_HARDWARE,
        "success": False,
        "status": "disabled",
    }

    if not RUN_IQM_HARDWARE:
        return result

    try:
        from iqm.qiskit_iqm import IQMProvider
        from qiskit import transpile

        server_url = os.getenv("IQM_SERVER_URL", "").strip()

        if not server_url:
            raise ValueError(
                "IQM_SERVER_URL is required when "
                "AQUA_RUN_IQM_HARDWARE=1."
            )

        provider = IQMProvider(server_url)
        backend_name = os.getenv("IQM_BACKEND_NAME", "").strip()
        backend = (
            provider.get_backend(backend_name)
            if backend_name
            else provider.get_backend()
        )

        transpiled = transpile(circuit, backend=backend)
        job = backend.run(transpiled, shots=shots)
        hardware_result = job.result()
        counts = hardware_result.get_counts()

        probabilities = counts_to_probabilities(
            counts,
            circuit.num_qubits,
        )
        bitstring = decode_best_bitstring(
            probabilities,
            circuit.num_qubits,
        )
        bits = np.array(
            [int(character) for character in bitstring[::-1]],
            dtype=int,
        )

        result.update(
            {
                "success": True,
                "status": "completed",
                "backend": str(backend.name),
                "job_id": str(job.job_id()),
                "shots": shots,
                "counts": counts,
                "best_bitstring": bitstring,
                "objective": qubo_value(qubo, bits),
                "feasible": feasible_assignment(bits),
                "transpiled_depth": int(transpiled.depth()),
                "transpiled_size": int(transpiled.size()),
            }
        )

    except Exception as exc:
        result.update(
            {
                "status": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )

    return result


def run_optional_hardware(
    qaoa_results: pd.DataFrame,
    qubos: list[QUBO],
) -> pd.DataFrame:
    best = best_qaoa_configuration(qaoa_results)

    if best is None:
        status = pd.DataFrame(
            [
                {
                    "provider": "IBM Quantum",
                    "attempted": RUN_IBM_HARDWARE,
                    "success": False,
                    "status": "No successful local QAOA configuration.",
                },
                {
                    "provider": "IQM",
                    "attempted": RUN_IQM_HARDWARE,
                    "success": False,
                    "status": "No successful local QAOA configuration.",
                },
            ]
        )
        status.to_csv(
            CSV_DIR / "hardware_backend_ablation.csv",
            index=False,
        )
        return status

    instance_id = int(best["instance_id"])
    qubo = qubos[instance_id]
    circuit = build_qaoa_circuit(
        qubo,
        gamma=np.asarray(best["gamma"], dtype=float),
        beta=np.asarray(best["beta"], dtype=float),
        measure=True,
    )

    results = [
        run_ibm_hardware_smoke_test(circuit, qubo, shots=512),
        run_iqm_hardware_smoke_test(circuit, qubo, shots=512),
    ]

    for record in results:
        safe_provider = record["provider"].lower().replace(" ", "_")
        with (
            BACKEND_DIR / f"{safe_provider}_execution.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2, default=str)

    table = pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if key not in {"counts", "traceback"}
            }
            for record in results
        ]
    )
    table.to_csv(CSV_DIR / "hardware_backend_ablation.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# Publication figures
# ---------------------------------------------------------------------------

def plot_prediction_ablation(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return

    columns = ["f1", "mcc", "roc_auc", "pr_auc"]
    data = metrics.set_index("model")[columns]

    fig, ax = plt.subplots()
    data.plot(kind="bar", ax=ax)
    ax.set_title("Classical, quantum, and hybrid prediction ablation")
    ax.set_xlabel("Prediction component")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "01_prediction_component_ablation")


def plot_calibration_ablation(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return

    columns = ["brier_score", "log_loss", "ece_10_bins"]
    data = metrics.set_index("model")[columns]

    fig, ax = plt.subplots()
    data.plot(kind="bar", ax=ax)
    ax.set_title("Prediction calibration and probabilistic loss")
    ax.set_xlabel("Prediction component")
    ax.set_ylabel("Lower is better")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "02_prediction_calibration_ablation")


def plot_scheduler_summary(summary: pd.DataFrame) -> None:
    metrics = [
        "sla_satisfaction_rate_mean",
        "mean_response_time_mean",
        "makespan_mean",
        "total_energy_mean",
        "total_cost_mean",
    ]
    available = [column for column in metrics if column in summary.columns]

    if not available:
        return

    normalized = summary[available].copy()

    for column in available:
        maximum = float(normalized[column].max())
        if maximum != 0:
            normalized[column] = normalized[column] / maximum

    normalized.index = summary["scheduler"].astype(str)

    fig, ax = plt.subplots()
    normalized.plot(kind="bar", ax=ax)
    ax.set_title("Normalized scheduler-performance comparison")
    ax.set_xlabel("Scheduler")
    ax.set_ylabel("Normalized metric")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "03_scheduler_ablation")


def plot_qaoa_exact_match(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    pivot = summary.pivot_table(
        index="depth",
        columns="shots",
        values="exact_match_rate",
        aggfunc="mean",
    )

    fig, ax = plt.subplots()
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(image, ax=ax, label="Exact-match rate")
    ax.set_xticks(
        range(len(pivot.columns)),
        labels=[str(value) for value in pivot.columns],
    )
    ax.set_yticks(
        range(len(pivot.index)),
        labels=[f"p={value}" for value in pivot.index],
    )
    ax.set_xlabel("Shots")
    ax.set_ylabel("QAOA depth")
    ax.set_title("QAOA exact-match sensitivity")
    fig.tight_layout()
    save_figure_jpeg(fig, "04_qaoa_exact_match_heatmap")


def plot_qaoa_objective_gap(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    for optimizer, group in summary.groupby("optimizer"):
        fig, ax = plt.subplots()

        for depth, depth_group in group.groupby("depth"):
            ordered = depth_group.sort_values("shots")
            ax.plot(
                ordered["shots"],
                ordered["objective_gap_mean"],
                marker="o",
                linewidth=1.8,
                label=f"p={depth}",
            )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Shots")
        ax.set_ylabel("Mean objective gap")
        ax.set_title(f"QAOA objective gap — {optimizer}")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        save_figure_jpeg(
            fig,
            f"05_qaoa_objective_gap_{optimizer.lower()}",
        )


def plot_qaoa_runtime(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    fig, ax = plt.subplots()

    for optimizer, group in summary.groupby("optimizer"):
        by_depth = (
            group.groupby("depth", as_index=False)[
                "runtime_seconds_mean"
            ]
            .mean()
            .sort_values("depth")
        )
        ax.plot(
            by_depth["depth"],
            by_depth["runtime_seconds_mean"],
            marker="o",
            linewidth=1.8,
            label=optimizer,
        )

    ax.set_xlabel("QAOA depth")
    ax.set_ylabel("Mean optimization runtime (seconds)")
    ax.set_title("QAOA computational-cost ablation")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "06_qaoa_runtime_ablation")


def plot_convergence(history: pd.DataFrame) -> None:
    if history.empty:
        return

    subset = history[
        (history["depth"] == history["depth"].min())
        & (history["shots"] == 512)
    ]

    fig, ax = plt.subplots()

    for optimizer, group in subset.groupby("optimizer"):
        curve = (
            group.groupby("evaluation", as_index=False)["objective"]
            .mean()
            .sort_values("evaluation")
        )
        ax.plot(
            curve["evaluation"],
            curve["objective"],
            linewidth=1.8,
            label=optimizer,
        )

    ax.set_xlabel("Objective evaluation")
    ax.set_ylabel("Mean sampled QUBO expectation")
    ax.set_title("QAOA optimizer convergence")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "07_qaoa_optimizer_convergence")


def plot_hardware_status(hardware: pd.DataFrame) -> None:
    if hardware.empty:
        return

    labels = hardware["provider"].astype(str).tolist()
    values = hardware["success"].astype(int).tolist()

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 1], labels=["Unavailable/failed", "Completed"])
    ax.set_title("Optional quantum-hardware execution status")
    ax.set_ylabel("Execution status")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure_jpeg(fig, "08_hardware_backend_status")


# ---------------------------------------------------------------------------
# Consolidated workbook/report
# ---------------------------------------------------------------------------

def export_workbook(
    prediction_metrics: pd.DataFrame,
    scheduling_summary: pd.DataFrame,
    qaoa_runs: pd.DataFrame,
    qaoa_summary: pd.DataFrame,
    qaoa_tests: pd.DataFrame,
    hardware: pd.DataFrame,
) -> None:
    try:
        with pd.ExcelWriter(
            OUTPUT_DIR / "aqua_sla_quantum_ablation.xlsx",
            engine="openpyxl",
        ) as writer:
            prediction_metrics.to_excel(
                writer,
                sheet_name="Prediction Ablation",
                index=False,
            )
            scheduling_summary.to_excel(
                writer,
                sheet_name="Scheduling Summary",
                index=False,
            )
            qaoa_runs.to_excel(
                writer,
                sheet_name="QAOA Runs",
                index=False,
            )
            qaoa_summary.to_excel(
                writer,
                sheet_name="QAOA Summary",
                index=False,
            )
            qaoa_tests.to_excel(
                writer,
                sheet_name="QAOA Tests",
                index=False,
            )
            hardware.to_excel(
                writer,
                sheet_name="Hardware",
                index=False,
            )
    except Exception as exc:
        (
            LOG_DIR / "excel_export_error.txt"
        ).write_text(str(exc), encoding="utf-8")


def write_report(
    prediction_metrics: pd.DataFrame,
    qaoa_summary: pd.DataFrame,
    hardware: pd.DataFrame,
) -> None:
    lines = [
        "# AQUA-SLA Quantum-Centric Ablation Report",
        "",
        "## Scope",
        "",
        (
            "The prediction-component analysis is computed from the locked "
            "classical, quantum, hybrid, and operational probabilities."
        ),
        (
            "The QAOA sensitivity analysis is a controlled 2-task × "
            "2-resource assignment-QUBO benchmark derived from saved task "
            "risk probabilities."
        ),
        (
            "Hardware execution is optional and is reported only when an "
            "authenticated backend successfully completes a job."
        ),
        "",
        "## Prediction-component ablation",
        "",
        prediction_metrics.to_markdown(index=False)
        if not prediction_metrics.empty
        else "No prediction ablation results.",
        "",
        "## QAOA sensitivity summary",
        "",
        qaoa_summary.to_markdown(index=False)
        if not qaoa_summary.empty
        else "No successful QAOA ablation runs.",
        "",
        "## Optional hardware status",
        "",
        hardware.to_markdown(index=False)
        if not hardware.empty
        else "No hardware status.",
        "",
        "## Claim discipline",
        "",
        (
            "Do not claim quantum advantage. Report simulation and hardware "
            "results separately, include backend/job metadata, and describe "
            "the QAOA experiment as a small-scale quantum scheduling "
            "proof-of-concept unless larger hardware evidence is obtained."
        ),
    ]

    (REPORT_DIR / "quantum_ablation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def save_environment_metadata(inputs: dict[str, Path]) -> None:
    versions: dict[str, str | None] = {}

    for package_name in [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "qiskit",
        "qiskit_machine_learning",
        "qiskit_ibm_runtime",
        "iqm.qiskit_iqm",
    ]:
        try:
            module = __import__(package_name)
            versions[package_name] = getattr(
                module,
                "__version__",
                "installed",
            )
        except Exception:
            versions[package_name] = None

    metadata = {
        "project": "AQUA-SLA",
        "study": "complete quantum-centric ablation suite",
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "inputs": {key: str(path) for key, path in inputs.items()},
        "qaoa_depths": QAOA_DEPTHS,
        "shot_counts": SHOT_COUNTS,
        "optimizers": OPTIMIZERS,
        "seeds": SEEDS,
        "max_qaoa_instances": MAX_QAOA_INSTANCES,
        "max_optimizer_iterations": MAX_OPT_ITER,
        "assignment_penalty": ASSIGNMENT_PENALTY,
        "objective_weights": OBJECTIVE_WEIGHTS,
        "ibm_hardware_requested": RUN_IBM_HARDWARE,
        "iqm_hardware_requested": RUN_IQM_HARDWARE,
    }

    with (
        REPORT_DIR / "complete_experiment_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_directories()
    inputs = resolve_inputs()
    save_environment_metadata(inputs)

    locked = pd.read_csv(inputs["locked_prediction_export"])
    assignments = pd.read_csv(inputs["scheduling_assignments"])
    batch_results = pd.read_csv(inputs["scheduling_batch_results"])
    scheduling_summary = pd.read_csv(inputs["scheduling_summary"])

    prediction_metrics, _ = run_prediction_ablation(locked)

    # Preserve all pre-existing scheduling evidence in the consolidated output.
    batch_results.to_csv(
        CSV_DIR / "locked_scheduling_batch_results.csv",
        index=False,
    )
    scheduling_summary.to_csv(
        CSV_DIR / "locked_scheduling_summary.csv",
        index=False,
    )
    assignments.to_csv(
        CSV_DIR / "locked_scheduling_assignments.csv",
        index=False,
    )

    for key in [
        "scheduler_improvements",
        "statistical_tests",
        "qaoa_exact_match_summary",
        "prediction_metrics",
    ]:
        if key in inputs:
            shutil.copy2(
                inputs[key],
                CSV_DIR / inputs[key].name,
            )

    qaoa_runs, convergence, qubos = run_qaoa_ablation(assignments)
    qaoa_summary = summarize_qaoa_ablation(qaoa_runs)
    qaoa_tests = qaoa_paired_tests(qaoa_runs)
    hardware = run_optional_hardware(qaoa_runs, qubos)

    plot_prediction_ablation(prediction_metrics)
    plot_calibration_ablation(prediction_metrics)
    plot_scheduler_summary(scheduling_summary)
    plot_qaoa_exact_match(qaoa_summary)
    plot_qaoa_objective_gap(qaoa_summary)
    plot_qaoa_runtime(qaoa_summary)
    plot_convergence(convergence)
    plot_hardware_status(hardware)

    export_workbook(
        prediction_metrics,
        scheduling_summary,
        qaoa_runs,
        qaoa_summary,
        qaoa_tests,
        hardware,
    )
    write_report(prediction_metrics, qaoa_summary, hardware)

    print("\nCompleted AQUA-SLA quantum-centric ablation suite.")
    print(f"Output directory:\n{OUTPUT_DIR}")
    print(f"CSV results:\n{CSV_DIR}")
    print(f"Saved models and QAOA parameters:\n{MODEL_DIR}")
    print(f"Circuits:\n{CIRCUIT_DIR}")
    print(f"JPEG figures:\n{FIGURE_DIR}")
    print(f"Hardware logs:\n{BACKEND_DIR}")
    print(f"Reports:\n{REPORT_DIR}")


if __name__ == "__main__":
    main()
