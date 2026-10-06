from __future__ import annotations

"""
AQUA-SLA V4 iQuantum calibrated risk-aware placement experiment.

Purpose
-------
This is a SEPARATE experiment. It does not overwrite the reviewer V2 run.
It does not select or discard runs according to which policy wins.

The experiment strengthens policy discrimination in a scientifically explicit way:
  * three heterogeneous QNodes (7, 15, 27 qubits),
  * independent QASM/circuit complexity,
  * identical exogenous background queue load for every policy,
  * identical foreground workload for every policy within a matched cell,
  * four policies: random, least_loaded, compatibility_aware, risk_aware,
  * 2 workload sizes x 5 seeds x 2 queue-stress regimes = 20 matched cells,
  * 80 iQuantum simulations in the default full experiment,
  * external SLO is calibrated on separate workloads and frozen before evaluation,
  * run-level uncertainty, paired bootstrap CIs, Wilcoxon tests, Holm correction,
    effect sizes, winner counts, and policy ranks are saved.

Important scientific boundary
-----------------------------
This script is designed to TEST whether a queue-aware, capability-reserving risk policy
improves high-risk outcomes under heterogeneous queue stress. It does not guarantee
that risk-aware placement will be best. All cells are retained, including cells where another policy performs better.

iQuantum is a discrete-event simulator. This experiment does not demonstrate physical
quantum-hardware speedup.

Recommended execution
---------------------
    conda activate aqua-sla
    cd /d E:\\other\\AQUA-SLA\\Code

Smoke test:
    python 32_aqua_sla_iquantum_risk_calibrated_v4.py --fast

Full experiment:
    python 32_aqua_sla_iquantum_risk_calibrated_v4.py
"""

import argparse
import itertools
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


# -----------------------------------------------------------------------------
# Paths / configuration
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(r"E:\other\AQUA-SLA")
DEFAULT_SOURCE_RUN = (
    PROJECT_DIR
    / "results"
    / "aqua_sla_review_v2"
    / "20260925_034909_reviewer_revision_v2"
)
OUTPUT_ROOT = PROJECT_DIR / "results" / "aqua_sla_iquantum_calibrated_v4"
IQUANTUM_HOME = PROJECT_DIR / "third_party" / "iQuantum"
EXAMPLES_POM = IQUANTUM_HOME / "modules" / "iquantum-examples" / "pom.xml"
JAVA_PACKAGE_DIR = (
    IQUANTUM_HOME
    / "modules"
    / "iquantum-examples"
    / "src"
    / "main"
    / "java"
    / "org"
    / "iquantum"
    / "examples"
    / "experimental"
)
JAVA_CLASS = "org.iquantum.examples.experimental.AquaSlaIQuantumBridgeV4"
JAVA_SOURCE = JAVA_PACKAGE_DIR / "AquaSlaIQuantumBridgeV4.java"

POLICIES = [
    "random",
    "least_loaded",
    "compatibility_aware",
    "risk_aware",
]
WORKLOAD_SIZES = [30, 60]
WORKLOAD_SEEDS = [101, 202, 303, 404, 505]
CALIBRATION_SEEDS = [9001, 9002, 9003]
QUEUE_SCENARIOS = ["balanced_background", "asymmetric_background"]
BOOTSTRAP_RESAMPLES = 5000
SLO_CALIBRATION_QUANTILE = 0.90
HIGH_RISK_THRESHOLD = 0.75
SEED = 42

# Policy-side node model. The Java bridge uses matching CLOPS values.
NODE_SPECS = {
    0: {"qubits": 7, "qv": 64, "clops": 1800.0},
    1: {"qubits": 15, "qv": 128, "clops": 2800.0},
    2: {"qubits": 27, "qv": 256, "clops": 4200.0},
}

# Workload complexity is intentionally independent of risk.
QUBIT_CYCLE = [4, 6, 8, 10, 12, 14, 20]
DEPTH_CYCLE = [16, 24, 32, 40, 48, 56, 64]
SHOT_CYCLE = [256, 512, 1024, 512, 256, 1024, 512]
GATE_SET = "CX|ID|RZ|SX|X"

# Foreground IDs are kept separate from background IDs so output can be filtered.
FOREGROUND_ID_BASE = 10000
BACKGROUND_ID_BASE = 0


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def save_json(obj: Any, path: Path) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(conv(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def locate_maven() -> str | None:
    candidates = [
        shutil.which("mvn.cmd"),
        shutil.which("mvn"),
        r"C:\Program Files\Apache\maven\bin\mvn.cmd",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def bootstrap_ci(
    values: Sequence[float],
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for i in range(resamples):
        draws[i] = rng.choice(x, size=len(x), replace=True).mean()
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def paired_cohens_dz(values: Sequence[float]) -> float:
    d = np.asarray(values, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    if np.isclose(sd, 0):
        return 0.0 if np.isclose(d.mean(), 0) else np.inf * np.sign(d.mean())
    return float(d.mean() / sd)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out


def normalized_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    return order.astype(float) / float(len(values) - 1)


# -----------------------------------------------------------------------------
# V3 Java bridge: three heterogeneous QNodes
# -----------------------------------------------------------------------------
def java_bridge_source() -> str:
    return r'''package org.iquantum.examples.experimental;

import org.iquantum.backends.quantum.QNode;
import org.iquantum.backends.quantum.qubittopologies.QubitTopology;
import org.iquantum.brokers.QCloudBroker;
import org.iquantum.core.iQuantum;
import org.iquantum.datacenters.QCloudDatacenter;
import org.iquantum.datacenters.QDatacenterCharacteristics;
import org.iquantum.gateways.CloudGateway;
import org.iquantum.policies.qtasks.QTaskSchedulerSpaceShared;
import org.iquantum.tasks.QTask;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;

public class AquaSlaIQuantumBridgeV4 {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            throw new IllegalArgumentException(
                "Usage: AquaSlaIQuantumBridgeV4 <workload.csv> <results.csv>"
            );
        }

        String inputPath = args[0];
        String outputPath = args[1];

        iQuantum.init(1, Calendar.getInstance(), false);

        List<QNode> qNodes = createQNodes();
        QDatacenterCharacteristics characteristics =
            new QDatacenterCharacteristics(qNodes, 0.0, 3.0);
        new QCloudDatacenter(
            "AQUA_SLA_iQuantum_Datacenter_V4",
            characteristics
        );

        QCloudBroker broker = new QCloudBroker("AQUA_SLA_QBroker_V4");
        CloudGateway gateway = new CloudGateway(
            "AQUA_SLA_CloudGateway_V4",
            broker
        );

        List<QTask> tasks = readTasks(inputPath, broker);
        gateway.submitQTasks(tasks);

        iQuantum.startSimulation();
        iQuantum.stopSimulation();

        writeResults(outputPath, broker.getQTaskReceivedList());
    }

    private static List<QNode> createQNodes() {
        List<QNode> nodes = new ArrayList<QNode>();

        ArrayList<String> gates = new ArrayList<String>(
            Arrays.asList("CX", "ID", "RZ", "SX", "X")
        );

        nodes.add(
            new QNode(
                0, 7, 64, 1800,
                gates, chainTopology(7),
                new QTaskSchedulerSpaceShared()
            )
        );

        nodes.add(
            new QNode(
                1, 15, 128, 2800,
                gates, chainTopology(15),
                new QTaskSchedulerSpaceShared()
            )
        );

        nodes.add(
            new QNode(
                2, 27, 256, 4200,
                gates, chainTopology(27),
                new QTaskSchedulerSpaceShared()
            )
        );

        return nodes;
    }

    private static QubitTopology chainTopology(int qubits) {
        List<int[]> edges = new ArrayList<int[]>();
        for (int i = 0; i < qubits - 1; i++) {
            edges.add(new int[]{i, i + 1});
            edges.add(new int[]{i + 1, i});
        }
        return new QubitTopology(qubits, edges);
    }

    private static List<QTask> readTasks(
        String inputPath,
        QCloudBroker broker
    ) throws Exception {
        List<QTask> tasks = new ArrayList<QTask>();

        try (BufferedReader reader = new BufferedReader(
            new FileReader(inputPath)
        )) {
            String header = reader.readLine();
            if (header == null) {
                return tasks;
            }

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] fields = line.split(",", -1);

                int qtaskId = Integer.parseInt(fields[0]);
                double risk = Double.parseDouble(fields[2]);
                int qubits = Integer.parseInt(fields[3]);
                int layers = Integer.parseInt(fields[4]);
                int shots = Integer.parseInt(fields[5]);
                int preferredNode = Integer.parseInt(fields[6]);
                String[] gateTokens = fields[7].split("\\|");
                String application = fields.length > 8 ? fields[8] : "AQUA-SLA-V4";

                ArrayList<String> gates = new ArrayList<String>(
                    Arrays.asList(gateTokens)
                );

                QTask task = new QTask(
                    qtaskId,
                    qubits,
                    layers,
                    shots,
                    gates,
                    chainTopology(qubits),
                    application,
                    "AQUA-SLA-risk-" + risk
                );

                task.setBrokerId(broker.getId());
                task.setQNodeId(preferredNode);
                tasks.add(task);
            }
        }

        return tasks;
    }

    private static void writeResults(
        String outputPath,
        List<QTask> tasks
    ) throws Exception {
        try (PrintWriter writer = new PrintWriter(
            new FileWriter(outputPath)
        )) {
            writer.println(
                "qtask_id,status,status_text,resource_id,qnode_id,"
                + "num_qubits,num_layers,num_shots,waiting_time,"
                + "actual_qpu_time,start_time,finish_time,cost"
            );

            for (QTask task : tasks) {
                writer.println(
                    task.getQTaskId() + ","
                    + task.getQTaskStatus() + ","
                    + csv(task.getQTaskStatusString()) + ","
                    + task.getResourceId() + ","
                    + task.getQNodeId() + ","
                    + task.getNumQubits() + ","
                    + task.getNumLayers() + ","
                    + task.getNumShots() + ","
                    + task.getWaitingTime() + ","
                    + task.getActualQPUTime() + ","
                    + task.getExecStartTime() + ","
                    + task.getFinishTime() + ","
                    + task.getCost()
                );
            }
        }
    }

    private static String csv(String value) {
        if (value == null) {
            return "";
        }
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }
}
'''


def install_java_bridge() -> None:
    JAVA_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    JAVA_SOURCE.write_text(java_bridge_source(), encoding="utf-8")


# -----------------------------------------------------------------------------
# Risk source and independent workloads
# -----------------------------------------------------------------------------
def load_risk_pool(source_run: Path) -> np.ndarray:
    candidates = [
        source_run / "02_prediction_hcqkl" / "matched_kernel_test_predictions.csv",
        source_run / "02_prediction_hcqkl" / "matched_kernel_test_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "risk_hcqkl" in df.columns:
                values = pd.to_numeric(df["risk_hcqkl"], errors="coerce").dropna().to_numpy(float)
                if len(values) >= 20:
                    return np.clip(values, 0.0, 1.0)
    raise FileNotFoundError(
        "Could not find matched_kernel_test_predictions.csv with risk_hcqkl in the source V2 run."
    )


def stratified_risk_sample(risk_pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Risk-stratified stress sample. Values stay on the original probability scale.

    This intentionally represents a stress regime rather than natural prevalence.
    It samples low, middle and high empirical risk strata, then shuffles them so
    risk is independent of circuit complexity/order.
    """
    rng = np.random.default_rng(seed)
    q25, q50, q75 = np.quantile(risk_pool, [0.25, 0.50, 0.75])
    low = risk_pool[risk_pool <= q25]
    mid = risk_pool[(risk_pool > q25) & (risk_pool < q75)]
    high = risk_pool[risk_pool >= q75]

    n_low = n // 3
    n_high = n // 3
    n_mid = n - n_low - n_high

    def draw(pool: np.ndarray, k: int) -> np.ndarray:
        if len(pool) == 0:
            pool = risk_pool
        return rng.choice(pool, size=k, replace=len(pool) < k)

    values = np.concatenate([
        draw(low, n_low),
        draw(mid, n_mid),
        draw(high, n_high),
    ])
    rng.shuffle(values)
    return values.astype(float)


def create_qasm_artifacts(
    n: int,
    seed: int,
    qasm_dir: Path,
    qtask_id_base: int,
    application: str,
) -> pd.DataFrame:
    from qiskit import QuantumCircuit, qasm2

    rng = np.random.default_rng(seed)
    qasm_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for tid in range(n):
        q = QUBIT_CYCLE[(tid + seed) % len(QUBIT_CYCLE)]
        target_depth = DEPTH_CYCLE[(tid + 2 * seed) % len(DEPTH_CYCLE)]
        shots = SHOT_CYCLE[(tid + 3 * seed) % len(SHOT_CYCLE)]

        qc = QuantumCircuit(q, q)
        for layer in range(target_depth):
            for qb in range(q):
                choice = (tid + layer + qb + seed) % 3
                if choice == 0:
                    qc.sx(qb)
                elif choice == 1:
                    qc.rz(float(rng.uniform(-np.pi, np.pi)), qb)
                else:
                    qc.x(qb)
            for qb in range(layer % 2, q - 1, 2):
                qc.cx(qb, qb + 1)
        qc.measure(range(q), range(q))

        qpath = qasm_dir / f"task_{qtask_id_base + tid:05d}.qasm"
        qasm2.dump(qc, str(qpath))

        rows.append({
            "qtask_id": qtask_id_base + tid,
            "source_task_id": tid,
            "risk_probability": 0.0,
            "num_qubits": q,
            "num_layers": int(qc.depth()),
            "num_shots": shots,
            "preferred_qnode_id": -1,
            "gate_set": GATE_SET,
            "application_name": application,
            "qasm_file": str(qpath),
        })

    return pd.DataFrame(rows)


def compatible_nodes(qubits: int) -> list[int]:
    return [node for node, spec in NODE_SPECS.items() if qubits <= spec["qubits"]]


def service_estimate(row: pd.Series, node: int) -> float:
    """Policy-side work estimate. It is not an iQuantum outcome."""
    work = float(row["num_layers"]) * float(row["num_shots"])
    return work / float(NODE_SPECS[node]["clops"])


def build_background(
    foreground_size: int,
    seed: int,
    scenario: str,
    qasm_dir: Path,
) -> pd.DataFrame:
    n_bg = max(6, foreground_size // 4)
    bg = create_qasm_artifacts(
        n=n_bg,
        seed=seed + 7000,
        qasm_dir=qasm_dir,
        qtask_id_base=BACKGROUND_ID_BASE,
        application="AQUA-SLA-V4-BACKGROUND",
    )

    if scenario == "balanced_background":
        nodes = [i % 3 for i in range(n_bg)]
    elif scenario == "asymmetric_background":
        # Deterministic seed-specific asymmetry, identical for all policies.
        heavy_node = seed % 3
        pattern = [heavy_node, heavy_node, heavy_node, (heavy_node + 1) % 3, (heavy_node + 2) % 3]
        nodes = [pattern[i % len(pattern)] for i in range(n_bg)]
    else:
        raise ValueError(f"Unknown queue scenario: {scenario}")

    # Ensure every background task is compatible with its fixed QNode.
    for i in range(n_bg):
        candidates = compatible_nodes(int(bg.loc[i, "num_qubits"]))
        if nodes[i] not in candidates:
            nodes[i] = max(candidates)

    bg["preferred_qnode_id"] = nodes
    bg["risk_probability"] = 0.0
    bg["risk_rank"] = 0.0
    bg["is_background"] = True
    return bg


def initial_background_load(background: pd.DataFrame) -> dict[int, float]:
    load = {node: 0.0 for node in NODE_SPECS}
    for _, row in background.iterrows():
        node = int(row["preferred_qnode_id"])
        load[node] += service_estimate(row, node)
    return load


def assign_policy(
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    policy: str,
    seed: int,
) -> pd.DataFrame:
    """Assign foreground QTasks to compatible QNodes.

    The V4 risk-aware policy is queue-aware and batch-aware. It does not change
    qubits, circuit depth, shots, or the external SLO. For low/mid-risk tasks it
    preserves scarce high-capability nodes when remaining high-risk tasks may
    need them. For high-risk tasks it prioritizes minimum projected completion
    time using the current estimated queue plus service time.

    This policy is intended to test whether risk information improves placement;
    it does not guarantee that risk-aware placement will outperform baselines.
    """
    rng = np.random.default_rng(seed)
    out = foreground.copy().reset_index(drop=True)
    load = initial_background_load(background)

    assigned: list[int] = []
    decision_rows: list[dict[str, Any]] = []

    def minmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        lo, hi = float(values.min()), float(values.max())
        if np.isclose(lo, hi):
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)

    for idx, row in out.iterrows():
        q = int(row["num_qubits"])
        candidates = compatible_nodes(q)
        if not candidates:
            assigned.append(-1)
            decision_rows.append({"selected_score": np.nan, "selected_projected_completion": np.nan})
            continue

        services = np.array([service_estimate(row, n) for n in candidates], dtype=float)
        loads = np.array([load[n] for n in candidates], dtype=float)
        projected = loads + services

        if policy == "random":
            node = int(rng.choice(candidates))
            selected_score = np.nan

        elif policy in {"least_loaded", "calibration_reference"}:
            # Neutral projected-completion reference. Used as a baseline and for
            # SLO calibration on separate calibration workloads.
            best = int(np.argmin(projected))
            node = int(candidates[best])
            selected_score = float(projected[best])

        elif policy == "compatibility_aware":
            # Smallest compatible QNode, independent of risk and queue state.
            node = min(candidates, key=lambda n: (NODE_SPECS[n]["qubits"], n))
            selected_score = float(NODE_SPECS[node]["qubits"])

        elif policy == "risk_aware":
            risk_rank = float(row["risk_rank"])

            # Look ahead only within the already-known scheduling batch. This is
            # not outcome leakage: it uses predicted risk and circuit compatibility,
            # not future simulator waiting/completion outcomes.
            remaining = out.iloc[idx + 1 :]
            future_high_need = []
            for node_id in candidates:
                count = 0
                for _, future in remaining.iterrows():
                    if float(future["risk_rank"]) < HIGH_RISK_THRESHOLD:
                        continue
                    if node_id in compatible_nodes(int(future["num_qubits"])):
                        count += 1
                future_high_need.append(float(count))

            projected_norm = minmax(projected)
            service_norm = minmax(services)
            capability = np.array(
                [NODE_SPECS[n]["clops"] for n in candidates], dtype=float
            )
            capability_norm = minmax(capability)
            reserve_norm = minmax(np.asarray(future_high_need, dtype=float))

            if risk_rank >= HIGH_RISK_THRESHOLD:
                # High-risk jobs explicitly minimize estimated time-to-completion.
                # A small service-time term breaks ties in favor of faster QNodes.
                scores = 0.90 * projected_norm + 0.10 * service_norm
            else:
                # For lower-risk jobs, avoid consuming a node that is both highly
                # capable and likely to be needed by remaining high-risk jobs unless
                # its projected completion advantage is substantial.
                urgency = risk_rank / max(HIGH_RISK_THRESHOLD, 1e-12)
                w_projected = 0.55 + 0.30 * urgency
                w_reserve = 0.35 * (1.0 - urgency)
                w_capability = 1.0 - w_projected - w_reserve
                scores = (
                    w_projected * projected_norm
                    + w_reserve * reserve_norm
                    + w_capability * capability_norm
                )

            best = int(np.argmin(scores))
            node = int(candidates[best])
            selected_score = float(scores[best])

        else:
            raise ValueError(f"Unknown policy: {policy}")

        assigned.append(node)
        load[node] += service_estimate(row, node)
        decision_rows.append({
            "selected_score": selected_score,
            "selected_projected_completion": float(
                load[node]
            ),
        })

    out["preferred_qnode_id"] = assigned
    out["policy"] = policy
    out["policy_selected_score"] = [x["selected_score"] for x in decision_rows]
    out["policy_projected_completion"] = [
        x["selected_projected_completion"] for x in decision_rows
    ]
    return out


# -----------------------------------------------------------------------------
# External SLO and manifest construction
# -----------------------------------------------------------------------------
def build_foreground(
    n: int,
    seed: int,
    risk_pool: np.ndarray,
    qasm_dir: Path,
    application: str = "AQUA-SLA-V4-FOREGROUND",
) -> pd.DataFrame:
    fg = create_qasm_artifacts(
        n=n,
        seed=seed,
        qasm_dir=qasm_dir,
        qtask_id_base=FOREGROUND_ID_BASE,
        application=application,
    )

    risks = stratified_risk_sample(risk_pool, n, seed + 8000)
    fg["risk_probability"] = risks
    fg["risk_rank"] = normalized_rank(risks)
    fg["is_background"] = False

    # Complexity-only reference service time. This is known before placement
    # and is independent of risk and of the policy that will be evaluated.
    fastest_reference = []
    for _, row in fg.iterrows():
        candidates = compatible_nodes(int(row["num_qubits"]))
        fastest_reference.append(
            min(service_estimate(row, node) for node in candidates)
        )
    fg["fastest_reference_service"] = np.asarray(fastest_reference, dtype=float)
    return fg


def apply_calibrated_slo(
    foreground: pd.DataFrame,
    multiplier: float,
) -> pd.DataFrame:
    out = foreground.copy()
    out["external_slo_budget"] = (
        float(multiplier)
        * pd.to_numeric(out["fastest_reference_service"], errors="coerce")
    )
    return out


def extract_calibration_ratios(
    foreground_manifest: pd.DataFrame,
    result_csv: Path,
) -> pd.DataFrame:
    if not result_csv.exists():
        return pd.DataFrame()
    results = pd.read_csv(result_csv)
    ids = set(foreground_manifest["qtask_id"].astype(int).tolist())
    fg = results[results["qtask_id"].astype(int).isin(ids)].copy()
    if fg.empty:
        return pd.DataFrame()
    fg = fg.merge(
        foreground_manifest[["qtask_id", "fastest_reference_service"]],
        on="qtask_id",
        how="left",
    )
    for c in ["waiting_time", "actual_qpu_time", "fastest_reference_service"]:
        fg[c] = pd.to_numeric(fg[c], errors="coerce")
    fg["completion_latency"] = fg["waiting_time"] + fg["actual_qpu_time"]
    fg["normalized_completion_ratio"] = (
        fg["completion_latency"]
        / fg["fastest_reference_service"].clip(lower=1e-12)
    )
    return fg


def calibrate_external_slo(
    out_dir: Path,
    risk_pool: np.ndarray,
    sizes: list[int],
    scenarios: list[str],
    quantile: float,
    fast: bool,
) -> tuple[dict[tuple[int, str], float], pd.DataFrame]:
    """Calibrate external SLO on separate workloads before evaluation.

    A neutral projected-completion reference policy is executed on calibration
    seeds that never appear in the evaluation set. For each workload-size and
    queue-scenario stratum, the SLO multiplier is the predeclared quantile of
    completion_latency / fastest_compatible_service. Evaluation then freezes
    this multiplier and applies it to every policy in that stratum.
    """
    calibration_root = out_dir / "00_slo_calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    calibration_seeds = [CALIBRATION_SEEDS[0]] if fast else CALIBRATION_SEEDS

    ratio_rows: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []

    for workload_size in sizes:
        for scenario in scenarios:
            for seed in calibration_seeds:
                cell = calibration_root / f"n{workload_size}_seed{seed}_{scenario}"
                qasm_fg = cell / "qasm_foreground"
                qasm_bg = cell / "qasm_background"
                cell.mkdir(parents=True, exist_ok=True)

                foreground = build_foreground(
                    n=workload_size,
                    seed=seed,
                    risk_pool=risk_pool,
                    qasm_dir=qasm_fg,
                    application="AQUA-SLA-V4-CALIBRATION-FOREGROUND",
                )
                # Calibration does not use risk to define the SLO.
                foreground["risk_probability"] = 0.0
                foreground["risk_rank"] = 0.0
                background = build_background(
                    foreground_size=workload_size,
                    seed=seed,
                    scenario=scenario,
                    qasm_dir=qasm_bg,
                )
                assigned = assign_policy(
                    foreground=foreground,
                    background=background,
                    policy="calibration_reference",
                    seed=seed + 12345,
                )
                combined = pd.concat([background, assigned], ignore_index=True, sort=False)
                input_csv = cell / "iquantum_workload.csv"
                bridge_columns(combined).to_csv(input_csv, index=False)
                result_csv = cell / "iquantum_task_results.csv"
                status = run_bridge(
                    input_csv=input_csv,
                    result_csv=result_csv,
                    log_path=cell / "maven_execution.log",
                )
                save_json(status, cell / "execution_status.json")
                run_rows.append({
                    "workload_size": workload_size,
                    "queue_scenario": scenario,
                    "calibration_seed": seed,
                    "success": bool(status.get("success", False)),
                    "elapsed_seconds": status.get("elapsed_seconds"),
                })
                if status.get("success"):
                    ratios = extract_calibration_ratios(assigned, result_csv)
                    if not ratios.empty:
                        ratios["workload_size"] = workload_size
                        ratios["queue_scenario"] = scenario
                        ratios["calibration_seed"] = seed
                        ratio_rows.append(ratios)

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(calibration_root / "calibration_run_status.csv", index=False)
    if not ratio_rows:
        raise RuntimeError("No successful SLO calibration results were produced.")

    ratios_df = pd.concat(ratio_rows, ignore_index=True)
    ratios_df.to_csv(calibration_root / "calibration_task_ratios.csv", index=False)

    multipliers: dict[tuple[int, str], float] = {}
    summary_rows: list[dict[str, Any]] = []
    for (size, scenario), group in ratios_df.groupby(["workload_size", "queue_scenario"]):
        values = pd.to_numeric(
            group["normalized_completion_ratio"], errors="coerce"
        ).dropna().to_numpy(float)
        if len(values) == 0:
            raise RuntimeError(
                f"No calibration ratios for workload={size}, scenario={scenario}"
            )
        multiplier = max(1.0, float(np.quantile(values, quantile)))
        multipliers[(int(size), str(scenario))] = multiplier
        summary_rows.append({
            "workload_size": int(size),
            "queue_scenario": str(scenario),
            "calibration_quantile": float(quantile),
            "n_calibration_tasks": int(len(values)),
            "slo_multiplier": multiplier,
            "ratio_mean": float(np.mean(values)),
            "ratio_median": float(np.median(values)),
            "ratio_p90": float(np.quantile(values, 0.90)),
            "ratio_p95": float(np.quantile(values, 0.95)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(calibration_root / "slo_calibration_summary.csv", index=False)
    save_json({
        "calibration_policy": "neutral projected-completion reference; no SLA risk used",
        "calibration_seeds": calibration_seeds,
        "evaluation_seeds": WORKLOAD_SEEDS,
        "quantile": float(quantile),
        "definition": (
            "For each workload-size and queue-scenario stratum, budget = frozen "
            "calibration multiplier × fastest-compatible service estimate. The "
            "multiplier is estimated only from separate calibration workloads."
        ),
        "multipliers": {
            f"n{size}|{scenario}": value
            for (size, scenario), value in multipliers.items()
        },
    }, calibration_root / "slo_calibration.json")
    return multipliers, summary_df


def complexity_risk_independence(fg: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["num_qubits", "num_layers", "num_shots"]:
        r, p = spearmanr(
            pd.to_numeric(fg["risk_probability"], errors="coerce"),
            pd.to_numeric(fg[col], errors="coerce"),
        )
        out[f"spearman_risk_vs_{col}"] = float(r)
        out[f"spearman_p_risk_vs_{col}"] = float(p)
    return out


def bridge_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "qtask_id",
        "source_task_id",
        "risk_probability",
        "num_qubits",
        "num_layers",
        "num_shots",
        "preferred_qnode_id",
        "gate_set",
        "application_name",
    ]
    return df.loc[:, cols].copy()


# -----------------------------------------------------------------------------
# iQuantum execution
# -----------------------------------------------------------------------------
def run_bridge(input_csv: Path, result_csv: Path, log_path: Path) -> dict[str, Any]:
    mvn = locate_maven()
    status = {
        "input": str(input_csv),
        "output": str(result_csv),
        "executed": False,
        "success": False,
        "maven": mvn,
        "pom": str(EXAMPLES_POM),
        "java_class": JAVA_CLASS,
    }

    if not mvn:
        status["reason"] = "Maven not found"
        return status
    if not EXAMPLES_POM.exists():
        status["reason"] = f"iQuantum examples pom not found: {EXAMPLES_POM}"
        return status

    cmd = [
        mvn,
        "-q",
        "-f",
        str(EXAMPLES_POM),
        "compile",
        "exec:java",
        f"-Dexec.mainClass={JAVA_CLASS}",
        f'-Dexec.args="{input_csv}" "{result_csv}"',
    ]

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(IQUANTUM_HOME),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    elapsed = time.perf_counter() - start
    log_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")

    status.update({
        "executed": True,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "command": cmd,
        "result_exists": result_csv.exists(),
        "success": proc.returncode == 0 and result_csv.exists(),
    })
    return status


# -----------------------------------------------------------------------------
# Result summarization
# -----------------------------------------------------------------------------
def summarize_run(
    policy: str,
    workload_size: int,
    workload_seed: int,
    scenario: str,
    foreground_manifest: pd.DataFrame,
    result_csv: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "policy": policy,
        "workload_size": workload_size,
        "workload_seed": workload_seed,
        "queue_scenario": scenario,
        "executed": bool(status.get("executed", False)),
        "success": bool(status.get("success", False)),
        "bridge_runtime_seconds": status.get("elapsed_seconds"),
    }

    if not row["success"] or not result_csv.exists():
        row["reason"] = status.get("reason") or status.get("returncode")
        return row

    results = pd.read_csv(result_csv)
    fg_ids = set(foreground_manifest["qtask_id"].astype(int).tolist())
    fg = results[results["qtask_id"].astype(int).isin(fg_ids)].copy()
    fg = fg.merge(
        foreground_manifest[
            [
                "qtask_id",
                "risk_probability",
                "risk_rank",
                "external_slo_budget",
                "preferred_qnode_id",
            ]
        ],
        on="qtask_id",
        how="left",
        suffixes=("", "_planned"),
    )

    if fg.empty:
        row["success"] = False
        row["reason"] = "No foreground tasks returned"
        return row

    for c in [
        "waiting_time",
        "actual_qpu_time",
        "start_time",
        "finish_time",
        "cost",
        "risk_probability",
        "risk_rank",
        "external_slo_budget",
    ]:
        fg[c] = pd.to_numeric(fg[c], errors="coerce")

    fg["completion_latency"] = fg["waiting_time"] + fg["actual_qpu_time"]
    fg["external_slo_miss"] = (
        fg["completion_latency"] > fg["external_slo_budget"]
    ).astype(int)

    high = fg[fg["risk_rank"] >= 0.75].copy()
    if high.empty:
        high = fg.nlargest(max(1, len(fg) // 4), "risk_rank").copy()

    row.update({
        "tasks_returned": int(len(fg)),
        "mean_waiting_time": float(fg["waiting_time"].mean()),
        "median_waiting_time": float(fg["waiting_time"].median()),
        "p95_waiting_time": float(fg["waiting_time"].quantile(0.95)),
        "mean_qpu_time": float(fg["actual_qpu_time"].mean()),
        "mean_completion_latency": float(fg["completion_latency"].mean()),
        "p95_completion_latency": float(fg["completion_latency"].quantile(0.95)),
        "makespan": float(fg["finish_time"].max()),
        "total_cost": float(fg["cost"].sum()),
        "external_slo_miss_rate": float(fg["external_slo_miss"].mean()),
        "high_risk_n": int(len(high)),
        "high_risk_mean_waiting": float(high["waiting_time"].mean()),
        "high_risk_p95_waiting": float(high["waiting_time"].quantile(0.95)),
        "high_risk_mean_completion": float(high["completion_latency"].mean()),
        "high_risk_p95_completion": float(high["completion_latency"].quantile(0.95)),
        "high_risk_external_slo_miss_rate": float(high["external_slo_miss"].mean()),
        "qnode0_count": int((fg["qnode_id"] == 0).sum()),
        "qnode1_count": int((fg["qnode_id"] == 1).sum()),
        "qnode2_count": int((fg["qnode_id"] == 2).sum()),
        "preferred_node_match_rate": float(
            (fg["qnode_id"].astype(int) == fg["preferred_qnode_id"].astype(int)).mean()
        ),
    })

    # Risk-weighted latency is descriptive, not the external SLO definition.
    weights = 0.25 + 0.75 * fg["risk_rank"].to_numpy(float)
    row["risk_weighted_completion"] = float(
        np.average(fg["completion_latency"].to_numpy(float), weights=weights)
    )

    return row


# -----------------------------------------------------------------------------
# Statistics / ranks
# -----------------------------------------------------------------------------
LOWER_IS_BETTER_METRICS = [
    "mean_waiting_time",
    "p95_waiting_time",
    "mean_completion_latency",
    "p95_completion_latency",
    "makespan",
    "total_cost",
    "external_slo_miss_rate",
    "high_risk_mean_waiting",
    "high_risk_p95_waiting",
    "high_risk_mean_completion",
    "high_risk_p95_completion",
    "high_risk_external_slo_miss_rate",
    "risk_weighted_completion",
]


def policy_uncertainty(summary: pd.DataFrame, out_dir: Path, bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    success = summary[summary["success"] == True].copy()

    for policy, g in success.groupby("policy"):
        for metric in LOWER_IS_BETTER_METRICS:
            if metric not in g.columns:
                continue
            values = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(float)
            if len(values) == 0:
                continue
            lo, hi = bootstrap_ci(values, bootstrap, SEED + len(rows))
            rows.append({
                "policy": policy,
                "metric": metric,
                "n_runs": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "median": float(np.median(values)),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
            })

    ans = pd.DataFrame(rows)
    ans.to_csv(out_dir / "policy_uncertainty.csv", index=False)
    return ans


def risk_aware_pairwise(summary: pd.DataFrame, out_dir: Path, bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    success = summary[summary["success"] == True].copy()
    keys = ["workload_size", "workload_seed", "queue_scenario"]

    risk = success[success["policy"] == "risk_aware"]
    for baseline in ["random", "least_loaded", "compatibility_aware"]:
        base = success[success["policy"] == baseline]
        for metric in LOWER_IS_BETTER_METRICS:
            if metric not in success.columns:
                continue
            a = risk[keys + [metric]].rename(columns={metric: "risk_aware"})
            b = base[keys + [metric]].rename(columns={metric: "baseline"})
            paired = a.merge(b, on=keys, how="inner")
            paired["risk_aware"] = pd.to_numeric(paired["risk_aware"], errors="coerce")
            paired["baseline"] = pd.to_numeric(paired["baseline"], errors="coerce")
            paired = paired.dropna(subset=["risk_aware", "baseline"])
            if paired.empty:
                continue

            d = paired["risk_aware"].to_numpy(float) - paired["baseline"].to_numpy(float)
            lo, hi = bootstrap_ci(d, bootstrap, SEED + len(rows))
            nz = d[~np.isclose(d, 0)]
            stat = np.nan
            p_raw = np.nan
            if len(nz) >= 6:
                try:
                    w = wilcoxon(nz, alternative="two-sided", zero_method="wilcox", method="auto")
                    stat = float(w.statistic)
                    p_raw = float(w.pvalue)
                except Exception:
                    pass

            rows.append({
                "baseline": baseline,
                "metric": metric,
                "pairs": len(paired),
                "risk_aware_mean": float(paired["risk_aware"].mean()),
                "baseline_mean": float(paired["baseline"].mean()),
                "mean_diff_risk_minus_baseline": float(d.mean()),
                "median_diff_risk_minus_baseline": float(np.median(d)),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
                "paired_cohens_dz": paired_cohens_dz(d),
                "wilcoxon_stat": stat,
                "p_raw": p_raw,
                "risk_aware_better_cells": int((d < 0).sum()),
                "ties": int(np.isclose(d, 0).sum()),
                "baseline_better_cells": int((d > 0).sum()),
            })

    ans = pd.DataFrame(rows)
    if not ans.empty:
        ans["p_holm"] = holm_adjust(ans["p_raw"].to_numpy(float))
        ans["significant_holm_0_05"] = ans["p_holm"] < 0.05
    ans.to_csv(out_dir / "risk_aware_vs_baselines.csv", index=False)
    return ans


def winner_and_rank_tables(summary: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = summary[summary["success"] == True].copy()
    keys = ["workload_size", "workload_seed", "queue_scenario"]
    winner_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []

    for key_values, cell in success.groupby(keys):
        key_dict = dict(zip(keys, key_values if isinstance(key_values, tuple) else [key_values]))
        for metric in LOWER_IS_BETTER_METRICS:
            if metric not in cell.columns:
                continue
            c = cell[["policy", metric]].copy()
            c[metric] = pd.to_numeric(c[metric], errors="coerce")
            c = c.dropna()
            if c.empty:
                continue
            min_value = c[metric].min()
            winners = c[np.isclose(c[metric], min_value)]["policy"].tolist()
            winner_rows.append({
                **key_dict,
                "metric": metric,
                "best_value": float(min_value),
                "winners": "|".join(sorted(winners)),
                "risk_aware_is_winner": "risk_aware" in winners,
            })

            c["rank"] = c[metric].rank(method="average", ascending=True)
            for _, rr in c.iterrows():
                rank_rows.append({
                    **key_dict,
                    "metric": metric,
                    "policy": rr["policy"],
                    "value": float(rr[metric]),
                    "rank": float(rr["rank"]),
                })

    winners = pd.DataFrame(winner_rows)
    ranks = pd.DataFrame(rank_rows)
    winners.to_csv(out_dir / "cell_metric_winners.csv", index=False)
    ranks.to_csv(out_dir / "cell_policy_ranks.csv", index=False)

    if not ranks.empty:
        rank_summary = (
            ranks.groupby(["policy", "metric"], as_index=False)
            .agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), cells=("rank", "size"))
        )
        rank_summary.to_csv(out_dir / "policy_rank_summary.csv", index=False)

    return winners, ranks


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def make_figures(summary: pd.DataFrame, uncertainty: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric in [
        "mean_waiting_time",
        "high_risk_mean_waiting",
        "external_slo_miss_rate",
        "makespan",
        "total_cost",
    ]:
        z = uncertainty[uncertainty["metric"] == metric].copy()
        if z.empty:
            continue
        z = z.sort_values("policy")
        x = np.arange(len(z))
        low = z["mean"].to_numpy(float) - z["bootstrap95_low"].to_numpy(float)
        high = z["bootstrap95_high"].to_numpy(float) - z["mean"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(x, z["mean"], yerr=np.vstack([low, high]), fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(z["policy"], rotation=20)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"iQuantum V4 policy comparison: {metric.replace('_', ' ')}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{metric}_bootstrap_ci.png", dpi=300)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def run_experiment(args: argparse.Namespace) -> Path:
    source_run = Path(args.source_run).resolve()
    if not source_run.exists():
        raise FileNotFoundError(f"Source V2 run not found: {source_run}")

    risk_pool = load_risk_pool(source_run)
    install_java_bridge()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{stamp}_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=False)

    sizes = [12] if args.fast else WORKLOAD_SIZES
    seeds = [101] if args.fast else WORKLOAD_SEEDS
    scenarios = ["asymmetric_background"] if args.fast else QUEUE_SCENARIOS
    bootstrap = min(1000, args.bootstrap) if args.fast else args.bootstrap

    # Freeze an external SLO from separate calibration workloads before any
    # evaluation-policy simulation is executed.
    slo_multipliers, slo_calibration_summary = calibrate_external_slo(
        out_dir=out_dir,
        risk_pool=risk_pool,
        sizes=sizes,
        scenarios=scenarios,
        quantile=args.slo_quantile,
        fast=args.fast,
    )

    save_json({
        "project": "AQUA-SLA",
        "experiment": "iquantum-risk-aware-calibrated-v4",
        "created": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "source_v2_run": str(source_run),
        "output": str(out_dir),
        "policies": POLICIES,
        "workload_sizes": sizes,
        "workload_seeds": seeds,
        "queue_scenarios": scenarios,
        "expected_simulations": len(POLICIES) * len(sizes) * len(seeds) * len(scenarios),
        "qnodes": NODE_SPECS,
        "risk_regime": (
            "Risk-stratified stress sample from existing HCQKL probabilities. "
            "Low/middle/high empirical strata are sampled and shuffled independently "
            "of circuit complexity. This is a stress test, not natural prevalence."
        ),
        "policy_fairness": (
            "Within each workload-size/seed/queue-scenario cell all policies receive "
            "identical foreground circuits, risk values, background tasks and background "
            "QNode assignments. Only the foreground placement rule differs."
        ),
        "external_slo": (
            "Frozen before evaluation from separate calibration workloads. For each "
            "workload-size/queue-scenario stratum, the budget equals a predeclared "
            "calibration quantile multiplier times the fastest-compatible service "
            "estimate. Calibration uses no SLA risk and no evaluation-policy outcome."
        ),
        "slo_calibration_quantile": float(args.slo_quantile),
        "slo_calibration_summary_rows": len(slo_calibration_summary),
        "claim_boundary": (
            "The experiment tests a stronger risk-aware policy under queue stress. It does "
            "not guarantee that risk-aware placement wins, and all runs are retained."
        ),
    }, out_dir / "protocol.json")

    summary_rows: list[dict[str, Any]] = []
    independence_rows: list[dict[str, Any]] = []

    for workload_size in sizes:
        for workload_seed in seeds:
            for scenario in scenarios:
                cell_dir = out_dir / f"n{workload_size}_seed{workload_seed}_{scenario}"
                qasm_fg = cell_dir / "qasm_foreground"
                qasm_bg = cell_dir / "qasm_background"
                cell_dir.mkdir(parents=True, exist_ok=True)

                foreground = build_foreground(
                    n=workload_size,
                    seed=workload_seed,
                    risk_pool=risk_pool,
                    qasm_dir=qasm_fg,
                    application="AQUA-SLA-V4-FOREGROUND",
                )
                slo_key = (int(workload_size), str(scenario))
                if slo_key not in slo_multipliers:
                    raise RuntimeError(f"Missing frozen SLO multiplier for {slo_key}")
                foreground = apply_calibrated_slo(
                    foreground,
                    slo_multipliers[slo_key],
                )
                background = build_background(
                    foreground_size=workload_size,
                    seed=workload_seed,
                    scenario=scenario,
                    qasm_dir=qasm_bg,
                )

                independence = complexity_risk_independence(foreground)
                independence_rows.append({
                    "workload_size": workload_size,
                    "workload_seed": workload_seed,
                    "queue_scenario": scenario,
                    **independence,
                })

                foreground.to_csv(cell_dir / "foreground_base_manifest.csv", index=False)
                background.to_csv(cell_dir / "background_manifest.csv", index=False)

                for pidx, policy in enumerate(POLICIES):
                    policy_dir = cell_dir / policy
                    policy_dir.mkdir(parents=True, exist_ok=True)

                    assigned_fg = assign_policy(
                        foreground=foreground,
                        background=background,
                        policy=policy,
                        seed=workload_seed + 1000 * pidx,
                    )
                    assigned_fg.to_csv(policy_dir / "foreground_policy_manifest.csv", index=False)

                    # Same background tasks first for every policy, then same foreground order.
                    combined = pd.concat([background, assigned_fg], ignore_index=True, sort=False)
                    input_csv = policy_dir / "iquantum_workload.csv"
                    bridge_columns(combined).to_csv(input_csv, index=False)

                    result_csv = policy_dir / "iquantum_task_results.csv"
                    status = run_bridge(
                        input_csv=input_csv,
                        result_csv=result_csv,
                        log_path=policy_dir / "maven_execution.log",
                    )
                    save_json(status, policy_dir / "execution_status.json")

                    summary = summarize_run(
                        policy=policy,
                        workload_size=workload_size,
                        workload_seed=workload_seed,
                        scenario=scenario,
                        foreground_manifest=assigned_fg,
                        result_csv=result_csv,
                        status=status,
                    )
                    summary_rows.append(summary)

                    # Checkpoint after each simulator invocation.
                    pd.DataFrame(summary_rows).to_csv(out_dir / "iquantum_v4_run_summary.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "iquantum_v4_run_summary.csv", index=False)
    pd.DataFrame(independence_rows).to_csv(out_dir / "risk_complexity_independence.csv", index=False)

    uncertainty = policy_uncertainty(summary_df, out_dir, bootstrap)
    pairwise = risk_aware_pairwise(summary_df, out_dir, bootstrap)
    winners, ranks = winner_and_rank_tables(summary_df, out_dir)
    make_figures(summary_df, uncertainty, out_dir)

    successful = int(summary_df["success"].fillna(False).astype(bool).sum()) if not summary_df.empty else 0
    expected = len(POLICIES) * len(sizes) * len(seeds) * len(scenarios)

    # Compact policy-level primary summary.
    primary_metrics = [
        "high_risk_mean_waiting",
        "high_risk_p95_completion",
        "external_slo_miss_rate",
        "risk_weighted_completion",
        "mean_waiting_time",
        "makespan",
        "total_cost",
    ]
    primary = uncertainty[uncertainty["metric"].isin(primary_metrics)].copy()
    primary.to_csv(out_dir / "primary_policy_summary.csv", index=False)

    status = {
        "completed": True,
        "output": str(out_dir),
        "expected_simulations": expected,
        "successful_simulations": successful,
        "failed_simulations": expected - successful,
        "summary_rows": len(summary_df),
        "uncertainty_rows": len(uncertainty),
        "paired_rows": len(pairwise),
        "winner_rows": len(winners),
        "notes": [
            "This is a separate stress-test experiment and does not overwrite reviewer V2 results.",
            "All matched cells are retained regardless of which policy wins.",
            "Circuit complexity is generated independently of SLA risk.",
            "Background queue state is identical across policies within a matched cell.",
            "The external SLO is calibrated on separate workloads and frozen before evaluation.",
            "iQuantum is discrete-event simulation, not physical quantum hardware.",
        ],
    }
    save_json(status, out_dir / "final_status.json")

    print("Finished AQUA-SLA iQuantum V4 calibrated policy experiment.")
    print("Output:", out_dir)
    print(json.dumps(status, indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--run-name", default="risk_aware_policy_calibrated")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--slo-quantile",
        type=float,
        default=SLO_CALIBRATION_QUANTILE,
        help="Predeclared calibration quantile for the external SLO multiplier.",
    )
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    if not (0.50 <= args.slo_quantile < 1.0):
        raise ValueError("--slo-quantile must be in [0.50, 1.0).")
    run_experiment(args)


if __name__ == "__main__":
    main()
