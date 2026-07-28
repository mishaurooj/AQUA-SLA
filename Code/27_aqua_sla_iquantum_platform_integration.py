from __future__ import annotations

r'''
AQUA-SLA + CLOUDS Lab iQuantum Integration
==========================================

This script integrates the CLOUDS Lab iQuantum toolkit
(https://github.com/Cloudslab/iQuantum) with AQUA-SLA.

It is distinct from IQM:
- iQuantum = Java/CloudSim discrete-event quantum-cloud simulator.
- IQM = quantum hardware/provider integration.

Workflow:
1. Read AQUA-SLA scheduling assignments.
2. Export an iQuantum workload CSV.
3. Generate a Java bridge in a local iQuantum clone.
4. Build and execute iQuantum with Maven.
5. Import task-level timing, backend, status, and cost results.
6. Save an iQuantum platform ablation and execution report.
7. Fail safely when Java, Maven, or iQuantum is unavailable.

Requirements:
- Java JDK 17+
- Maven 3.9+
- Local clone of https://github.com/Cloudslab/iQuantum

Environment:
    IQUANTUM_HOME=D:\other\AQUA-SLA\third_party\iQuantum
    AQUA_RUN_IQUANTUM=1
'''

import json
import os
import subprocess
import time
import traceback
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"D:\other\AQUA-SLA")
RESULT_ROOT = PROJECT_DIR / "results" / "aqua_sla_final"
OUTPUT_DIR = RESULT_ROOT / "iquantum_platform_integration"
CSV_DIR = OUTPUT_DIR / "csv"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

IQUANTUM_HOME = Path(
    os.getenv(
        "IQUANTUM_HOME",
        str(PROJECT_DIR / "third_party" / "iQuantum"),
    )
)

RUN_IQUANTUM = os.getenv("AQUA_RUN_IQUANTUM", "1") == "1"


def resolve_java_command() -> str | None:
    """Return the Java executable path visible to Python."""
    candidates = [shutil.which("java.exe"), shutil.which("java")]
    java_home = os.getenv("JAVA_HOME")
    if java_home:
        candidates.extend([
            str(Path(java_home) / "bin" / "java.exe"),
            str(Path(java_home) / "bin" / "java"),
        ])
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def resolve_maven_command() -> str | None:
    """Return the Maven launcher path visible to Python."""
    candidates = [
        shutil.which("mvn.cmd"),
        shutil.which("mvn.exe"),
        shutil.which("mvn"),
    ]
    maven_home = os.getenv("MAVEN_HOME")
    if maven_home:
        candidates.extend([
            str(Path(maven_home) / "bin" / "mvn.cmd"),
            str(Path(maven_home) / "bin" / "mvn.exe"),
            str(Path(maven_home) / "bin" / "mvn"),
        ])
    candidates.extend([
        r"C:\Program Files\Apache\maven\bin\mvn.cmd",
        r"C:\Program Files\Apache Maven\bin\mvn.cmd",
    ])
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


JAVA_COMMAND = resolve_java_command()
MAVEN_COMMAND = resolve_maven_command()

JAVA_PACKAGE = "org.iquantum.examples.experimental"
JAVA_CLASS = "AquaSlaIQuantumBridge"

JAVA_RELATIVE_PATH = (
    Path("modules")
    / "iquantum-examples"
    / "src"
    / "main"
    / "java"
    / "org"
    / "iquantum"
    / "examples"
    / "experimental"
    / f"{JAVA_CLASS}.java"
)

WORKLOAD_PATH = CSV_DIR / "aqua_sla_iquantum_workload.csv"
IQUANTUM_RESULT_PATH = CSV_DIR / "iquantum_task_results.csv"


def ensure_dirs() -> None:
    for directory in [OUTPUT_DIR, CSV_DIR, REPORT_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def find_result_file(filename: str) -> Path:
    preferred = [
        RESULT_ROOT / "quantum_ablation_complete" / "csv" / filename,
        RESULT_ROOT / filename,
    ]

    for path in preferred:
        if path.exists():
            return path

    matches = list((PROJECT_DIR / "results").rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not locate {filename}")


def command_version(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        text = (completed.stdout + "\n" + completed.stderr).strip()
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "output": text,
        }
    except Exception as exc:
        return {
            "available": False,
            "returncode": None,
            "output": f"{type(exc).__name__}: {exc}",
        }


def normalize_resource_id(value: Any) -> int:
    text = str(value).strip().lower()

    if text in {"0", "resource_0", "qnode_0", "node_0"}:
        return 0
    if text in {"1", "resource_1", "qnode_1", "node_1"}:
        return 1

    try:
        return int(float(text)) % 2
    except Exception:
        return 0


def build_workload(assignments: pd.DataFrame) -> pd.DataFrame:
    frame = assignments.copy()

    if "scheduler" in frame.columns:
        qaoa = frame[
            frame["scheduler"].astype(str).str.contains(
                "qaoa",
                case=False,
                na=False,
            )
        ].copy()
        if not qaoa.empty:
            frame = qaoa

    sort_columns = [
        column
        for column in ["seed", "batch", "task_id"]
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns)

    frame = frame.head(250).reset_index(drop=True)

    risk = (
        frame["risk_probability"].astype(float)
        if "risk_probability" in frame.columns
        else pd.Series(np.full(len(frame), 0.5))
    )

    task_id = (
        frame["task_id"].astype(int)
        if "task_id" in frame.columns
        else pd.Series(np.arange(len(frame)))
    )

    assigned = None
    for candidate in [
        "assigned_resource",
        "resource_id",
        "server_id",
        "node_id",
        "assignment",
    ]:
        if candidate in frame.columns:
            assigned = frame[candidate]
            break

    if assigned is None:
        assigned = pd.Series(np.arange(len(frame)) % 2)

    workload = pd.DataFrame(
        {
            "qtask_id": np.arange(len(frame), dtype=int),
            "source_task_id": task_id.to_numpy(),
            "risk_probability": risk.to_numpy(),
            "num_qubits": np.where(risk >= 0.75, 6, 4).astype(int),
            "num_layers": np.maximum(
                10,
                np.round(20 + 80 * risk).astype(int),
            ),
            "num_shots": np.where(risk >= 0.75, 1024, 512).astype(int),
            "preferred_qnode_id": [
                normalize_resource_id(value) for value in assigned
            ],
            "gate_set": "CX|RZ|SX|X",
            "application_name": "AQUA-SLA",
        }
    )

    workload.to_csv(WORKLOAD_PATH, index=False)
    return workload


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

public class AquaSlaIQuantumBridge {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            throw new IllegalArgumentException(
                "Usage: AquaSlaIQuantumBridge <workload.csv> <results.csv>"
            );
        }

        String inputPath = args[0];
        String outputPath = args[1];

        iQuantum.init(1, Calendar.getInstance(), false);

        List<QNode> qNodes = createQNodes();
        QDatacenterCharacteristics characteristics =
            new QDatacenterCharacteristics(qNodes, 0.0, 3.0);
        new QCloudDatacenter(
            "AQUA_SLA_iQuantum_Datacenter",
            characteristics
        );

        QCloudBroker broker = new QCloudBroker("AQUA_SLA_QBroker");
        CloudGateway gateway = new CloudGateway(
            "AQUA_SLA_CloudGateway",
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

        QubitTopology topology7 = chainTopology(7);
        QubitTopology topology27 = chainTopology(27);

        ArrayList<String> gates = new ArrayList<String>(
            Arrays.asList("CX", "ID", "RZ", "SX", "X")
        );

        nodes.add(
            new QNode(
                0,
                7,
                128,
                2600,
                gates,
                topology7,
                new QTaskSchedulerSpaceShared()
            )
        );

        nodes.add(
            new QNode(
                1,
                27,
                256,
                3400,
                gates,
                topology27,
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
                    "AQUA-SLA-QAOA",
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


def install_bridge() -> Path:
    destination = IQUANTUM_HOME / JAVA_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(java_bridge_source(), encoding="utf-8")
    return destination


def run_iquantum() -> dict[str, Any]:
    java_status = (
        command_version([JAVA_COMMAND, "-version"])
        if JAVA_COMMAND
        else {
            "available": False,
            "returncode": None,
            "output": "Java executable not found. Check JAVA_HOME and PATH.",
        }
    )

    maven_status = (
        command_version([MAVEN_COMMAND, "-version"])
        if MAVEN_COMMAND
        else {
            "available": False,
            "returncode": None,
            "output": "Maven launcher not found. Check MAVEN_HOME and PATH.",
        }
    )

    status: dict[str, Any] = {
        "platform": "CLOUDS Lab iQuantum",
        "repository": "Cloudslab/iQuantum",
        "requested": RUN_IQUANTUM,
        "success": False,
        "iquantum_home": str(IQUANTUM_HOME),
        "java_command": JAVA_COMMAND,
        "maven_command": MAVEN_COMMAND,
        "java_home": os.getenv("JAVA_HOME"),
        "maven_home": os.getenv("MAVEN_HOME"),
        "path": os.getenv("PATH"),
        "java": java_status,
        "maven": maven_status,
    }

    if not RUN_IQUANTUM:
        status["status"] = "disabled_by_AQUA_RUN_IQUANTUM"
        return status

    if not IQUANTUM_HOME.exists():
        status["status"] = (
            "iQuantum repository not found. Clone "
            "https://github.com/Cloudslab/iQuantum and set IQUANTUM_HOME."
        )
        return status

    if not (IQUANTUM_HOME / "pom.xml").exists():
        status["status"] = "IQUANTUM_HOME does not contain pom.xml."
        return status

    if not java_status["available"] or not maven_status["available"]:
        status["status"] = "Java or Maven is unavailable."
        return status

    bridge_path = install_bridge()
    status["bridge_source"] = str(bridge_path)

    args_value = f'"{WORKLOAD_PATH}" "{IQUANTUM_RESULT_PATH}"'
    examples_pom = (
        IQUANTUM_HOME
        / "modules"
        / "iquantum-examples"
        / "pom.xml"
    )

    if not examples_pom.exists():
        status["status"] = (
            "iQuantum examples module pom.xml was not found: "
            f"{examples_pom}"
        )
        return status

    command = [
        MAVEN_COMMAND,
        "-e",
        "-f",
        str(examples_pom),
        "compile",
        "exec:java",
        f"-Dexec.mainClass={JAVA_PACKAGE}.{JAVA_CLASS}",
        f"-Dexec.args={args_value}",
    ]

    start = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=IQUANTUM_HOME,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
            shell=False,
        )
        elapsed = time.perf_counter() - start

        (LOG_DIR / "iquantum_stdout.txt").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (LOG_DIR / "iquantum_stderr.txt").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        (LOG_DIR / "iquantum_combined.log").write_text(
            completed.stdout + "\n\n--- STDERR ---\n" + completed.stderr,
            encoding="utf-8",
        )

        status.update(
            {
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "command": command,
                "examples_pom": str(examples_pom),
                "result_file_exists": IQUANTUM_RESULT_PATH.exists(),
            }
        )

        if completed.returncode == 0 and IQUANTUM_RESULT_PATH.exists():
            status["success"] = True
            status["status"] = "completed"
        else:
            status["status"] = "build_or_execution_failed"

    except Exception as exc:
        status.update(
            {
                "status": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )

    return status


def summarize_results(
    workload: pd.DataFrame,
    status: dict[str, Any],
) -> pd.DataFrame:
    if not status.get("success", False):
        summary = pd.DataFrame(
            [
                {
                    "platform": "iQuantum",
                    "executed": False,
                    "tasks_submitted": len(workload),
                    "tasks_completed": 0,
                    "success_rate": np.nan,
                    "mean_waiting_time": np.nan,
                    "mean_qpu_time": np.nan,
                    "makespan": np.nan,
                    "total_cost": np.nan,
                    "status": status.get("status", "unknown"),
                }
            ]
        )
        summary.to_csv(
            CSV_DIR / "iquantum_platform_summary.csv",
            index=False,
        )
        return summary

    results = pd.read_csv(IQUANTUM_RESULT_PATH)
    merged = workload.merge(results, on="qtask_id", how="left")
    merged.to_csv(
        CSV_DIR / "aqua_sla_iquantum_merged_results.csv",
        index=False,
    )

    completed = results["status"].astype(int) == 4

    summary = pd.DataFrame(
        [
            {
                "platform": "iQuantum",
                "executed": True,
                "tasks_submitted": len(results),
                "tasks_completed": int(completed.sum()),
                "success_rate": float(completed.mean()),
                "mean_waiting_time": float(
                    results["waiting_time"].mean()
                ),
                "mean_qpu_time": float(
                    results["actual_qpu_time"].mean()
                ),
                "makespan": float(results["finish_time"].max()),
                "total_cost": float(results["cost"].sum()),
                "node_0_tasks": int((results["qnode_id"] == 0).sum()),
                "node_1_tasks": int((results["qnode_id"] == 1).sum()),
                "status": "completed",
            }
        ]
    )

    summary.to_csv(
        CSV_DIR / "iquantum_platform_summary.csv",
        index=False,
    )
    return summary


def save_ablation(
    iquantum_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    try:
        scheduler_summary = pd.read_csv(
            find_result_file("locked_scheduling_summary.csv")
        )

        for _, row in scheduler_summary.iterrows():
            rows.append(
                {
                    "experiment_layer": "AQUA-SLA_scheduler",
                    "method": row.get("scheduler", "unknown"),
                    "sla_satisfaction": row.get(
                        "sla_satisfaction_rate_mean",
                        np.nan,
                    ),
                    "mean_response_time": row.get(
                        "mean_response_time_mean",
                        np.nan,
                    ),
                    "makespan": row.get("makespan_mean", np.nan),
                    "energy": row.get("total_energy_mean", np.nan),
                    "cost": row.get("total_cost_mean", np.nan),
                    "note": "Locked AQUA-SLA scheduling result",
                }
            )
    except Exception:
        pass

    iq = iquantum_summary.iloc[0]
    rows.append(
        {
            "experiment_layer": "iQuantum_platform",
            "method": "CLOUDS Lab iQuantum discrete-event simulation",
            "sla_satisfaction": iq.get("success_rate", np.nan),
            "mean_response_time": iq.get("mean_qpu_time", np.nan),
            "makespan": iq.get("makespan", np.nan),
            "energy": np.nan,
            "cost": iq.get("total_cost", np.nan),
            "note": (
                "Platform simulation metrics; not directly interchangeable "
                "with the locked analytical scheduler metrics."
            ),
        }
    )

    ablation = pd.DataFrame(rows)
    ablation.to_csv(
        CSV_DIR / "aqua_sla_iquantum_platform_ablation.csv",
        index=False,
    )
    return ablation


def write_report(
    status: dict[str, Any],
    summary: pd.DataFrame,
) -> None:
    lines = [
        "# AQUA-SLA iQuantum Platform Integration",
        "",
        "This integration uses the CLOUDS Lab iQuantum Java/CloudSim toolkit.",
        "It is separate from IQM quantum hardware execution.",
        "",
        "## Execution status",
        "",
        "```json",
        json.dumps(status, indent=2, default=str),
        "```",
        "",
        "## iQuantum summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Claim boundary",
        "",
        (
            "Report iQuantum as a discrete-event quantum-cloud simulation "
            "platform. Do not describe it as physical quantum hardware. "
            "Qiskit provides circuit/QAOA execution; iQuantum evaluates "
            "resource allocation, task timing, backend assignment, and cost."
        ),
    ]

    (REPORT_DIR / "iquantum_integration_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()

    assignments_path = find_result_file(
        "locked_scheduling_assignments.csv"
    )
    assignments = pd.read_csv(assignments_path)
    workload = build_workload(assignments)

    status = run_iquantum()

    with (REPORT_DIR / "iquantum_execution_status.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(status, file, indent=2, default=str)

    summary = summarize_results(workload, status)
    save_ablation(summary)
    write_report(status, summary)

    print("\nAQUA-SLA iQuantum integration completed.")
    print(f"iQuantum requested: {RUN_IQUANTUM}")
    print(f"iQuantum success: {status.get('success', False)}")
    print(f"Status: {status.get('status')}")
    print(f"Java command: {status.get('java_command')}")
    print(f"Maven command: {status.get('maven_command')}")
    print(f"iQuantum home: {status.get('iquantum_home')}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()