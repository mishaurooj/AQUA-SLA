from __future__ import annotations

r"""
AQUA-SLA: save UPDATED quantum-circuit figures only.

This script does NOT retrain HCQKL, does NOT rerun iQuantum, and does NOT
rerun the scheduling experiments. It reuses the trained HCQKL bundle and,
when available, the validated compact-QUBO coefficients from the targeted
QAOA rerun to regenerate publication-ready circuit figures.

Saved circuit panels:
  01 State preparation and measurement
  02 Full hybrid / quantum feature map
  03 Quantum state evolution in AQUA-SLA
  04 Hardware-efficient variational ansatz
  05 Fidelity quantum-kernel circuit
  06 SWAP-test quantum-kernel circuit
  07 Compact-QUBO QAOA cost Hamiltonian layer
  08 QAOA mixer layer
  09 Full QAOA circuit, p=1
  10 Full QAOA circuit, p=2
  11 Full QAOA circuit, p=3

It also saves one 11-panel composite figure matching the style of the
earlier paper figure.

Recommended:
    conda activate aqua-sla
    cd /d E:\other\AQUA-SLA\Code
    python 33_aqua_sla_save_updated_quantum_circuits.py

Optional:
    python 33_aqua_sla_save_updated_quantum_circuits.py ^
      --run-dir "E:\other\AQUA-SLA\results\aqua_sla_review_v2\20260925_034909_reviewer_revision_v2"

Requirements:
    pip install qiskit matplotlib pylatexenc joblib numpy
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.visualization import circuit_drawer


PROJECT_DIR = Path(r"E:\other\AQUA-SLA")
DEFAULT_V2_ROOT = PROJECT_DIR / "results" / "aqua_sla_review_v2"

FILENAMES = [
    "01_State_Preparation_to_Measurement",
    "02_Full_Hybrid_Feature_Map",
    "03_Quantum_State_Evolution",
    "04_Hardware_Efficient_Variational_Ansatz",
    "05_Fidelity_Quantum_Kernel_Circuit",
    "06_SWAP_Test_Quantum_Kernel",
    "07_Compact_QUBO_QAOA_Cost_Hamiltonian_Layer",
    "08_QAOA_Mixer_Layer",
    "09_Full_QAOA_Circuit_p_1",
    "10_Full_QAOA_Circuit_p_2",
    "11_Full_QAOA_Circuit_p_3",
]


def latest_v2_run(v2_root: Path) -> Path:
    candidates = [
        p for p in v2_root.iterdir()
        if p.is_dir() and (p / "03_models" / "hcqkl_v2_bundle.joblib").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No V2 run containing hcqkl_v2_bundle.joblib was found under {v2_root}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def locate_hcqkl_bundle(run_dir: Path) -> Path:
    p = run_dir / "03_models" / "hcqkl_v2_bundle.joblib"
    if not p.exists():
        raise FileNotFoundError(f"HCQKL bundle not found: {p}")
    return p


def locate_latest_qaoa_validation(run_dir: Path) -> Path | None:
    root = run_dir / "11_targeted_qaoa_iquantum_revision"
    if not root.exists():
        return None

    files = list(root.rglob("block_00_qubo_validation.json"))
    if not files:
        files = list(root.rglob("*qubo_validation.json"))
    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)


def zz_feature_map_circuit(n_qubits: int, reps: int = 2) -> QuantumCircuit:
    try:
        from qiskit.circuit.library import zz_feature_map
        qc = zz_feature_map(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement="linear",
        )
        qc.name = "ZZFeatureMap"
        return qc
    except Exception:
        from qiskit.circuit.library import ZZFeatureMap
        qc = ZZFeatureMap(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement="linear",
        )
        qc.name = "ZZFeatureMap"
        return qc


def bind_feature_map(
    n_qubits: int,
    values: np.ndarray,
    reps: int = 2,
) -> QuantumCircuit:
    fmap = zz_feature_map_circuit(n_qubits, reps=reps)
    values = np.asarray(values, dtype=float).ravel()

    params = list(fmap.parameters)
    if len(values) < len(params):
        values = np.resize(values, len(params))
    mapping = {
        p: float(values[i % len(values)])
        for i, p in enumerate(params)
    }
    return fmap.assign_parameters(mapping, inplace=False)


def make_state_preparation(values: np.ndarray) -> QuantumCircuit:
    values = np.asarray(values, dtype=float).ravel()
    n = len(values)

    qc = QuantumCircuit(n, n, name="StatePrep")
    for q, v in enumerate(values):
        qc.h(q)
        qc.ry(float(v), q)
        qc.rz(float(v / 2.0), q)

    qc.barrier()
    qc.measure(range(n), range(n))
    return qc


def make_hybrid_feature_map(values: np.ndarray) -> QuantumCircuit:
    n = len(values)
    base = bind_feature_map(n, values, reps=2)

    qc = QuantumCircuit(n, name="HybridFeatureMap")
    for q, v in enumerate(values):
        qc.ry(float(v), q)
    qc.barrier(label="encode")
    qc.compose(base, inplace=True)
    return qc


def make_state_evolution(values: np.ndarray) -> QuantumCircuit:
    n = len(values)
    fmap = bind_feature_map(n, values, reps=1)

    qc = QuantumCircuit(n, name="AQUA-SLA evolution")
    for q in range(n):
        qc.h(q)
    qc.barrier(label="prepare")
    qc.compose(fmap, inplace=True)
    qc.barrier(label="feature map")

    for q in range(n):
        qc.ry(0.35 + 0.08 * q, q)
        qc.rz(0.15 + 0.05 * q, q)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.barrier(label="variational")
    return qc


def make_hardware_efficient_ansatz(n_qubits: int, reps: int = 2) -> QuantumCircuit:
    theta = ParameterVector("θ", length=2 * n_qubits * reps)
    qc = QuantumCircuit(n_qubits, name="HEA")

    k = 0
    for layer in range(reps):
        for q in range(n_qubits):
            qc.ry(theta[k], q)
            k += 1
            qc.rz(theta[k], q)
            k += 1

        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)

        if layer != reps - 1:
            qc.barrier()

    return qc


def make_fidelity_kernel_circuit(
    x: np.ndarray,
    y: np.ndarray,
) -> QuantumCircuit:
    n = len(x)
    ux = bind_feature_map(n, x, reps=2)
    uy = bind_feature_map(n, y, reps=2)

    qc = QuantumCircuit(n, n, name="FidelityKernel")
    qc.compose(ux, inplace=True)
    qc.barrier(label="φ(x)")
    qc.compose(uy.inverse(), inplace=True)
    qc.barrier(label="φ(y)†")
    qc.measure(range(n), range(n))
    return qc


def make_swap_test_kernel(
    x: np.ndarray,
    y: np.ndarray,
) -> QuantumCircuit:
    n = len(x)
    ux = bind_feature_map(n, x, reps=1)
    uy = bind_feature_map(n, y, reps=1)

    total = 1 + 2 * n
    qc = QuantumCircuit(total, 1, name="SWAPTest")

    anc = 0
    reg_x = list(range(1, 1 + n))
    reg_y = list(range(1 + n, 1 + 2 * n))

    qc.h(anc)
    qc.compose(ux, qubits=reg_x, inplace=True)
    qc.compose(uy, qubits=reg_y, inplace=True)
    qc.barrier()

    for qx, qy in zip(reg_x, reg_y):
        qc.cswap(anc, qx, qy)

    qc.h(anc)
    qc.measure(anc, 0)
    return qc


def representative_qubo_coefficients() -> tuple[
    float,
    dict[str, float],
    dict[tuple[str, str], float],
]:
    penalty = 10.0
    constant = 2.0 * penalty

    linear = {
        "x_0_0": 0.30 - penalty,
        "x_0_1": 0.65 - penalty,
        "x_1_0": 0.45 - penalty,
        "x_1_1": 0.25 - penalty,
    }

    quadratic = {
        ("x_0_0", "x_0_1"): 2.0 * penalty,
        ("x_1_0", "x_1_1"): 2.0 * penalty,
        ("x_0_0", "x_1_0"): penalty,
    }
    return constant, linear, quadratic


def load_qubo_coefficients(
    validation_path: Path | None,
) -> tuple[
    float,
    dict[str, float],
    dict[tuple[str, str], float],
    str,
]:
    if validation_path is None:
        c, lin, quad = representative_qubo_coefficients()
        return c, lin, quad, "representative compact-QUBO fallback"

    try:
        data = json.loads(validation_path.read_text(encoding="utf-8"))

        constant = float(data["constant"])
        linear = {
            str(k): float(v)
            for k, v in data["linear"].items()
        }

        quadratic = {}
        for key, value in data["quadratic"].items():
            if "|" in key:
                a, b = key.split("|", 1)
            else:
                stripped = (
                    key.strip("()[] ")
                    .replace("'", "")
                    .replace('"', "")
                )
                parts = [x.strip() for x in stripped.split(",")]
                if len(parts) != 2:
                    continue
                a, b = parts
            quadratic[(a, b)] = float(value)

        return constant, linear, quadratic, str(validation_path)
    except Exception:
        c, lin, quad = representative_qubo_coefficients()
        return c, lin, quad, "representative compact-QUBO fallback"


QUBO_VAR_ORDER = ["x_0_0", "x_0_1", "x_1_0", "x_1_1"]


def add_qaoa_cost_layer(
    qc: QuantumCircuit,
    gamma: Any,
    linear: dict[str, float],
    quadratic: dict[tuple[str, str], float],
) -> None:
    idx = {name: i for i, name in enumerate(QUBO_VAR_ORDER)}

    for name, a in linear.items():
        if name not in idx or abs(a) < 1e-12:
            continue
        qc.rz(-gamma * float(a), idx[name])

    for (a_name, b_name), b in quadratic.items():
        if a_name not in idx or b_name not in idx or abs(b) < 1e-12:
            continue
        i, j = idx[a_name], idx[b_name]
        coef = float(b)
        qc.rz(-gamma * coef / 2.0, i)
        qc.rz(-gamma * coef / 2.0, j)
        qc.rzz(gamma * coef / 2.0, i, j)


def add_qaoa_mixer_layer(
    qc: QuantumCircuit,
    beta: Any,
) -> None:
    for q in range(4):
        qc.rx(2.0 * beta, q)


def make_qaoa_cost_layer(
    linear: dict[str, float],
    quadratic: dict[tuple[str, str], float],
) -> QuantumCircuit:
    gamma = Parameter("γ")
    qc = QuantumCircuit(4, name="Cost")
    add_qaoa_cost_layer(qc, gamma, linear, quadratic)
    return qc


def make_qaoa_mixer_layer() -> QuantumCircuit:
    beta = Parameter("β")
    qc = QuantumCircuit(4, name="Mixer")
    add_qaoa_mixer_layer(qc, beta)
    return qc


def make_full_qaoa(
    p: int,
    linear: dict[str, float],
    quadratic: dict[tuple[str, str], float],
) -> QuantumCircuit:
    gammas = ParameterVector("γ", p)
    betas = ParameterVector("β", p)

    qc = QuantumCircuit(4, 4, name=f"QAOA_p{p}")
    qc.h(range(4))

    for layer in range(p):
        qc.barrier(label=f"p={layer+1}")
        add_qaoa_cost_layer(
            qc,
            gammas[layer],
            linear,
            quadratic,
        )
        add_qaoa_mixer_layer(
            qc,
            betas[layer],
        )

    qc.barrier()
    qc.measure(range(4), range(4))
    return qc


def draw_and_save(
    qc: QuantumCircuit,
    title: str,
    stem: Path,
    fold: int = 120,
) -> None:
    fig = circuit_drawer(
        qc,
        output="mpl",
        fold=fold,
        idle_wires=False,
    )

    try:
        fig.suptitle(title, fontsize=12, y=1.02)
    except Exception:
        pass

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        stem.with_suffix(".jpeg"),
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def make_composite(
    png_paths: list[Path],
    titles: list[str],
    output_stem: Path,
) -> None:
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(
        3,
        12,
        height_ratios=[1, 1, 1.15],
        hspace=0.60,
        wspace=0.45,
    )

    positions = [
        (0, slice(0, 3)),
        (0, slice(3, 6)),
        (0, slice(6, 9)),
        (0, slice(9, 12)),
        (1, slice(0, 3)),
        (1, slice(3, 6)),
        (1, slice(6, 9)),
        (1, slice(9, 12)),
        (2, slice(0, 4)),
        (2, slice(4, 8)),
        (2, slice(8, 12)),
    ]

    labels = list("abcdefghijk")

    for path, title, pos, label in zip(
        png_paths,
        titles,
        positions,
        labels,
    ):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"({label}) {title}",
            fontsize=10,
            pad=6,
        )

    fig.suptitle(
        "Updated AQUA-SLA Quantum-Circuit Pipeline",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=350,
        bbox_inches="tight",
    )
    fig.savefig(
        output_stem.with_suffix(".jpeg"),
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing reviewer V2 run containing 03_models/hcqkl_v2_bundle.joblib.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory.",
    )
    parser.add_argument(
        "--max-feature-qubits",
        type=int,
        default=6,
        help="Maximum feature-map qubits to draw. Default 6.",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_v2_run(DEFAULT_V2_ROOT)

    bundle_path = locate_hcqkl_bundle(run_dir)
    bundle = joblib.load(bundle_path)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = run_dir / "12_updated_quantum_circuit_figures"

    out_dir.mkdir(parents=True, exist_ok=True)

    z_train = np.asarray(bundle["z_train"], dtype=float)
    if z_train.ndim != 2 or len(z_train) < 2:
        raise RuntimeError(
            "The trained HCQKL bundle does not contain at least two transformed training vectors."
        )

    n = min(
        args.max_feature_qubits,
        z_train.shape[1],
    )
    if n < 2:
        raise RuntimeError(
            f"Need at least 2 feature-map dimensions; found {n}."
        )

    x = z_train[0, :n]
    y = z_train[1, :n]

    validation_path = locate_latest_qaoa_validation(run_dir)
    _, linear, quadratic, qubo_source = load_qubo_coefficients(
        validation_path
    )

    circuits: list[QuantumCircuit] = []
    titles: list[str] = []

    circuits.append(make_state_preparation(x))
    titles.append("State preparation and measurement")

    circuits.append(make_hybrid_feature_map(x))
    titles.append("Hybrid feature map")

    circuits.append(make_state_evolution(x))
    titles.append("Quantum state evolution")

    circuits.append(make_hardware_efficient_ansatz(n, reps=2))
    titles.append("Hardware-efficient ansatz")

    circuits.append(make_fidelity_kernel_circuit(x, y))
    titles.append("Fidelity quantum-kernel circuit")

    circuits.append(make_swap_test_kernel(x, y))
    titles.append("SWAP-test kernel circuit")

    circuits.append(make_qaoa_cost_layer(linear, quadratic))
    titles.append("Compact-QUBO QAOA cost layer")

    circuits.append(make_qaoa_mixer_layer())
    titles.append("QAOA mixer layer")

    circuits.append(make_full_qaoa(1, linear, quadratic))
    titles.append(r"Full QAOA, $p=1$")

    circuits.append(make_full_qaoa(2, linear, quadratic))
    titles.append(r"Full QAOA, $p=2$")

    circuits.append(make_full_qaoa(3, linear, quadratic))
    titles.append(r"Full QAOA, $p=3$")

    png_paths: list[Path] = []

    for qc, filename, title in zip(
        circuits,
        FILENAMES,
        titles,
    ):
        stem = out_dir / filename
        draw_and_save(
            qc,
            title,
            stem,
            fold=140,
        )
        png_paths.append(stem.with_suffix(".png"))
        print("Saved:", stem.with_suffix(".png"))

    make_composite(
        png_paths,
        titles,
        out_dir / "AQUA_SLA_Updated_Quantum_Circuits_Composite",
    )

    print()
    print("Finished.")
    print("HCQKL source:", bundle_path)
    print("QAOA coefficient source:", qubo_source)
    print("Circuit output:", out_dir)
    print("No model retraining or experiment rerun was performed.")


if __name__ == "__main__":
    main()
