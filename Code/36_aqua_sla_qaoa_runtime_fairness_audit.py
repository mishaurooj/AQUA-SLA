from __future__ import annotations

r"""
AQUA-SLA FINAL QAOA RUNTIME-FAIRNESS AUDIT
==========================================

Purpose
-------
This is a targeted final audit for the remaining runtime question only.

It does NOT retrain HCQKL.
It does NOT rebuild the future target.
It does NOT rerun classical scheduling.
It does NOT rerun iQuantum.
It does NOT change the QUBO or select instances based on QAOA performance.

Instead, it reuses the exact heterogeneous compact-QUBO instances already
saved by the repaired final AQUA-SLA run and measures QAOA timing fairly after
removing Windows multiprocessing/process-spawn overhead.

For each saved final QAOA block it compares:
    * QAOA solver-only runtime
    * QAOA warm-process end-to-end runtime
    * MILP solver-only runtime
    * MILP end-to-end runtime
    * RiskGreedy runtime
    * LeastLoaded runtime

Three QAOA efficiency configurations are audited:
    primary_512s_50iter       : p=1, shots=512, COBYLA maxiter=50
    efficient_256s_20iter     : p=1, shots=256, COBYLA maxiter=20
    efficient_128s_10iter     : p=1, shots=128, COBYLA maxiter=10

A faster QAOA configuration is accepted as "reliable" only if it retains:
    >=95% successful completion,
    >=95% raw feasibility,
    >=95% exact objective match to MILP.

A QAOA time advantage is declared ONLY if the fastest reliable configuration
also has a paired block-bootstrap 95% CI for the end-to-end QAOA/MILP runtime
ratio entirely below 1.0.

If this condition is not met, the generated status file explicitly states:
    "No QAOA runtime advantage was observed."

This prevents a false quantum-speedup claim.

Recommended final run
---------------------
conda activate aqua-sla
cd /d E:\other\AQUA-SLA\Code

python 36_aqua_sla_qaoa_runtime_fairness_audit.py ^
  --run-dir "E:\other\AQUA-SLA\results\aqua_sla_final_reviewer_complete\20261005_150107_final_reviewer_complete_fixed" ^
  --repeats 3 ^
  --bootstrap 5000

Smoke test
----------
python 36_aqua_sla_qaoa_runtime_fairness_audit.py ^
  --run-dir "E:\other\AQUA-SLA\results\aqua_sla_final_reviewer_complete\20261005_150107_final_reviewer_complete_fixed" ^
  --fast

Outputs
-------
<run-dir>\16_qaoa_runtime_fairness_audit\<timestamp>\
    qaoa_runtime_runs.csv
    qaoa_runtime_block_level.csv
    qaoa_runtime_summary.csv
    qaoa_runtime_paired_statistics.csv
    qaoa_runtime_fairness_status.json
    qaoa_runtime_comparison.png
    qaoa_runtime_ratio.png
"""

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import wilcoxon


SEED = 42


# =============================================================================
# General utilities
# =============================================================================

def save_json(obj: Any, path: Path) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
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


def bootstrap_mean_ci(
    values: Sequence[float],
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    stats = np.empty(resamples, dtype=float)

    for b in range(resamples):
        sample = rng.choice(
            x,
            size=len(x),
            replace=True,
        )
        stats[b] = np.mean(sample)

    return (
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
    )


def paired_cohens_dz(differences: Sequence[float]) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]

    if len(d) < 2:
        return np.nan

    sd = float(np.std(d, ddof=1))

    if np.isclose(sd, 0):
        if np.isclose(float(np.mean(d)), 0):
            return 0.0
        return float(
            np.sign(np.mean(d)) * np.inf
        )

    return float(
        np.mean(d) / sd
    )


def holm_adjust(
    p_values: Sequence[float],
) -> np.ndarray:
    p = np.asarray(
        p_values,
        dtype=float,
    )

    ans = np.full(
        len(p),
        np.nan,
    )

    valid = np.where(
        np.isfinite(p)
    )[0]

    if len(valid) == 0:
        return ans

    order = valid[
        np.argsort(
            p[valid]
        )
    ]

    running = 0.0
    m = len(order)

    for rank, idx in enumerate(order):
        adjusted = min(
            1.0,
            (m - rank) * p[idx],
        )
        running = max(
            running,
            adjusted,
        )
        ans[idx] = running

    return ans


# =============================================================================
# Saved final-instance loading
# =============================================================================

def locate_block_directory(
    run_dir: Path,
) -> Path:
    candidates = [
        run_dir
        / "06_compact_qaoa"
        / "qaoa_compact_qubo"
        / "blocks",
    ]

    candidates.extend(
        run_dir.rglob(
            "qaoa_compact_qubo/blocks"
        )
    )

    for p in candidates:
        if (
            p.exists()
            and any(
                p.glob(
                    "block_*_manifest.json"
                )
            )
        ):
            return p

    raise FileNotFoundError(
        "Could not find saved final QAOA block manifests under "
        f"{run_dir}"
    )


def block_number(
    path: Path,
) -> int:
    stem = path.stem

    # block_00_manifest
    parts = stem.split("_")

    for p in parts:
        if p.isdigit():
            return int(p)

    raise ValueError(
        f"Cannot parse block number from {path.name}"
    )


def load_saved_blocks(
    run_dir: Path,
    max_blocks: int | None = None,
) -> list[dict[str, Any]]:
    block_dir = locate_block_directory(
        run_dir
    )

    manifests = sorted(
        block_dir.glob(
            "block_*_manifest.json"
        ),
        key=block_number,
    )

    blocks = []

    for manifest_path in manifests:
        b = block_number(
            manifest_path
        )

        selected_path = (
            block_dir
            / f"block_{b:02d}_selected_tasks.csv"
        )

        if not selected_path.exists():
            # Compatibility with an alternate name.
            alt = (
                block_dir
                / f"block_{b:02d}_tasks.csv"
            )
            if alt.exists():
                selected_path = alt

        validation_path = (
            block_dir
            / f"block_{b:02d}_qubo_validation.json"
        )

        qaoa_result_path = (
            block_dir
            / f"block_{b:02d}_qaoa.json"
        )

        if not validation_path.exists():
            continue

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        validation = json.loads(
            validation_path.read_text(
                encoding="utf-8"
            )
        )

        if selected_path.exists():
            tasks = pd.read_csv(
                selected_path,
                low_memory=False,
            )
        else:
            task_records = manifest.get(
                "tasks",
                [],
            )
            tasks = pd.DataFrame(
                task_records
            )

        if tasks.empty:
            raise RuntimeError(
                f"Saved QAOA block {b} has no task data."
            )

        resources = manifest.get(
            "resources",
            []
        )

        C = np.asarray(
            manifest.get(
                "cost_matrix"
            ),
            dtype=float,
        )

        if C.shape != (2, 2):
            raise RuntimeError(
                f"Block {b}: expected 2x2 cost matrix, got {C.shape}."
            )

        if len(resources) != 2:
            raise RuntimeError(
                f"Block {b}: expected two resources."
            )

        constant = float(
            validation[
                "constant"
            ]
        )

        linear = {
            str(k): float(v)
            for k, v in validation[
                "linear"
            ].items()
        }

        quadratic = {}

        for key, value in validation[
            "quadratic"
        ].items():
            if "|" in key:
                a, c = key.split(
                    "|",
                    1,
                )
            else:
                stripped = (
                    key.strip(
                        "()[] "
                    )
                    .replace(
                        "'",
                        "",
                    )
                    .replace(
                        '"',
                        "",
                    )
                )
                parts = [
                    x.strip()
                    for x in stripped.split(
                        ","
                    )
                ]
                if len(parts) != 2:
                    raise ValueError(
                        f"Unsupported quadratic key: {key}"
                    )
                a, c = parts

            quadratic[
                (a, c)
            ] = float(value)

        old_qaoa = None

        if qaoa_result_path.exists():
            try:
                old_qaoa = json.loads(
                    qaoa_result_path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                old_qaoa = None

        blocks.append(
            {
                "block": b,
                "tasks": tasks,
                "resources": resources,
                "C": C,
                "milp_objective_saved": float(
                    manifest.get(
                        "milp_objective",
                        np.nan,
                    )
                ),
                "constant": constant,
                "linear": linear,
                "quadratic": quadratic,
                "validation": validation,
                "old_qaoa": old_qaoa,
                "manifest_path": str(
                    manifest_path
                ),
            }
        )

    if max_blocks is not None:
        blocks = blocks[
            :max_blocks
        ]

    if not blocks:
        raise RuntimeError(
            "No saved QAOA blocks were loaded."
        )

    return blocks


# =============================================================================
# Exact classical comparison solvers
# =============================================================================

def task_demands(
    tasks: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    cpu = pd.to_numeric(
        tasks[
            "requested_cpu"
        ],
        errors="coerce",
    ).fillna(0).to_numpy(float)

    mem = pd.to_numeric(
        tasks[
            "requested_memory"
        ],
        errors="coerce",
    ).fillna(0).to_numpy(float)

    return cpu, mem


def resource_capacity(
    resource: dict[str, Any],
    name: str,
) -> float:
    aliases = {
        "cpu": [
            "cpu_capacity",
            "cpu",
            "capacity_cpu",
        ],
        "memory": [
            "memory_capacity",
            "memory",
            "capacity_memory",
        ],
    }

    for key in aliases[
        name
    ]:
        if key in resource:
            return float(
                resource[
                    key
                ]
            )

    raise KeyError(
        f"Resource lacks {name} capacity: {resource}"
    )


def solve_milp_saved_instance(
    tasks: pd.DataFrame,
    resources: list[dict[str, Any]],
    C: np.ndarray,
) -> dict[str, Any]:
    n, m = C.shape

    c = C.ravel()

    cpu, mem = task_demands(
        tasks
    )

    rows = []
    lb = []
    ub = []

    for i in range(n):
        row = np.zeros(
            n * m,
            dtype=float,
        )
        row[
            i * m:(i + 1) * m
        ] = 1.0
        rows.append(
            row
        )
        lb.append(
            1.0
        )
        ub.append(
            1.0
        )

    for j, r in enumerate(
        resources
    ):
        rcpu = np.zeros(
            n * m,
            dtype=float,
        )
        rmem = np.zeros(
            n * m,
            dtype=float,
        )

        for i in range(n):
            rcpu[
                i * m + j
            ] = cpu[i]
            rmem[
                i * m + j
            ] = mem[i]

        rows.extend(
            [
                rcpu,
                rmem,
            ]
        )

        lb.extend(
            [
                -np.inf,
                -np.inf,
            ]
        )

        ub.extend(
            [
                resource_capacity(
                    r,
                    "cpu",
                ),
                resource_capacity(
                    r,
                    "memory",
                ),
            ]
        )

    solve_start = time.perf_counter()

    result = milp(
        c=c,
        integrality=np.ones(
            n * m,
            dtype=int,
        ),
        bounds=Bounds(
            np.zeros(
                n * m
            ),
            np.ones(
                n * m
            ),
        ),
        constraints=LinearConstraint(
            np.vstack(
                rows
            ),
            np.asarray(
                lb,
                float,
            ),
            np.asarray(
                ub,
                float,
            ),
        ),
        options={
            "time_limit": 60,
        },
    )

    solver_seconds = (
        time.perf_counter()
        - solve_start
    )

    if result.x is None:
        return {
            "success": False,
            "solver_seconds": solver_seconds,
            "message": str(
                result.message
            ),
        }

    assignment = (
        np.argmax(
            result.x.reshape(
                n,
                m,
            ),
            axis=1,
        )
        .astype(int)
        .tolist()
    )

    objective = float(
        C[
            np.arange(n),
            np.asarray(
                assignment,
                int,
            ),
        ].sum()
    )

    return {
        "success": True,
        "solver_seconds": solver_seconds,
        "assignment": assignment,
        "objective": objective,
        "message": str(
            result.message
        ),
    }


def assignment_feasible(
    assignment: Sequence[int],
    tasks: pd.DataFrame,
    resources: list[dict[str, Any]],
) -> bool:
    if assignment is None:
        return False

    cpu, mem = task_demands(
        tasks
    )

    for j, r in enumerate(
        resources
    ):
        ids = [
            i
            for i, jj in enumerate(
                assignment
            )
            if jj == j
        ]

        if not ids:
            continue

        if (
            cpu[
                ids
            ].sum()
            > resource_capacity(
                r,
                "cpu",
            )
            + 1e-9
        ):
            return False

        if (
            mem[
                ids
            ].sum()
            > resource_capacity(
                r,
                "memory",
            )
            + 1e-9
        ):
            return False

    return True


def objective(
    assignment: Sequence[int],
    C: np.ndarray,
) -> float:
    return float(
        C[
            np.arange(
                len(
                    assignment
                )
            ),
            np.asarray(
                assignment,
                int,
            ),
        ].sum()
    )


def risk_greedy(
    tasks: pd.DataFrame,
    resources: list[dict[str, Any]],
    C: np.ndarray,
) -> list[int] | None:
    risk = pd.to_numeric(
        tasks[
            "risk_hcqkl"
        ],
        errors="coerce",
    ).fillna(0).to_numpy(float)

    cpu, mem = task_demands(
        tasks
    )

    used_cpu = np.zeros(
        len(resources)
    )
    used_mem = np.zeros(
        len(resources)
    )

    assignment = [
        -1
    ] * len(
        tasks
    )

    order = np.argsort(
        -risk
    )

    for i in order:
        candidates = []

        for j, r in enumerate(
            resources
        ):
            if (
                used_cpu[j]
                + cpu[i]
                <= resource_capacity(
                    r,
                    "cpu",
                )
                + 1e-9
                and used_mem[j]
                + mem[i]
                <= resource_capacity(
                    r,
                    "memory",
                )
                + 1e-9
            ):
                candidates.append(
                    j
                )

        if not candidates:
            return None

        j = min(
            candidates,
            key=lambda jj: (
                C[
                    i,
                    jj,
                ],
                jj,
            ),
        )

        assignment[
            i
        ] = int(
            j
        )

        used_cpu[j] += cpu[i]
        used_mem[j] += mem[i]

    return assignment


def least_loaded(
    tasks: pd.DataFrame,
    resources: list[dict[str, Any]],
    C: np.ndarray,
) -> list[int] | None:
    cpu, mem = task_demands(
        tasks
    )

    if (
        "_estimated_duration_selection"
        in tasks.columns
    ):
        duration = pd.to_numeric(
            tasks[
                "_estimated_duration_selection"
            ],
            errors="coerce",
        ).fillna(1.0).to_numpy(float)
    else:
        duration = np.ones(
            len(tasks),
            dtype=float,
        )

    used_cpu = np.zeros(
        len(resources)
    )
    used_mem = np.zeros(
        len(resources)
    )
    load = np.zeros(
        len(resources)
    )

    assignment = [
        -1
    ] * len(
        tasks
    )

    for i in range(
        len(tasks)
    ):
        candidates = []

        for j, r in enumerate(
            resources
        ):
            if (
                used_cpu[j]
                + cpu[i]
                <= resource_capacity(
                    r,
                    "cpu",
                )
                + 1e-9
                and used_mem[j]
                + mem[i]
                <= resource_capacity(
                    r,
                    "memory",
                )
                + 1e-9
            ):
                candidates.append(
                    j
                )

        if not candidates:
            return None

        def speed(j: int) -> float:
            return float(
                resources[
                    j
                ].get(
                    "speed",
                    1.0,
                )
            )

        j = min(
            candidates,
            key=lambda jj: (
                load[jj],
                C[
                    i,
                    jj,
                ],
                jj,
            ),
        )

        assignment[
            i
        ] = int(
            j
        )

        used_cpu[j] += cpu[i]
        used_mem[j] += mem[i]
        load[j] += (
            duration[i]
            / max(
                speed(j),
                1e-12,
            )
        )

    return assignment


# =============================================================================
# QAOA warm-process timing
# =============================================================================

def make_sampler(
    shots: int,
    seed: int,
):
    try:
        from qiskit.primitives import StatevectorSampler

        return (
            StatevectorSampler(
                default_shots=int(
                    shots
                ),
                seed=int(
                    seed
                ),
            ),
            "StatevectorSampler",
        )

    except Exception:
        from qiskit.primitives import Sampler

        return (
            Sampler(
                options={
                    "shots": int(
                        shots
                    ),
                    "seed": int(
                        seed
                    ),
                }
            ),
            "Sampler",
        )


def build_qp(
    block: dict[str, Any],
):
    from qiskit_optimization import QuadraticProgram

    qp = QuadraticProgram(
        "aqua_sla_runtime_fairness_qubo"
    )

    for i in range(2):
        for j in range(2):
            qp.binary_var(
                f"x_{i}_{j}"
            )

    qp.minimize(
        constant=float(
            block[
                "constant"
            ]
        ),
        linear={
            k: float(v)
            for k, v in block[
                "linear"
            ].items()
        },
        quadratic={
            (
                a,
                b,
            ): float(v)
            for (
                a,
                b,
            ), v in block[
                "quadratic"
            ].items()
        },
    )

    return qp


def run_qaoa_once(
    block: dict[str, Any],
    reps: int,
    shots: int,
    maxiter: int,
    seed: int,
) -> dict[str, Any]:
    try:
        from qiskit_algorithms.minimum_eigensolvers import QAOA
        from qiskit_algorithms.optimizers import COBYLA

        try:
            from qiskit_algorithms.utils import algorithm_globals

            algorithm_globals.random_seed = int(
                seed
            )
        except Exception:
            pass

    except Exception:
        from qiskit.algorithms.minimum_eigensolvers import QAOA
        from qiskit.algorithms.optimizers import COBYLA

    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    e2e_start = time.perf_counter()

    qp = build_qp(
        block
    )

    sampler, sampler_name = make_sampler(
        shots,
        seed,
    )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(
            maxiter=int(
                maxiter
            )
        ),
        reps=int(
            reps
        ),
    )

    optimizer = MinimumEigenOptimizer(
        qaoa
    )

    solve_start = time.perf_counter()

    result = optimizer.solve(
        qp
    )

    solver_seconds = (
        time.perf_counter()
        - solve_start
    )

    end_to_end_seconds = (
        time.perf_counter()
        - e2e_start
    )

    x = np.asarray(
        result.x,
        dtype=float,
    )

    bits = (
        x >= 0.5
    ).astype(int)

    bit_matrix = bits.reshape(
        2,
        2,
    )

    assignment = []

    for i in range(2):
        ones = np.where(
            bit_matrix[
                i
            ] == 1
        )[0]

        if len(
            ones
        ) != 1:
            assignment = None
            break

        assignment.append(
            int(
                ones[0]
            )
        )

    return {
        "success": True,
        "sampler": sampler_name,
        "solver_seconds": solver_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "raw_x": x.tolist(),
        "raw_bits": bits.tolist(),
        "assignment": assignment,
        "fval": (
            float(
                result.fval
            )
            if getattr(
                result,
                "fval",
                None,
            )
            is not None
            else None
        ),
    }


# =============================================================================
# Analysis
# =============================================================================

def paired_runtime_statistics(
    block_level: pd.DataFrame,
    bootstrap: int,
) -> pd.DataFrame:
    rows = []

    for config, g in block_level.groupby(
        "config"
    ):
        for q_col, c_col, label in [
            (
                "qaoa_solver_seconds",
                "milp_solver_seconds",
                "solver_only",
            ),
            (
                "qaoa_end_to_end_seconds",
                "milp_end_to_end_seconds",
                "end_to_end",
            ),
        ]:
            z = g[
                [
                    q_col,
                    c_col,
                ]
            ].dropna()

            if z.empty:
                continue

            q = z[
                q_col
            ].to_numpy(float)

            c = z[
                c_col
            ].to_numpy(float)

            d = q - c

            ratio = q / np.maximum(
                c,
                1e-12,
            )

            dlo, dhi = bootstrap_mean_ci(
                d,
                bootstrap,
                SEED
                + len(
                    rows
                ),
            )

            rlo, rhi = bootstrap_mean_ci(
                ratio,
                bootstrap,
                SEED
                + 1000
                + len(
                    rows
                ),
            )

            nz = d[
                ~np.isclose(
                    d,
                    0,
                )
            ]

            stat = np.nan
            p = np.nan

            if len(
                nz
            ) >= 6:
                try:
                    w = wilcoxon(
                        nz,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    )
                    stat = float(
                        w.statistic
                    )
                    p = float(
                        w.pvalue
                    )
                except Exception:
                    pass

            rows.append(
                {
                    "config": config,
                    "comparison": label,
                    "paired_blocks": len(
                        z
                    ),
                    "mean_qaoa_seconds": float(
                        q.mean()
                    ),
                    "mean_milp_seconds": float(
                        c.mean()
                    ),
                    "mean_difference_qaoa_minus_milp": float(
                        d.mean()
                    ),
                    "bootstrap95_difference_low": dlo,
                    "bootstrap95_difference_high": dhi,
                    "mean_runtime_ratio_qaoa_over_milp": float(
                        ratio.mean()
                    ),
                    "bootstrap95_ratio_low": rlo,
                    "bootstrap95_ratio_high": rhi,
                    "paired_cohens_dz": paired_cohens_dz(
                        d
                    ),
                    "wilcoxon_stat": stat,
                    "p_raw": p,
                }
            )

    ans = pd.DataFrame(
        rows
    )

    if not ans.empty:
        ans[
            "p_holm"
        ] = holm_adjust(
            ans[
                "p_raw"
            ].to_numpy(float)
        )

        ans[
            "significant_holm_0_05"
        ] = (
            ans[
                "p_holm"
            ] < 0.05
        )

    return ans


def make_figures(
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if summary.empty:
        return

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    x = np.arange(
        len(
            summary
        )
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        summary[
            "mean_qaoa_end_to_end_seconds"
        ],
        width,
        label="QAOA",
    )

    ax.bar(
        x + width / 2,
        summary[
            "mean_milp_end_to_end_seconds"
        ],
        width,
        label="MILP",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        summary[
            "config"
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Mean end-to-end runtime (s)"
    )

    ax.set_title(
        "QAOA vs MILP runtime fairness audit"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "qaoa_runtime_comparison.png",
        dpi=300,
    )

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.axhline(
        1.0,
        linestyle="--",
    )

    ax.bar(
        summary[
            "config"
        ],
        summary[
            "mean_qaoa_vs_milp_end_to_end_ratio"
        ],
    )

    ax.set_ylabel(
        "QAOA / MILP end-to-end runtime ratio"
    )

    ax.set_title(
        "Runtime ratio (<1 would indicate QAOA faster)"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "qaoa_runtime_ratio.png",
        dpi=300,
    )

    plt.close(
        fig
    )


# =============================================================================
# Main audit
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AQUA-SLA final QAOA runtime-fairness audit on saved "
            "heterogeneous compact-QUBO instances."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Final AQUA-SLA result folder containing the repaired "
            "06_compact_qaoa block files."
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help=(
            "Repeated timed executions per saved block and QAOA configuration."
        ),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Smoke test: first 3 blocks, one repeat, 128-shot/10-iteration "
            "configuration only, 1000 bootstrap resamples."
        ),
    )

    args = parser.parse_args()

    run_dir = Path(
        args.run_dir
    ).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(
            run_dir
        )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    out_dir = (
        run_dir
        / "16_qaoa_runtime_fairness_audit"
        / stamp
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    max_blocks = (
        3
        if args.fast
        else None
    )

    repeats = (
        1
        if args.fast
        else max(
            1,
            int(
                args.repeats
            ),
        )
    )

    bootstrap = (
        min(
            1000,
            args.bootstrap,
        )
        if args.fast
        else args.bootstrap
    )

    blocks = load_saved_blocks(
        run_dir,
        max_blocks=max_blocks,
    )

    configurations = [
        {
            "config": "primary_512s_50iter",
            "reps": 1,
            "shots": 512,
            "maxiter": 50,
        },
        {
            "config": "efficient_256s_20iter",
            "reps": 1,
            "shots": 256,
            "maxiter": 20,
        },
        {
            "config": "efficient_128s_10iter",
            "reps": 1,
            "shots": 128,
            "maxiter": 10,
        },
    ]

    if args.fast:
        configurations = [
            configurations[-1]
        ]

    save_json(
        {
            "project": "AQUA-SLA",
            "experiment": "qaoa-runtime-fairness-audit",
            "created": datetime.now().isoformat(),
            "source_run": str(
                run_dir
            ),
            "output": str(
                out_dir
            ),
            "python": sys.version,
            "platform": platform.platform(),
            "blocks": len(
                blocks
            ),
            "repeats": repeats,
            "configurations": configurations,
            "timing_rule": (
                "One QAOA warm-up solve is excluded. Timed QAOA then runs "
                "in the same Python process, removing Windows process-spawn "
                "overhead. Solver-only and end-to-end times are both saved."
            ),
        },
        out_dir
        / "runtime_audit_protocol.json",
    )

    # Warm up Qiskit/QAOA once. This is deliberately excluded.
    warm_block = blocks[
        0
    ]

    warm_cfg = configurations[
        -1
    ]

    warm_start = time.perf_counter()

    warm_result = run_qaoa_once(
        warm_block,
        reps=warm_cfg[
            "reps"
        ],
        shots=warm_cfg[
            "shots"
        ],
        maxiter=warm_cfg[
            "maxiter"
        ],
        seed=SEED
        + 999999,
    )

    warmup_seconds = (
        time.perf_counter()
        - warm_start
    )

    rows = []

    for block in blocks:
        b = block[
            "block"
        ]

        tasks = block[
            "tasks"
        ]

        resources = block[
            "resources"
        ]

        C = block[
            "C"
        ]

        for repeat in range(
            repeats
        ):
            milp_e2e_start = time.perf_counter()

            milp_result = solve_milp_saved_instance(
                tasks,
                resources,
                C,
            )

            milp_e2e_seconds = (
                time.perf_counter()
                - milp_e2e_start
            )

            if not milp_result[
                "success"
            ]:
                raise RuntimeError(
                    f"MILP failed on saved QAOA block {b}: "
                    f"{milp_result.get('message')}"
                )

            saved_obj = block[
                "milp_objective_saved"
            ]

            if (
                np.isfinite(
                    saved_obj
                )
                and not np.isclose(
                    milp_result[
                        "objective"
                    ],
                    saved_obj,
                    atol=1e-8,
                )
            ):
                raise RuntimeError(
                    f"Block {b}: rebuilt MILP objective "
                    f"{milp_result['objective']} differs from saved "
                    f"{saved_obj}."
                )

            rg_start = time.perf_counter()

            rg_assignment = risk_greedy(
                tasks,
                resources,
                C,
            )

            risk_greedy_seconds = (
                time.perf_counter()
                - rg_start
            )

            ll_start = time.perf_counter()

            ll_assignment = least_loaded(
                tasks,
                resources,
                C,
            )

            least_loaded_seconds = (
                time.perf_counter()
                - ll_start
            )

            for cfg in configurations:
                seed = (
                    SEED
                    + 100000
                    + b * 1000
                    + repeat * 100
                    + cfg[
                        "shots"
                    ]
                )

                try:
                    q = run_qaoa_once(
                        block,
                        reps=cfg[
                            "reps"
                        ],
                        shots=cfg[
                            "shots"
                        ],
                        maxiter=cfg[
                            "maxiter"
                        ],
                        seed=seed,
                    )

                    a = q[
                        "assignment"
                    ]

                    feasible = bool(
                        a is not None
                        and assignment_feasible(
                            a,
                            tasks,
                            resources,
                        )
                    )

                    qobj = (
                        objective(
                            a,
                            C,
                        )
                        if a
                        is not None
                        else np.nan
                    )

                    gap = (
                        qobj
                        - milp_result[
                            "objective"
                        ]
                        if np.isfinite(
                            qobj
                        )
                        else np.nan
                    )

                    rows.append(
                        {
                            "block": b,
                            "repeat": repeat,
                            "config": cfg[
                                "config"
                            ],
                            "reps": cfg[
                                "reps"
                            ],
                            "shots": cfg[
                                "shots"
                            ],
                            "maxiter": cfg[
                                "maxiter"
                            ],
                            "qaoa_success": True,
                            "qaoa_raw_feasible": feasible,
                            "qaoa_optimal_match": bool(
                                feasible
                                and np.isclose(
                                    gap,
                                    0.0,
                                    atol=1e-8,
                                )
                            ),
                            "qaoa_objective": qobj,
                            "milp_objective": milp_result[
                                "objective"
                            ],
                            "qaoa_objective_gap": gap,
                            "qaoa_solver_seconds": q[
                                "solver_seconds"
                            ],
                            "qaoa_end_to_end_seconds": q[
                                "end_to_end_seconds"
                            ],
                            "milp_solver_seconds": milp_result[
                                "solver_seconds"
                            ],
                            "milp_end_to_end_seconds": milp_e2e_seconds,
                            "risk_greedy_seconds": risk_greedy_seconds,
                            "risk_greedy_feasible": (
                                assignment_feasible(
                                    rg_assignment,
                                    tasks,
                                    resources,
                                )
                                if rg_assignment
                                is not None
                                else False
                            ),
                            "least_loaded_seconds": least_loaded_seconds,
                            "least_loaded_feasible": (
                                assignment_feasible(
                                    ll_assignment,
                                    tasks,
                                    resources,
                                )
                                if ll_assignment
                                is not None
                                else False
                            ),
                            "qaoa_vs_milp_solver_ratio": (
                                q[
                                    "solver_seconds"
                                ]
                                / max(
                                    milp_result[
                                        "solver_seconds"
                                    ],
                                    1e-12,
                                )
                            ),
                            "qaoa_vs_milp_end_to_end_ratio": (
                                q[
                                    "end_to_end_seconds"
                                ]
                                / max(
                                    milp_e2e_seconds,
                                    1e-12,
                                )
                            ),
                            "qaoa_vs_risk_greedy_ratio": (
                                q[
                                    "end_to_end_seconds"
                                ]
                                / max(
                                    risk_greedy_seconds,
                                    1e-12,
                                )
                            ),
                        }
                    )

                except Exception as exc:
                    rows.append(
                        {
                            "block": b,
                            "repeat": repeat,
                            "config": cfg[
                                "config"
                            ],
                            "reps": cfg[
                                "reps"
                            ],
                            "shots": cfg[
                                "shots"
                            ],
                            "maxiter": cfg[
                                "maxiter"
                            ],
                            "qaoa_success": False,
                            "qaoa_raw_feasible": False,
                            "qaoa_optimal_match": False,
                            "qaoa_objective": np.nan,
                            "milp_objective": milp_result[
                                "objective"
                            ],
                            "qaoa_objective_gap": np.nan,
                            "qaoa_solver_seconds": np.nan,
                            "qaoa_end_to_end_seconds": np.nan,
                            "milp_solver_seconds": milp_result[
                                "solver_seconds"
                            ],
                            "milp_end_to_end_seconds": milp_e2e_seconds,
                            "risk_greedy_seconds": risk_greedy_seconds,
                            "least_loaded_seconds": least_loaded_seconds,
                            "qaoa_vs_milp_solver_ratio": np.nan,
                            "qaoa_vs_milp_end_to_end_ratio": np.nan,
                            "qaoa_vs_risk_greedy_ratio": np.nan,
                            "error": repr(
                                exc
                            ),
                        }
                    )

    runs = pd.DataFrame(
        rows
    )

    runs.to_csv(
        out_dir
        / "qaoa_runtime_runs.csv",
        index=False,
    )

    # Repeated timings are correlated. Aggregate to one median observation
    # per temporal workload block before statistical inference.
    block_level = (
        runs
        .groupby(
            [
                "block",
                "config",
            ],
            as_index=False,
        )
        .agg(
            attempts=(
                "repeat",
                "count",
            ),
            qaoa_success_rate=(
                "qaoa_success",
                "mean",
            ),
            qaoa_raw_feasibility_rate=(
                "qaoa_raw_feasible",
                "mean",
            ),
            qaoa_optimal_match_rate=(
                "qaoa_optimal_match",
                "mean",
            ),
            qaoa_objective_gap=(
                "qaoa_objective_gap",
                "mean",
            ),
            qaoa_solver_seconds=(
                "qaoa_solver_seconds",
                "median",
            ),
            qaoa_end_to_end_seconds=(
                "qaoa_end_to_end_seconds",
                "median",
            ),
            milp_solver_seconds=(
                "milp_solver_seconds",
                "median",
            ),
            milp_end_to_end_seconds=(
                "milp_end_to_end_seconds",
                "median",
            ),
            risk_greedy_seconds=(
                "risk_greedy_seconds",
                "median",
            ),
            least_loaded_seconds=(
                "least_loaded_seconds",
                "median",
            ),
        )
    )

    block_level[
        "qaoa_vs_milp_solver_ratio"
    ] = (
        block_level[
            "qaoa_solver_seconds"
        ]
        / block_level[
            "milp_solver_seconds"
        ].clip(
            lower=1e-12
        )
    )

    block_level[
        "qaoa_vs_milp_end_to_end_ratio"
    ] = (
        block_level[
            "qaoa_end_to_end_seconds"
        ]
        / block_level[
            "milp_end_to_end_seconds"
        ].clip(
            lower=1e-12
        )
    )

    block_level.to_csv(
        out_dir
        / "qaoa_runtime_block_level.csv",
        index=False,
    )

    summary_rows = []

    for config, g in block_level.groupby(
        "config"
    ):
        ratio = g[
            "qaoa_vs_milp_end_to_end_ratio"
        ].to_numpy(float)

        rlo, rhi = bootstrap_mean_ci(
            ratio,
            bootstrap,
            SEED
            + len(
                summary_rows
            ),
        )

        solver_ratio = g[
            "qaoa_vs_milp_solver_ratio"
        ].to_numpy(float)

        slo, shi = bootstrap_mean_ci(
            solver_ratio,
            bootstrap,
            SEED
            + 100
            + len(
                summary_rows
            ),
        )

        summary_rows.append(
            {
                "config": config,
                "blocks": len(
                    g
                ),
                "success_rate": float(
                    g[
                        "qaoa_success_rate"
                    ].mean()
                ),
                "raw_feasibility_rate": float(
                    g[
                        "qaoa_raw_feasibility_rate"
                    ].mean()
                ),
                "optimal_match_rate": float(
                    g[
                        "qaoa_optimal_match_rate"
                    ].mean()
                ),
                "mean_objective_gap": float(
                    g[
                        "qaoa_objective_gap"
                    ].mean()
                ),
                "mean_qaoa_solver_seconds": float(
                    g[
                        "qaoa_solver_seconds"
                    ].mean()
                ),
                "mean_qaoa_end_to_end_seconds": float(
                    g[
                        "qaoa_end_to_end_seconds"
                    ].mean()
                ),
                "mean_milp_solver_seconds": float(
                    g[
                        "milp_solver_seconds"
                    ].mean()
                ),
                "mean_milp_end_to_end_seconds": float(
                    g[
                        "milp_end_to_end_seconds"
                    ].mean()
                ),
                "mean_risk_greedy_seconds": float(
                    g[
                        "risk_greedy_seconds"
                    ].mean()
                ),
                "mean_least_loaded_seconds": float(
                    g[
                        "least_loaded_seconds"
                    ].mean()
                ),
                "mean_qaoa_vs_milp_solver_ratio": float(
                    solver_ratio.mean()
                ),
                "bootstrap95_qaoa_vs_milp_solver_ratio_low": slo,
                "bootstrap95_qaoa_vs_milp_solver_ratio_high": shi,
                "mean_qaoa_vs_milp_end_to_end_ratio": float(
                    ratio.mean()
                ),
                "bootstrap95_qaoa_vs_milp_end_to_end_ratio_low": rlo,
                "bootstrap95_qaoa_vs_milp_end_to_end_ratio_high": rhi,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        out_dir
        / "qaoa_runtime_summary.csv",
        index=False,
    )

    stats = paired_runtime_statistics(
        block_level,
        bootstrap,
    )

    stats.to_csv(
        out_dir
        / "qaoa_runtime_paired_statistics.csv",
        index=False,
    )

    reliable = summary[
        (summary[
            "success_rate"
        ] >= 0.95)
        & (
            summary[
                "raw_feasibility_rate"
            ] >= 0.95
        )
        & (
            summary[
                "optimal_match_rate"
            ] >= 0.95
        )
    ].copy()

    best = None

    if not reliable.empty:
        best = (
            reliable
            .sort_values(
                "mean_qaoa_end_to_end_seconds"
            )
            .iloc[
                0
            ]
            .to_dict()
        )

    time_advantage = bool(
        best is not None
        and best[
            "mean_qaoa_vs_milp_end_to_end_ratio"
        ] < 1.0
        and best[
            "bootstrap95_qaoa_vs_milp_end_to_end_ratio_high"
        ] < 1.0
    )

    if time_advantage:
        conclusion = (
            "A warm-process QAOA runtime advantage over MILP was observed "
            "under the strict reliability and paired-bootstrap criterion. "
            "This remains simulator-based software timing and is not a "
            "physical quantum-speedup claim."
        )
    else:
        conclusion = (
            "No QAOA runtime advantage over MILP was observed under the "
            "strict reliability and paired-bootstrap criterion. The final "
            "paper should report QAOA as reliable solution recovery for the "
            "compact benchmark, not as a runtime or quantum-speedup result."
        )

    old_cold_path = (
        run_dir
        / "06_compact_qaoa"
        / "qaoa_compact_qubo"
        / "qaoa_execution_summary_compact.csv"
    )

    cold_reference = None

    if old_cold_path.exists():
        try:
            old = pd.read_csv(
                old_cold_path
            )

            successful = old[
                old[
                    "success"
                ].fillna(
                    False
                ).astype(
                    bool
                )
            ]

            if (
                not successful.empty
                and "wall_seconds"
                in successful.columns
            ):
                cold_reference = {
                    "successful_runs": int(
                        len(
                            successful
                        )
                    ),
                    "mean_cold_process_wall_seconds": float(
                        successful[
                            "wall_seconds"
                        ].mean()
                    ),
                    "median_cold_process_wall_seconds": float(
                        successful[
                            "wall_seconds"
                        ].median()
                    ),
                }
        except Exception:
            cold_reference = None

    status = {
        "completed": True,
        "source_run": str(
            run_dir
        ),
        "output": str(
            out_dir
        ),
        "saved_final_qaoa_blocks": len(
            blocks
        ),
        "repeats_per_block": repeats,
        "warmup_seconds_excluded": warmup_seconds,
        "warmup_success": bool(
            warm_result.get(
                "success",
                False,
            )
        ),
        "configurations": configurations,
        "best_reliable_configuration": best,
        "cold_process_reference": cold_reference,
        "qaoa_end_to_end_time_advantage_over_milp": time_advantage,
        "conclusion": conclusion,
        "decision_rule": (
            "A time advantage is accepted only when a configuration has "
            ">=95% success, raw feasibility, and exact MILP-objective match, "
            "and the paired block-bootstrap 95% CI for QAOA/MILP end-to-end "
            "runtime ratio lies completely below 1."
        ),
        "scientific_boundary": (
            "All QAOA measurements are simulator/software timings. They do "
            "not demonstrate physical quantum computational advantage."
        ),
    }

    save_json(
        status,
        out_dir
        / "qaoa_runtime_fairness_status.json",
    )

    make_figures(
        summary,
        out_dir,
    )

    print()
    print(
        "AQUA-SLA QAOA RUNTIME FAIRNESS AUDIT FINISHED."
    )
    print(
        "Output:",
        out_dir,
    )
    print(
        json.dumps(
            status,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
