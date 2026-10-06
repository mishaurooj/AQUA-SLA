from __future__ import annotations

r"""
AQUA-SLA FINAL RESUME + iQUANTUM PATH FIX + QUANTUM-VALUE AUDIT
===============================================================

This targeted finalizer is for an already completed AQUA-SLA final run where:
  * lifecycle-censored prediction is complete,
  * HCQKL is trained,
  * heterogeneous capacity scheduling is complete,
  * compact QAOA is repaired/completed,
  * the remaining integrated failure is iQuantum path length on Windows.

It does NOT retrain the predictor and does NOT rerun the completed scheduling
or QAOA studies.

It performs four final tasks:

1) Runs calibrated iQuantum V4 in a SHORT Windows staging path, avoiding the
   MAX_PATH failure caused by deeply nested QASM directories.

2) Preserves the complete raw iQuantum staging result as a ZIP inside the
   final AQUA-SLA run, while copying paper-facing summaries/figures into a
   shallow final directory.

3) Runs a predeclared matched-kernel quantum-value audit using the already
   saved untouched temporal test predictions:
       Quantum fidelity kernel vs matched RBF kernel
       HCQKL vs matched RBF kernel
   using paired bootstrap CIs for ROC-AUC and PR-AUC.
   It DOES NOT alter alpha or choose a metric after seeing the result.

4) Regenerates the final paper figures, QAOA/HCQKL circuit figures, QPY/QASM
   artifacts, final result index, and final freeze status.

IMPORTANT SCIENTIFIC BOUNDARY
-----------------------------
This script will not manufacture or guarantee a "quantum advantage".

The quantum-value criterion is fixed before analysis:
  A matched predictive quantum advantage is supported only if BOTH
  ROC-AUC and PR-AUC improvements of the quantum fidelity kernel over
  the matched RBF kernel have paired-bootstrap 95% CIs strictly above zero.

Even if that criterion is met, the claim is:
  "predictive advantage over the matched RBF kernel on this evaluation"

It is NOT:
  "quantum computational advantage" or "quantum speedup".

The already completed QAOA runtime audit remains authoritative for runtime.

RECOMMENDED RUN
---------------
conda activate aqua-sla
cd /d E:\other\AQUA-SLA\Code

python 37_aqua_sla_resume_iquantum_finalize_quantum_value.py ^
  --run-dir "E:\other\AQUA-SLA\results\aqua_sla_final_reviewer_complete\20261005_150107_final_reviewer_complete_fixed" ^
  --stage-root "E:\AQUA_IQ4_TMP" ^
  --bootstrap 5000

SMOKE TEST
----------
python 37_aqua_sla_resume_iquantum_finalize_quantum_value.py ^
  --run-dir "E:\other\AQUA-SLA\results\aqua_sla_final_reviewer_complete\20261005_150107_final_reviewer_complete_fixed" ^
  --stage-root "E:\AQUA_IQ4_TMP" ^
  --bootstrap 500 ^
  --fast

OUTPUTS ADDED TO EXISTING FINAL RUN
-----------------------------------
08_iquantum_calibrated_v4\
    final_iq4\
        final_status.json
        protocol.json
        primary_policy_summary.csv
        policy_uncertainty.csv
        risk_aware_vs_baselines.csv
        iquantum_v4_run_summary.csv
        ...
    iquantum_v4_complete_raw.zip

10_figures\
    final AQUA-SLA figures
    copied iQuantum figures

12_updated_quantum_circuit_figures\
14_quantum_circuit_artifacts\

17_quantum_value_audit\
    matched_kernel_quantum_value_summary.csv
    matched_kernel_quantum_value_bootstrap.csv
    quantum_value_status.json

00_manifest\
    FINAL_RESULTS_INDEX.csv
    final_status.json
    FINAL_FREEZE_STATUS.json
"""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# =============================================================================
# Utilities
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
        if isinstance(x, (list, tuple, set)):
            return [conv(v) for v in x]
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(conv(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def import_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_master_script(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    here = Path(__file__).resolve().parent
    candidates = [
        here / "35_aqua_sla_final_reviewer_complete.py",
        Path(r"E:\other\AQUA-SLA\Code\35_aqua_sla_final_reviewer_complete.py"),
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not locate 35_aqua_sla_final_reviewer_complete.py. "
        "Place script 37 in the same Code directory or pass --master-script."
    )


def shortest_relative_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        parts = list(rel.parts)
    except Exception:
        parts = [path.name]

    if len(parts) <= 1:
        return path.name

    # Shallow evidence names while retaining origin context.
    parent = "_".join(parts[-3:-1])
    if parent:
        return f"{parent}__{path.name}"
    return path.name


# =============================================================================
# iQuantum path-safe staging and final packaging
# =============================================================================

IQ_KEY_FILES = [
    "final_status.json",
    "protocol.json",
    "primary_policy_summary.csv",
    "policy_uncertainty.csv",
    "risk_aware_vs_baselines.csv",
    "cell_metric_winners.csv",
    "cell_policy_ranks.csv",
    "policy_rank_summary.csv",
    "risk_complexity_independence.csv",
    "iquantum_v4_run_summary.csv",
    "slo_calibration_summary.csv",
    "slo_calibration.json",
    "calibration_run_status.csv",
]


def verify_iquantum_run(iq_run: Path) -> dict[str, Any]:
    status_candidates = list(iq_run.rglob("final_status.json"))
    if not status_candidates:
        raise RuntimeError(
            f"iQuantum staging run contains no final_status.json: {iq_run}"
        )

    # Prefer top-level status.
    status_path = iq_run / "final_status.json"
    if not status_path.exists():
        status_path = min(status_candidates, key=lambda p: len(p.parts))

    status = load_json(status_path, {})
    if not status:
        raise RuntimeError(f"Could not read iQuantum final status: {status_path}")

    if status.get("completed") is False:
        raise RuntimeError(
            "iQuantum staging run finished with completed=false:\n"
            + json.dumps(status, indent=2)
        )

    # Strong explicit check when the helper exposes these values.
    expected = status.get("expected_simulations")
    successful = status.get("successful_simulations")
    failed = status.get("failed_simulations")

    if expected is not None and successful is not None:
        if int(successful) != int(expected):
            raise RuntimeError(
                f"iQuantum evaluation incomplete: {successful}/{expected} successful."
            )

    if failed is not None and int(failed) != 0:
        raise RuntimeError(f"iQuantum reports {failed} failed simulations.")

    summary_paths = list(iq_run.rglob("iquantum_v4_run_summary.csv"))
    if summary_paths:
        summary = pd.read_csv(summary_paths[0], low_memory=False)
        if len(summary) < 80:
            raise RuntimeError(
                f"Expected at least 80 final iQuantum evaluation rows; found {len(summary)}."
            )
        if "success" in summary.columns:
            success = (
                summary["success"]
                .fillna(False)
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            if int(success.sum()) < 80:
                raise RuntimeError(
                    f"Only {int(success.sum())} successful iQuantum evaluation rows."
                )

    return {
        "status_path": str(status_path),
        "status": status,
        "verified_complete": True,
    }


def package_iquantum_run(
    iq_run: Path,
    final_iq_root: Path,
    figures_root: Path,
) -> Path:
    final_iq_root.mkdir(parents=True, exist_ok=True)

    shallow = final_iq_root / "final_iq4"
    if shallow.exists():
        shutil.rmtree(shallow)
    shallow.mkdir(parents=True, exist_ok=True)

    # Copy important paper-facing evidence into a shallow directory.
    copied = []

    for filename in IQ_KEY_FILES:
        matches = list(iq_run.rglob(filename))
        for src in matches:
            dst_name = filename

            # Avoid collisions when calibration + top-level use same names.
            dst = shallow / dst_name
            if dst.exists():
                dst_name = shortest_relative_name(src, iq_run)
                dst = shallow / dst_name

            shutil.copy2(src, dst)
            copied.append(
                {
                    "source": str(src),
                    "destination": str(dst),
                }
            )

    # Copy all generated iQuantum plots shallowly.
    figures_root.mkdir(parents=True, exist_ok=True)
    fig_manifest = []

    for ext in ("*.png", "*.jpeg", "*.jpg", "*.pdf"):
        for src in iq_run.rglob(ext):
            if "qasm" in {x.lower() for x in src.parts}:
                continue

            safe_name = "iquantum_" + shortest_relative_name(src, iq_run)
            dst = figures_root / safe_name

            # Very defensive collision handling.
            k = 1
            base = dst.stem
            while dst.exists():
                dst = dst.with_name(f"{base}_{k}{dst.suffix}")
                k += 1

            shutil.copy2(src, dst)
            fig_manifest.append(
                {
                    "source": str(src),
                    "destination": str(dst),
                }
            )

    pd.DataFrame(copied).to_csv(
        shallow / "paper_evidence_copy_manifest.csv",
        index=False,
    )
    pd.DataFrame(fig_manifest).to_csv(
        shallow / "figure_copy_manifest.csv",
        index=False,
    )

    # Preserve every raw result without expanding deep paths inside the final run.
    archive_base = final_iq_root / "iquantum_v4_complete_raw"
    archive_path = archive_base.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()

    shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=str(iq_run.parent),
        base_dir=iq_run.name,
    )

    if not archive_path.exists():
        raise RuntimeError("Failed to create complete iQuantum raw ZIP archive.")

    return shallow


def run_iquantum_short_path(
    master,
    out,
    helpers: dict[str, Path],
    stage_root: Path,
    bootstrap: int,
    fast: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    stage_root = stage_root.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)

    # Keep path deliberately very short.
    helper32 = master.load_module(
        helpers["iquantum_v4"],
        "aqua_sla_iq4_short_stage",
    )
    helper32.OUTPUT_ROOT = stage_root

    args = argparse.Namespace(
        source_run=str(out.root),
        run_name="iq4",
        bootstrap=min(1000, bootstrap) if fast else bootstrap,
        slo_quantile=0.90,
        fast=fast,
    )

    start = time.perf_counter()
    iq_run = helper32.run_experiment(args)
    elapsed = time.perf_counter() - start

    iq_run = Path(iq_run).resolve()

    verification = verify_iquantum_run(iq_run)
    verification["elapsed_seconds"] = elapsed
    verification["staging_root"] = str(stage_root)
    verification["staging_run"] = str(iq_run)

    shallow = package_iquantum_run(
        iq_run=iq_run,
        final_iq_root=out.iquantum,
        figures_root=out.figures,
    )

    verification["final_shallow_evidence"] = str(shallow)
    verification["raw_zip"] = str(
        out.iquantum / "iquantum_v4_complete_raw.zip"
    )

    save_json(
        verification,
        shallow / "path_safe_resume_verification.json",
    )

    return iq_run, shallow, verification


# =============================================================================
# Predeclared matched-kernel quantum-value audit
# =============================================================================

def paired_bootstrap_metric_difference(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: str,
    resamples: int,
    seed: int,
) -> tuple[float, float, float, pd.DataFrame]:
    y = np.asarray(y, int)
    p_a = np.asarray(p_a, float)
    p_b = np.asarray(p_b, float)

    if metric == "roc_auc":
        fn = roc_auc_score
        observed = float(fn(y, p_a) - fn(y, p_b))
    elif metric == "pr_auc":
        fn = average_precision_score
        observed = float(fn(y, p_a) - fn(y, p_b))
    elif metric == "brier_improvement":
        # Positive means A has LOWER/better Brier than B.
        observed = float(
            brier_score_loss(y, p_b)
            - brier_score_loss(y, p_a)
        )
        fn = None
    else:
        raise ValueError(metric)

    rng = np.random.default_rng(seed)
    diffs = []

    # Class-preserving paired bootstrap. Rejection only when a rare resample
    # contains one class, preventing undefined AUC values.
    attempts = 0
    max_attempts = max(resamples * 10, resamples + 100)

    while len(diffs) < resamples and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, len(y), size=len(y))
        ys = y[idx]

        if len(np.unique(ys)) < 2:
            continue

        aa = p_a[idx]
        bb = p_b[idx]

        if metric == "brier_improvement":
            diff = (
                brier_score_loss(ys, bb)
                - brier_score_loss(ys, aa)
            )
        else:
            diff = fn(ys, aa) - fn(ys, bb)

        diffs.append(float(diff))

    if len(diffs) < max(100, resamples // 2):
        raise RuntimeError(
            f"Too few valid bootstrap resamples for {metric}: {len(diffs)}"
        )

    arr = np.asarray(diffs, float)
    lo = float(np.quantile(arr, 0.025))
    hi = float(np.quantile(arr, 0.975))

    sample_df = pd.DataFrame(
        {
            "metric": metric,
            "difference_a_minus_b": arr,
        }
    )

    return observed, lo, hi, sample_df


def quantum_value_audit(
    out,
    bootstrap: int,
    fast: bool,
) -> dict[str, Any]:
    audit_dir = out.root / "17_quantum_value_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    pred_candidates = [
        out.prediction / "matched_kernel_test_predictions_strict_subset.csv",
        out.prediction / "matched_kernel_test_predictions.csv",
    ]

    pred_path = next((p for p in pred_candidates if p.exists()), None)
    if pred_path is None:
        raise FileNotFoundError(
            "Could not locate matched-kernel test predictions for quantum-value audit."
        )

    df = pd.read_csv(pred_path, low_memory=False)

    required = [
        "future_sla_event",
        "p_RBF_kernel",
        "p_Quantum_fidelity_kernel",
        "p_HCQKL",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Matched-kernel prediction file is missing columns: {missing}"
        )

    y = df["future_sla_event"].to_numpy(int)

    comparisons = [
        (
            "Quantum_fidelity_vs_RBF",
            df["p_Quantum_fidelity_kernel"].to_numpy(float),
            df["p_RBF_kernel"].to_numpy(float),
        ),
        (
            "HCQKL_vs_RBF",
            df["p_HCQKL"].to_numpy(float),
            df["p_RBF_kernel"].to_numpy(float),
        ),
    ]

    resamples = min(500, bootstrap) if fast else bootstrap

    summary_rows = []
    boot_parts = []

    for comp_idx, (name, a, b) in enumerate(comparisons):
        for metric_idx, metric in enumerate(
            ["roc_auc", "pr_auc", "brier_improvement"]
        ):
            observed, lo, hi, samples = paired_bootstrap_metric_difference(
                y=y,
                p_a=a,
                p_b=b,
                metric=metric,
                resamples=resamples,
                seed=42 + comp_idx * 100 + metric_idx,
            )

            samples.insert(0, "comparison", name)
            boot_parts.append(samples)

            summary_rows.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "difference_definition": (
                        "A_minus_B"
                        if metric != "brier_improvement"
                        else "Brier_B_minus_Brier_A_positive_means_A_better"
                    ),
                    "observed_difference": observed,
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "ci_strictly_positive": bool(lo > 0.0),
                    "test_rows": len(df),
                    "positives": int(y.sum()),
                    "prevalence": float(y.mean()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        audit_dir / "matched_kernel_quantum_value_summary.csv",
        index=False,
    )

    if boot_parts:
        pd.concat(boot_parts, ignore_index=True).to_csv(
            audit_dir / "matched_kernel_quantum_value_bootstrap.csv",
            index=False,
        )

    q = summary[
        summary["comparison"] == "Quantum_fidelity_vs_RBF"
    ].set_index("metric")

    # Locked criterion: BOTH primary ranking metrics must improve with
    # paired-bootstrap 95% lower bounds > 0.
    predictive_advantage = bool(
        "roc_auc" in q.index
        and "pr_auc" in q.index
        and bool(q.loc["roc_auc", "ci_strictly_positive"])
        and bool(q.loc["pr_auc", "ci_strictly_positive"])
    )

    alpha_path = out.prediction / "hcqkl_alpha_sweep.csv"
    best_alpha = None

    if alpha_path.exists():
        alpha = pd.read_csv(alpha_path)
        score_col = next(
            (
                c
                for c in [
                    "mean_selection_score",
                    "selection_score",
                ]
                if c in alpha.columns
            ),
            None,
        )

        if score_col and "alpha_rbf" in alpha.columns:
            row = alpha.loc[alpha[score_col].idxmax()]
            best_alpha = {
                "alpha_rbf": float(row["alpha_rbf"]),
                "alpha_quantum": float(
                    row.get(
                        "alpha_quantum",
                        1.0 - float(row["alpha_rbf"]),
                    )
                ),
                "selection_score": float(row[score_col]),
            }

    status = {
        "completed": True,
        "prediction_file": str(pred_path),
        "bootstrap_resamples": resamples,
        "primary_predeclared_comparison": "Quantum_fidelity_vs_RBF",
        "primary_metrics": ["roc_auc", "pr_auc"],
        "criterion": (
            "Both paired-bootstrap 95% confidence intervals for "
            "Quantum_fidelity minus matched RBF must lie strictly above zero."
        ),
        "matched_quantum_predictive_advantage_supported": predictive_advantage,
        "selected_hcqkl_alpha": best_alpha,
        "allowed_claim_if_true": (
            "The quantum fidelity kernel showed a predictive ranking advantage "
            "over the matched RBF kernel on the predefined temporal evaluation."
        ),
        "prohibited_claim": (
            "This result does not establish quantum computational advantage, "
            "hardware speedup, or superiority over the strongest classical ML baseline."
        ),
    }

    if not predictive_advantage:
        status["interpretation"] = (
            "The strict matched-kernel predictive-advantage criterion was not met. "
            "Report the observed matched-kernel differences without claiming quantum advantage."
        )
    else:
        status["interpretation"] = (
            "The strict matched-kernel predictive-advantage criterion was met. "
            "The paper may claim incremental predictive value relative to the matched RBF kernel, "
            "while explicitly stating that this is not computational quantum advantage."
        )

    save_json(
        status,
        audit_dir / "quantum_value_status.json",
    )

    return status


# =============================================================================
# Final figures, circuits, index, freeze status
# =============================================================================

def regenerate_final_artifacts(master, out, helpers: dict[str, Path]) -> dict[str, Any]:
    # Regenerate general final figures from existing final evidence.
    audit = pd.read_csv(
        out.target / "global_vs_task_lifecycle_audit.csv"
    )
    baseline = pd.read_csv(
        out.prediction / "classical_baseline_metrics.csv"
    )
    kernel = pd.read_csv(
        out.prediction / "matched_kernel_test_metrics.csv"
    )
    scheduling_summary = pd.read_csv(
        out.scheduling / "capacity_scheduling_summary.csv"
    )
    alpha = pd.read_csv(
        out.prediction / "hcqkl_alpha_sweep.csv"
    )

    master.make_final_figures(
        audit,
        baseline,
        kernel,
        scheduling_summary,
        alpha,
        out,
    )

    # Circuit figures from repaired QAOA + final HCQKL.
    circuit_log = out.logs / "quantum_circuit_generation_final_resume.log"

    cmd = [
        sys.executable,
        str(helpers["circuits"]),
        "--run-dir",
        str(out.root),
        "--output-dir",
        str(out.circuits),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    circuit_log.write_text(
        (
            "COMMAND:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + proc.stdout
            + "\n\nSTDERR:\n"
            + proc.stderr
        ),
        encoding="utf-8",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            "Final quantum-circuit figure regeneration failed. "
            f"Inspect {circuit_log}"
        )

    helper33 = master.load_module(
        helpers["circuits"],
        "aqua_sla_final_circuit_artifact_module",
    )

    circuit_manifest = master.save_quantum_circuit_artifacts(
        out,
        helper33,
    )

    return {
        "general_figures": len(list(out.figures.glob("*"))),
        "circuit_rendered_files": len(list(out.circuits.glob("*"))),
        "circuit_objects": int(len(circuit_manifest)),
    }


def latest_runtime_audit_status(run_dir: Path) -> dict[str, Any] | None:
    root = run_dir / "16_qaoa_runtime_fairness_audit"
    if not root.exists():
        return None

    candidates = list(root.rglob("qaoa_runtime_fairness_status.json"))
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    status = load_json(latest, None)

    if status is not None:
        status["_status_path"] = str(latest)

    return status


def update_final_status_and_freeze(
    master,
    out,
    final_iq: Path,
    iq_verification: dict[str, Any],
    quantum_value: dict[str, Any],
    artifact_status: dict[str, Any],
) -> dict[str, Any]:
    status_path = out.manifest / "final_status.json"
    old = load_json(status_path, {}) or {}

    previous_error = old.get("error") or old.get("fatal_error")

    stage_path = out.manifest / "stage_status.json"
    stages = load_json(stage_path, {}) or {}

    stages["iquantum_calibrated_v4"] = "completed_path_safe_resume"
    stages["quantum_value_audit"] = "completed"
    stages["final_figures"] = "completed"
    stages["quantum_circuit_figures"] = "completed_finalized"
    stages["quantum_circuit_artifacts"] = "completed_finalized"
    stages["final_index"] = "completed"
    stages.pop("fatal_error", None)

    save_json(stages, stage_path)

    runtime_status = latest_runtime_audit_status(out.root)

    final = dict(old)
    final.update(
        {
            "completed": True,
            "error": None,
            "fatal_error": None,
            "output": str(out.root),
            "resume_finalization_completed": True,
            "resume_finalization_time": datetime.now().isoformat(),
            "previous_integrated_error_preserved_for_audit": previous_error,
            "iquantum_output": str(final_iq),
            "iquantum_path_safe_resume": iq_verification,
            "quantum_value_audit": quantum_value,
            "qaoa_runtime_fairness_audit": runtime_status,
            "artifact_finalization": artifact_status,
            "stage_status": stages,
            "claim_boundaries": [
                "The future disruption target is project-defined and is not a Google contractual SLA field.",
                "Validation/test prediction metrics use natural temporal prevalence.",
                "HCQKL performs kernel-level RBF/quantum blending with alpha selected by temporal validation.",
                "Any matched-kernel quantum predictive advantage is relative to the matched RBF kernel only.",
                "The strongest classical ML baseline remains a separate benchmark and must not be hidden.",
                "External scheduling SLO outcomes are evaluated independently after assignment.",
                "Raw and repaired QAOA are reported separately.",
                "The QAOA runtime-fairness audit determines whether any runtime advantage exists; it must not be overridden.",
                "Qiskit quantum-kernel/QAOA results are simulator based.",
                "iQuantum is discrete-event simulation, not physical quantum hardware.",
            ],
        }
    )

    save_json(final, status_path)

    # Refresh reviewer coverage and result index.
    master.reviewer_coverage(out)
    master.write_result_index(out, final_iq)

    # Freeze checklist.
    required = {
        "target_metadata": out.target / "future_target_metadata.json",
        "classical_metrics": out.prediction / "classical_baseline_metrics.csv",
        "kernel_metrics": out.prediction / "matched_kernel_test_metrics.csv",
        "alpha_sweep": out.prediction / "hcqkl_alpha_sweep.csv",
        "scheduling_summary": out.scheduling / "capacity_scheduling_summary.csv",
        "scheduling_statistics": out.statistics / "capacity_scheduling_paired_statistics.csv",
        "qaoa_reliability": (
            out.qaoa
            / "qaoa_compact_qubo"
            / "qaoa_reliability_summary.json"
        ),
        "iquantum_status": final_iq / "final_status.json",
        "quantum_value_status": (
            out.root
            / "17_quantum_value_audit"
            / "quantum_value_status.json"
        ),
        "circuit_manifest": (
            out.circuit_artifacts / "quantum_circuit_manifest.csv"
        ),
        "result_index": out.manifest / "FINAL_RESULTS_INDEX.csv",
    }

    checks = [
        {
            "artifact": k,
            "path": str(v),
            "exists": v.exists(),
        }
        for k, v in required.items()
    ]

    all_present = all(x["exists"] for x in checks)

    freeze = {
        "experiment_frozen": bool(all_present),
        "created": datetime.now().isoformat(),
        "run": str(out.root),
        "all_required_artifacts_present": all_present,
        "checks": checks,
        "iquantum_verified_complete": bool(
            iq_verification.get("verified_complete", False)
        ),
        "quantum_value_result": quantum_value,
        "qaoa_runtime_result": runtime_status,
        "next_action": (
            "Freeze experiments and rewrite the manuscript from this result package."
            if all_present
            else "Do not freeze; one or more required final artifacts are missing."
        ),
    }

    save_json(
        freeze,
        out.manifest / "FINAL_FREEZE_STATUS.json",
    )

    return freeze


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume iQuantum through a short Windows staging path, package "
            "the completed V4 evidence, audit matched-kernel quantum value, "
            "and finalize the AQUA-SLA result folder."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing final AQUA-SLA run to resume/finalize.",
    )

    parser.add_argument(
        "--stage-root",
        default=r"E:\AQUA_IQ4_TMP",
        help="SHORT temporary root used only for iQuantum execution.",
    )

    parser.add_argument(
        "--master-script",
        default=None,
        help="Optional path to 35_aqua_sla_final_reviewer_complete.py",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--fast",
        action="store_true",
    )

    parser.add_argument(
        "--delete-staging-after-zip",
        action="store_true",
        help=(
            "Delete the short-path raw staging result only after verification "
            "and successful ZIP packaging into the final run."
        ),
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    master_script = find_master_script(args.master_script)
    master = import_module_from_path(
        master_script,
        "aqua_sla_master_finalizer",
    )

    out = master.Paths.from_existing(run_dir)
    helpers = master.write_embedded_helpers(out)

    resume_manifest = {
        "project": "AQUA-SLA",
        "experiment": "path-safe-iQuantum-resume-finalize-quantum-value",
        "created": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "run": str(run_dir),
        "master_script": str(master_script),
        "stage_root": str(Path(args.stage_root).resolve()),
        "bootstrap": args.bootstrap,
        "fast": bool(args.fast),
    }

    save_json(
        resume_manifest,
        out.manifest / "resume_finalization_manifest.json",
    )

    print("1/4 Running iQuantum V4 from short staging path...")

    iq_run, final_iq, iq_verification = run_iquantum_short_path(
        master=master,
        out=out,
        helpers=helpers,
        stage_root=Path(args.stage_root),
        bootstrap=args.bootstrap,
        fast=args.fast,
    )

    print("2/4 Running predeclared matched-kernel quantum-value audit...")

    quantum_value = quantum_value_audit(
        out=out,
        bootstrap=args.bootstrap,
        fast=args.fast,
    )

    print("3/4 Regenerating final figures and quantum circuit artifacts...")

    artifact_status = regenerate_final_artifacts(
        master=master,
        out=out,
        helpers=helpers,
    )

    print("4/4 Updating final status, evidence index, and freeze checklist...")

    freeze = update_final_status_and_freeze(
        master=master,
        out=out,
        final_iq=final_iq,
        iq_verification=iq_verification,
        quantum_value=quantum_value,
        artifact_status=artifact_status,
    )

    if args.delete_staging_after_zip:
        raw_zip = out.iquantum / "iquantum_v4_complete_raw.zip"
        if (
            raw_zip.exists()
            and iq_verification.get("verified_complete")
            and freeze.get("all_required_artifacts_present")
        ):
            shutil.rmtree(iq_run)
            print("Deleted verified staging run:", iq_run)

    result = {
        "completed": True,
        "final_run": str(out.root),
        "iquantum_final_evidence": str(final_iq),
        "iquantum_raw_zip": str(
            out.iquantum / "iquantum_v4_complete_raw.zip"
        ),
        "quantum_value_audit": quantum_value,
        "artifact_status": artifact_status,
        "freeze_status": freeze,
    }

    print()
    print("AQUA-SLA FINAL RESUME/FINALIZATION FINISHED.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
