
#!/usr/bin/env python3
"""
AQUA-SLA Script 53
Multi-seed prediction optimization and reproducible evaluation.

Original Script 51 remains unchanged.
All model selection uses validation data.
Test scores are used for reporting, not model selection.

Requirements:
    numpy pandas scipy scikit-learn
    matplotlib joblib xgboost jinja2

Run:
    python 53_aqua_sla_multiseed_optimization.py
"""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import inspect
import json
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import sklearn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "Code"
DATA = ROOT / "Dataset" / "processed"
OUTPUT = ROOT / "results" / "53_multiseed_optimization"

DEFAULT_SEEDS = [5101, 5201, 5301, 5401, 5501]


def log(message):
    print(f"[53] {message}", flush=True)


def dump(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            default=lambda x: (
                x.item()
                if isinstance(x, np.generic)
                else str(x)
            )
        ),
        encoding="utf-8"
    )


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def locate_script51():

    files = sorted(
        CODE.glob("51_aqua_sla_joint_new_ablations*.py")
    )

    if not files:
        raise FileNotFoundError(
            "Script 51 not found in the Code directory."
        )

    return files[0]


def load_script51():

    source = locate_script51()

    spec = importlib.util.spec_from_file_location(
        "aqua51", source
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def replace_checked(source, old, new):

    if source.count(old) != 1:
        raise RuntimeError(
            "Script 51 source differs from expected version. "
            f"Cannot safely apply patch:\n{old}"
        )

    return source.replace(old, new, 1)


def instrument_prediction(module):

    source = inspect.getsource(module.prediction)

    # Fixed evaluation population across all training seeds.
    source = replace_checked(
        source,
        "rng=np.random.default_rng(a.seed)",
        """rng=np.random.default_rng(a.seed)
    split_rng=np.random.default_rng(a.split_seed)"""
    )

    source = replace_checked(
        source,
        "ids=np.sort(rng.choice(len(df),n,replace=False))",
        "ids=np.sort(split_rng.choice(len(df),n,replace=False))"
    )

    # Save fitted preprocessing transformations.
    source = replace_checked(
        source,
        "Z,ZA,ZB,ZT=[emb(x) for x in (X,A,B,T)]",
        """Z,ZA,ZB,ZT=[emb(x) for x in (X,A,B,T)]

    modeldir=out/'trained_models'
    modeldir.mkdir(parents=True,exist_ok=True)

    joblib.dump({
        'imputer':imp,
        'robust_scaler':robust,
        'pca':pca,
        'quantile_transformer':transform,
        'features':list(features)
    },modeldir/'preprocessing.joblib')

    for split_name,frame in [
        ('train',t0),('valA',v1),
        ('valB',v2),('test',ext)
    ]:
        pd.DataFrame({
            'source_row':frame._source_row.to_numpy(),
            'time_seconds':frame.time_seconds.to_numpy(),
            'y_true':frame.future_sla_event.to_numpy()
        }).to_csv(
            out/f'partition_{split_name}.csv',
            index=False
        )"""
    )

    # Save all baseline estimators, including XGBoost.
    source = replace_checked(
        source,
        "tic=time.perf_counter();model.fit(X,ytr)",
        """tic=time.perf_counter();model.fit(X,ytr)
        modeldir=out/'trained_models'
        modeldir.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,modeldir/f'baseline_{name}.joblib')"""
    )

    # Save every trained kernel configuration.
    source = replace_checked(
        source,
        "mdl.fit(np.column_stack([X,kt]),ytr)",
        """mdl.fit(np.column_stack([X,kt]),ytr)

            joblib.dump({
                'model':mdl,
                'landmark_indices':ids,
                'landmark_features':Z[ids],
                'landmark_states':sr[kind][0][ids],
                'kind':kind,
                'alpha':alpha,
                'gamma':gamma,
                'kernel_name':cname,
                'bandwidth':theta
            },modeldir/f'{item}.joblib')"""
    )

    # Export actual ROC-AUC and PR-AUC for baselines.
    source = replace_checked(
        source,
        "# Strong baseline selected on VA.",
        """pd.DataFrame([
        {
            'model':name,
            'valA_roc_auc':roc_auc_score(y1,rec['a']),
            'valA_pr_auc':average_precision_score(y1,rec['a']),
            'selection_objective':score(y1,rec['a'])
        }
        for name,rec in baseline.items()
    ]).to_csv(
        out/'53_baseline_validationA.csv',
        index=False
    )

    # Strong baseline selected on VA."""
    )

    # Save Validation-B probabilities for ensemble selection.
    source = replace_checked(
        source,
        "for name,rec in baseline.items():vbpred['classical_'+name]=rec['b']",
        """for name,rec in baseline.items():vbpred['classical_'+name]=rec['b']

    pd.DataFrame({
        'source_row':v2._source_row.to_numpy(),
        'time_seconds':v2.time_seconds.to_numpy(),
        'y_true':y2,
        **vbpred
    }).to_csv(
        out/'53_validationB_scores.csv',
        index=False
    )"""
    )

    environment = module.__dict__.copy()
    environment["joblib"] = joblib

    exec(
        compile(
            source,
            "<instrumented_script51>",
            "exec"
        ),
        environment
    )

    return environment["prediction"]


def selection_score(y, p):

    return float(
        roc_auc_score(y, p)
        + 0.20 * average_precision_score(y, p)
    )


def choose_threshold(y, p):

    candidates = np.unique(
        np.r_[
            0.0,
            np.quantile(
                p,
                np.linspace(0.01, 0.99, 150)
            ),
            1.0
        ]
    )

    return float(
        max(
            candidates,
            key=lambda t: f1_score(
                y,
                p >= t,
                zero_division=0
            )
        )
    )


def metrics(y, p, threshold):

    yhat = (p >= threshold).astype(int)

    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(
            average_precision_score(y, p)
        ),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, yhat)
        ),
        "precision": float(
            precision_score(
                y, yhat, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y, yhat, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y, yhat, zero_division=0
            )
        ),
        "mcc": float(
            matthews_corrcoef(y, yhat)
        ),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold)
    }


def export(df, name, output):

    csv_path = output / "tables" / f"{name}.csv"
    tex_path = output / "latex" / f"{name}.tex"

    df.to_csv(csv_path, index=False)

    latex = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.4f}"
    )

    tex_path.write_text(
        "\\begingroup\n"
        "\\color{blue}\n"
        "\\scriptsize\n"
        + latex
        + "\n\\endgroup\n",
        encoding="utf-8"
    )

    log(f"Exported {name}")


def make_experiment(args, seed, out):

    return argparse.Namespace(
        data=str(args.data),
        external_test=args.external_test,
        output=str(out),
        mode="both",

        train_n=args.train_n,
        val_n=args.val_n,
        test_n=args.test_n,

        align_evals=args.align_evals,
        landmarks=[24, 48],
        score_batch=1000,

        task_sizes=[10, 12],
        blocks=12,
        maxfev=55,
        shots=1024,

        bootstrap=500,

        seed=seed,
        split_seed=args.split_seed,
        workers=args.workers,
        smoke=args.smoke
    )


def require_same_population(reference, current):

    keys = [
        "source_row",
        "time_seconds",
        "y_true"
    ]

    pd.testing.assert_frame_equal(
        reference[keys].reset_index(drop=True),
        current[keys].reset_index(drop=True),
        check_dtype=False
    )


def run_prediction_seeds(module, args, output):

    predictor = instrument_prediction(module)

    all_metrics = []
    all_va = []
    all_vb = []
    all_baselines = []
    validation_scores = []
    test_scores = []

    for seed in args.seeds:

        out = output / "seeds" / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)

        log("=" * 65)
        log(f"STARTING PREDICTION SEED {seed}")

        experiment = make_experiment(
            args, seed, out
        )

        predictor(experiment, out)

        va = pd.read_csv(
            out /
            "51_20_prediction_variants_validationA.csv"
        )

        vb = pd.read_csv(
            out / "51_fusion_validationB.csv"
        )

        baseline = pd.read_csv(
            out / "53_baseline_validationA.csv"
        )

        final = pd.read_csv(
            out / "51_prediction_final_metrics.csv"
        )

        val_scores = pd.read_csv(
            out / "53_validationB_scores.csv"
        )

        scores = pd.read_csv(
            out / "51_prediction_test_scores.csv"
        )

        if test_scores:

            require_same_population(
                test_scores[0],
                scores
            )

            require_same_population(
                validation_scores[0],
                val_scores
            )

        for df in (va, vb, baseline, final):
            df.insert(0, "seed", seed)

        # Recalculate reported test metrics independently.
        y = scores["y_true"].to_numpy()

        extra = []

        for _, row in final.iterrows():

            p = scores[row["model"]].to_numpy()

            recalculated = metrics(
                y,
                p,
                float(row["threshold"])
            )

            for key in (
                "roc_auc",
                "pr_auc",
                "f1",
                "precision",
                "recall",
                "balanced_accuracy",
                "accuracy"
            ):

                if not np.isclose(
                    recalculated[key],
                    float(row[key]),
                    atol=1e-8
                ):
                    raise ValueError(
                        f"Metric mismatch: {key}"
                    )

            extra.append(recalculated["mcc"])

        final["mcc"] = extra

        all_metrics.append(final)
        all_va.append(va)
        all_vb.append(vb)
        all_baselines.append(baseline)

        validation_scores.append(val_scores)
        test_scores.append(scores)

        log(f"SEED {seed}: TEST RESULTS")

        print(
            final[
                [
                    "model",
                    "roc_auc",
                    "pr_auc",
                    "f1",
                    "balanced_accuracy"
                ]
            ].to_string(index=False),
            flush=True
        )

    return (
        pd.concat(all_metrics, ignore_index=True),
        pd.concat(all_va, ignore_index=True),
        pd.concat(all_vb, ignore_index=True),
        pd.concat(all_baselines, ignore_index=True),
        validation_scores,
        test_scores
    )


def seed_summary(df):

    measures = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc"
    ]

    result = df.groupby("model")[measures].agg(
        ["mean", "std"]
    )

    result.columns = [
        f"{metric}_{stat}"
        for metric, stat in result.columns
    ]

    return result.reset_index()


def build_ensembles(validation_scores, test_scores):

    identifiers = {
        "source_row",
        "time_seconds",
        "y_true"
    }

    common = set(validation_scores[0].columns)

    for frame in validation_scores + test_scores:
        common &= set(frame.columns)

    names = sorted(common - identifiers)

    yv = validation_scores[0]["y_true"].to_numpy()
    yt = test_scores[0]["y_true"].to_numpy()

    ensemble_validation = {}
    ensemble_test = {}

    for name in names:

        ensemble_validation[name] = np.mean(
            [
                frame[name].to_numpy()
                for frame in validation_scores
            ],
            axis=0
        )

        ensemble_test[name] = np.mean(
            [
                frame[name].to_numpy()
                for frame in test_scores
            ],
            axis=0
        )

    # Classical comparator selected on Validation-B only.
    classical = [
        name for name in names
        if name.startswith("classical_")
    ]

    if not classical:
        raise ValueError(
            "No common classical baseline across seeds."
        )

    best_classical = max(
        classical,
        key=lambda name: selection_score(
            yv,
            ensemble_validation[name]
        )
    )

    quantum_name = "selected_quantum_augmented"
    kernel_name = "selected_classical_kernel_augmented"

    fusion_results = []
    selected_weights = {}

    for family, component in [
        ("HCQKL", quantum_name),
        ("Classical_kernel", kernel_name)
    ]:

        if component not in ensemble_validation:
            continue

        trials = []

        for weight in np.linspace(0, 1, 21):

            pv = (
                (1 - weight)
                * ensemble_validation[best_classical]
                + weight
                * ensemble_validation[component]
            )

            row = {
                "family": family,
                "classical_baseline": best_classical,
                "kernel_component": component,
                "weight": float(weight),
                "valB_roc_auc": roc_auc_score(
                    yv, pv
                ),
                "valB_pr_auc": average_precision_score(
                    yv, pv
                ),
                "selection_objective": selection_score(
                    yv, pv
                )
            }

            trials.append(row)
            fusion_results.append(row)

        selected = max(
            trials,
            key=lambda x: x["selection_objective"]
        )

        weight = selected["weight"]
        name = f"ensemble_{family}_fusion"

        ensemble_validation[name] = (
            (1 - weight)
            * ensemble_validation[best_classical]
            + weight
            * ensemble_validation[component]
        )

        ensemble_test[name] = (
            (1 - weight)
            * ensemble_test[best_classical]
            + weight
            * ensemble_test[component]
        )

        selected_weights[family] = selected

    records = []

    # Model identity and thresholds selected before
    # evaluating their test probabilities.
    selected_model = max(
        ensemble_validation,
        key=lambda name: selection_score(
            yv,
            ensemble_validation[name]
        )
    )

    for name, p in ensemble_test.items():

        threshold = choose_threshold(
            yv,
            ensemble_validation[name]
        )

        records.append({
            "model": name,
            "valB_roc_auc": roc_auc_score(
                yv,
                ensemble_validation[name]
            ),
            "valB_pr_auc": average_precision_score(
                yv,
                ensemble_validation[name]
            ),
            "selected_on_validationB": (
                name == selected_model
            ),
            **metrics(yt, p, threshold)
        })

    ensemble_frame = test_scores[0][
        ["source_row", "time_seconds", "y_true"]
    ].copy()

    for name, values in ensemble_test.items():
        ensemble_frame[name] = values

    return (
        pd.DataFrame(records),
        pd.DataFrame(fusion_results),
        ensemble_frame,
        selected_model,
        selected_weights
    )


def paired_ci(scores, method, comparator, seed=9011):

    y = scores["y_true"].to_numpy()
    p = scores[method].to_numpy()
    q = scores[comparator].to_numpy()

    chunks = np.array_split(
        np.arange(len(y)),
        12
    )

    rng = np.random.default_rng(seed)

    differences = []

    for _ in range(1000):

        ids = np.concatenate([
            chunks[i]
            for i in rng.integers(
                0, len(chunks), len(chunks)
            )
        ])

        if len(np.unique(y[ids])) < 2:
            continue

        differences.append([
            roc_auc_score(y[ids], p[ids])
            - roc_auc_score(y[ids], q[ids]),

            average_precision_score(y[ids], p[ids])
            - average_precision_score(y[ids], q[ids])
        ])

    if not differences:
        raise ValueError(
            "Insufficient classes for bootstrap."
        )

    bounds = np.quantile(
        np.asarray(differences),
        [0.025, 0.975],
        axis=0
    )

    return {
        "method": method,
        "comparator": comparator,
        "delta_roc_auc": (
            roc_auc_score(y, p)
            - roc_auc_score(y, q)
        ),
        "roc_low": bounds[0, 0],
        "roc_high": bounds[1, 0],
        "delta_pr_auc": (
            average_precision_score(y, p)
            - average_precision_score(y, q)
        ),
        "pr_low": bounds[0, 1],
        "pr_high": bounds[1, 1]
    }


def create_figures(summary, va, scores, results, output):

    figures = output / "figures"

    # Multi-seed mean and standard deviation.
    plot = summary.sort_values("roc_auc_mean")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.barh(
        plot["model"],
        plot["roc_auc_mean"],
        xerr=plot["roc_auc_std"].fillna(0),
        capsize=3
    )

    ax.axvline(
        0.5,
        linestyle=":",
        color="gray"
    )

    ax.axvline(
        0.8,
        linestyle="--",
        color="black",
        label="Research target"
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean test ROC-AUC")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        figures / "multiseed_test_performance.png",
        dpi=600
    )

    plt.close(fig)

    # Figure 3: Validation-A kernel sensitivity.
    sub = va[
        va["kernel"].str.startswith("HCQKL_orig")
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for landmarks, group in sub.groupby("landmarks"):

        values = group.groupby("alpha")[
            "valA_roc_auc"
        ].agg(["mean", "std"]).reset_index()

        ax.errorbar(
            values["alpha"],
            values["mean"],
            yerr=values["std"].fillna(0),
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=f"{landmarks} landmarks"
        )

    ax.set_xlabel("Kernel mixing coefficient")
    ax.set_ylabel("Validation-A ROC-AUC")
    ax.set_xticks([0.25, 0.50, 0.75])
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(
        figures / "figure_3_kernel_sensitivity.png",
        dpi=600
    )

    plt.close(fig)

    y = scores["y_true"].to_numpy()

    # ROC and precision-recall curves.
    for curve in ("roc", "pr"):

        fig, ax = plt.subplots(figsize=(8, 6))

        for _, row in results.iterrows():

            name = row["model"]
            p = scores[name].to_numpy()

            if curve == "roc":

                x, z, _ = roc_curve(y, p)
                label = (
                    f"{name} "
                    f"({row['roc_auc']:.4f})"
                )

            else:

                z, x, _ = precision_recall_curve(
                    y, p
                )

                label = (
                    f"{name} "
                    f"({row['pr_auc']:.4f})"
                )

            ax.plot(x, z, label=label)

        if curve == "roc":

            ax.plot(
                [0, 1],
                [0, 1],
                "--",
                color="gray"
            )

            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")

        else:

            ax.axhline(
                y.mean(),
                linestyle="--",
                color="gray"
            )

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")

        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        fig.savefig(
            figures / f"ensemble_{curve}.png",
            dpi=600
        )

        plt.close(fig)


def run_scheduling(module, args, output):

    records = []
    raw_records = []

    # Each seed is evaluated using its original
    # configuration-selection/evaluation protocol.
    for seed in args.seeds:

        out = output / "seeds" / f"seed_{seed}"

        log(f"STARTING QAOA SEED {seed}")

        experiment = make_experiment(
            args, seed, out
        )

        module.scheduling(experiment, out)

        paired = pd.read_csv(
            out /
            "51_qaoa_validation_selected_paired.csv"
        )

        raw = pd.read_csv(
            out / "51_qaoa_instance_results.csv"
        )

        paired.insert(0, "seed", seed)
        raw.insert(0, "seed", seed)

        records.append(paired)
        raw_records.append(raw)

    df = pd.concat(records, ignore_index=True)
    raw = pd.concat(raw_records, ignore_index=True)

    # Keep QAOA results paired with the actual
    # evaluation blocks and comparators.
    rows = []

    for (seed, tasks), group in df.groupby(
        ["seed", "tasks"]
    ):

        first = group.iloc[0]

        rows.append({
            "seed": seed,
            "tasks": tasks,
            "method": "Selected QAOA",
            "optimal_recovery_pct": (
                100 * first["qaoa_recovery"]
            ),
            "runtime_s": first[
                "qaoa_mean_runtime_s"
            ]
        })

        for _, row in group.iterrows():

            rows.append({
                "seed": seed,
                "tasks": tasks,
                "method": row["comparator"],
                "optimal_recovery_pct": (
                    100 * row["baseline_recovery"]
                ),
                "runtime_s": row[
                    "baseline_mean_runtime_s"
                ]
            })

    selected = pd.DataFrame(rows)

    summary = selected.groupby(
        ["tasks", "method"],
        as_index=False
    ).agg(
        recovery_mean=(
            "optimal_recovery_pct", "mean"
        ),
        recovery_std=(
            "optimal_recovery_pct", "std"
        ),
        runtime_mean=(
            "runtime_s", "mean"
        ),
        runtime_std=(
            "runtime_s", "std"
        )
    )

    export(
        df,
        "QAOA_paired_comparisons",
        output
    )

    export(
        selected,
        "QAOA_per_seed",
        output
    )

    export(
        summary,
        "table_VIII_D_QAOA_multiseed",
        output
    )

    raw.to_csv(
        output / "tables" / "QAOA_raw_results.csv",
        index=False
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        default=str(
            DATA / "aqua_sla_prepared.csv"
        )
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT)
    )

    parser.add_argument(
        "--external-test",
        default=None
    )

    parser.add_argument(
        "--train-n",
        type=int,
        default=60000
    )

    parser.add_argument(
        "--val-n",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--test-n",
        type=int,
        default=6000
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=5101
    )

    parser.add_argument(
        "--align-evals",
        type=int,
        default=100
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--skip-scheduling",
        action="store_true"
    )

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    args = parser.parse_args()

    if args.smoke:

        args.train_n = 5000
        args.val_n = 900
        args.test_n = 1600
        args.align_evals = 30
        args.seeds = args.seeds[:2]
        args.skip_scheduling = True

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds.")

    if not Path(args.data).exists():
        raise FileNotFoundError(args.data)

    output = Path(args.output).resolve()

    for name in (
        "seeds",
        "tables",
        "latex",
        "figures"
    ):

        (output / name).mkdir(
            parents=True,
            exist_ok=True
        )

    start = time.perf_counter()

    module = load_script51()

    audit = {
        "script": "53_multiseed_optimization",
        "script51_sha256": sha256(
            locate_script51()
        ),
        "dataset_sha256": sha256(args.data),
        "external_dataset_sha256": (
            sha256(args.external_test)
            if args.external_test
            else None
        ),
        "arguments": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "target_test_roc_auc": 0.80,
        "target_is_guaranteed": False,
        "test_selection_permitted": False,
        "fixed_prediction_population": True,
        "known_trace_exploratory": (
            args.external_test is None
        )
    }

    dump(
        output / "53_configuration.json",
        audit
    )

    (
        final,
        va,
        vb,
        baseline,
        validation_scores,
        test_scores
    ) = run_prediction_seeds(
        module, args, output
    )

    summary = seed_summary(final)

    export(
        final,
        "prediction_all_seeds",
        output
    )

    export(
        summary,
        "table_III_common_protocol_multiseed",
        output
    )

    va_summary = va.groupby(
        ["name", "alpha", "landmarks"],
        as_index=False
    ).agg(
        roc_mean=("valA_roc_auc", "mean"),
        roc_std=("valA_roc_auc", "std"),
        pr_mean=("valA_pr_auc", "mean"),
        pr_std=("valA_pr_auc", "std"),
        selection_mean=(
            "selection_objective", "mean"
        )
    )

    export(
        va_summary,
        "table_VIII_A_validationA_multiseed",
        output
    )

    export(
        va,
        "validationA_all_seeds",
        output
    )

    export(
        vb,
        "validationB_all_seeds",
        output
    )

    export(
        baseline,
        "baseline_validationA_all_seeds",
        output
    )

    (
        ensemble_results,
        fusion_grid,
        scores,
        selected_model,
        weights
    ) = build_ensembles(
        validation_scores,
        test_scores
    )

    export(
        fusion_grid,
        "table_VIII_B_ensemble_fusion",
        output
    )

    export(
        ensemble_results,
        "table_VIII_C_ensemble_test_results",
        output
    )

    scores.to_csv(
        output / "ensemble_test_scores.csv",
        index=False
    )

    dump(
        output / "53_selected_models.json",
        {
            "selected_on_validationB": selected_model,
            "fusion_weights": weights,
            "thresholds": dict(
                zip(
                    ensemble_results["model"],
                    ensemble_results["threshold"]
                )
            )
        }
    )

    comparisons = []

    if {
        "ensemble_HCQKL_fusion",
        "ensemble_Classical_kernel_fusion"
    }.issubset(scores.columns):

        comparisons.append(
            paired_ci(
                scores,
                "ensemble_HCQKL_fusion",
                "ensemble_Classical_kernel_fusion"
            )
        )

    if weights.get("HCQKL"):

        comparator = weights[
            "HCQKL"
        ]["classical_baseline"]

        comparisons.append(
            paired_ci(
                scores,
                "ensemble_HCQKL_fusion",
                comparator
            )
        )

    if comparisons:

        export(
            pd.DataFrame(comparisons),
            "prediction_paired_CI",
            output
        )

    create_figures(
        summary,
        va,
        scores,
        ensemble_results,
        output
    )

    if not args.skip_scheduling:

        run_scheduling(
            module,
            args,
            output
        )

    selected = ensemble_results[
        ensemble_results["model"] == selected_model
    ].iloc[0]

    audit.update({
        "elapsed_seconds": (
            time.perf_counter() - start
        ),
        "verified_prediction_metrics": True,
        "selected_model": selected_model,
        "selected_test_roc_auc": float(
            selected["roc_auc"]
        ),
        "target_80_percent_achieved": bool(
            selected["roc_auc"] >= 0.80
        ),
        "fusion_weights": weights,
        "scheduling_completed": (
            not args.skip_scheduling
        ),
        "manuscript_boundaries": {
            "table_III": (
                "New common prediction protocol, "
                "not original full/matched protocol."
            ),
            "table_IV": (
                "Original RiskGreedy not rerun."
            ),
            "table_VI": (
                "Original four-bit QUBO not rerun."
            ),
            "table_VII": (
                "Original sensitivity not rerun."
            ),
            "figure_4": (
                "Original iQuantum not rerun."
            )
        }
    })

    dump(
        output / "53_reproducibility_audit.json",
        audit
    )

    log("=" * 65)
    log("FINAL MULTI-SEED RESULTS")

    print(
        summary[
            [
                "model",
                "roc_auc_mean",
                "roc_auc_std",
                "pr_auc_mean",
                "f1_mean",
                "balanced_accuracy_mean"
            ]
        ].to_string(index=False)
    )

    log("FINAL ENSEMBLE RESULTS")

    print(
        ensemble_results[
            [
                "model",
                "valB_roc_auc",
                "roc_auc",
                "pr_auc",
                "f1",
                "balanced_accuracy"
            ]
        ].to_string(index=False)
    )

    log(f"Validation-selected model: {selected_model}")

    log(
        f"Selected model test ROC-AUC: "
        f"{selected['roc_auc']:.4f}"
    )

    log(
        "80% research target achieved: "
        + str(
            audit["target_80_percent_achieved"]
        )
    )

    log(f"ALL RESULTS SAVED TO {output}")


if __name__ == "__main__":
    main()
