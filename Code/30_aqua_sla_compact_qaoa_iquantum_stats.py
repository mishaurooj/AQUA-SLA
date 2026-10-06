from __future__ import annotations

"""
AQUA-SLA targeted reviewer revision.

Purpose
-------
1. Rerun ONLY the reformulated compact QAOA scheduling experiment.
2. Compute richer statistics from the already completed 24 iQuantum runs.

This script does not retrain HCQKL and does not rerun iQuantum.

Recommended full run
--------------------
conda activate aqua-sla
cd /d E:\\other\\AQUA-SLA\\Code
python 30_aqua_sla_compact_qaoa_iquantum_stats.py ^
  --run-dir "E:\\other\\AQUA-SLA\\results\\aqua_sla_review_v2\\20260925_034909_reviewer_revision_v2" ^
  --qaoa-blocks 20 --shots 512 --maxiter 50 --reps 1 ^
  --timeout 120 --bootstrap 5000

Smoke test
----------
python 30_aqua_sla_compact_qaoa_iquantum_stats.py --fast
"""

import argparse
import itertools
import json
import multiprocessing as mp
import platform
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import wilcoxon
from sklearn.metrics.pairwise import rbf_kernel

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_DIR = Path(r"E:\other\AQUA-SLA")
DEFAULT_RUN_DIR = (
    PROJECT_DIR / "results" / "aqua_sla_review_v2" /
    "20260925_034909_reviewer_revision_v2"
)
TIME_COLUMN = "time_seconds"
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
SEED = 42


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
    path.write_text(json.dumps(conv(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def numeric(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    x = frame.loc[:, cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def temporal_split(df: pd.DataFrame):
    df = df.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    n = len(df)
    a = int(n * TRAIN_FRACTION)
    b = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def bootstrap_ci(values: Sequence[float], resamples: int, seed: int):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for i in range(resamples):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_cohens_dz(d: Sequence[float]) -> float:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    if np.isclose(sd, 0):
        return 0.0 if np.isclose(np.mean(d), 0) else float(np.sign(np.mean(d)) * np.inf)
    return float(np.mean(d) / sd)


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
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


# ---------------------------------------------------------------------------
# Existing HCQKL inference only
# ---------------------------------------------------------------------------

def statevectors(Z: np.ndarray) -> np.ndarray:
    from qiskit.quantum_info import Statevector
    try:
        from qiskit.circuit.library import zz_feature_map
        fmap = zz_feature_map(feature_dimension=Z.shape[1], reps=2, entanglement="linear")
    except Exception:
        from qiskit.circuit.library import ZZFeatureMap
        fmap = ZZFeatureMap(feature_dimension=Z.shape[1], reps=2, entanglement="linear")
    states = []
    for row in Z:
        qc = fmap.assign_parameters(row, inplace=False)
        states.append(np.asarray(Statevector.from_instruction(qc).data, complex))
    return np.vstack(states)


def fidelity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs(A @ B.conj().T) ** 2


def score_hcqkl(frame: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    feats = bundle["features"]
    X = bundle["scaler"].transform(bundle["imputer"].transform(numeric(frame, feats)))
    Z = bundle["pca"].transform(X)
    Z = bundle["angle_scaler"].transform(Z)
    Kr = rbf_kernel(Z, bundle["z_train"], gamma=bundle["rbf_gamma"])
    S = statevectors(Z)
    Kq = fidelity(S, bundle["statevectors_train"])
    alpha = float(bundle["best_alpha"])
    Kh = alpha * Kr + (1 - alpha) * Kq
    d = bundle["hcqkl_svc"].decision_function(Kh)
    return bundle["hcqkl_calibrator"].predict_proba(np.asarray(d).reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Scheduling model and independent external SLO evaluation
# ---------------------------------------------------------------------------

@dataclass
class Resource:
    resource_id: int
    cpu_capacity: float
    memory_capacity: float
    speed: float
    energy_rate: float
    cost_rate: float


def resources_for(batch: pd.DataFrame, m: int = 2) -> list[Resource]:
    cpu = np.clip(pd.to_numeric(batch["requested_cpu"], errors="coerce").fillna(0).to_numpy(float), 1e-6, None)
    mem = np.clip(pd.to_numeric(batch["requested_memory"], errors="coerce").fillna(0).to_numpy(float), 1e-6, None)
    ac, am = cpu.sum() / m, mem.sum() / m
    cm = np.linspace(.95, 1.55, m)
    mm = np.linspace(1.10, 1.45, m)
    sp = np.linspace(.85, 1.45, m)
    en = np.linspace(.85, 1.20, m)
    co = np.linspace(.75, 1.25, m)
    return [
        Resource(j, max(ac * cm[j] * 1.35, cpu.max()), max(am * mm[j] * 1.35, mem.max()), sp[j], en[j], co[j])
        for j in range(m)
    ]


def train_duration_tables(train: pd.DataFrame):
    d = pd.to_numeric(train["duration_seconds"], errors="coerce")
    d = d[(d > 0) & np.isfinite(d)]
    global_median = float(np.median(d)) if len(d) else 60.0
    global_slo = float(np.quantile(d, .90)) if len(d) else 300.0
    medians, slos = {}, {}
    if "scheduling_class" in train:
        t = train.copy()
        t["_d"] = pd.to_numeric(t["duration_seconds"], errors="coerce")
        for cls, g in t.groupby("scheduling_class", dropna=False):
            v = g["_d"]
            v = v[(v > 0) & np.isfinite(v)]
            if len(v) >= 50:
                medians[str(cls)] = float(np.median(v))
                slos[str(cls)] = float(np.quantile(v, .90))
    return medians, global_median, slos, global_slo


def cost_matrix(batch, resources, medians, global_median):
    risk = np.clip(batch["risk_hcqkl"].to_numpy(float), 0, 1)
    cpu = pd.to_numeric(batch["requested_cpu"], errors="coerce").fillna(0).to_numpy(float)
    mem = pd.to_numeric(batch["requested_memory"], errors="coerce").fillna(0).to_numpy(float)
    cls = batch["scheduling_class"].astype(str) if "scheduling_class" in batch else pd.Series(["global"] * len(batch))
    dur = np.array([medians.get(str(c), global_median) for c in cls], float)
    n, m = len(batch), len(resources)
    L = np.zeros((n, m)); E = np.zeros((n, m)); M = np.zeros((n, m)); R = np.zeros((n, m))
    for i in range(n):
        for j, r in enumerate(resources):
            ex = dur[i] / r.speed
            pressure = max(cpu[i] / r.cpu_capacity, mem[i] / r.memory_capacity)
            L[i, j] = ex * (1 + .35 * pressure)
            E[i, j] = ex * r.energy_rate
            M[i, j] = ex * r.cost_rate
            R[i, j] = risk[i] * L[i, j]
    def norm(x):
        lo, hi = x.min(), x.max()
        return np.zeros_like(x) if np.isclose(lo, hi) else (x - lo) / (hi - lo)
    return .50 * norm(R) + .25 * norm(L) + .15 * norm(E) + .10 * norm(M)


def assignment_feasible(a, batch, resources):
    if a is None or len(a) != len(batch) or any(j < 0 or j >= len(resources) for j in a):
        return False
    cpu = pd.to_numeric(batch["requested_cpu"], errors="coerce").fillna(0).to_numpy(float)
    mem = pd.to_numeric(batch["requested_memory"], errors="coerce").fillna(0).to_numpy(float)
    for j, r in enumerate(resources):
        ids = [i for i, x in enumerate(a) if x == j]
        if ids and (cpu[ids].sum() > r.cpu_capacity + 1e-9 or mem[ids].sum() > r.memory_capacity + 1e-9):
            return False
    return True


def assignment_objective(a, C):
    return float(C[np.arange(len(a)), np.asarray(a, int)].sum())


def solve_milp(batch, resources, C):
    n, m = C.shape
    c = C.ravel()
    A, lb, ub = [], [], []
    for i in range(n):
        row = np.zeros(n * m); row[i*m:(i+1)*m] = 1
        A.append(row); lb.append(1); ub.append(1)
    cpu = pd.to_numeric(batch["requested_cpu"], errors="coerce").fillna(0).to_numpy(float)
    mem = pd.to_numeric(batch["requested_memory"], errors="coerce").fillna(0).to_numpy(float)
    for j, r in enumerate(resources):
        rc = np.zeros(n*m); rm = np.zeros(n*m)
        for i in range(n):
            rc[i*m+j] = cpu[i]; rm[i*m+j] = mem[i]
        A += [rc, rm]; lb += [-np.inf, -np.inf]; ub += [r.cpu_capacity, r.memory_capacity]
    start = time.perf_counter()
    ans = milp(c=c, integrality=np.ones(n*m, int), bounds=Bounds(np.zeros(n*m), np.ones(n*m)),
               constraints=LinearConstraint(np.vstack(A), np.asarray(lb), np.asarray(ub)), options={"time_limit": 60})
    runtime = time.perf_counter() - start
    if ans.x is None:
        return None, np.nan, runtime, str(ans.message)
    a = np.argmax(ans.x.reshape(n, m), axis=1).astype(int).tolist()
    return a, assignment_objective(a, C), runtime, str(ans.message)


def risk_greedy(batch, resources, C):
    start = time.perf_counter()
    order = np.argsort(-batch["risk_hcqkl"].to_numpy(float))
    cpu = pd.to_numeric(batch["requested_cpu"], errors="coerce").fillna(0).to_numpy(float)
    mem = pd.to_numeric(batch["requested_memory"], errors="coerce").fillna(0).to_numpy(float)
    uc = np.zeros(len(resources)); um = np.zeros(len(resources)); a = [-1] * len(batch)
    for i in order:
        cand = [j for j, r in enumerate(resources) if uc[j] + cpu[i] <= r.cpu_capacity + 1e-9 and um[j] + mem[i] <= r.memory_capacity + 1e-9]
        if not cand:
            return None, time.perf_counter() - start
        j = min(cand, key=lambda jj: C[i, jj])
        a[i] = int(j); uc[j] += cpu[i]; um[j] += mem[i]
    return a, time.perf_counter() - start


def random_feasible(batch, resources, rng):
    start = time.perf_counter()
    for _ in range(1000):
        a = rng.integers(0, len(resources), size=len(batch)).tolist()
        if assignment_feasible(a, batch, resources):
            return a, time.perf_counter() - start
    return None, time.perf_counter() - start


def evaluate_external_slo(batch, resources, a, slos, global_slo):
    arr = pd.to_numeric(batch[TIME_COLUMN], errors="coerce").to_numpy(float)
    arr = arr - np.nanmin(arr)
    dur = pd.to_numeric(batch["duration_seconds"], errors="coerce").to_numpy(float)
    good = np.isfinite(dur) & (dur > 0)
    fill = np.nanmedian(dur[good]) if good.any() else 60.0
    dur = np.where(good, dur, fill)
    wait = np.zeros(len(batch)); finish = np.zeros(len(batch)); energy = np.zeros(len(batch)); cost = np.zeros(len(batch)); miss = np.zeros(len(batch), int)
    for j, r in enumerate(resources):
        ids = sorted([i for i, x in enumerate(a) if x == j], key=lambda i: arr[i])
        avail = 0.0
        for i in ids:
            start = max(arr[i], avail)
            wait[i] = start - arr[i]
            service = dur[i] / r.speed
            finish[i] = start + service; avail = finish[i]
            energy[i] = service * r.energy_rate; cost[i] = service * r.cost_rate
            cls = str(batch["scheduling_class"].iloc[i]) if "scheduling_class" in batch else "global"
            miss[i] = int(finish[i] > arr[i] + slos.get(cls, global_slo))
    return {
        "slo_miss_rate": float(miss.mean()),
        "mean_waiting": float(wait.mean()),
        "mean_completion": float(np.mean(finish-arr)),
        "makespan": float(finish.max()-arr.min()),
        "total_energy": float(energy.sum()),
        "total_cost": float(cost.sum()),
    }


def dense_window_starts(pool, batch_size, blocks):
    n = len(pool)
    if n < batch_size:
        return []
    max_start = n - batch_size
    edges = np.linspace(0, max_start + 1, blocks + 1, dtype=int)
    times = pool[TIME_COLUMN].to_numpy(float)
    starts = []
    for b in range(blocks):
        lo = int(edges[b]); hi = min(max(lo + 1, int(edges[b+1])), max_start + 1)
        cand = np.arange(lo, hi, dtype=int)
        if len(cand):
            spans = times[cand + batch_size - 1] - times[cand]
            starts.append(int(cand[int(np.argmin(spans))]))
    return starts


# ---------------------------------------------------------------------------
# Exact compact QUBO for 2 tasks x 2 resources
# ---------------------------------------------------------------------------

def vname(i, j):
    return f"x_{i}_{j}"


def decode_bits(bits):
    B = np.asarray(bits, int).reshape(2, 2)
    a = []
    for i in range(2):
        ones = np.where(B[i] == 1)[0]
        if len(ones) != 1:
            return None
        a.append(int(ones[0]))
    return a


def compact_qubo_coefficients(batch, resources, C, penalty):
    """Exact for 2 tasks: assignment penalties plus exact forbidden-subset capacity penalties."""
    if len(batch) != 2 or len(resources) != 2 or C.shape != (2, 2):
        raise ValueError("Compact exact QUBO requires exactly 2 tasks x 2 resources.")
    constant = 0.0
    linear = {vname(i,j): float(C[i,j]) for i in range(2) for j in range(2)}
    quadratic = {}
    # Exactly one resource per task: P(x_i0+x_i1-1)^2
    for i in range(2):
        constant += penalty
        a, b = vname(i,0), vname(i,1)
        linear[a] -= penalty; linear[b] -= penalty
        quadratic[(a,b)] = quadratic.get((a,b), 0.0) + 2.0 * penalty
    cpu = pd.to_numeric(batch["requested_cpu"], errors="coerce").fillna(0).to_numpy(float)
    mem = pd.to_numeric(batch["requested_memory"], errors="coerce").fillna(0).to_numpy(float)
    audit = []
    for j, r in enumerate(resources):
        single_bad = []
        for i in range(2):
            bad = bool(cpu[i] > r.cpu_capacity + 1e-9 or mem[i] > r.memory_capacity + 1e-9)
            single_bad.append(bad)
            if bad:
                linear[vname(i,j)] += penalty
            audit.append({"resource":j,"subset":f"task_{i}","violates":bad,"cpu_sum":float(cpu[i]),"memory_sum":float(mem[i]),"cpu_capacity":float(r.cpu_capacity),"memory_capacity":float(r.memory_capacity)})
        pair_bad = bool(cpu.sum() > r.cpu_capacity + 1e-9 or mem.sum() > r.memory_capacity + 1e-9)
        if pair_bad and not single_bad[0] and not single_bad[1]:
            key = (vname(0,j), vname(1,j))
            quadratic[key] = quadratic.get(key, 0.0) + penalty
        audit.append({"resource":j,"subset":"task_0+task_1","violates":pair_bad,"cpu_sum":float(cpu.sum()),"memory_sum":float(mem.sum()),"cpu_capacity":float(r.cpu_capacity),"memory_capacity":float(r.memory_capacity)})
    return constant, linear, quadratic, audit


def qubo_energy(bits, constant, linear, quadratic):
    bits = np.asarray(bits, int)
    vals = {vname(i,j): int(bits[i*2+j]) for i in range(2) for j in range(2)}
    e = constant + sum(coef * vals[k] for k, coef in linear.items())
    e += sum(coef * vals[a] * vals[b] for (a,b), coef in quadratic.items())
    return float(e)


def validate_qubo(batch, resources, C, penalty):
    constant, linear, quadratic, audit = compact_qubo_coefficients(batch, resources, C, penalty)
    rows = []
    for bits in itertools.product([0,1], repeat=4):
        a = decode_bits(bits)
        feas = a is not None and assignment_feasible(a, batch, resources)
        obj = assignment_objective(a, C) if a is not None else np.nan
        rows.append({"bits":"".join(map(str,bits)),"qubo_energy":qubo_energy(bits,constant,linear,quadratic),"assignment":json.dumps(a) if a is not None else None,"feasible":bool(feas),"assignment_objective":obj})
    df = pd.DataFrame(rows).sort_values(["qubo_energy","bits"]).reset_index(drop=True)
    feas = df[df["feasible"]]
    if feas.empty:
        return df, {"valid":False,"reason":"No feasible assignment","capacity_audit":audit}
    best = df.iloc[0]
    best_obj = float(feas["assignment_objective"].min())
    valid = bool(best["feasible"] and np.isclose(float(best["assignment_objective"]), best_obj, atol=1e-8))
    return df, {"valid":valid,"best_bits":best["bits"],"best_assignment":best["assignment"],"best_assignment_objective":float(best["assignment_objective"]) if np.isfinite(best["assignment_objective"]) else None,"best_feasible_objective":best_obj,"capacity_audit":audit,"constant":constant,"linear":linear,"quadratic":{f"{a}|{b}":v for (a,b),v in quadratic.items()}}


def exact_repair(bits, batch, resources, C):
    raw = np.asarray(bits, int)
    candidates = []
    for a0 in range(2):
        for a1 in range(2):
            a = [a0,a1]
            if not assignment_feasible(a,batch,resources):
                continue
            target = np.zeros(4,int); target[a0] = 1; target[2+a1] = 1
            h = int(np.abs(raw-target).sum())
            candidates.append((h, assignment_objective(a,C), a))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x:(x[0],x[1],x[2]))
    return candidates[0][2], candidates[0][0]


def _sampler(shots, seed):
    try:
        from qiskit.primitives import StatevectorSampler
        return StatevectorSampler(default_shots=shots, seed=seed), "StatevectorSampler"
    except Exception:
        from qiskit.primitives import Sampler
        return Sampler(options={"shots":shots,"seed":seed}), "Sampler"


def _qaoa_worker(payload, queue):
    try:
        try:
            from qiskit_algorithms.minimum_eigensolvers import QAOA
            from qiskit_algorithms.optimizers import COBYLA
        except Exception:
            from qiskit.algorithms.minimum_eigensolvers import QAOA
            from qiskit.algorithms.optimizers import COBYLA
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        qp = QuadraticProgram("aqua_sla_compact_qubo")
        for i in range(2):
            for j in range(2):
                qp.binary_var(vname(i,j))
        qdict = {}
        for key, value in payload["quadratic"].items():
            a,b = key.split("|",1); qdict[(a,b)] = float(value)
        qp.minimize(constant=float(payload["constant"]), linear={k:float(v) for k,v in payload["linear"].items()}, quadratic=qdict)
        sampler, sname = _sampler(int(payload["shots"]), int(payload["seed"]))
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=int(payload["maxiter"])), reps=int(payload["reps"]))
        opt = MinimumEigenOptimizer(qaoa)
        start = time.perf_counter(); result = opt.solve(qp); elapsed = time.perf_counter()-start
        x = np.asarray(result.x,float); bits = (x >= .5).astype(int).tolist()
        queue.put({"success":True,"timed_out":False,"sampler":sname,"elapsed_seconds":elapsed,"status":str(getattr(result,"status","unknown")),"raw_x":x.tolist(),"raw_bits":bits,"fval":float(result.fval) if getattr(result,"fval",None) is not None else None})
    except Exception as exc:
        queue.put({"success":False,"timed_out":False,"error_stage":"qaoa_worker","error":repr(exc)})


def run_qaoa_timeout(payload, timeout_seconds):
    ctx = mp.get_context("spawn"); q = ctx.Queue(); p = ctx.Process(target=_qaoa_worker,args=(payload,q))
    start = time.perf_counter(); p.start(); p.join(timeout_seconds); wall = time.perf_counter()-start
    if p.is_alive():
        p.terminate(); p.join(10)
        return {"success":False,"timed_out":True,"timeout_seconds":timeout_seconds,"wall_seconds":wall,"error_stage":"wall_clock_timeout","error":f"Exceeded {timeout_seconds}s"}
    if not q.empty():
        ans = q.get(); ans["wall_seconds"] = wall; return ans
    return {"success":False,"timed_out":False,"wall_seconds":wall,"error_stage":"worker_no_result","error":f"Worker exit code {p.exitcode}"}


def add_result(rows, block, solver, batch, resources, C, a, runtime, milp_obj, slos, global_slo, extra=None):
    extra = extra or {}
    if a is None:
        rows.append({"block":block,"solver":solver,"completed":False,"feasible":False,"runtime_seconds":runtime,"objective":np.nan,"milp_objective":milp_obj,"objective_gap":np.nan,"relative_objective_gap":np.nan,**extra}); return
    feas = assignment_feasible(a,batch,resources); obj = assignment_objective(a,C)
    ext = evaluate_external_slo(batch,resources,a,slos,global_slo) if feas else {k:np.nan for k in ["slo_miss_rate","mean_waiting","mean_completion","makespan","total_energy","total_cost"]}
    gap = obj-milp_obj if np.isfinite(milp_obj) else np.nan
    rows.append({"block":block,"solver":solver,"completed":True,"feasible":bool(feas),"runtime_seconds":runtime,"objective":obj,"milp_objective":milp_obj,"objective_gap":gap,"relative_objective_gap":gap/max(abs(milp_obj),1e-12) if np.isfinite(gap) else np.nan,"assignment":json.dumps(a),**ext,**extra})


def paired_stats(results, out_dir, bootstrap):
    metrics = ["objective","slo_miss_rate","mean_waiting","mean_completion","makespan","total_energy","total_cost","runtime_seconds"]
    comps = [("QAOA_raw","MILP"),("QAOA_repaired","MILP"),("RiskGreedy","MILP"),("RandomFeasible","MILP"),("QAOA_raw","RiskGreedy"),("QAOA_repaired","RiskGreedy")]
    rows = []
    for a,b in comps:
        for metric in metrics:
            aa = results[(results.solver==a)&results[metric].notna()][["block",metric]].rename(columns={metric:"a"})
            bb = results[(results.solver==b)&results[metric].notna()][["block",metric]].rename(columns={metric:"b"})
            p = aa.merge(bb,on="block").dropna()
            if p.empty: continue
            d = p.a.to_numpy(float)-p.b.to_numpy(float); lo,hi = bootstrap_ci(d,bootstrap,SEED+len(rows))
            nz = d[~np.isclose(d,0)]; stat=pv=np.nan
            if len(nz)>=6:
                try:
                    w=wilcoxon(nz,alternative="two-sided",zero_method="wilcox",method="auto"); stat=float(w.statistic); pv=float(w.pvalue)
                except Exception: pass
            rows.append({"solver_a":a,"solver_b":b,"metric":metric,"pairs":len(p),"mean_a":float(p.a.mean()),"mean_b":float(p.b.mean()),"mean_diff_a_minus_b":float(d.mean()),"median_diff_a_minus_b":float(np.median(d)),"bootstrap95_low":lo,"bootstrap95_high":hi,"paired_cohens_dz":paired_cohens_dz(d),"wilcoxon_stat":stat,"p_raw":pv})
    ans=pd.DataFrame(rows)
    if not ans.empty:
        ans["p_holm"]=holm_adjust(ans.p_raw.to_numpy(float)); ans["significant_holm_0_05"]=ans.p_holm<.05
    ans.to_csv(out_dir/"qaoa_paired_statistics_compact.csv",index=False)
    return ans


def run_compact_qaoa(run_dir, target_dir, blocks, reps, shots, maxiter, timeout, bootstrap):
    qdir=target_dir/"qaoa_compact_qubo"; bdir=qdir/"blocks"; bdir.mkdir(parents=True,exist_ok=True)
    future=run_dir/"01_future_target"/"future_target_dataset.csv"; model=run_dir/"03_models"/"hcqkl_v2_bundle.joblib"
    if not future.exists(): raise FileNotFoundError(f"Missing {future}")
    if not model.exists(): raise FileNotFoundError(f"Missing {model}")
    data=pd.read_csv(future,low_memory=False); train,_,test=temporal_split(data); bundle=joblib.load(model)
    req=[TIME_COLUMN,"requested_cpu","requested_memory","duration_seconds"]
    pool=test.copy()
    for c in req: pool[c]=pd.to_numeric(pool[c],errors="coerce")
    pool=pool.dropna(subset=req); pool=pool[(pool.requested_cpu>0)&(pool.requested_memory>0)&(pool.duration_seconds>0)].sort_values(TIME_COLUMN,kind="mergesort").reset_index(drop=True)
    meds,gmed,slos,gslo=train_duration_tables(train)
    save_json({"external_slo":"training-period 90th percentile duration by scheduling class","assignment_objective_uses_realized_slo":False,"test_label_used_in_assignment":False,"global_slo_seconds":gslo},qdir/"external_slo_independence.json")
    starts=dense_window_starts(pool,2,blocks); rng=np.random.default_rng(SEED); rows=[]; exec_rows=[]; val_rows=[]
    for block,start_idx in enumerate(starts):
        batch=pool.iloc[start_idx:start_idx+2].copy().reset_index(drop=True); batch["risk_hcqkl"]=score_hcqkl(batch,bundle)
        resources=resources_for(batch,2); C=cost_matrix(batch,resources,meds,gmed)
        milp_a,milp_obj,milp_rt,milp_status=solve_milp(batch,resources,C)
        if milp_a is None: continue
        penalty=max(10.0,10.0*(1.0+float(np.abs(C).sum())))
        enum,validation=validate_qubo(batch,resources,C,penalty); enum.to_csv(bdir/f"block_{block:02d}_qubo_enumeration.csv",index=False)
        match=bool(validation.get("valid",False) and np.isclose(validation.get("best_feasible_objective",np.nan),milp_obj,atol=1e-8))
        val_rows.append({"block":block,"penalty":penalty,"qubo_valid":bool(validation.get("valid",False)),"qubo_matches_milp":match,"milp_objective":milp_obj,"qubo_best_feasible_objective":validation.get("best_feasible_objective")})
        save_json(validation,bdir/f"block_{block:02d}_qubo_validation.json")
        add_result(rows,block,"MILP",batch,resources,C,milp_a,milp_rt,milp_obj,slos,gslo,{"milp_status":milp_status,"qubo_valid":match})
        ga,grt=risk_greedy(batch,resources,C); add_result(rows,block,"RiskGreedy",batch,resources,C,ga,grt,milp_obj,slos,gslo,{"qubo_valid":match})
        ra,rrt=random_feasible(batch,resources,rng); add_result(rows,block,"RandomFeasible",batch,resources,C,ra,rrt,milp_obj,slos,gslo,{"qubo_valid":match})
        if not match:
            qi={"block":block,"attempted":False,"success":False,"timed_out":False,"error_stage":"qubo_validation","error":"Compact QUBO did not exactly match MILP."}; exec_rows.append(qi); save_json(qi,bdir/f"block_{block:02d}_qaoa.json")
            for s in ["QAOA_raw","QAOA_repaired"]: add_result(rows,block,s,batch,resources,C,None,None,milp_obj,slos,gslo,{"qaoa_attempted":False})
            continue
        constant,linear,quadratic,audit=compact_qubo_coefficients(batch,resources,C,penalty)
        payload={"constant":constant,"linear":linear,"quadratic":{f"{a}|{b}":v for (a,b),v in quadratic.items()},"reps":reps,"shots":shots,"maxiter":maxiter,"seed":SEED+1000+block}
        qi=run_qaoa_timeout(payload,timeout); qi.update({"block":block,"attempted":True,"penalty":penalty,"reps":reps,"shots":shots,"maxiter":maxiter,"timeout_seconds":timeout,"qubo_validated_against_milp":True})
        raw_a=rep_a=None; hamming=None
        if qi.get("success"):
            raw_a=decode_bits(qi["raw_bits"]); raw_feas=raw_a is not None and assignment_feasible(raw_a,batch,resources)
            rep_a,hamming=exact_repair(qi["raw_bits"],batch,resources,C)
            qi.update({"raw_assignment":raw_a,"raw_one_hot":raw_a is not None,"raw_feasible":bool(raw_feas),"repair_required":bool(not raw_feas or hamming not in (None,0)),"repair_hamming_changes":hamming,"repaired_assignment":rep_a,"repaired_feasible":bool(rep_a is not None and assignment_feasible(rep_a,batch,resources))})
            if raw_a is not None: qi.update({"raw_objective":assignment_objective(raw_a,C),"raw_objective_gap":assignment_objective(raw_a,C)-milp_obj})
            if rep_a is not None: qi.update({"repaired_objective":assignment_objective(rep_a,C),"repaired_objective_gap":assignment_objective(rep_a,C)-milp_obj})
        exec_rows.append(qi); save_json(qi,bdir/f"block_{block:02d}_qaoa.json")
        if qi.get("success"):
            add_result(rows,block,"QAOA_raw",batch,resources,C,raw_a,qi.get("wall_seconds"),milp_obj,slos,gslo,{"qaoa_success":True,"qaoa_timeout":False,"repair_required":qi.get("repair_required")})
            add_result(rows,block,"QAOA_repaired",batch,resources,C,rep_a,qi.get("wall_seconds"),milp_obj,slos,gslo,{"qaoa_success":True,"qaoa_timeout":False,"repair_required":qi.get("repair_required"),"repair_hamming_changes":hamming})
        else:
            for s in ["QAOA_raw","QAOA_repaired"]: add_result(rows,block,s,batch,resources,C,None,qi.get("wall_seconds"),milp_obj,slos,gslo,{"qaoa_success":False,"qaoa_timeout":bool(qi.get("timed_out",False)),"qaoa_error":qi.get("error")})
        save_json({"block":block,"source_start_index":start_idx,"tasks":batch[[c for c in ["record_id",TIME_COLUMN,"requested_cpu","requested_memory","duration_seconds","scheduling_class","risk_hcqkl"] if c in batch]].to_dict("records"),"resources":[asdict(r) for r in resources],"cost_matrix":C,"milp_assignment":milp_a,"milp_objective":milp_obj,"capacity_audit":audit},bdir/f"block_{block:02d}_manifest.json")
        pd.DataFrame(rows).to_csv(qdir/"qaoa_solver_results_compact.csv",index=False); pd.DataFrame(exec_rows).to_csv(qdir/"qaoa_execution_summary_compact.csv",index=False); pd.DataFrame(val_rows).to_csv(qdir/"qubo_encoding_validation.csv",index=False)
    results=pd.DataFrame(rows); execution=pd.DataFrame(exec_rows); validation_df=pd.DataFrame(val_rows)
    results.to_csv(qdir/"qaoa_solver_results_compact.csv",index=False); execution.to_csv(qdir/"qaoa_execution_summary_compact.csv",index=False); validation_df.to_csv(qdir/"qubo_encoding_validation.csv",index=False)
    if not results.empty:
        summary=results.groupby("solver",as_index=False).agg(attempted_blocks=("block","count"),completion_rate=("completed","mean"),feasibility_rate=("feasible","mean"),mean_runtime_seconds=("runtime_seconds","mean"),median_runtime_seconds=("runtime_seconds","median"),mean_objective=("objective","mean"),mean_objective_gap=("objective_gap","mean"),median_objective_gap=("objective_gap","median"),mean_relative_gap=("relative_objective_gap","mean"),mean_slo_miss=("slo_miss_rate","mean"),mean_waiting=("mean_waiting","mean"),mean_completion=("mean_completion","mean"),mean_makespan=("makespan","mean"),mean_energy=("total_energy","mean"),mean_cost=("total_cost","mean"))
        summary.to_csv(qdir/"qaoa_solver_summary_compact.csv",index=False)
    stats=paired_stats(results,qdir,bootstrap)
    successes=int(execution.success.fillna(False).astype(bool).sum()) if not execution.empty and "success" in execution else 0
    timeouts=int(execution.timed_out.fillna(False).astype(bool).sum()) if not execution.empty and "timed_out" in execution else 0
    raw_feas=int(execution.raw_feasible.fillna(False).astype(bool).sum()) if not execution.empty and "raw_feasible" in execution else 0
    repairs=int(execution.repair_required.fillna(False).astype(bool).sum()) if not execution.empty and "repair_required" in execution else 0
    reliability={"requested_blocks":blocks,"attempted_qaoa_blocks":len(execution),"successful_qaoa_blocks":successes,"timeout_blocks":timeouts,"qaoa_completion_rate":successes/len(execution) if len(execution) else np.nan,"raw_feasible_blocks":raw_feas,"raw_feasibility_rate_among_successes":raw_feas/successes if successes else np.nan,"repair_required_blocks":repairs,"repair_rate_among_successes":repairs/successes if successes else np.nan,"all_qubo_encodings_validated":bool(not validation_df.empty and validation_df.qubo_matches_milp.all()),"interpretation_boundary":"If QAOA completion or raw feasibility remains poor, report it as exploratory rather than a principal performance claim."}
    save_json(reliability,qdir/"qaoa_reliability_summary.json")
    save_json({"benchmark":"compact exact 2-task x 2-resource QUBO","logical_assignment_variables":4,"assignment_constraint":"P*(x_i0+x_i1-1)^2","capacity_encoding":"Exact forbidden-subset penalties for the 2-task case; original floating-point CPU/memory constraints are not removed or approximated.","validation":"All 16 logical bitstrings are exhaustively enumerated per block; QAOA executes only when QUBO optimum matches original-capacity MILP objective.","reps":reps,"shots":shots,"maxiter":maxiter,"timeout_seconds":timeout,"blocks":blocks,"external_slo_independent_of_assignment_objective":True},qdir/"qaoa_protocol.json")
    return results,stats,reliability


# ---------------------------------------------------------------------------
# Rich statistics from existing iQuantum 24-run design
# ---------------------------------------------------------------------------

def load_iq(run_dir):
    p=run_dir/"08_iquantum_policies"/"iquantum_policy_summary.csv"
    if not p.exists(): raise FileNotFoundError(f"Missing {p}")
    d=pd.read_csv(p)
    for c in ["policy","workload_size","workload_seed","success"]:
        if c not in d.columns: raise ValueError(f"iQuantum summary missing {c}")
    if d.success.dtype != bool:
        d["success"]=d.success.fillna(False).astype(str).str.lower().isin(["true","1","yes"])
    return d


def iq_metrics(df):
    pref=["mean_waiting_time","mean_qpu_time","makespan","total_cost","tasks_returned"]
    return [c for c in pref if c in df and pd.to_numeric(df[c],errors="coerce").notna().any()]


def run_iq_stats(run_dir,target_dir,bootstrap):
    out=target_dir/"iquantum_rich_statistics"; out.mkdir(parents=True,exist_ok=True)
    d=load_iq(run_dir); d.to_csv(out/"iquantum_reconstructed_run_summary.csv",index=False)
    policies=["random","least_loaded","compatibility_aware","risk_aware"]; sizes=[30,60]; seeds=[101,202,303]
    expected=pd.DataFrame(list(itertools.product(policies,sizes,seeds)),columns=["policy","workload_size","workload_seed"])
    actual=d.groupby(["policy","workload_size","workload_seed"],as_index=False).agg(rows=("success","size"),successful=("success","max"))
    design=expected.merge(actual,on=["policy","workload_size","workload_seed"],how="left"); design["rows"]=design.rows.fillna(0).astype(int); design["successful"]=design.successful.fillna(False).astype(bool); design["present"]=design.rows>0
    design.to_csv(out/"iquantum_design_completeness.csv",index=False)
    s=d[d.success].copy(); metrics=iq_metrics(s)
    rows=[]
    for policy,g in s.groupby("policy"):
        for metric in metrics:
            v=pd.to_numeric(g[metric],errors="coerce").dropna().to_numpy(float)
            if len(v)==0: continue
            lo,hi=bootstrap_ci(v,bootstrap,SEED+len(rows))
            rows.append({"policy":policy,"metric":metric,"n_runs":len(v),"mean":float(v.mean()),"std":float(v.std(ddof=1)) if len(v)>1 else np.nan,"standard_error":float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else np.nan,"median":float(np.median(v)),"min":float(v.min()),"max":float(v.max()),"bootstrap95_low":lo,"bootstrap95_high":hi})
    uncertainty=pd.DataFrame(rows); uncertainty.to_csv(out/"iquantum_policy_uncertainty.csv",index=False)
    rows=[]
    for (size,policy),g in s.groupby(["workload_size","policy"]):
        for metric in metrics:
            v=pd.to_numeric(g[metric],errors="coerce").dropna().to_numpy(float)
            if len(v)==0: continue
            lo,hi=bootstrap_ci(v,bootstrap,SEED+int(size)+len(rows))
            rows.append({"workload_size":size,"policy":policy,"metric":metric,"n_seeds":len(v),"mean":float(v.mean()),"std":float(v.std(ddof=1)) if len(v)>1 else np.nan,"median":float(np.median(v)),"bootstrap95_low":lo,"bootstrap95_high":hi})
    pd.DataFrame(rows).to_csv(out/"iquantum_policy_by_workload_uncertainty.csv",index=False)
    rows=[]
    pols=sorted(s.policy.unique())
    for a,b in itertools.combinations(pols,2):
        for metric in metrics:
            aa=s[s.policy==a][["workload_size","workload_seed",metric]].rename(columns={metric:"a"}); bb=s[s.policy==b][["workload_size","workload_seed",metric]].rename(columns={metric:"b"})
            p=aa.merge(bb,on=["workload_size","workload_seed"]).dropna()
            if p.empty: continue
            p["a"]=pd.to_numeric(p.a,errors="coerce"); p["b"]=pd.to_numeric(p.b,errors="coerce"); p=p.dropna()
            if p.empty: continue
            diff=p.a.to_numpy(float)-p.b.to_numpy(float); lo,hi=bootstrap_ci(diff,bootstrap,SEED+len(rows)); nz=diff[~np.isclose(diff,0)]; stat=pv=np.nan
            if len(nz)>=6:
                try:
                    w=wilcoxon(nz,alternative="two-sided",zero_method="wilcox",method="auto"); stat=float(w.statistic); pv=float(w.pvalue)
                except Exception: pass
            pct=np.where(np.abs(p.b.to_numpy(float))>1e-12,100*diff/p.b.to_numpy(float),np.nan)
            rows.append({"policy_a":a,"policy_b":b,"metric":metric,"pairs":len(p),"mean_a":float(p.a.mean()),"mean_b":float(p.b.mean()),"mean_diff_a_minus_b":float(diff.mean()),"median_diff_a_minus_b":float(np.median(diff)),"mean_percent_diff_vs_b":float(np.nanmean(pct)),"bootstrap95_low":lo,"bootstrap95_high":hi,"paired_cohens_dz":paired_cohens_dz(diff),"wilcoxon_stat":stat,"p_raw":pv})
    pairwise=pd.DataFrame(rows)
    if not pairwise.empty:
        pairwise["p_holm"]=holm_adjust(pairwise.p_raw.to_numpy(float)); pairwise["significant_holm_0_05"]=pairwise.p_holm<.05
    pairwise.to_csv(out/"iquantum_all_policy_pairwise_statistics.csv",index=False)
    risk=pairwise[(pairwise.policy_a=="risk_aware")|(pairwise.policy_b=="risk_aware")].copy() if not pairwise.empty else pd.DataFrame()
    if not risk.empty:
        flip=risk.policy_b=="risk_aware"
        for c in ["mean_diff_a_minus_b","median_diff_a_minus_b","paired_cohens_dz"]: risk.loc[flip,c]=-risk.loc[flip,c]
        lo=risk.bootstrap95_low.copy(); hi=risk.bootstrap95_high.copy(); risk.loc[flip,"bootstrap95_low"]=-hi[flip]; risk.loc[flip,"bootstrap95_high"]=-lo[flip]
        risk["baseline_policy"]=np.where(risk.policy_a=="risk_aware",risk.policy_b,risk.policy_a); risk["difference_definition"]="risk_aware_minus_baseline"
    risk.to_csv(out/"iquantum_risk_aware_vs_baselines.csv",index=False)
    manifest={"expected_design":"4 policies x 2 workload sizes x 3 seeds = 24 runs","expected_cells":24,"complete_successful_cells":int((design.present&design.successful).sum()),"successful_rows":int(d.success.sum()),"metrics_analyzed":metrics,"uncertainty":f"{bootstrap}-resample nonparametric bootstrap 95% CI","pairing":"Policy comparisons paired by workload_size and workload_seed","multiplicity":"Holm correction applied to available pairwise Wilcoxon tests","claim_boundary":"iQuantum is discrete-event simulation, not physical quantum hardware."}
    save_json(manifest,out/"iquantum_statistics_manifest.json")
    return uncertainty,risk,manifest


def figures(qres,iq,target_dir):
    try: import matplotlib.pyplot as plt
    except Exception: return
    fdir=target_dir/"figures"; fdir.mkdir(parents=True,exist_ok=True)
    if not qres.empty:
        z=qres.dropna(subset=["objective_gap"]).groupby("solver",as_index=False).objective_gap.mean()
        if not z.empty:
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(z.solver,z.objective_gap); ax.set_ylabel("Mean objective gap to MILP"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(fdir/"qaoa_compact_objective_gaps.png",dpi=300); plt.close(fig)
        z=qres.dropna(subset=["runtime_seconds"]).groupby("solver",as_index=False).runtime_seconds.mean()
        if not z.empty:
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(z.solver,z.runtime_seconds); ax.set_ylabel("Mean runtime (s)"); ax.tick_params(axis="x",rotation=20); fig.tight_layout(); fig.savefig(fdir/"qaoa_compact_runtime.png",dpi=300); plt.close(fig)
    if not iq.empty:
        for metric in ["makespan","mean_waiting_time","total_cost"]:
            z=iq[iq.metric==metric].copy()
            if z.empty: continue
            x=np.arange(len(z)); yerr=np.vstack([z["mean"]-z.bootstrap95_low,z.bootstrap95_high-z["mean"]])
            fig,ax=plt.subplots(figsize=(8,4.5)); ax.errorbar(x,z["mean"],yerr=yerr,fmt="o",capsize=4); ax.set_xticks(x); ax.set_xticklabels(z.policy,rotation=20); ax.set_ylabel(metric.replace("_"," ")); fig.tight_layout(); fig.savefig(fdir/f"iquantum_{metric}_bootstrap_ci.png",dpi=300); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description="Compact-QAOA rerun + richer statistics for existing 24 iQuantum runs")
    ap.add_argument("--run-dir",default=str(DEFAULT_RUN_DIR)); ap.add_argument("--qaoa-blocks",type=int,default=20); ap.add_argument("--reps",type=int,default=1); ap.add_argument("--shots",type=int,default=512); ap.add_argument("--maxiter",type=int,default=50); ap.add_argument("--timeout",type=int,default=120); ap.add_argument("--bootstrap",type=int,default=5000); ap.add_argument("--fast",action="store_true")
    args=ap.parse_args()
    if args.fast:
        args.qaoa_blocks=min(args.qaoa_blocks,3); args.shots=min(args.shots,128); args.maxiter=min(args.maxiter,10); args.timeout=min(args.timeout,60); args.bootstrap=min(args.bootstrap,1000)
    run_dir=Path(args.run_dir).resolve()
    if not run_dir.exists(): raise FileNotFoundError(f"Run dir not found: {run_dir}")
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); target=run_dir/"11_targeted_qaoa_iquantum_revision"/f"{stamp}_compact_qaoa_iquantum_stats"; target.mkdir(parents=True,exist_ok=False)
    save_json({"project":"AQUA-SLA","experiment":"targeted-compact-qaoa-and-iquantum-stats","created":datetime.now().isoformat(),"python":sys.version,"platform":platform.platform(),"source_run":str(run_dir),"output":str(target),"fast":args.fast,"qaoa":{"tasks":2,"resources":2,"blocks":args.qaoa_blocks,"reps":args.reps,"shots":args.shots,"maxiter":args.maxiter,"timeout_seconds":args.timeout},"bootstrap_resamples":args.bootstrap},target/"targeted_run_manifest.json")
    err=None; qres=pd.DataFrame(); qstats=pd.DataFrame(); reliability={}; iqu=pd.DataFrame(); iqr=pd.DataFrame(); iqm={}
    try:
        print("Running compact QUBO QAOA...")
        qres,qstats,reliability=run_compact_qaoa(run_dir,target,args.qaoa_blocks,args.reps,args.shots,args.maxiter,args.timeout,args.bootstrap)
        print("Computing richer statistics from existing 24 iQuantum runs...")
        iqu,iqr,iqm=run_iq_stats(run_dir,target,args.bootstrap)
        figures(qres,iqu,target)
    except Exception as exc:
        err=repr(exc); raise
    finally:
        save_json({"completed":err is None,"error":err,"source_run":str(run_dir),"output":str(target),"qaoa_solver_rows":len(qres),"qaoa_statistical_rows":len(qstats),"qaoa_reliability":reliability,"iquantum_uncertainty_rows":len(iqu),"iquantum_risk_aware_comparison_rows":len(iqr),"iquantum_manifest":iqm,"notes":["No HCQKL retraining was performed.","No iQuantum simulator rerun was performed.","Compact QUBO is exhaustively validated against MILP before QAOA.","Raw and repaired QAOA are reported separately.","External SLO outcomes are post-assignment outcomes, not the assignment objective."]},target/"final_targeted_status.json")
    print("Finished. Output:",target)


if __name__ == "__main__":
    mp.freeze_support()
    main()
