#!/usr/bin/env python3
r"""AQUA-SLA independent QAOA circuit and fairness experiment (Script 46B).

Uses actual workload CPU/memory requests, constructs independently disclosed
2-host capacity-constrained assignment benchmarks, and compares 8 explicitly
named QAOA ablations to exact MILP, greedy, local search and matched random
search. Original RX-mixer architecture is e^{-i gamma H_C} followed by RX
layers, with p=1/2. Feasible-subspace XY mixer is a separate modification.

Generated diagonal cost H_C uses an exact nonlinear capacity penalty; it is not
an exact 2-local QUBO. CPU statevector circuit simulation does not demonstrate
quantum acceleration. Result files do not overwrite original 4-bit evidence.

Install: conda install numpy scipy pandas scikit-learn
Usage:
 python 46_qaoa_independent_ablation.py --data ..\\Dataset\\processed\\aqua_sla_prepared.csv --out ..\\results\\46_qaoa
 python 46_qaoa_independent_ablation.py --data processed.zip --fast --out qaoa_fast
"""
from __future__ import annotations
import argparse, json, time, sys, platform, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, milp, Bounds, LinearConstraint

COLUMNS=['collection_id','instance_index','time_seconds','requested_cpu','requested_memory','priority']
SEED=4712

def read_tasks(path):
    path=Path(path)
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            members=[x for x in z.namelist() if x.endswith('aqua_sla_prepared.csv')]
            if not members:raise FileNotFoundError('Expected aqua_sla_prepared.csv')
            with z.open(members[0]) as f:df=pd.read_csv(f,usecols=lambda x:x in COLUMNS)
    else:df=pd.read_csv(path,usecols=lambda x:x in COLUMNS)
    df=df.dropna(subset=['time_seconds','requested_cpu','requested_memory']).copy()
    df=df[(df.requested_cpu>0)&(df.requested_memory>0)].copy()
    df=df.sort_values('time_seconds',kind='mergesort')
    df=df.drop_duplicates(['collection_id','instance_index'])
    return df.reset_index(drop=True)

def make_instance(chunk,seed):
    rng=np.random.default_rng(seed)
    n=len(chunk);m=2
    c=chunk.requested_cpu.to_numpy(float); mem=chunk.requested_memory.to_numpy(float)
    # Map to normalized heterogeneity, preserve observed relative demand.
    c=np.clip(c/max(np.median(c),1e-8),.3,2.0)
    mem=np.clip(mem/max(np.median(mem),1e-8),.3,2.0)
    pri=pd.to_numeric(chunk.priority,errors='coerce').fillna(0).to_numpy(float)
    rel=(pri-np.min(pri))/(max(1e-9,np.ptp(pri)))
    duration=np.maximum(.2,.75*c+.6*mem)
    delays=np.column_stack([duration*(1.07+.12*rel),duration*(.91+.15*(1-rel))])
    cost=delays+np.column_stack([.18*c+.05*mem,.1*c+.12*mem])
    # Deadline is evaluated separately from optimization cost.
    deadline=duration*(1.02+.35*rel)
    cpu_caps=np.array([.56,.58])*c.sum()
    mem_caps=np.array([.57,.57])*mem.sum()
    # Make sure at least one feasible assignment exists, retaining nontrivial packing.
    assignments=((np.arange(2**n)[:,None]>>np.arange(n))&1).astype(int)
    for _ in range(12):
        feasible=[]
        for x in assignments:
            feasible.append(all(c[x==r].sum()<=cpu_caps[r]+1e-9 and mem[x==r].sum()<=mem_caps[r]+1e-9 for r in range(m)))
        if any(feasible):break
        cpu_caps*=1.07;mem_caps*=1.07
    if not any(feasible):raise RuntimeError('Generated problem infeasible')
    return {'n':n,'m':m,'cpu':c,'mem':mem,'cpu_caps':cpu_caps,'mem_caps':mem_caps,
        'cost':cost,'delays':delays,'deadline':deadline,'ids':chunk[['collection_id','instance_index']].values.tolist(),
        'source_t_min':float(chunk.time_seconds.min()),'source_t_max':float(chunk.time_seconds.max())}

def assess(p,assignment):
    """Assessment independent from penalty Hamiltonian; returns raw resource violations."""
    x=np.asarray(assignment,int);cpu=p['cpu'];mem=p['mem'];n=p['n']
    cpu_load=np.array([cpu[x==r].sum() for r in range(2)])
    mem_load=np.array([mem[x==r].sum() for r in range(2)])
    cpu_over=np.maximum(0,cpu_load-p['cpu_caps'])
    mem_over=np.maximum(0,mem_load-p['mem_caps'])
    valid=bool(cpu_over.max()<1e-8 and mem_over.max()<1e-8)
    raw_cost=float(p['cost'][np.arange(n),x].sum())
    # Report separately; excluded from optimization phase cost.
    slo=float(np.mean(p['delays'][np.arange(n),x]>p['deadline']))
    makespan=float(max(p['delays'][x==r,r].sum() for r in range(2)))
    return dict(feasible=valid,objective=raw_cost,cpu_overload=float(cpu_over.sum()),
      memory_overload=float(mem_over.sum()),cpu_utilization=float(max(cpu_load/p['cpu_caps'])),
      memory_utilization=float(max(mem_load/p['mem_caps'])),slo_miss_rate=slo,
      makespan=makespan,total_energy=float(np.sum(p['cost'][np.arange(n),x] *p['cpu'])),
      assignment=''.join(map(str,x.tolist())))

def enumerate_optimum(p):
    n=p['n'];best=None
    for xnum in range(2**n):
        assignment=np.array([(xnum>>i)&1 for i in range(n)],int)
        q=assess(p,assignment)
        if q['feasible'] and (best is None or q['objective']<best[0]):best=(q['objective'],assignment)
    if best is None:raise RuntimeError('No feasible solution')
    return best

def milp_optimum(p):
    """Exact linear MILP with one-hot and cumulative CPU/memory constraints."""
    n=p['n'];m=2
    coeff=np.asarray(p['cost']).reshape(-1)
    A=[];upper=[];lower=[]
    for t in range(n):
        line=np.zeros(n*m);line[t*m:(t+1)*m]=1;A.append(line);lower.append(1);upper.append(1)
    for r in range(m):
        for name,cap in [('cpu','cpu_caps'),('mem','mem_caps')]:
            line=np.zeros(n*m);line[r::m]=p[name]
            A.append(line);lower.append(0);upper.append(p[cap][r])
    tic=time.perf_counter()
    res=milp(c=coeff,integrality=np.ones(n*m),bounds=Bounds(0,1),
             constraints=LinearConstraint(np.array(A),np.array(lower),np.array(upper)),
             options={'time_limit':20.})
    elapsed=time.perf_counter()-tic
    if res.x is None:raise RuntimeError(f'MILP failure {res.message}')
    assignment=np.argmax(res.x.reshape(n,m),axis=1).astype(int)
    return assignment,elapsed,res.status

def full_diagonal(p):
    n=p['n'];q=n*2;indices=np.arange(2**q,dtype=np.uint32)
    bits=((indices[:,None]>>np.arange(q,dtype=np.uint32)[None,:])&1).astype(np.float64).reshape(-1,n,2)
    ones=((bits.sum(axis=2)-1)**2).sum(axis=1)
    objective=(bits*p['cost'][None,:,:]).sum((1,2))
    c_load=np.einsum('nti,t->ni',bits,p['cpu']);m_load=np.einsum('nti,t->ni',bits,p['mem'])
    c_excess=np.maximum(0,c_load/p['cpu_caps'][None,:]-1)
    m_excess=np.maximum(0,m_load/p['mem_caps'][None,:]-1)
    penalty=max(20.,10.*float(p['cost'].sum()))
    energy=objective+penalty*(ones+(c_excess**2+m_excess**2).sum(axis=1))
    # Normalized energy is for numerical conditioning, not a different argmin.
    return energy.astype(np.float64),float(penalty)

def feasible_diagonal(p):
    n=p['n'];idx=np.arange(2**n)
    assignments=((idx[:,None]>>np.arange(n))&1).astype(int)
    e=[]
    for x in assignments:
        a=assess(p,x)
        # Same constrained cost, no one-hot penalty because all basis states are one-hot.
        penalty=max(20.,10.*float(p['cost'].sum()))
        extra=penalty*((a['cpu_overload']/p['cpu_caps'].min())**2+(a['memory_overload']/p['mem_caps'].min())**2)
        e.append(a['objective']+extra)
    return np.asarray(e)

def apply_rx(state,q,beta):
    # RX(2*beta), matching the standard QAOA exp(-i beta X).
    b=state.reshape(-1,2**(q+1))
    ind0=np.arange(2**q);ind1=ind0+2**q
    a=b[:,ind0].copy();c=b[:,ind1].copy()
    co=np.cos(beta);si=np.sin(beta)
    b[:,ind0]=co*a-1j*si*c;b[:,ind1]=co*c-1j*si*a

def qaoa_dist(energies,n_qubits,depth,angles,feasible_subspace):
    # Both mixer variants are genuine statevector amplitudes, no classical
    # Gibbs/softmax distribution masquerading as a quantum circuit.
    size=2**n_qubits
    state=np.full(size,1/np.sqrt(size),dtype=np.complex128)
    e=(energies-energies.min())/max(np.ptp(energies),1e-9)
    for d in range(depth):
        gamma=angles[d];beta=angles[depth+d]
        state*=np.exp(-1j*gamma*e)
        for j in range(n_qubits):apply_rx(state,j,beta)
    probs=np.abs(state)**2
    return probs/np.sum(probs)

def optimize_quantum(p,mixer,depth,maxfev,seed):
    e,_=full_diagonal(p) if mixer=='RX' else (feasible_diagonal(p),None)
    q=p['n']*2 if mixer=='RX' else p['n']
    e_norm=(e-e.min())/max(np.ptp(e),1e-9)
    rng=np.random.default_rng(seed)
    initial=np.r_[rng.uniform(.4,2.,depth),rng.uniform(.2,1.1,depth)]
    cnt=[0]
    def objective(theta):
        cnt[0]+=1
        prob=qaoa_dist(e,q,depth,theta,mixer=='XY')
        return float(prob@e_norm)
    tic=time.perf_counter()
    opt=minimize(objective,initial,method='Nelder-Mead',options={'maxfev':maxfev,'xatol':.06,'fatol':.0001})
    duration=time.perf_counter()-tic
    prob=qaoa_dist(e,q,depth,opt.x,mixer=='XY')
    return prob,duration,int(cnt[0]),float(opt.fun)

def recover(p,ids,mixer):
    n=p['n'];ids=np.asarray(ids,int)
    if mixer=='RX':
        bit=((ids[:,None]>>np.arange(2*n)[None,:])&1).reshape(-1,n,2)
        onehot=(bit.sum(axis=2)==1).all(axis=1)
        if not onehot.any():return None
        assign=bit[onehot,:,1].astype(int)
    else:assign=((ids[:,None]>>np.arange(n)[None,:])&1).astype(int)
    best=None
    for x in np.unique(assign,axis=0):
        info=assess(p,x)
        if not info['feasible']:continue
        if best is None or info['objective']<best[0]:best=(info['objective'],x)
    return None if best is None else best[1]

def best_of_random(p,rng,draws,uniform_bits=False):
    n=p['n']
    if uniform_bits:
        bit=rng.integers(0,2,size=(draws,n,2),dtype=np.int8)
        ok=(bit.sum(axis=2)==1).all(axis=1)
        choices=bit[ok,:,1]
    else:choices=rng.integers(0,2,size=(draws,n),dtype=np.int8)
    if not len(choices):return None
    best=None
    # Avoid repeated evaluation of duplicate bitstrings.
    for x in np.unique(choices,axis=0):
        info=assess(p,x)
        if not info['feasible']:continue
        if best is None or info['objective']<best[0]:best=(info['objective'],x.copy())
    return None if best is None else best[1]

def greedy(p):
    n=p['n'];x=np.full(n,-1,int)
    order=np.argsort(-np.maximum(p['cpu']/p['cpu_caps'].mean(),p['mem']/p['mem_caps'].mean()))
    for t in order:
        valid=[]
        for r in range(2):
            alt=x.copy();alt[t]=r
            sub=alt>=0
            cc=np.array([p['cpu'][sub&(alt==j)].sum() for j in range(2)])
            mm=np.array([p['mem'][sub&(alt==j)].sum() for j in range(2)])
            if (cc<=p['cpu_caps']+1e-9).all() and (mm<=p['mem_caps']+1e-9).all():valid.append(r)
        if not valid:return None
        x[t]=min(valid,key=lambda r:p['cost'][t,r])
    return x

def local_search(p,start,passes=8):
    if start is None:return None
    x=np.asarray(start,int).copy()
    for _ in range(passes):
        current=assess(p,x);best=None
        for t in range(len(x)):
            cand=x.copy();cand[t]^=1
            s=assess(p,cand)
            if s['feasible'] and s['objective']+1e-9<current['objective']:
                if best is None or s['objective']<best[0]:best=(s['objective'],cand)
        if best is None:break
        x=best[1]
    return x

def choose_blocks(df,n,count,seed):
    rng=np.random.default_rng(seed);out=[]
    # At least two temporal strata. No outcome labels used for choice.
    chunks=np.array_split(np.arange(len(df)),max(2,count))
    for j,segment in enumerate(chunks[:count]):
        if len(segment)<n:continue
        ids=np.sort(rng.choice(segment,size=n,replace=False))
        sub=df.iloc[ids]
        # Keep task heterogeneity and provenance; no synthetic task feature data.
        out.append((j,sub))
    return out

def reference_summary(args,out):
    if not args.reference_zip:return
    zpath=Path(args.reference_zip)
    with zipfile.ZipFile(zpath) as z:
        f=[n for n in z.namelist() if n.endswith('43_qaoa_summary.csv')]
        if f:
            with z.open(f[0]) as src:
                tab=pd.read_csv(src)
                tab.to_csv(out/'46_original_four_bit_reference_ONLY.csv',index=False)
                print('Original 4-bit results copied for comparison only, NOT re-executed')

def stress_runs(tasks,args,out):
    """Larger n=10/12 logical XY tests; 2n-qubit RX cannot fit CPU budget.

    XY statevector uses 2**n feasible logical basis states, logically equivalent
    to one-hot 2-qubit XY mixing for two machines. All strong classical
    baselines are retained. This experiment is a separate scalability study.
    """
    if not args.stress_sizes:return []
    rows=[]
    for n in args.stress_sizes:
        for block,chunk in choose_blocks(tasks,n,args.stress_blocks,args.seed+n+503):
            p=make_instance(chunk,args.seed+n*170+block)
            optcost,xopt=enumerate_optimum(p)
            milp_sol,milp_time,_=milp_optimum(p)
            if abs(assess(p,milp_sol)['objective']-optcost)>1e-5:raise AssertionError('MILP wrong')
            def record(name,x,t,draws,evals):
                info=assess(p,x) if x is not None else dict(feasible=False,objective=np.nan,slo_miss_rate=np.nan)
                rows.append(dict(tasks=n,block=block,method=name,
                    optimal_recovery=int(info['feasible'] and abs(info['objective']-optcost)<1e-6),
                    feasible_rate=int(info['feasible']),
                    objective_gap_pct=100*(info['objective']/optcost-1) if info['feasible'] else np.nan,
                    slo_miss_rate=info['slo_miss_rate'],runtime_seconds=t,draws=draws,expectation_evals=evals))
            record('MILP',milp_sol,milp_time,0,0)
            tic=time.perf_counter();gre=greedy(p);record('Greedy',gre,time.perf_counter()-tic,0,0)
            tic=time.perf_counter();record('Greedy+local_search',local_search(p,gre),time.perf_counter()-tic,0,0)
            for shots in (512,2048,max(args.maxfev*512,512)):
                label='Random_onehot_'+('optimizer_budget' if shots>2048 else str(shots))
                tic=time.perf_counter();x=best_of_random(p,np.random.default_rng(args.seed+n*40+block*13+shots),shots)
                record(label,x,time.perf_counter()-tic,shots,0)
            for depth in (1,2):
                prob,train_time,neval,_=optimize_quantum(p,'XY',depth,args.maxfev,args.seed+n*900+block*33+depth)
                for shots in (512,1024):
                    tic=time.perf_counter()
                    ids=np.random.default_rng(args.seed+n*400+block*15+depth*7+shots).choice(len(prob),size=shots,p=prob)
                    x=recover(p,ids,'XY')
                    record(f'XY_p{depth}_shots{shots}',x,train_time+time.perf_counter()-tic,shots,neval)
            print(f'STRESS {n} tasks, block {block} completed',flush=True)
    tab=pd.DataFrame(rows)
    tab.to_csv(out/'46_xy_scalability_instances.csv',index=False)
    tab.groupby(['tasks','method'],as_index=False).agg(instances=('block','count'),
       optimal_recovery=('optimal_recovery','mean'),feasibility=('feasible_rate','mean'),
       mean_gap_pct=('objective_gap_pct','mean'),mean_runtime_s=('runtime_seconds','mean'))\
       .to_csv(out/'46_xy_scalability_summary.csv',index=False)
    return rows

def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tasks=read_tasks(args.data)
    print('Unique real tasks with CPU/mem:',len(tasks),flush=True)
    experiments=[];meta=[]
    for n in args.sizes:
        for block,chunk in choose_blocks(tasks,n,args.blocks,args.seed+n):
            p=make_instance(chunk,args.seed+n*100+block)
            optcost,xopt=enumerate_optimum(p)
            xmilp,milp_time,status=milp_optimum(p)
            if abs(assess(p,xmilp)['objective']-optcost)>1e-5:raise AssertionError('MILP not exact')
            meta.append({'tasks':n,'block':block,'source_t_min':p['source_t_min'],'source_t_max':p['source_t_max'],
                'cpu_caps':p['cpu_caps'].tolist(),'mem_caps':p['mem_caps'].tolist(),
                'task_ids':p['ids'],'milp_status':int(status),'optimal_obj':float(optcost)})
            def record(method,x,elapsed,evals,shots,energy_evals=0,optimizer_sharing=False):
                if x is None:
                    info={'feasible':False,'objective':None,'cpu_overload':None,'memory_overload':None,
                          'cpu_utilization':None,'memory_utilization':None,'slo_miss_rate':None,
                          'makespan':None,'total_energy':None,'assignment':None}
                else:info=assess(p,x)
                recovery=bool(info['feasible'] and abs(info['objective']-optcost)<1e-6)
                experiments.append({'tasks':n,'block':block,'method':method,**info,
                   'optimal_recovery':int(recovery),'objective_gap_pct':float(100*(info['objective']/optcost-1)) if info['feasible'] else None,
                   'runtime_seconds':elapsed,'candidate_budget':evals,'shots':shots,
                   'expectation_evals':energy_evals,'optimizer_params_shared_across_shots':optimizer_sharing})
            record('MILP',xmilp,milp_time,0,0)
            tic=time.perf_counter();gx=greedy(p);record('Greedy',gx,time.perf_counter()-tic,0,0)
            tic=time.perf_counter();lx=local_search(p,gx);record('Greedy+local_search',lx,time.perf_counter()-tic,0,0)
            for label,num,bit in [('Random_bits_512',512,True),('Random_onehot_512',512,False),('Random_onehot_2048',2048,False)]:
                rng=np.random.default_rng(args.seed+n*10000+block*20+num+(1 if bit else 0))
                tic=time.perf_counter();x=best_of_random(p,rng,num,uniform_bits=bit)
                record(label,x,time.perf_counter()-tic,num,num)
            configs=[('RX',1,128),('RX',1,512),('RX',2,512),('RX',2,1024),
                     ('XY',1,128),('XY',1,512),('XY',2,512),('XY',2,1024)]
            opt_cache={}
            for mixer,depth,shots in configs:
                if (mixer,depth) not in opt_cache:
                    opt_cache[(mixer,depth)]=optimize_quantum(p,mixer,depth,args.maxfev,
                         args.seed+n*1000+block*15+(depth*3)+(1 if mixer=='XY' else 0))
                prob,train_time,evals,value=opt_cache[(mixer,depth)]
                rng=np.random.default_rng(args.seed+n*17000+block*117+depth*13+shots+(0 if mixer=='RX' else 50000))
                tic=time.perf_counter();draws=rng.choice(len(prob),size=shots,p=prob)
                sol=recover(p,draws,mixer)
                rt=train_time+time.perf_counter()-tic
                record(f'{mixer}_p{depth}_shots{shots}',sol,rt,shots,shots,evals,
                       optimizer_sharing=True)
            # Fairer comparison: classical one-hot evaluates at least as many
            # candidates as the QAOA total of objective evaluations and shots.
            total_budget=max(args.maxfev*512,512)
            rng=np.random.default_rng(args.seed+block+n*33)
            tic=time.perf_counter();rb=best_of_random(p,rng,total_budget)
            record('Random_onehot_optimizer_budget',rb,time.perf_counter()-tic,total_budget,total_budget)
            print(f'{n} tasks block {block}: MILP={optcost:.3f}; {len(experiments)} total solver rows',flush=True)
    table=pd.DataFrame(experiments)
    table.to_csv(out/'46_qaoa_instance_results.csv',index=False)
    summary=table.groupby(['tasks','method'],as_index=False).agg(instances=('block','count'),
       recovery=('optimal_recovery','mean'),feasible_rate=('feasible','mean'),
       mean_gap_pct=('objective_gap_pct','mean'),mean_slo_miss=('slo_miss_rate','mean'),
       mean_makespan=('makespan','mean'),mean_cpu_util=('cpu_utilization','mean'),
       mean_mem_util=('memory_utilization','mean'),
       mean_energy=('total_energy','mean'),mean_runtime_s=('runtime_seconds','mean'),
       mean_evals=('expectation_evals','mean'))
    summary.to_csv(out/'46_qaoa_summary.csv',index=False)
    (out/'46_instances.json').write_text(json.dumps(meta,indent=2))
    # Instance-paired gaps vs each strong comparator, not aggregate unpaired percent.
    comparisons=[]
    quantum=[x for x in table.method.unique() if x.startswith('RX_') or x.startswith('XY_')]
    baselines=['MILP','Greedy','Greedy+local_search','Random_onehot_512',
               'Random_onehot_2048','Random_onehot_optimizer_budget']
    for n in args.sizes:
        for name in quantum:
            aa=table[(table.tasks==n)&(table.method==name)].set_index('block')
            for b in baselines:
                bb=table[(table.tasks==n)&(table.method==b)].set_index('block')
                ix=aa.index.intersection(bb.index)
                if len(ix)==0:continue
                aa0=aa.loc[ix];bb0=bb.loc[ix]
                comparisons.append({'tasks':n,'qaoa_method':name,'baseline':b,
                     'paired_instances':len(ix),
                     'delta_optimal_recovery':float((aa0.optimal_recovery-bb0.optimal_recovery).mean()),
                     'delta_feasibility':float((aa0.feasible.astype(int)-bb0.feasible.astype(int)).mean()),
                     'runtime_ratio':float(aa0.runtime_seconds.mean()/max(bb0.runtime_seconds.mean(),1e-12))})
    pd.DataFrame(comparisons).to_csv(out/'46_qaoa_paired.csv',index=False)
    reference_summary(args,out)
    stress=stress_runs(tasks,args,out)
    strict_all=all(all(c['delta_optimal_recovery']>0 for c in comparisons if c['tasks']==n and c['qaoa_method']==q and c['baseline'] in baselines)
                  for n in args.sizes for q in quantum)
    verdict={'sizes':args.sizes,'blocks_per_size':args.blocks,'eight_qaoa_configs':quantum,
        'no_optimization_on_test_labels':True,'scheduling_instances_from_real_cpu_memory_requests':True,
        'nonlinear_diagonal_capacity_penalty_not_qubo':True,
        'original_four_bit_reference_separate':bool(args.reference_zip),
        'quantum_beats_all_classical':bool(strict_all),
        'hardware_quantum_advantage_demonstrated':False,
        'gpu_available':False,'simulator':'numpy complex statevector CPU',
        'xy_stress_sizes':args.stress_sizes,'xy_stress_rows':len(stress),
        'warning':'Original RX circuit architecture retained, but enlarged problem H_C uses multi-body diagonal penalties. XY is a modified one-hot mixer; neither is proof of computational advantage. SLO outcomes for generated cases are model-derived, not observed cloud SLA measurements.'}
    (out/'46_qaoa_verdict.json').write_text(json.dumps(verdict,indent=2))
    print('RESULT FILES',out.resolve())
    return verdict

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=r'E:\other\AQUA-SLA\Dataset\processed\aqua_sla_prepared.csv')
    p.add_argument('--out',default=r'E:\other\AQUA-SLA\results\46_qaoa_independent')
    p.add_argument('--reference-zip',default=None)
    p.add_argument('--sizes',type=int,nargs='+',default=[4,6])
    p.add_argument('--blocks',type=int,default=12)
    p.add_argument('--maxfev',type=int,default=60)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--stress-sizes',type=int,nargs='*',default=[10,12])
    p.add_argument('--stress-blocks',type=int,default=6)
    p.add_argument('--fast',action='store_true')
    a=p.parse_args()
    if a.fast:a.blocks=4;a.maxfev=22;a.sizes=[4,6];a.stress_sizes=[10];a.stress_blocks=3
    run(a)
if __name__=='__main__':main()
