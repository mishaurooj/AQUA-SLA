#!/usr/bin/env python3
r"""AQUA-SLA Script 51: 20 matched predictive-ablation variants + 20 QAOA variants.

Research experiments, NEVER a guarantee of superiority. On the original trace,
all evaluation periods have been examined in earlier Scripts 35-50: report these
results as EXPLORATORY. A new external trace is needed for independent evidence.

Prediction: original six-qubit 2-repetition ZZFeatureMap, train-only bandwidth
alignment, 20 landmark-feature augmented boosted-tree models with equivalent
classical RBF feature controls, and matched pure-kernel SVM comparisons.
Prediction ablation selection on Validation A; fusion weights on Validation B;
held-out time block only for evaluation. Original early-time features only.

Scheduling: 20 assignment-preserving XY-QAOA variants (depth 1..3, ordinary
mean/CVaR objectives, warm starts) versus exact MILP, greedy + local search,
512/2048-draw and optimizer-budget-matched classical search. Reports full
optimizer runtime and repair time. Simulated statevectors are NOT a quantum
hardware benchmark. Inputs are observed tasks but generated two-host scenarios.

Requirements: numpy pandas scipy scikit-learn, plus the three companion files
47_hcqkl_bandwidth_ablation.py, 46_qaoa_independent_ablation.py,
47_qaoa_warmstart_ablation.py alongside this script.

Examples:
  python 51_aqua_sla_joint_new_ablations.py --data processed.zip --mode both --smoke
  python 51_aqua_sla_joint_new_ablations.py --data ..\Dataset\processed\aqua_sla_prepared.csv --mode both --output ..\results\51_final
"""
from __future__ import annotations
import argparse, importlib.util, json, time, warnings, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

HERE=Path(__file__).resolve().parent

def dep(file, name):
    p=HERE/file
    if not p.exists():raise FileNotFoundError(f'Required companion file missing: {p}')
    spec=importlib.util.spec_from_file_location(name,p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def tojson(x):
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x))

def dump(path,thing):Path(path).write_text(json.dumps(thing,indent=2,default=tojson),encoding='utf8')

def score(y,p):
    if len(np.unique(y))<2:return -999
    return float(roc_auc_score(y,p)+.20*average_precision_score(y,p))

def aug_model(seed,fast):
    return HistGradientBoostingClassifier(max_iter=65 if fast else 160,
        learning_rate=.065,max_leaf_nodes=31,l2_regularization=1.,random_state=seed)

def prediction(a,out):
    m=dep('47_hcqkl_bandwidth_ablation.py','aqua47_pred')
    data,meta=m.lifecycle_300(m.read_data(a.data))
    cut=[int(len(data)*v) for v in (.6,.7,.8)]
    t0=data.iloc[:cut[0]].copy(); v1=data.iloc[cut[0]:cut[1]].copy()
    v2=data.iloc[cut[1]:cut[2]].copy(); ext=data.iloc[cut[2]:].copy()
    # Strict 300-second embargo at every chronological boundary.
    t0=t0[t0.time_seconds+300<v1.time_seconds.min()]
    v1=v1[v1.time_seconds+300<v2.time_seconds.min()]
    v2=v2[v2.time_seconds+300<ext.time_seconds.min()]
    if a.external_test:
        ext,external_meta=m.lifecycle_300(m.read_data(a.external_test))
        meta['external_meta']=external_meta
        origin='external_supplied_new_trace_independence_user_must_verify'
    else:origin='internal_already_examined_exploratory'
    rng=np.random.default_rng(a.seed)
    def choose(df,n):
        if n>=len(df) or n<=0:return df.copy()
        ids=np.sort(rng.choice(len(df),n,replace=False));return df.iloc[ids].copy()
    t0=choose(t0,a.train_n);v1=choose(v1,a.val_n);v2=choose(v2,a.val_n);ext=choose(ext,a.test_n)
    ytr=t0.future_sla_event.to_numpy(int)
    y1=v1.future_sla_event.to_numpy(int)
    y2=v2.future_sla_event.to_numpy(int)
    yte=ext.future_sla_event.to_numpy(int)
    for tag,y in [('train',ytr),('valA',y1),('valB',y2),('test',yte)]:
        if len(np.unique(y))!=2:raise ValueError(f'{tag}: only one class. Increase sample size.')
    features=m.FEATURES
    imp=SimpleImputer(strategy='median')
    X=imp.fit_transform(t0[features]);A=imp.transform(v1[features]);B=imp.transform(v2[features]);T=imp.transform(ext[features])
    print('[51-PRED] sample sizes',list(map(len,(ytr,y1,y2,yte))),'test prevalence',yte.mean(),flush=True)
    baseline={}
    models={
      'HGB31':aug_model(a.seed,a.smoke),
      'HGB63':HistGradientBoostingClassifier(max_iter=80 if a.smoke else 160,max_leaf_nodes=63,learning_rate=.04,l2_regularization=3.,random_state=a.seed),
      'RF':RandomForestClassifier(n_estimators=75 if a.smoke else 200,max_depth=16,min_samples_leaf=8,n_jobs=min(a.workers,4),random_state=a.seed),
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost']=XGBClassifier(n_estimators=85 if a.smoke else 220,max_depth=5,learning_rate=.06,
            subsample=.9,colsample_bytree=.9,tree_method='hist',n_jobs=min(a.workers,4),
            random_state=a.seed,eval_metric='logloss')
    except ImportError:pass
    for name,model in models.items():
        tic=time.perf_counter();model.fit(X,ytr)
        baseline[name]={'a':model.predict_proba(A)[:,1],'b':model.predict_proba(B)[:,1],
                        't':model.predict_proba(T)[:,1], 'runtime_s':time.perf_counter()-tic}
        print('[51-PRED] baseline',name,round(score(y1,baseline[name]['a']),4),flush=True)
    # Leakage-free encoding; quantile mapping avoids out-of-range MinMax extrapolation.
    robust=RobustScaler().fit(X);pca=PCA(n_components=6,random_state=a.seed).fit(robust.transform(X))
    transform=QuantileTransformer(n_quantiles=min(300,len(X)),output_distribution='uniform',random_state=a.seed)
    transform.fit(pca.transform(robust.transform(X)))
    def emb(z):return np.pi*np.clip(transform.transform(pca.transform(robust.transform(z))),0,1)
    Z,ZA,ZB,ZT=[emb(x) for x in (X,A,B,T)]
    # Script 47 internally chooses a label-stratified subset *from training only*.
    # A random small sample often contains no positives (2.54% or lower),
    # erroneously making "trainable" quantum maps identical to originals.
    # This avoids that failure while retaining validation/test isolation.
    theta,fit=m.fit_trainable(Z,ytr,a.seed,a.align_evals)
    dump(out/'51_quantum_alignment.json',fit)
    np.savez_compressed(out/'51_quantum_parameters.npz',bandwidth=theta,
        pca_components=pca.components_,explained_variance=pca.explained_variance_ratio_)
    m.export_zz_qasm(Z[0]*theta,out/'51_original_architecture_trained_example.qasm')
    largest=min(max(a.landmarks),len(X))
    landmarks=np.sort(rng.choice(len(X),largest,replace=False))
    # Quantum state preparation is shared by all 20 configurations.
    S0=m.original_zz_states(Z);Sa0=m.original_zz_states(ZA);Sb0=m.original_zz_states(ZB)
    S1=m.original_zz_states(Z*theta);Sa1=m.original_zz_states(ZA*theta);Sb1=m.original_zz_states(ZB*theta)
    sr={ 'orig':(S0,Sa0,Sb0), 'train':(S1,Sa1,Sb1)}
    # Gamma values fixed by training landmark distribution, never by validation/test labels.
    d=np.sum((Z[landmarks,None,:]-Z[None,landmarks,:])**2,axis=2)
    median=float(np.median(d[d>0])) if np.any(d>0) else 1.
    gammas={'g1':1/max(1e-6,median)}
    cfg=[('Q_orig','orig',0.,'g1'),('Q_train','train',0.,'g1')]
    cfg += [(family,'orig',1.,'g1') for family in ('RBF','Laplacian','Poly2')]
    cfg += [(f'HCQKL_orig_a{int(alpha*100)}','orig',alpha,'g1') for alpha in (.25,.50,.75)]
    cfg += [(f'HCQKL_train_a{int(alpha*100)}','train',alpha,'g1') for alpha in (.25,.50)]
    assert len(cfg)==10
    # To avoid test-peeking, materialize only training and validation kernels now.
    def kernel_parts(which,alpha,gamma,ids,method='RBF'):
        ss,sa,sb=sr[which]
        sref=ss[ids]
        def compute(states,x):
            kq=np.abs(states@sref.conj().T)**2 if alpha<1 else 0.
            if alpha:
                if method=='Laplacian':
                    dist=np.sum(np.abs(x[:,None,:]-Z[ids][None,:,:]),axis=2)
                    kr=np.exp(-np.sqrt(gamma)*dist)
                elif method=='Poly2':
                    kr=((1.+(x@Z[ids].T)/(6*np.pi*np.pi))/2.)**2
                else:kr=rbf_kernel(x,Z[ids],gamma=gamma)
            else:kr=0.
            return np.asarray(alpha*kr+(1-alpha)*kq,dtype=np.float32)
        return compute(ss,Z),compute(sa,ZA),compute(sb,ZB)
    configs=[];stored={};idx_map={}
    for cname,kind,alpha,gname in cfg:
        gamma=gammas[gname]
        for nlan in a.landmarks:
            ids=landmarks[:min(nlan,len(landmarks))]
            item=f'{cname}_L{len(ids)}'
            tic=time.perf_counter();kt,ka,kb=kernel_parts(kind,alpha,gamma,ids,cname)
            mdl=aug_model(a.seed,a.smoke)
            mdl.fit(np.column_stack([X,kt]),ytr)
            pa=mdl.predict_proba(np.column_stack([A,ka]))[:,1]
            pb=mdl.predict_proba(np.column_stack([B,kb]))[:,1]
            rec=dict(name=item,kernel=cname,kind=kind,alpha=alpha,gamma=gamma,landmarks=len(ids),
                     family='classical_kernel' if alpha==1 else 'quantum_augmented',
                     valA_roc_auc=float(roc_auc_score(y1,pa)),valA_pr_auc=float(average_precision_score(y1,pa)),
                     selection_objective=score(y1,pa),training_s=time.perf_counter()-tic)
            configs.append(rec);stored[item]=(mdl,ids,kind,alpha,gamma,pa,pb,cname)
            print('[51-PRED]',item,'VA_ROC',round(rec['valA_roc_auc'],4),flush=True)
    assert len(configs)==20
    pd.DataFrame(configs).sort_values('selection_objective',ascending=False).to_csv(out/'51_20_prediction_variants_validationA.csv',index=False)
    # Strong baseline selected on VA. Quantum-vs-classical landmark controls likewise.
    best_base=max(baseline,key=lambda k:score(y1,baseline[k]['a']))
    qa=[v for v in configs if v['family']=='quantum_augmented']
    ca=[v for v in configs if v['family']=='classical_kernel']
    best_q=max(qa,key=lambda v:v['selection_objective'])['name']
    best_c=max(ca,key=lambda v:v['selection_objective'])['name']
    selected={'best_classical_base':best_base,'quantum_augmentation':best_q,'classical_kernel_augmentation':best_c}
    print('[51-PRED] selected on validation-A',selected,flush=True)
    # Fusion weight selection uses VALIDATION B only, without test scores.
    weights=(0.,.05,.10,.25,.50,.75,1.)
    grid=[];chosen={}
    for group,name in [('quantum',best_q),('classical_kernel',best_c)]:
        vbase=baseline[best_base]['b'];vkernel=stored[name][-2]
        for w in weights:
            p=(1-w)*vbase+w*vkernel
            item={'group':group,'base':best_base,'variant':name,'quantum_or_RBF_weight':w,
                  'valB_roc_auc':float(roc_auc_score(y2,p)),
                  'valB_pr_auc':float(average_precision_score(y2,p)),
                  'selection_objective':score(y2,p)}
            grid.append(item)
        chosen[group]=max((r for r in grid if r['group']==group),key=lambda r:r['selection_objective'])
    pd.DataFrame(grid).to_csv(out/'51_fusion_validationB.csv',index=False)
    # Freeze all choices. Score test once, after selection.
    def test_aug(name):
        mdl,ids,kind,alpha,gamma,pa,pb,kname=stored[name]
        sref=sr[kind][0][ids]
        preds=[]
        for j in range(0,len(T),max(64,a.score_batch)):
            z=ZT[j:j+a.score_batch]
            kq=np.abs(m.original_zz_states(z*(theta if kind=='train' else 1.))@sref.conj().T)**2 if alpha<1 else 0
            if alpha:
                if kname=='Laplacian':
                    dist=np.sum(np.abs(z[:,None,:]-Z[ids][None,:,:]),axis=2)
                    kr=np.exp(-np.sqrt(gamma)*dist)
                elif kname=='Poly2':kr=((1.+(z@Z[ids].T)/(6*np.pi*np.pi))/2.)**2
                else:kr=rbf_kernel(z,Z[ids],gamma=gamma)
            else:kr=0.
            k=alpha*kr+(1-alpha)*kq
            preds.append(mdl.predict_proba(np.column_stack([T[j:j+a.score_batch],k]))[:,1])
        return np.concatenate(preds)
    qt=test_aug(best_q);ct=test_aug(best_c)
    predictions={'classical_'+best_base:baseline[best_base]['t'],
                 'selected_quantum_augmented':qt,
                 'selected_classical_kernel_augmented':ct,
                 'selected_quantum_fusion':(1-chosen['quantum']['quantum_or_RBF_weight'])*baseline[best_base]['t']+chosen['quantum']['quantum_or_RBF_weight']*qt,
                 'selected_classical_kernel_fusion':(1-chosen['classical_kernel']['quantum_or_RBF_weight'])*baseline[best_base]['t']+chosen['classical_kernel']['quantum_or_RBF_weight']*ct}
    # Keep every classical baseline, including the strongest on TEST (diagnostic only).
    for key,val in baseline.items():predictions['classical_'+key]=val['t']
    # Training-only Platt model would be ideal but all model probabilities are native.
    # F1 threshold chosen via validation B for every recorded model.
    vbpred={'selected_quantum_augmented':stored[best_q][-2],
            'selected_classical_kernel_augmented':stored[best_c][-2],
            'selected_quantum_fusion':(1-chosen['quantum']['quantum_or_RBF_weight'])*baseline[best_base]['b']+chosen['quantum']['quantum_or_RBF_weight']*stored[best_q][-2],
            'selected_classical_kernel_fusion':(1-chosen['classical_kernel']['quantum_or_RBF_weight'])*baseline[best_base]['b']+chosen['classical_kernel']['quantum_or_RBF_weight']*stored[best_c][-2]}
    for name,rec in baseline.items():vbpred['classical_'+name]=rec['b']
    from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score,precision_score,recall_score,brier_score_loss
    report=[]
    for name,p in predictions.items():
        w=vbpred[name]
        cutoffs=np.quantile(w,np.linspace(.02,.99,90));cutoffs=np.unique(cutoffs)
        threshold=float(max(cutoffs,key=lambda t:f1_score(y2,w>=t,zero_division=0)))
        yhat=(p>=threshold)
        k=max(1,int(.10*len(yte)));top=np.argsort(-p,kind='mergesort')[:k]
        report.append(dict(model=name,n=len(yte),prevalence=float(yte.mean()),
            roc_auc=float(roc_auc_score(yte,p)),pr_auc=float(average_precision_score(yte,p)),
            accuracy=float(accuracy_score(yte,yhat)),balanced_accuracy=float(balanced_accuracy_score(yte,yhat)),
            f1=float(f1_score(yte,yhat,zero_division=0)),precision=float(precision_score(yte,yhat,zero_division=0)),
            recall=float(recall_score(yte,yhat,zero_division=0)),brier=float(brier_score_loss(yte,p)),
            top10_recall=float(yte[top].sum()/max(1,yte.sum())),threshold=threshold))
    pd.DataFrame(report).sort_values('roc_auc',ascending=False).to_csv(out/'51_prediction_final_metrics.csv',index=False)
    pd.DataFrame({'source_row':ext._source_row.to_numpy(),'time_seconds':ext.time_seconds.to_numpy(),
        'y_true':yte,**predictions}).to_csv(out/'51_prediction_test_scores.csv',index=False)
    # Temporal moving-block paired CI, no fake independence from seed repetitions.
    rng2=np.random.default_rng(a.seed+170)
    def ci(name,control):
        pn=predictions[name];pc=predictions[control]
        delta=float(roc_auc_score(yte,pn)-roc_auc_score(yte,pc));ap=float(average_precision_score(yte,pn)-average_precision_score(yte,pc))
        blocks=np.array_split(np.arange(len(yte)),min(12,max(2,len(yte)//30)))
        ds=[]
        for _ in range(a.bootstrap):
            ii=np.concatenate([blocks[j] for j in rng2.integers(0,len(blocks),len(blocks))])
            if len(np.unique(yte[ii]))<2:continue
            ds.append((roc_auc_score(yte[ii],pn[ii])-roc_auc_score(yte[ii],pc[ii]),
                       average_precision_score(yte[ii],pn[ii])-average_precision_score(yte[ii],pc[ii])))
        if not ds:return dict(method=name,control=control,delta_roc_auc=delta,delta_pr_auc=ap)
        x=np.asarray(ds);bounds=np.quantile(x,[.025,.975],axis=0)
        return dict(method=name,control=control,delta_roc_auc=delta,delta_pr_auc=ap,
                    roc_low=float(bounds[0,0]),roc_high=float(bounds[1,0]),
                    pr_low=float(bounds[0,1]),pr_high=float(bounds[1,1]),
                    support_roc_superiority=bool(bounds[0,0]>0),support_pr_superiority=bool(bounds[0,1]>0))
    qs='selected_quantum_fusion'; rs='selected_classical_kernel_fusion'
    compare=[ci(qs,'classical_'+best_base),ci(qs,rs),ci('selected_quantum_augmented','selected_classical_kernel_augmented')]
    strongest=max((x for x in predictions if x.startswith('classical_')),
                  key=lambda x:roc_auc_score(yte,predictions[x]))
    if strongest!='classical_'+best_base:compare.append(ci(qs,strongest))
    pd.DataFrame(compare).to_csv(out/'51_prediction_paired_CI.csv',index=False)
    verdict={'data_origin':origin,'data_meta':meta,'prediction_variants':20,
             'test_n':int(len(yte)),'test_positive':int(yte.sum()),'valA_model_choices':selected,
             'valB_fusion_choices':chosen,'strongest_classical_on_test_DIAGNOSTIC':strongest,
             'paired':compare,'independent_confirmation':False,
             'predictive_superiority_supported_exploratory':bool(all(
                 row.get('roc_low',-99)>0 and row.get('pr_low',-99)>0 for row in compare[:2])),
             'claim_boundary':'Old trace has repeatedly been inspected, so even significant CIs are exploratory.'}
    dump(out/'51_prediction_verdict.json',verdict)
    print('[51-PRED] final AUC',[(x['model'],round(x['roc_auc'],4)) for x in report],flush=True)
    return verdict


def cvar(energy,probs,order,alpha):
    probs=probs[order];ener=energy[order]
    cum=np.cumsum(probs);prev=cum-probs
    taken=np.clip(alpha-prev,0.,probs)
    return float((taken@ener)/alpha)


def scheduling(a,out):
    q=dep('46_qaoa_independent_ablation.py','aqua46_sch')
    w=dep('47_qaoa_warmstart_ablation.py','aqua47_sch')
    taskdf=q.read_tasks(a.data)
    # 3 objectives x 3 depths x 2 starts + 2 original uniform-start = 20.
    variants=[(obj,depth,strength) for obj in ('mean','cvar20','cvar50')
              for depth in (1,2,3) for strength in (.60,.90)]
    variants += [('mean',1,None),('mean',2,None)]
    assert len(variants)==20
    rows=[];origin=[]
    for n in a.task_sizes:
        if n>16:raise ValueError('CPU statevector simulation limited to <=16 logical task bits')
        for block,chunk in q.choose_blocks(taskdf,n,a.blocks,a.seed+n+5151):
            p=q.make_instance(chunk,a.seed+n*53+block)
            milp,t_exact,status=q.milp_optimum(p)
            if status!=0:raise ValueError('MILP failed to certify optimum')
            opt=float(q.assess(p,milp)['objective'])
            e=q.feasible_diagonal(p)
            orig=time.perf_counter();center=q.greedy(p)
            repair='greedy'
            if center is None:
                center,repair_cost=w.repair_feasible_seed(p)
                init_cost=time.perf_counter()-orig+repair_cost
                repair='capacity_only_MILP_repair'
            else:init_cost=time.perf_counter()-orig
            if not q.assess(p,center)['feasible']:raise AssertionError('Invalid warm-start assignment')
            def save(name,x,t,evals,shots):
                result=q.assess(p,x) if x is not None else None
                rows.append(dict(tasks=n,block=block,method=name,optimal_recovery=int(result is not None and result['feasible'] and abs(result['objective']-opt)<1e-7),
                       feasible=int(result is not None and result['feasible']),
                       objective=float(result['objective']) if result else np.nan,
                       relative_gap_pct=100*(result['objective']/opt-1) if result and result['feasible'] else np.nan,
                       slo_miss_rate=float(result['slo_miss_rate']) if result else np.nan,
                       makespan=float(result['makespan']) if result else np.nan,
                       cpu_utilization=float(result['cpu_utilization']) if result else np.nan,
                       memory_utilization=float(result['memory_utilization']) if result else np.nan,
                       energy=float(result['total_energy']) if result else np.nan,
                       runtime_s=float(t),optimizer_evaluations=int(evals),shots=int(shots)))
            save('MILP',milp,t_exact,0,0)
            save('warm_seed',center,init_cost,0,0)
            start=time.perf_counter();local=q.local_search(p,center)
            save('greedy_local_search',local,init_cost+time.perf_counter()-start,0,0)
            for draw in (512,2048,max(4096,a.maxfev*a.shots)):
                rg=np.random.default_rng(a.seed+n*7+block*17+draw)
                start=time.perf_counter();pick=q.best_of_random(p,rg,draw,uniform_bits=False)
                save('random_budget' if draw>2048 else 'random_'+str(draw),pick,time.perf_counter()-start,draw,draw)
                start=time.perf_counter();pick=w.best_neighborhood(p,q,center,rg,draw)
                save('neighborhood_budget' if draw>2048 else 'neighborhood_'+str(draw),pick,init_cost+time.perf_counter()-start,draw,draw)
            ee=(e-e.min())/max(np.ptp(e),1e-12)
            order=np.argsort(ee,kind='mergesort')
            for obj,depth,strength in variants:
                name=f'QAOA_{obj}_p{depth}_s'+('uniform' if strength is None else str(int(strength*100)))
                rg=np.random.default_rng(a.seed+n*411+block*101+depth*17+int((strength or 0)*100)+len(obj)*13)
                start=np.r_[rg.uniform(.4,1.8,depth),rg.uniform(.2,1.0,depth)]
                calls=[0]
                def loss(angle):
                    calls[0]+=1
                    prob=w.amplitude_distribution(e,n,depth,angle,center if strength is not None else None,strength or .5)
                    return float(prob@ee) if obj=='mean' else cvar(ee,prob,order,.2 if obj=='cvar20' else .5)
                tic=time.perf_counter()
                res=minimize(loss,start,method='Nelder-Mead',options={'maxfev':a.maxfev,'xatol':.075,'fatol':.00025})
                prob=w.amplitude_distribution(e,n,depth,res.x,center if strength is not None else None,strength or .5)
                ids=rg.choice(len(prob),size=a.shots,p=prob)
                best=q.recover(p,ids,'XY')
                save(name,best,time.perf_counter()-tic+(init_cost if strength is not None else 0),calls[0],a.shots)
            origin.append(dict(tasks=n,block=block,start_type=repair,optimum=opt,
                               earliest_task_time=float(chunk.time_seconds.min()),latest_task_time=float(chunk.time_seconds.max())))
            print(f'[51-QAOA] tasks={n} block={block} all 20 configurations completed',flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(out/'51_qaoa_instance_results.csv',index=False)
    summary=raw.groupby(['tasks','method'],as_index=False).agg(
        optimal_recovery=('optimal_recovery','mean'),feasible_rate=('feasible','mean'),
        mean_gap_pct=('relative_gap_pct','mean'),mean_slo_miss=('slo_miss_rate','mean'),
        mean_makespan=('makespan','mean'),mean_cpu_util=('cpu_utilization','mean'),
        mean_mem_util=('memory_utilization','mean'),mean_energy=('energy','mean'),
        mean_runtime_s=('runtime_s','mean'),instances=('block','count'))
    summary.to_csv(out/'51_qaoa_summary.csv',index=False)
    chosen=[]
    for n in a.task_sizes:
        sub=raw[raw.tasks==n];block_ids=sorted(sub.block.unique());pivot=max(1,len(block_ids)//2)
        training=block_ids[:pivot];testing=block_ids[pivot:]
        if not testing:continue
        candidates=sub[sub.method.str.startswith('QAOA_')]
        chosen_name=str(candidates[candidates.block.isin(training)].groupby('method',as_index=False)
               .optimal_recovery.mean().sort_values(['optimal_recovery','method'],ascending=[False,True]).iloc[0]['method'])
        onhold=sub[sub.block.isin(testing)];qa=onhold[onhold.method==chosen_name].set_index('block')
        for control in ['MILP','greedy_local_search','random_512','random_2048','random_budget','neighborhood_budget']:
            ref=onhold[onhold.method==control].set_index('block');both=qa.index.intersection(ref.index)
            if not len(both):continue
            di=qa.loc[both,'optimal_recovery'].to_numpy()-ref.loc[both,'optimal_recovery'].to_numpy()
            chosen.append(dict(tasks=n,selection_blocks=len(training),evaluation_blocks=len(both),
                selected_QAOA=chosen_name,comparator=control,
                qaoa_recovery=float(qa.loc[both,'optimal_recovery'].mean()),
                baseline_recovery=float(ref.loc[both,'optimal_recovery'].mean()),
                delta_recovery=float(di.mean()),wins=int((di>0).sum()),losses=int((di<0).sum()),
                qaoa_mean_runtime_s=float(qa.loc[both,'runtime_s'].mean()),
                baseline_mean_runtime_s=float(ref.loc[both,'runtime_s'].mean())))
    pd.DataFrame(chosen).to_csv(out/'51_qaoa_validation_selected_paired.csv',index=False)
    dump(out/'51_qaoa_instance_provenance.json',origin)
    verdict={'qaoa_variants':20,'original_four_bit_QUBOs_rerun':False,
             'two_host_generated_scheduling_instances_using_observed_task_resources':True,
             'simulator':'CPU 2**n logical statevector, NOT quantum hardware',
             'optimization':'mean/CVaR lower-energy objective, fixed optimization evaluations and final shots',
             'paired':chosen,'computational_advantage_proven':False,
             'warning':'MILP exact optimum; matching MILP is not beating it. Random budget counts optimizer evaluations times shots; simulated QAOA has substantial extra classical cost.'}
    dump(out/'51_qaoa_verdict.json',verdict)
    return verdict


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default=r'..\Dataset\processed\aqua_sla_prepared.csv')
    ap.add_argument('--external-test',default=None,help='Separate untouched source trace; must contain lifecycle fields')
    ap.add_argument('--output',default=r'..\results\51_reviewer_experiments')
    ap.add_argument('--mode',choices=['prediction','scheduling','both'],default='both')
    ap.add_argument('--train-n',type=int,default=18000)
    ap.add_argument('--val-n',type=int,default=2500)
    ap.add_argument('--test-n',type=int,default=6000)
    ap.add_argument('--align-evals',type=int,default=55)
    ap.add_argument('--landmarks',type=int,nargs='+',default=[24,48])
    ap.add_argument('--score-batch',type=int,default=1000)
    ap.add_argument('--task-sizes',type=int,nargs='+',default=[10,12])
    ap.add_argument('--blocks',type=int,default=12)
    ap.add_argument('--maxfev',type=int,default=55)
    ap.add_argument('--shots',type=int,default=1024)
    ap.add_argument('--bootstrap',type=int,default=500)
    ap.add_argument('--seed',type=int,default=5101)
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.smoke:
        a.train_n=5000;a.val_n=900;a.test_n=1600;a.align_evals=30
        a.task_sizes=[8];a.blocks=4;a.maxfev=15;a.shots=256;a.bootstrap=100
        a.landmarks=[12,24]
    results={'args':vars(a),'protocol':'EXPLORATORY on known dataset; no test-driven selection permitted'}
    tic=time.perf_counter()
    if a.mode in ('prediction','both'):
        results['prediction']=prediction(a,out)
    if a.mode in ('scheduling','both'):
        results['scheduling']=scheduling(a,out)
    results['elapsed_s']=time.perf_counter()-tic
    dump(out/'51_overall_verdict.json',results)
    print('[51] COMPLETE',out,'elapsed_s',round(results['elapsed_s'],2),flush=True)

if __name__=='__main__':main()
