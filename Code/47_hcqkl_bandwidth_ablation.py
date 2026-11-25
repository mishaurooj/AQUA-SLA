#!/usr/bin/env python3
r"""AQUA-SLA independent HCQKL experiment (Script 47A).

Exact original feature-map architecture: Qiskit ZZFeatureMap, 6 qubits,
2 repetitions, linear entanglement, statevector fidelity. An algebraically
identical CPU statevector implementation is provided when Qiskit is absent.
Training-only feature preprocessing, validation-only hyperparameter selection,
one fixed chronological holdout, and Nine predefined kernel ablations including a genuinely bandwidth-trained ZZFeatureMap.

Install: conda install numpy pandas scipy scikit-learn; pip install xgboost
Run from Code folder:
 python 47_hcqkl_independent_ablation.py --data ..\\Dataset\\processed\\aqua_sla_prepared.csv --out ..\\results\\47_hcqkl
Run directly on uploaded archive:
 python 47_hcqkl_independent_ablation.py --data processed.zip --out results46 --fast

No positive result is guaranteed. No test score is used for model selection.
This dataset's 300s outcome is an operational future disruption surrogate,
not a contractual SLA label.
"""
from __future__ import annotations
import argparse, csv, io, json, os, platform, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score,average_precision_score,accuracy_score,
    balanced_accuracy_score,precision_score,recall_score,f1_score,brier_score_loss,
    log_loss,confusion_matrix)
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel

SEED=4701
FEATURES=['scheduling_class','collection_type','priority','vertical_scaling','scheduler',
          'requested_cpu','requested_memory','requested_cpu_missing','requested_memory_missing']
KEYS=['collection_id','instance_index']
REQUIRED=KEYS+['time_seconds','event','end_time_seconds']+FEATURES
D_EVENTS={'FAIL','LOST','EVICT','KILL'}


def read_data(path):
    path=Path(path)
    if not path.exists():raise FileNotFoundError(path)
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            matches=[n for n in z.namelist() if n.endswith('aqua_sla_prepared.csv')]
            if not matches:raise FileNotFoundError('Missing aqua_sla_prepared.csv inside ZIP')
            with z.open(matches[0]) as f:return pd.read_csv(f,usecols=lambda c:c in REQUIRED,low_memory=False)
    return pd.read_csv(path,usecols=lambda c:c in REQUIRED,low_memory=False)


def lifecycle_300(raw):
    """Reference-equivalent copy of Script 35 lifecycle target logic.

    Original tasks sorted by key, time and original row; future disruptive and
    terminal events computed strictly after the current event (same-time
    future rows are checked as in original). Do NOT use raw sla_failure or
    sla_violation as early-time target features.
    """
    df=raw.copy()
    miss=set(REQUIRED)-set(df)
    if miss:raise ValueError(f'Missing required columns {miss}')
    df['_source_row']=np.arange(len(df),dtype=np.int64)
    df['time_seconds']=pd.to_numeric(df.time_seconds,errors='coerce')
    df['end_time_seconds']=pd.to_numeric(df.end_time_seconds,errors='coerce')
    df['event']=df.event.astype(str).str.upper().str.strip()
    for c in KEYS:df[c]=df[c].astype(str)
    df=df.dropna(subset=['time_seconds']).sort_values(KEYS+['time_seconds','_source_row'],kind='mergesort').reset_index(drop=True)
    g=df.groupby(KEYS,sort=False,dropna=False)
    disruption=df.event.isin(D_EVENTS)
    terminal=df.event.eq('FINISH')
    # bfill(next disruptive after current row), including strict row ordering.
    nex_dis=g['time_seconds'].transform(lambda x: x) if False else None
    dis_t=df.time_seconds.where(disruption)
    fin_t=df.time_seconds.where(terminal)
    nex_dis=dis_t.groupby([df[k] for k in KEYS],sort=False).bfill().groupby([df[k] for k in KEYS],sort=False).shift(-1).to_numpy(float)
    nex_fin=fin_t.groupby([df[k] for k in KEYS],sort=False).bfill().groupby([df[k] for k in KEYS],sort=False).shift(-1).to_numpy(float)
    prior=(disruption.astype(int).groupby([df[k] for k in KEYS],sort=False).cumsum().to_numpy()-disruption.to_numpy(dtype=int))>0
    last_event=g.time_seconds.transform('max').to_numpy(float)
    end_max=g.end_time_seconds.transform('max').to_numpy(float)
    obs=np.fmax(last_event,end_max)
    t=df.time_seconds.to_numpy(float); h=t+300.; global_end=np.max(t)
    positive=np.isfinite(nex_dis)&(nex_dis>t)&(nex_dis<=h)
    clean=np.isfinite(nex_fin)&(nex_fin>t)&(nex_fin<=h)&(~np.isfinite(nex_dis)|(nex_fin<nex_dis))
    positive &= ~clean
    observed=np.isfinite(obs)&(obs>=h)&(h<=global_end)
    censored=(~positive)&(~clean)&(~observed)
    keep=(~disruption.to_numpy())&(~prior)&(~censored)
    result=df.loc[keep,FEATURES+['time_seconds','_source_row']].copy()
    result['future_sla_event']=positive[keep].astype(int)
    result=result.sort_values(['time_seconds','_source_row'],kind='mergesort').reset_index(drop=True)
    meta={'source_rows':len(df),'eligible_rows':len(result),'positives':int(result.future_sla_event.sum()),
          'prevalence':float(result.future_sla_event.mean()),'task_lifecycle_censored_rows':int(censored.sum()),
          'reference':'Code/35_aqua_sla_final_reviewer_complete.py; prepare_target_frame -> lifecycle_annotations -> build_lifecycle_target'}
    return result,meta


def h_gate(states,j,q):
    b=states.reshape((len(states),2**(q-j-1),2,2**j))
    a=b[:,:,0,:].copy();c=b[:,:,1,:].copy()
    b[:,:,0,:]=(a+c)*0.7071067811865476
    b[:,:,1,:]=(a-c)*0.7071067811865476


def original_zz_states(z, single_scale=1.,zz_scale=1.):
    """Statevector of ZZFeatureMap(reps=2, entanglement='linear').

    Qiskit circuit per repetition: H each, P(2*x_j) each,
    CX(j,j+1), P(2*(pi-x_j)*(pi-x_j1)), CX(j,j+1).
    Finite-sample parameterized variant changes only the two phase scales.
    """
    z=np.asarray(z,dtype=float)
    if z.ndim!=2:raise ValueError('z must be a 2D array')
    n,q=z.shape
    if q<2 or q>9:raise ValueError('Quantum dimension must be 2..9')
    s=np.zeros((n,2**q),complex);s[:,0]=1.
    basis=np.arange(2**q)
    for _ in range(2):
        for j in range(q):h_gate(s,j,q)
        for j in range(q):
            mask=(basis>>j)&1
            s[:,mask.astype(bool)]*=np.exp(2j*single_scale*z[:,j,None])
        for j in range(q-1):
            # CX-P-CX: P on the XOR parity bit of adjacent qubits.
            mask=(((basis>>j)^(basis>>(j+1)))&1).astype(bool)
            phi=2*zz_scale*(np.pi-z[:,j])*(np.pi-z[:,j+1])
            s[:,mask]*=np.exp(1j*phi[:,None])
    return s


def export_zz_qasm(z,filename,single_scale=1.,zz_scale=1.):
    """Write a concrete OpenQASM 2.0 example of the original ZZ circuit.

    It is the exact H-P-CX-P-CX circuit (two repetitions, linear adjacency).
    This is an example data row, not a trained-predictor QPY or hardware run.
    """
    z=np.asarray(z,float);q=len(z)
    lines=['OPENQASM 2.0;','include "qelib1.inc";',f'qreg q[{q}];']
    for _ in range(2):
        for j in range(q):lines.append(f'h q[{j}];')
        for j in range(q):lines.append(f'u1({2*single_scale*z[j]:.16g}) q[{j}];')
        for j in range(q-1):
            phase=2*zz_scale*(np.pi-z[j])*(np.pi-z[j+1])
            lines.extend([f'cx q[{j}],q[{j+1}];',f'u1({phase:.16g}) q[{j+1}];',f'cx q[{j}],q[{j+1}];'])
    Path(filename).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def alignment(k,y):
    k=k-k.mean(0,keepdims=True)-k.mean(1,keepdims=True)+k.mean()
    yy=np.where(y==1,1.,-1.)
    yy=yy-yy.mean()
    return float((yy@k@yy)/(max(1e-12,np.linalg.norm(k))*max(1e-12,np.dot(yy,yy))))


def fit_trainable(z, y, seed, fev):
    """Fit six independent feature-encoding bandwidths using TRAINING labels only.

    Unlike Script 46, this really changes the ZZFeatureMap's encoded input.
    Alignment is a training surrogate, not evidence of test generalization.
    """
    rng = np.random.default_rng(seed)
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    n=min(56,len(pos),len(neg))
    original=np.ones(z.shape[1], dtype=float)
    if n<4:
        return original,{'status':'insufficient_training_cases',
                         'alignment_before':None,'alignment_after':None,'fit_n':2*n,'nfev':0}
    ids=np.r_[rng.choice(pos,n,replace=False),rng.choice(neg,n,replace=False)]
    yy=np.asarray(y[ids],int);zz=np.asarray(z[ids],float)
    count=[0]
    def loss(theta):
        count[0]+=1
        states=original_zz_states(zz*np.asarray(theta)[None,:])
        k=np.abs(states @ states.conj().T)**2
        return -alignment(k,yy)
    before=-loss(original)
    # Preregistered low-dimensional bandwidth starts; no validation/test labels.
    starts=[original,original*.70,original*1.30]
    losses=[loss(s) for s in starts]
    chosen=np.array(starts[int(np.argmin(losses))],copy=True)
    candidate=minimize(loss,chosen,method='Powell',
         bounds=[(.35,1.65)]*len(chosen),
         options={'maxfev':max(30,int(fev)),'xtol':.055,'ftol':.0008})
    fitted=np.clip(candidate.x,.35,1.65)
    if loss(fitted)<loss(chosen):chosen=fitted
    after=-loss(chosen)
    # A fall-back to the original map is valid if training fails, but report it.
    if after<before:chosen=original;after=before
    return chosen, {'alignment_before':float(before),'alignment_after':float(after),
        'alignment_gain':float(after-before),'fit_n':int(2*n),
        'nfev':int(count[0]),'training_only':True,
        'feature_wise_encoding_scales':chosen.tolist(),
        'nontrivial_map':bool(np.max(np.abs(chosen-original))>1e-6)}


def sel_indices(y,n,seed,case_control=False):
    y=np.asarray(y)
    rng=np.random.default_rng(seed)
    if n>=len(y):return np.arange(len(y))
    if case_control:
        pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        p=min(len(pos),max(1,n//4)); n0=min(len(neg),n-p)
        ix=np.r_[rng.choice(pos,p,replace=False),rng.choice(neg,n0,replace=False)]
        return np.sort(ix)
    return np.sort(rng.choice(len(y),size=n,replace=False))


def safe_metrics(y,p,threshold):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    pred=p>=threshold
    res=dict(n=len(y),positives=int(y.sum()),prevalence=float(y.mean()),
      accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
      precision=float(precision_score(y,pred,zero_division=0)),
      recall=float(recall_score(y,pred,zero_division=0)),
      f1=float(f1_score(y,pred,zero_division=0)),
      brier=float(brier_score_loss(y,p)),
      specificity=float(np.mean(~pred[y==0])) if (y==0).any() else np.nan,
      threshold=float(threshold))
    res['roc_auc']=float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan
    res['pr_auc']=float(average_precision_score(y,p)) if y.sum()>0 else np.nan
    n=max(1,int(np.ceil(.1*len(y))))
    res['top10_recall']=float(y[np.argsort(-p)[:n]].sum()/max(1,y.sum()))
    bins=np.minimum(np.floor(p*10).astype(int),9)
    res['ece10']=float(sum(np.sum(bins==b)/len(y)*abs(y[bins==b].mean()-p[bins==b].mean()) for b in range(10) if np.any(bins==b)))
    return res


def select_threshold(y,p):
    thresholds=np.unique(np.r_[np.linspace(.02,.8,30),np.quantile(p,np.linspace(.1,.99,45))])
    scores=np.asarray([f1_score(y,p>=x,zero_division=0) for x in thresholds])
    return float(thresholds[np.argmax(scores)])


def calibrate(m_val,y_val,m_test):
    # Validation calibration only; never trained on test labels.
    if np.unique(y_val).size<2:return expit(m_val),expit(m_test)
    fit=LogisticRegression(solver='lbfgs',max_iter=400)
    fit.fit(np.asarray(m_val).reshape(-1,1),y_val)
    return fit.predict_proba(np.asarray(m_val).reshape(-1,1))[:,1],fit.predict_proba(np.asarray(m_test).reshape(-1,1))[:,1]


def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    raw=read_data(a.data)
    data,meta=lifecycle_300(raw)
    del raw
    (out/'47_target_audit.json').write_text(json.dumps(meta,indent=2))
    print('TARGET',meta,flush=True)
    n=len(data);t_end=int(n*.6);v_end=int(n*.8)
    tr=data.iloc[:t_end].copy();va=data.iloc[t_end:v_end].copy();te=data.iloc[v_end:].copy()
    ytr_all=tr.future_sla_event.to_numpy(int)
    tr_ix=sel_indices(ytr_all,min(a.kernel_train,len(tr)),a.seed,case_control=True)
    va_ix=sel_indices(va.future_sla_event.to_numpy(int),min(a.valid,len(va)),a.seed+1)
    te_ix=sel_indices(te.future_sla_event.to_numpy(int),min(a.test,len(te)),a.seed+2)
    tsub=tr.iloc[tr_ix].copy();vsub=va.iloc[va_ix].copy();esub=te.iloc[te_ix].copy()
    ytr=tsub.future_sla_event.to_numpy(int);yv=vsub.future_sla_event.to_numpy(int);yt=esub.future_sla_event.to_numpy(int)
    if min(np.unique(ytr).size,np.unique(yv).size,np.unique(yt).size)<2:
        raise RuntimeError('At least one selected split has only one class. Increase --valid/--test/--kernel-train.')
    # True chronological disjoint partitions before any sampling.
    split={'train_raw':len(tr),'validation_raw':len(va),'test_raw':len(te),
           'train_selected':len(tsub),'val_selected':len(vsub),'test_selected':len(esub),
           'training_sample_positive_rate':float(ytr.mean()),'validation_positive_rate':float(yv.mean()),
           'test_positive_rate':float(yt.mean()),'temporal_train_max':float(tr.time_seconds.max()),
           'temporal_valid_min':float(va.time_seconds.min()),'temporal_valid_max':float(va.time_seconds.max()),
           'temporal_test_min':float(te.time_seconds.min()),'split_design':'chronological 60/20/20 before subsampling'}
    (out/'47_split_audit.json').write_text(json.dumps(split,indent=2))
    print('SPLIT',split,flush=True)
    # Original preprocessing and 6-dimensional quantum circuit.
    Xtrain=tsub[FEATURES].apply(pd.to_numeric,errors='coerce')
    Xi=SimpleImputer(strategy='median');sc=RobustScaler();pca=PCA(n_components=6,random_state=a.seed);ang=MinMaxScaler(feature_range=(0,np.pi))
    ztr=ang.fit_transform(pca.fit_transform(sc.fit_transform(Xi.fit_transform(Xtrain))))
    def transform(d):return ang.transform(pca.transform(sc.transform(Xi.transform(d[FEATURES].apply(pd.to_numeric,errors='coerce')))))
    zv=transform(vsub);ze=transform(esub)
    ztr=np.clip(ztr,-2*np.pi,3*np.pi);zv=np.clip(zv,-2*np.pi,3*np.pi);ze=np.clip(ze,-2*np.pi,3*np.pi)
    t0=time.perf_counter();st=original_zz_states(ztr);sv=original_zz_states(zv);se=original_zz_states(ze)
    KQ=np.abs(st@st.conj().T)**2;VQ=np.abs(sv@st.conj().T)**2;TQ=np.abs(se@st.conj().T)**2
    quantum_build=time.perf_counter()-t0
    params,align_info=fit_trainable(ztr,ytr,a.seed,a.align_fev)
    export_zz_qasm(ztr[0],out/'47_original_ZZFeatureMap_example.qasm')
    export_zz_qasm(ztr[0]*params,out/'47_trainable_ZZFeatureMap_example.qasm')
    np.savez_compressed(out/'47_quantum_encoding_parameters.npz',
        original_scale=np.array([1.,1.]),trainable_scale=np.array(params),
        pca_components=pca.components_,scaler_center=sc.center_,scaler_scale=sc.scale_,
        angle_data_min=ang.data_min_,angle_data_max=ang.data_max_)

    sts=original_zz_states(ztr*params)
    svs=original_zz_states(zv*params)
    ses=original_zz_states(ze*params)
    KQT=np.abs(sts@sts.conj().T)**2;VQT=np.abs(svs@sts.conj().T)**2;TQT=np.abs(ses@sts.conj().T)**2
    # Training-only median distance / original gamma, no test labels.
    sub=ztr[np.linspace(0,len(ztr)-1,min(200,len(ztr)),dtype=int)]
    dist=((sub[:,None,:]-sub[None,:,:])**2).sum(-1)
    med=float(np.median(dist[np.triu_indices(len(sub),1)]))
    g0=1/6;gmed=1/max(.001,med)
    gammas=sorted(set([g0,gmed*.5,gmed,gmed*2]))
    kvsplit=int(.7*len(vsub));kvsplit=max(20,min(kvsplit,len(vsub)-20))
    if len(np.unique(yv[:kvsplit]))<2 or len(np.unique(yv[kvsplit:]))<2:
        # No label-driven rerouting to the test set. Larger validation sample needed.
        raise RuntimeError('Validation selection/calibration blocks need both classes. Increase --valid.')
    from sklearn.svm import SVC
    candidates=[];stored={};runstart=time.perf_counter()
    # Predefined nine ablation variants. Tuned versions choose only on first 70% validation.
    specs=[('Q_original',0.,False,False),('RBF_original',1.,False,False),
           ('RBF_tuned',1.,True,False),('HCQKL_a025',.25,True,False),
           ('HCQKL_a050',.5,True,False),('HCQKL_a075',.75,True,False),
           ('HCQKL_alpha_selected',None,True,False),('Q_trainable_map',0.,False,True),('HCQKL_trainable_map',None,True,True)]
    for name,alpha,tune_gamma,learned in specs:
        aa=[alpha] if alpha is not None else ([0.,.25,.5,.75] if learned else [0.,.25,.5,.75,1.])
        gg=gammas if tune_gamma else [g0]
        best=None;best_value=-np.inf
        for gam in gg:
            kR=rbf_kernel(ztr,ztr,gamma=gam);vR=rbf_kernel(zv,ztr,gamma=gam)
            for alpha_c in aa:
                K=alpha_c*kR+(1-alpha_c)*(KQT if learned else KQ)
                V=alpha_c*vR+(1-alpha_c)*(VQT if learned else VQ)
                for C in ([.1,1.,10.] if tune_gamma else [10.]):
                    startfit=time.perf_counter()
                    model=SVC(kernel='precomputed',C=C,class_weight='balanced',cache_size=256)
                    model.fit(K,ytr)
                    margin=model.decision_function(V)
                    if np.unique(yv[:kvsplit]).size!=2: continue
                    score=float(average_precision_score(yv[:kvsplit],margin[:kvsplit]))
                    # If tied, deterministic selection order, no test evaluation.
                    if score>best_value+1e-12:
                        best_value=score
                        best=(gam,alpha_c,C,model,margin,time.perf_counter()-startfit)
        gam,alpha_c,C,model,v_margin,fit_s=best
        Kte=alpha_c*rbf_kernel(ze,ztr,gamma=gam)+(1-alpha_c)*(TQT if learned else TQ)
        teststart=time.perf_counter();te_margin=model.decision_function(Kte);test_s=time.perf_counter()-teststart
        # Calibrate on validation calibration tail only.
        vp,tp=calibrate(v_margin[kvsplit:],yv[kvsplit:],te_margin)
        thr=select_threshold(yv[kvsplit:],vp)
        row=dict(model=name,kind='HCQKL' if alpha_c<1 and alpha_c>0 else ('Quantum' if alpha_c==0 else 'RBF'),
                 alpha=float(alpha_c),gamma=float(gam),C=float(C),validation_selection_ap=best_value,
                 fit_seconds=float(fit_s),inference_seconds=float(test_s),
                 circuit_qubits=6,circuit_repetitions=2,**safe_metrics(yt,tp,thr))
        candidates.append(row)
        stored[name]=tp
        print(name, 'ROC',round(row['roc_auc'],4),'PR',round(row['pr_auc'],4), 'alpha',alpha_c,flush=True)
    # Classical baselines: matched-size and additional full-training controls.
    Xi_all=SimpleImputer(strategy='median').fit(tsub[FEATURES]);xs=StandardScaler().fit(Xi_all.transform(tsub[FEATURES]))
    xx=xs.transform(Xi_all.transform(tsub[FEATURES]));xv=xs.transform(Xi_all.transform(vsub[FEATURES]));xt=xs.transform(Xi_all.transform(esub[FEATURES]))
    models={'RF_matched':RandomForestClassifier(n_estimators=120,min_samples_leaf=2,max_features='sqrt',n_jobs=-1,random_state=a.seed),
            'MLP_matched':MLPClassifier(hidden_layer_sizes=(64,32),max_iter=120,early_stopping=True,random_state=a.seed),
            'HGB_matched':HistGradientBoostingClassifier(max_iter=130,l2_regularization=.1,random_state=a.seed)}
    try:
        from xgboost import XGBClassifier
        models['XGB_matched']=XGBClassifier(n_estimators=130,max_depth=4,learning_rate=.045,subsample=.9,colsample_bytree=.95,
            n_jobs=2,tree_method='hist',eval_metric='logloss',random_state=a.seed)
    except ImportError:pass
    from sklearn.base import clone
    # The fuller classical controls use a natural-prevalence sample from training only;
    # they must NOT be confused with the size-matched case-control controls.
    full_ix=sel_indices(ytr_all,min(a.baseline_train,len(tr)),a.seed+13,case_control=False)
    xfull=tr.iloc[full_ix][FEATURES].apply(pd.to_numeric,errors='coerce')
    yfull=ytr_all[full_ix]
    if len(np.unique(yfull))==2:
        xf_imp=SimpleImputer(strategy='median').fit(xfull)
        xf_sc=StandardScaler().fit(xf_imp.transform(xfull))
        xf=xf_sc.transform(xf_imp.transform(xfull))
        xvf=xf_sc.transform(xf_imp.transform(vsub[FEATURES]))
        xtf=xf_sc.transform(xf_imp.transform(esub[FEATURES]))
        models['RF_full_train']=clone(models['RF_matched'])
        models['HGB_full_train']=clone(models['HGB_matched'])
        if 'XGB_matched' in models:models['XGB_full_train']=clone(models['XGB_matched'])
        models['MLP_full_train']=clone(models['MLP_matched'])
    for name,model in models.items():
        this_full=name.endswith('_full_train')
        fitx=xf if this_full else xx;fity=yfull if this_full else ytr
        valx=xvf if this_full else xv;testx=xtf if this_full else xt
        tfit=time.perf_counter();model.fit(fitx,fity);fit_s=time.perf_counter()-tfit
        ttest=time.perf_counter();pred_test=model.predict_proba(testx)[:,1];test_s=time.perf_counter()-ttest
        pred_val=model.predict_proba(valx[kvsplit:])[:,1]
        # Calibration on validation only; default probability scores remain raw to preserve rank.
        thr=select_threshold(yv[kvsplit:],pred_val)
        row=dict(model=name,kind='Classical',alpha=np.nan,gamma=np.nan,C=np.nan,
            validation_selection_ap=float(average_precision_score(yv[:kvsplit],model.predict_proba(valx[:kvsplit])[:,1])),
            fit_seconds=fit_s,inference_seconds=test_s,circuit_qubits=0,circuit_repetitions=0,**safe_metrics(yt,pred_test,thr))
        candidates.append(row);stored[name]=pred_test
        print(name,'ROC',round(row['roc_auc'],4),'PR',round(row['pr_auc'],4),flush=True)
    full=pd.DataFrame(candidates)
    full.to_csv(out/'47_prediction_all_models.csv',index=False)
    # Temporal robustness: independently report contiguous test thirds, do not pick winner from them.
    subrows=[]
    for model,scores in stored.items():
        for block,idx in enumerate(np.array_split(np.arange(len(yt)),3),1):
            m=safe_metrics(yt[idx],scores[idx],float(full.loc[full.model==model,'threshold'].iloc[0]))
            subrows.append(dict(model=model,temporal_block=block,**m))
    pd.DataFrame(subrows).to_csv(out/'47_temporal_blocks.csv',index=False)
    pd.DataFrame({'time_seconds':esub.time_seconds.to_numpy(),'y_true':yt,**stored}).to_csv(out/'47_test_predictions.csv',index=False)
    # Bootstrap paired by contiguous test blocks: fixed seeded bootstrap only, no selective interpretation.
    rng=np.random.default_rng(a.seed+7);blocks=np.array_split(np.arange(len(yt)),12)
    selected_candidates=full[full.model.str.startswith('HCQKL')].sort_values('validation_selection_ap',ascending=False)
    quantum_name=str(selected_candidates.iloc[0]['model'])
    control_name='HGB_matched'
    diffs=[]
    for _ in range(a.bootstrap):
        ix=np.concatenate([blocks[k] for k in rng.integers(0,len(blocks),len(blocks))])
        if np.unique(yt[ix]).size<2:continue
        qa=stored[quantum_name][ix];cl=stored[control_name][ix]; yy=yt[ix]
        diffs.append([roc_auc_score(yy,qa)-roc_auc_score(yy,cl),
                     average_precision_score(yy,qa)-average_precision_score(yy,cl)])
    cis=np.quantile(np.asarray(diffs),[.025,.975],axis=0).tolist() if diffs else None
    selected_name=quantum_name
    baseline_names=[x for x in stored if x in models]+['RBF_tuned']
    eval_ci=dict(reference=f'{quantum_name} vs HGB_matched',bootstrap_ci=cis,
       strict_both_metric_superiority=bool(cis is not None and cis[0][0]>0 and cis[0][1]>0
           and 0<float(full.loc[full.model==quantum_name,'alpha'].iloc[0])<1
           and all(float(full.loc[full.model==quantum_name,'roc_auc'].iloc[0])>v
                   for v in full[full.model.isin([k for k in stored if k in models]+['RBF_tuned'])].roc_auc)),
       selected_validation_kernel=str(selected_name),
       selected_validation_alpha=float(full.loc[full.model==selected_name,'alpha'].iloc[0]),
       selected_is_genuine_hybrid=bool(0<float(full.loc[full.model==selected_name,'alpha'].iloc[0])<1),
       matched_baseline_train_size=len(tsub),full_classical_train_size=len(yfull) if 'yfull' in locals() else None,
       quantum_vs_all_classical_in_test_roc=bool(full[full.model==quantum_name].roc_auc.iloc[0]>full[full.model.isin(baseline_names)].roc_auc.max()),
       quantum_vs_all_classical_in_test_pr=bool(full[full.model==quantum_name].pr_auc.iloc[0]>full[full.model.isin(baseline_names)].pr_auc.max()),
       test_used_for_tuning=False,
       caution='Exploratory real-data run; repeated use of a previously examined temporal holdout is not confirmatory. Simulator only.')
    manifest={'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
                'gpu_available':False,'qiskit_used':False},'target':meta,'split':split,'trainable_circuit':{**align_info,'optimized_scales':list(params)},
              'circuit':'Qiskit ZZFeatureMap(feature_dimension=6,reps=2,entanglement=linear) exact gate-level CPU statevector',
              'quantum_kernel_build_seconds':quantum_build,'elapsed_seconds':time.perf_counter()-start,
              'verdict':eval_ci,'kernel_ablation_variants':[x[0] for x in specs]}
    (out/'47_manifest.json').write_text(json.dumps(manifest,indent=2,default=float))
    print('RESULT FILES:',out.resolve(),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=r'E:\other\AQUA-SLA\Dataset\processed\aqua_sla_prepared.csv')
    p.add_argument('--out',default=r'E:\other\AQUA-SLA\results\47_hcqkl_independent')
    p.add_argument('--kernel-train',type=int,default=1200)
    p.add_argument('--valid',type=int,default=2400)
    p.add_argument('--test',type=int,default=3000)
    p.add_argument('--align-fev',type=int,default=150)
    p.add_argument('--baseline-train',type=int,default=15000)
    p.add_argument('--bootstrap',type=int,default=250)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--fast',action='store_true')
    a=p.parse_args()
    if a.fast:a.kernel_train=360;a.valid=1800;a.test=1500;a.align_fev=65;a.bootstrap=50;a.baseline_train=3000
    run(a)
if __name__=='__main__':main()
