#!/usr/bin/env python3
"""Frozen-validation state-allocation diagnostic. No fitting or test access."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import resource,signal
resource.setrlimit(resource.RLIMIT_AS,(2147483648,2147483648))
signal.alarm(600)
try:os.nice(10)
except PermissionError:pass
import json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SRC=Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
IDENTITY=SRC/'data/identity_audit_v2.json'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def save(obj,p):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def pair_stats(err,pred,y,mask):
    a=err[:,:-1];b=err[:,1:]
    sep=a*a+b*b
    swap=(pred[:,:-1]-y[:,1:])**2+(pred[:,1:]-y[:,:-1])**2
    v=np.stack([np.ones_like(a),a,b,a*a,b*b,a*b,np.maximum(sep-swap,0)],axis=-1)
    return (v*mask[:,:,None]).sum(axis=1)
def describe(v):
    n,sa,sb,aa,bb,ab,gain=map(float,v)
    if n==0:return {'pairs':0}
    sep=aa+bb;var_a=aa-sa*sa/n;var_b=bb-sb*sb/n
    corr=(ab-sa*sb/n)/np.sqrt(var_a*var_b) if var_a>0 and var_b>0 else None
    return {'pairs':int(round(n)),'separate_state_sse':sep,'pair_sum_sse':sep+2*ab,
        'pair_sum_sse_ratio':(sep+2*ab)/sep if sep>0 else None,
        'adjacent_error_correlation':float(corr) if corr is not None else None,
        'mean_pair_sum_error':(sa+sb)/n,'mean_cross_error_product':ab/n,
        'oracle_local_swap_sse_reduction':gain,
        'oracle_local_swap_reduction_fraction':gain/sep if sep>0 else None}
def interval(a):
    a=np.asarray(a);assert np.isfinite(a).all()
    return {'q025_q50_q975':np.quantile(a,[.025,.5,.975]).tolist(),
            'fraction_positive':float((a>0).mean())}
def main():
    started=time.time();cfg=json.loads((ROOT/'PROTOCOL.json').read_text())
    assert cfg['models']==['mto_eta0','mto_eta01','mto_eta1','equal_three']
    with np.load(SRC/'data/dataset.npz',allow_pickle=False) as z:
        ix=z['val'].copy();ids=z['ids'][ix].copy()
    assert len(ids)==6686 and len(set(ids.tolist()))==len(ids)
    assert len(set(ix.tolist()))==len(ix)
    with np.load(SRC/'data/raw_labels.npz',allow_pickle=False) as raw:
        assert np.array_equal(raw['ids'][ix],ids)
        raw_truth={k:raw[k][ix].copy() for k in ('f','E','mask_f','mask_E','mask_A')}
    assert all(raw_truth[k].all() for k in ('mask_f','mask_E','mask_A'))
    predictions={};energies={};hashes={};y=e=None
    hashes[str(SRC/'data/raw_labels.npz')]=sha(SRC/'data/raw_labels.npz')
    for name in cfg['models'][:-1]:
        path=SRC/'runs'/name/'val_predictions.npz'
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z['ids'],ids) and np.array_equal(z['indices'],ix)
            for key,saved in (('f','f_true'),('E','E_true'),
                              ('mask_f','mask_f_true'),('mask_E','mask_E_true'),
                              ('mask_A','mask_A_true')):
                assert z[saved].dtype==raw_truth[key].dtype
                assert np.array_equal(z[saved],raw_truth[key])
            assert z['mask_f_true'].all() and z['mask_E_true'].all()
            yy=z['f_true'].astype(np.float64);ee=z['E_true'].astype(np.float64)
            if y is None:y=yy;e=ee
            else:assert np.array_equal(y,yy) and np.array_equal(e,ee)
            predictions[name]=z['f'].astype(np.float64);energies[name]=z['E'].astype(np.float64)
        hashes[str(path)]=sha(path)
    assert y.shape==(6686,10) and np.isfinite(y).all() and np.isfinite(e).all()
    assert all(p.shape==y.shape and np.isfinite(p).all() for p in predictions.values())
    predictions['equal_three']=sum(predictions.values())/3
    energies['equal_three']=sum(energies.values())/3
    audit=json.loads(IDENTITY.read_text());keys=[]
    for index,id_ in zip(ix,ids):
        row=audit[int(index)];assert int(row[0])==int(id_) and row[2] is True;keys.append(row[1])
    unique,groups=np.unique(np.asarray(keys),return_inverse=True);G=len(unique)
    gap=np.diff(e,axis=1);assert (gap>=0).all()
    low,high=cfg['gap_cuts_eV'];tol=cfg['gap_boundary_tolerance_eV']
    gap_masks={'low':gap<=low+tol,'middle':(gap>low+tol)&(gap<high-tol),'high':gap>=high-tol}
    assert np.stack(list(gap_masks.values())).sum(axis=0).min()==1
    assert np.stack(list(gap_masks.values())).sum(axis=0).max()==1
    brightness=y[:,:-1]+y[:,1:];b50,b90=np.quantile(brightness,[.5,.9])
    brightness_masks={'low':brightness<=b50,'middle':(brightness>b50)&(brightness<b90),'high':brightness>=b90}
    offsets={}
    for name,pairs in cfg['disjoint_offsets'].items():
        used=[s for pair in pairs for s in pair];assert len(used)==len(set(used))
        mask=np.zeros_like(gap,dtype=bool)
        for a,b in pairs:assert b==a+1;mask[:,a-1]=True
        offsets[name]=mask
    sst=float(np.square(y-y.mean()).sum());results={};sufficient={}
    for name,pred in predictions.items():
        err=pred-y;sse=float(np.square(err).sum())
        # Primary accuracy always retains original state order and raw labels.
        primary={'raw_native_f_r2':1-sse/sst,'sse':sse,'mae':float(np.abs(err).mean()),
                 'rmse':float(np.sqrt(np.square(err).mean())),
                 'classification':'ensemble' if name=='equal_three' else 'single_model',
                 'predicted_energy_adjacent_inversions':int((np.diff(energies[name],axis=1)<0).sum())}
        slices=[]
        for j in range(9):
            pair_mask=np.zeros_like(gap,dtype=bool);pair_mask[:,j]=True
            for gl,gm in gap_masks.items():
                for bl,bm in brightness_masks.items():
                    mask=pair_mask&gm&bm;st=pair_stats(err,pred,y,mask).sum(axis=0)
                    entry={'states':[j+1,j+2],'gap_bin':gl,'brightness_bin':bl,
                           **describe(st),'truth_pair_strength_sum':float(brightness[mask].sum())}
                    slices.append(entry)
        disjoint={}
        for on,om in offsets.items():
            disjoint[on]={'physical_pairs':cfg['disjoint_offsets'][on],'gap_bins':{}}
            for gl,gm in {'all':np.ones_like(gap,dtype=bool),**gap_masks}.items():
                st=pair_stats(err,pred,y,om&gm)
                disjoint[on]['gap_bins'][gl]=describe(st.sum(axis=0))
                if name in ('mto_eta0','equal_three') and gl in ('low','high'):
                    sufficient[(name,on,gl)]=st
        results[name]={'primary':primary,'state_pair_gap_brightness_slices':slices,'disjoint_offsets':disjoint}
    # A fixed-offset oracle is the exact minimum over independent swap/no-swap choices.
    tiny_y=np.array([[1.,3.,2.,5.]]);tiny_p=np.array([[3.,1.,5.,2.]])
    tiny_mask=np.array([[True,False,True]])
    tiny=describe(pair_stats(tiny_p-tiny_y,tiny_p,tiny_y,tiny_mask).sum(axis=0))
    assert tiny['pairs']==2 and tiny['oracle_local_swap_sse_reduction']==tiny['separate_state_sse']
    # Aggregate sufficient statistics by existing connectivity cluster, never by transition.
    labels=list(sufficient);columns=[]
    for label in labels:
        agg=np.zeros((G,7));np.add.at(agg,groups,sufficient[label]);columns.append(agg)
    primary_stats=np.column_stack([np.full(len(y),10),y.sum(axis=1),(y*y).sum(axis=1)]+
        [((p-y)**2).sum(axis=1) for p in predictions.values()])
    agg_primary=np.zeros((G,primary_stats.shape[1]));np.add.at(agg_primary,groups,primary_stats)
    matrix=np.concatenate(columns+[agg_primary],axis=1)
    B=cfg['bootstrap']['replicates'];rng=np.random.default_rng(cfg['bootstrap']['seed'])
    samples=[]
    for start in range(0,B,100):
        draws=rng.integers(0,G,(min(100,B-start),G));weights=np.zeros((len(draws),G))
        for j,draw in enumerate(draws):weights[j]=np.bincount(draw,minlength=G)
        samples.append(weights@matrix)
    samples=np.concatenate(samples);assert len(samples)==B
    boot={};lookup={label:i for i,label in enumerate(labels)}
    for on in offsets:
        boot[on]={}
        for gl in ('low','high'):
            a=samples[:,lookup[('mto_eta0',on,gl)]*7:lookup[('mto_eta0',on,gl)]*7+7]
            b=samples[:,lookup[('equal_three',on,gl)]*7:lookup[('equal_three',on,gl)]*7+7]
            aa=a[:,3]+a[:,4];bb=b[:,3]+b[:,4]
            acov=a[:,5]/a[:,0]-a[:,1]*a[:,2]/a[:,0]**2
            bcov=b[:,5]/b[:,0]-b[:,1]*b[:,2]/b[:,0]**2
            boot[on][gl]={'ensemble_relative_separate_sse_reduction':interval((aa-bb)/aa),
                'ensemble_minus_eta0_pair_sum_sse_ratio':interval(2*b[:,5]/bb-2*a[:,5]/aa),
                'ensemble_minus_eta0_adjacent_error_covariance':interval(bcov-acov),
                'ensemble_minus_eta0_oracle_swap_fraction':interval(b[:,6]/bb-a[:,6]/aa)}
    v=samples[:,-7:];bsst=v[:,2]-v[:,1]**2/v[:,0];assert (bsst>0).all()
    primary_boot={name:interval((v[:,3]-v[:,3+j])/bsst) for j,name in enumerate(predictions)}
    result={'classification':'Exploratory frozen-validation state-allocation diagnostics; no fitting/test access',
        'created':time.time(),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'script_sha256':sha(Path(__file__)),
        'input_sha256':hashes,'identity_sha256':sha(IDENTITY),'molecules':len(ids),'connectivity_groups':G,
        'group_largest':int(np.bincount(groups).max()),'repeated_groups':int((np.bincount(groups)>1).sum()),
        'gap_cuts_eV':[low,high],'brightness_pair_sum_q50_q90':[float(b50),float(b90)],'models':results,
        'group_bootstrap_ensemble_vs_eta0':boot,'group_bootstrap_ordered_delta_r2_vs_eta0':primary_boot,
        'bootstrap':cfg['bootstrap'],'unit_check_disjoint_oracle_passed':True,
        'limits':['Initial eta0 feasibility was viewed before this exploratory multi-model protocol.',
                  'True gaps/brightness and swaps are label-informed diagnostics, never inference inputs or relabelled accuracy.',
                  'The two fixed offsets are analyzed separately; they overlap with each other and cannot be summed.',
                  'Bootstrap conditions on these frozen selected checkpoints and does not remove selection or training-seed uncertainty.',
                  'Negative error correlation is not proof of physical state mixing or incorrect labels.'],
        'elapsed_seconds':time.time()-started}
    save(result,ROOT/'RESULTS.json')
    lines=['# Validation state-allocation diagnostic','','Primary metrics retain the original state ordering and printed raw oscillator strengths. Equal-three is an ensemble; the eta models are single models. No training or test predictions were used.','',
        '| Model | Raw native-f validation R² | MAE | Energy-order inversions |','| --- | ---: | ---: | ---: |']
    for name,v in results.items():
        x=v['primary'];lines.append(f"| {name} | {x['raw_native_f_r2']:.6f} | {x['mae']:.6f} | {x['predicted_energy_adjacent_inversions']} |")
    lines+=['','## Separate disjoint-pair diagnostics','',
        'Ratio = pair-summed SSE / separate-state SSE. Oracle fraction is a truth-informed local swap bound, not attainable accuracy or R². Offsets must not be pooled.','',
        '| Offset | Gap | Model | Pairs | SSE | Sum ratio | Error correlation | Oracle swap fraction |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for on in offsets:
        for gl in ('low','high'):
            for name in ('mto_eta0','equal_three'):
                x=results[name]['disjoint_offsets'][on]['gap_bins'][gl]
                lines.append(f"| {on} | {gl} | {name} | {x['pairs']} | {x['separate_state_sse']:.5f} | {x['pair_sum_sse_ratio']:.4f} | {x['adjacent_error_correlation']:.4f} | {x['oracle_local_swap_reduction_fraction']:.4f} |")
    lines+=['','## Connectivity-group uncertainty','',
        '| Offset | Gap | Ensemble separate-SSE reduction, 95% interval | Change in sum ratio, 95% interval |',
        '| --- | --- | --- | --- |']
    for on in offsets:
        for gl in ('low','high'):
            x=boot[on][gl];a=x['ensemble_relative_separate_sse_reduction']['q025_q50_q975'];b=x['ensemble_minus_eta0_pair_sum_sse_ratio']['q025_q50_q975']
            lines.append(f'| {on} | {gl} | [{a[0]:+.4f}, {a[2]:+.4f}] | [{b[0]:+.4f}, {b[2]:+.4f}] |')
    lines+=['',f'Gap cuts: {low:.5f} and {high:.5f} eV. Pair-strength q50/q90: {b50:.6f}/{b90:.6f}. Full state-pair × gap × brightness strata are in RESULTS.json. Group bootstrap uses2000 replicates over'+str(G)+' connectivity groups.',
        '', 'These are conditional exploratory diagnostics. Small-gap cancellation can reflect intensity allocation, state composition, rounding or prediction bias; it does not establish physical mixing or justify relabelling. Root should review strata alongside current objective/readout results before choosing another architecture.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],'primary':{k:v['primary'] for k,v in results.items()},'group_bootstrap_ensemble_vs_eta0':boot},allow_nan=False))
if __name__=='__main__':main()
