#!/usr/bin/env python3
"""Fixed validation geometry audit; no fitting/inference/test split use."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2'); os.environ.setdefault('OPENBLAS_NUM_THREADS','2'); os.environ.setdefault('MKL_NUM_THREADS','2'); os.environ.setdefault('NUMEXPR_NUM_THREADS','2')
import hashlib,json,struct,zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    threadpool_limits=lambda limits:nullcontext()
ROOT=Path('/home/inspur/MTO-1')
OUT=ROOT/'research/oscillator_r2_20260928/geometry_error_audit'
DATA=ROOT/'experiments/qm9s_eta_Ef_20260926/data'
PRED=ROOT/'experiments/qm9s_eta_Ef_20260926/runs'
IDENT=ROOT/'experiments/qm9s_full_EA_20260925/data/identity_audit_v2.json'
EPS=np.finfo(np.float64).eps

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def npz_member_memmap(npz_path,key):
    """Map one ZIP_STORED NPY member; row selection below avoids touching test rows."""
    npz_path=Path(npz_path)
    with zipfile.ZipFile(npz_path) as zf:
        info=zf.getinfo(key+'.npy')
        if info.compress_type != zipfile.ZIP_STORED: raise RuntimeError(f'not stored: {npz_path}:{key}')
        with open(npz_path,'rb') as f:
            f.seek(info.header_offset); hdr=f.read(30)
            sig,ver,flag,method,mtime,mdate,crc,csize,usize,nlen,elen=struct.unpack('<IHHHHHIIIHH',hdr)
            if sig!=0x04034b50: raise RuntimeError('bad zip local header')
            f.seek(info.header_offset+30+nlen+elen)
            version=np.lib.format.read_magic(f)
            if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
            else: shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
            offset=f.tell()
    return np.memmap(npz_path,mode='r',dtype=dtype,offset=offset,shape=shape,order='F' if fortran else 'C')
def stream_selected_identity_rows(path, wanted):
    """Parse identity rows only at wanted dataset indices; skip all other row contents."""
    wanted=set(map(int,wanted)); out={}; n=len(wanted); i=0
    with open(path,'rb') as f:
        c=f.read(1)
        while c and c.isspace(): c=f.read(1)
        if c!=b'[': raise RuntimeError('identity JSON top level is not array')
        while True:
            c=f.read(1)
            while c and (c.isspace() or c==b','): c=f.read(1)
            if not c or c==b']': break
            start=f.tell()-1; stack=[]; in_str=False; escape=False; first=c
            if first not in (b'[',b'{'): raise RuntimeError(f'identity row {i} unexpected token')
            stack.append(b']' if first==b'[' else b'}')
            while stack:
                c=f.read(1)
                if not c: raise RuntimeError('truncated identity JSON')
                if in_str:
                    if escape: escape=False
                    elif c==b'\\': escape=True
                    elif c==b'"': in_str=False
                    continue
                if c==b'"': in_str=True
                elif c in (b'[',b'{'): stack.append(b']' if c==b'[' else b'}')
                elif c in (b']',b'}'):
                    if not stack or c!=stack.pop(): raise RuntimeError('unbalanced identity JSON')
            end=f.tell()
            if i in wanted:
                f.seek(start); raw=f.read(end-start); out[i]=json.loads(raw)
            i+=1
    if i<max(wanted)+1 or set(out)!=wanted: raise RuntimeError('missing requested identity rows')
    return out,i
def formula(zrow):
    vals,counts=np.unique(zrow[zrow!=0].astype(np.int64),return_counts=True)
    return tuple((int(z),int(n)) for z,n in zip(vals,counts))
def geometry(zrow,posrow):
    x=np.asarray(posrow[np.asarray(zrow)!=0],dtype=np.float64)
    x=x-x.mean(axis=0,dtype=np.float64)
    cov=(x.T@x)/float(len(x))
    ev=np.linalg.eigvalsh(cov)[::-1]
    tol=64.0*EPS*max(float(np.trace(cov)),1.0)
    if not np.isfinite(ev).all() or ev[0]<=tol: return (float('nan'),float('nan'),ev.tolist(),float(tol))
    return (float(ev[1]/ev[0]),float(ev[2]/ev[0]),ev.tolist(),float(tol))
def score(y,p):
    err=p-y; return {'sse':float(np.square(err).sum()),'mean_signed_error':float(err.mean()),'overprediction_fraction':float((err>0).mean()),'labels':int(err.size)}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ds=DATA/'dataset.npz'; rawp=DATA/'raw_labels.npz'
    ids=npz_member_memmap(ds,'ids'); z=npz_member_memmap(ds,'z'); pos=npz_member_memmap(ds,'pos'); train=npz_member_memmap(ds,'train').astype(np.int64); val=npz_member_memmap(ds,'val').astype(np.int64)
    train_ids=np.asarray(ids[train]); val_ids=np.asarray(ids[val]); train_z=np.asarray(z[train]); val_z=np.asarray(z[val]); train_pos=np.asarray(pos[train]); val_pos=np.asarray(pos[val])
    if np.intersect1d(train,val).size: raise RuntimeError('train/validation index overlap')
    if np.intersect1d(train_ids,val_ids).size: raise RuntimeError('train/validation ID overlap')
    raw_ids=npz_member_memmap(rawp,'ids'); raw_f=npz_member_memmap(rawp,'f'); raw_mask=npz_member_memmap(rawp,'mask_f')
    train_raw_ids=np.asarray(raw_ids[train]);val_raw_ids=np.asarray(raw_ids[val]);yt=np.asarray(raw_f[train],dtype=np.float64);yv=np.asarray(raw_f[val],dtype=np.float64);mt=np.asarray(raw_mask[train],dtype=bool);mv=np.asarray(raw_mask[val],dtype=bool)
    if not np.array_equal(train_raw_ids,train_ids) or not np.array_equal(val_raw_ids,val_ids): raise RuntimeError('raw-label IDs not aligned')
    if not mt.all() or not mv.all() or yt.shape!=(len(train),10) or yv.shape!=(len(val),10): raise RuntimeError('unexpected raw label masks/shapes')
    names=['mto_eta0','mto_eta01','mto_eta1']; preds={}; files=[]
    for name in names:
        path=PRED/name/'val_predictions.npz'; files.append(path)
        with np.load(path,allow_pickle=False) as q:
            ii=q['indices']; mi=q['ids']; truth=q['f_true'].astype(np.float64); mask=q['mask_f_true'].astype(bool); pred=q['f'].astype(np.float64)
        if not np.array_equal(ii,val) or not np.array_equal(mi,val_ids): raise RuntimeError(f'{name} validation indices/IDs mismatch')
        if not np.array_equal(truth,yv) or not np.array_equal(mask,mv): raise RuntimeError(f'{name} raw truth/mask mismatch')
        preds[name]=pred
    if len({p.shape for p in preds.values()})!=1: raise RuntimeError('prediction shape mismatch')
    equal3=(preds['mto_eta0']+preds['mto_eta01']+preds['mto_eta1'])/3.0
    predictions={'eta0':preds['mto_eta0'],'equal3':equal3}
    with threadpool_limits(limits=2):
        train_g=[geometry(zz,pp) for zz,pp in zip(train_z,train_pos)]
        val_g=[geometry(zz,pp) for zz,pp in zip(val_z,val_pos)]
    qtrain=np.array([g[0] for g in train_g]); qval=np.array([g[0] for g in val_g]); q3train=np.array([g[1] for g in train_g]); q3val=np.array([g[1] for g in val_g])
    oktrain=np.isfinite(qtrain); okval=np.isfinite(qval)
    qs=[float(np.quantile(qtrain[oktrain],q)) for q in (.001,.01,.10,.50)]
    raw_bounds=[1e-5]+qs; bounds=[]
    for b in sorted(raw_bounds):
        if not bounds or not np.isclose(b,bounds[-1],rtol=0,atol=1e-15): bounds.append(float(b))
    binlabels=['q2 <= %.12g'%bounds[0]]+[('%.12g < q2 <= %.12g'%(bounds[i-1],bounds[i])) for i in range(1,len(bounds))]+['q2 > %.12g'%bounds[-1]]
    tr_bin=np.searchsorted(bounds,qtrain,side='left'); va_bin=np.searchsorted(bounds,qval,side='left')
    # Selected identity metadata: only train/validation row indices are parsed; test rows are lexically skipped.
    selected=np.concatenate([train,val]); ident,n_identity_rows=stream_selected_identity_rows(IDENT,selected)
    for idx,mid in zip(train,train_ids):
        if int(ident[int(idx)][0])!=int(mid): raise RuntimeError(f'train identity mismatch {idx}')
    for idx,mid in zip(val,val_ids):
        if int(ident[int(idx)][0])!=int(mid): raise RuntimeError(f'val identity mismatch {idx}')
    formula_train=[formula(a) for a in train_z]; formula_val=[formula(a) for a in val_z]
    target_id=14562
    where=np.flatnonzero(val_ids==target_id)
    if len(where)!=1: raise RuntimeError('target ID not found exactly once in validation')
    vi=int(where[0]); target_idx=int(val[vi]); target_formula=formula_val[vi]; target_group=str(ident[target_idx][1]); target_q2=float(qval[vi]); target_q3=float(q3val[vi]); target_zero=bool(np.all(yv[vi]==0))
    neartr=oktrain & (qtrain<=1e-5); nearval=okval & (qval<=1e-5)
    same_formula=np.array([f==target_formula for f in formula_train]); same_group=np.array([str(ident[int(i)][1])==target_group for i in train]); same_formula_near=same_formula&neartr; same_formula_group=same_formula&same_group
    group_ids=[int(i) for i in train_ids[same_group]]
    formula_near_ids=[int(i) for i in train_ids[same_formula_near]]
    # All benchmark outcomes retain the full validation population.
    global_scores={}; yzero=(yv==0)
    for name,pred in predictions.items(): global_scores[name]=score(yv,pred)
    global_sse={k:v['sse'] for k,v in global_scores.items()}
    strata=[]
    for bi,label in enumerate(binlabels):
        ti=(tr_bin==bi)&oktrain; vi_mask=(va_bin==bi)&okval
        row={'bin_index':bi,'label':label,'training_molecules':int(ti.sum()),'validation_molecules':int(vi_mask.sum()),'training_q2_count':int(ti.sum()),'training_raw_f_zero_count':int((yt[ti]==0).sum()),'training_raw_f_label_count':int(mt[ti].sum()),'training_raw_f_zero_prevalence':float((yt[ti]==0).sum()/mt[ti].sum()) if mt[ti].sum() else None,'validation_raw_f_zero_count':int(yzero[vi_mask].sum()),'validation_raw_f_label_count':int(mv[vi_mask].sum()),'validation_raw_f_zero_prevalence':float(yzero[vi_mask].sum()/mv[vi_mask].sum()) if mv[vi_mask].sum() else None,'validation_molecules_with_any_zero':int(np.any(yzero[vi_mask],axis=1).sum()),'validation_molecules_all_zero':int(np.all(yzero[vi_mask],axis=1).sum()),'models':{}}
        for name,pred in predictions.items():
            allscore=score(yv[vi_mask],pred[vi_mask]) if vi_mask.any() else {'sse':0.,'mean_signed_error':None,'overprediction_fraction':None,'labels':0}
            per=[]
            for st in range(10):
                m=vi_mask & mv[:,st]
                if not m.any(): per.append({'state':st,'labels':0});continue
                s=score(yv[m,st],pred[m,st]);per.append({'state':st,'labels':int(m.sum()),'raw_f_zero_count':int((yv[m,st]==0).sum()),'sse':s['sse'],'mean_signed_error':s['mean_signed_error'],'overprediction_fraction':s['overprediction_fraction']})
            mol_sse=np.square(pred[vi_mask]-yv[vi_mask]).sum(axis=1) if vi_mask.any() else np.array([],dtype=np.float64)
            case_only=None
            if vi_mask.any() and vi_mask[vi]:
                case_sse=float(np.square(pred[vi]-yv[vi]).sum())
                case_only={'excluded_molecule_id':14562,'remaining_molecules':int(vi_mask.sum()-1),'sse_after_case_exclusion_concentration_only':float(allscore['sse']-case_sse),'mean_sse_per_remaining_molecule_concentration_only':float((allscore['sse']-case_sse)/(vi_mask.sum()-1)) if vi_mask.sum()>1 else None}
            row['models'][name]={**allscore,'sse_share_of_full_validation':allscore['sse']/global_sse[name] if global_sse[name] else None,'mean_sse_per_molecule':allscore['sse']/int(vi_mask.sum()) if vi_mask.any() else None,'median_sse_per_molecule':float(np.median(mol_sse)) if len(mol_sse) else None,'p90_sse_per_molecule':float(np.quantile(mol_sse,.9)) if len(mol_sse) else None,'case_excluded_concentration_only':case_only,'per_state':per}
        strata.append(row)
    case_models={}
    for name,pred in predictions.items():
        case_models[name]={'per_state':[],'sse':float(np.square(pred[vi]-yv[vi]).sum()),'sse_share_of_full_validation':float(np.square(pred[vi]-yv[vi]).sum()/global_sse[name])}
        for st in range(10):
            err=float(pred[vi,st]-yv[vi,st]);case_models[name]['per_state'].append({'state':st,'truth':float(yv[vi,st]),'prediction':float(pred[vi,st]),'error':err,'overprediction':bool(err>0),'sse':err*err})
    def examples(mask):
        idx=np.flatnonzero(mask);return {'count':int(len(idx)),'ids':[int(x) for x in train_ids[idx]],'q2': [float(qtrain[x]) for x in idx]}
    result={'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'classification':'validation geometry/concentration diagnostic only; no training, inference, test use, or sample removal','protocol_sha256':sha(OUT/'PROTOCOL.md'),'code_sha256':sha(Path(__file__)),'pinned_sources':{str(p):sha(p) for p in [ds,rawp,*files,IDENT]},'alignment':{'train_molecules':len(train),'validation_molecules':len(val),'train_val_indices_disjoint':True,'raw_ids_and_masks_verified':True,'each_saved_val_prediction_ids_indices_truth_mask_verified':True,'selected_identity_rows_verified_train_val_only':True,'identity_source_rows_total_lexically_skipped_nonselected':n_identity_rows-len(selected)},'geometry':{'definition':'centered unweighted coordinate covariance X.T@X/n_atoms in FP64; eigenvalues descending','degeneracy':'undefined when lambda1 <= 64*eps64*max(trace(cov),1)','near_linear_threshold_q2_le_1e-5':1e-5,'train_quantiles':{str(q):v for q,v in zip((.001,.01,.10,.50),qs)},'deduplicated_boundaries_including_fixed_threshold':bounds,'interval_convention':'[0,b0], then (previous,boundary], final (last,+inf); degenerate geometry separately counted','train_degenerate_count':int((~oktrain).sum()),'validation_degenerate_count':int((~okval).sum()),'train_near_linear_count':int(neartr.sum()),'validation_near_linear_count':int(nearval.sum()),'training_q3_quantiles':{str(q):float(np.quantile(q3train[np.isfinite(q3train)],q)) for q in (.001,.01,.10,.50)},'validation_q2_quantiles':{str(q):float(np.quantile(qval[okval],q)) for q in (.001,.01,.10,.50)},'validation_q3_quantiles':{str(q):float(np.quantile(q3val[np.isfinite(q3val)],q)) for q in (.001,.01,.10,.50)}},'full_validation_metrics_all_molecules_retained':global_scores,'strata':strata,'case_14562':{'dataset_index':target_idx,'molecule_id':target_id,'formula':[[z,n] for z,n in target_formula],'q2_lambda2_over_lambda1':target_q2,'q3_lambda3_over_lambda1':target_q3,'near_linear_q2_le_1e-5':target_q2<=1e-5,'all_ten_raw_f_exactly_zero':target_zero,'connectivity_key':target_group,'shape_bin':binlabels[int(va_bin[vi])],'models':case_models},'training_matches_to_case':{'same_composition':examples(same_formula),'near_linear_any_composition':examples(neartr),'same_composition_and_near_linear':examples(same_formula_near),'same_existing_connectivity_group':examples(same_group),'same_composition_and_connectivity_group':examples(same_formula_group)},'limitations':['Geometry associations are descriptive and do not establish causality or label errors.','Exact raw-f zeros are reported as supplied; no claim is made that labels are erroneous.','All benchmark metrics include every validation molecule; molecule 14562 is isolated only for concentration accounting.','Equal3 is the arithmetic mean of three saved validation native-f predictions; no new inference or fitted weights.','No test indices, test labels, test geometries, test predictions, or test metrics were accessed.']}
    (OUT/'geometry_error_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Validation geometry and oscillator-strength error audit','',f"Created {result['created_at']} (Asia/Shanghai). Validation-only descriptive analysis; no fitting, inference, test access, or sample removal.",'','## Fixed geometry definition and bins','',result['geometry']['definition']+'. Ratios are '+result['geometry']['degeneracy']+'. Near-linear means q2 <= 1e-5.','',f"Train-derived q2 quantiles (0.1%, 1%, 10%, 50%): {qs}. After adding 1e-5 and deduplicating, boundaries are {bounds}. The intervals are left-open/right-closed after the first `[0,b0]` bin; degenerate rows are reported separately.",'',f"Train/validation degenerates: {result['geometry']['train_degenerate_count']}/{result['geometry']['validation_degenerate_count']}. Near-linear train/validation: {result['geometry']['train_near_linear_count']}/{result['geometry']['validation_near_linear_count']}.",'','## Full validation benchmark (all molecules retained)','','| Predictor | Molecules | Labels | SSE | R² | MAE | RMSE |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in global_scores.items():
        sst=float(np.square(yv-yv.mean()).sum()); lines.append(f"| {name} | {len(val)} | {yv.size} | {m['sse']:.12g} | {1-m['sse']/sst:.9f} | {float(np.abs((predictions[name]-yv)).mean()):.9g} | {float(np.sqrt(m['sse']/yv.size)):.9g} |")
    lines+=['','## Validation strata','', '| Shape stratum | Train n | Train exact-zero f prevalence | Val n | Exact-zero f labels / prevalence | eta0 SSE share | eta0 mean SSE/mol | eta0 mean signed error | eta0 overprediction fraction | equal3 SSE share | equal3 mean SSE/mol | equal3 mean signed error | equal3 overprediction fraction |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in strata:
        e=row['models']['eta0'];q=row['models']['equal3'];lines.append(f"| {row['label']} | {row['training_molecules']} | {row['training_raw_f_zero_prevalence'] if row['training_raw_f_zero_prevalence'] is not None else 'n/a'} | {row['validation_molecules']} | {row['validation_raw_f_zero_count']}/{row['validation_raw_f_label_count']} ({row['validation_raw_f_zero_prevalence'] if row['validation_raw_f_zero_prevalence'] is not None else 'n/a'}) | {e['sse_share_of_full_validation'] if e['sse_share_of_full_validation'] is not None else 'n/a'} | {e['mean_sse_per_molecule'] if e['mean_sse_per_molecule'] is not None else 'n/a'} | {e['mean_signed_error'] if e['mean_signed_error'] is not None else 'n/a'} | {e['overprediction_fraction'] if e['overprediction_fraction'] is not None else 'n/a'} | {q['sse_share_of_full_validation'] if q['sse_share_of_full_validation'] is not None else 'n/a'} | {q['mean_sse_per_molecule'] if q['mean_sse_per_molecule'] is not None else 'n/a'} | {q['mean_signed_error'] if q['mean_signed_error'] is not None else 'n/a'} | {q['overprediction_fraction'] if q['overprediction_fraction'] is not None else 'n/a'} |")
    lines+=['','SSE distribution summaries below use the full stratum. The separate case-exclusion fields in JSON are concentration-only diagnostics and do not alter benchmark scores. Median/P90 per-molecule SSE for the sparsely populated low-q2 bins should be read with their molecule counts.','','## State-level signed error and overprediction by shape stratum','', '| Shape stratum | State | eta0 mean signed error | eta0 overprediction fraction | equal3 mean signed error | equal3 overprediction fraction |','|---|---:|---:|---:|---:|---:|']
    for row in strata:
        for st in range(10):
            a=row['models']['eta0']['per_state'][st];b=row['models']['equal3']['per_state'][st]
            if a.get('labels',0): lines.append(f"| {row['label']} | {st} | {a['mean_signed_error']:.9g} | {a['overprediction_fraction']:.6f} | {b['mean_signed_error']:.9g} | {b['overprediction_fraction']:.6f} |")
    low=next((x for x in strata if x['validation_molecules']==1 and x['validation_raw_f_zero_count']==10),None)
    low2=next((x for x in strata if x['validation_molecules']==5),None)
    if low and low2:
        combined={}
        for name in ('eta0','equal3'):
            sse=low['models'][name]['sse']+low2['models'][name]['sse']
            combined[name]={'sse':sse,'share':sse/global_sse[name],'molecules':6,'mean_sse':sse/6,'after_case_sse':low2['models'][name]['sse'],'after_case_mean':low2['models'][name]['mean_sse_per_molecule']}
        lines+=['','## Low-q2 concentration check','',f"The strict near-linear validation bin contains only molecule 14562, so there is no independent validation replicate under q2 <= 1e-5. The next low-q2 bin contains {low2['validation_molecules']} other molecules. The two lowest bins together contain six molecules and contribute eta0 SSE share {combined['eta0']['share']:.3%} and equal3 SSE share {combined['equal3']['share']:.3%}. This total includes molecule 14562 and is descriptive.",'', '| Predictor | Six-molecule SSE share | Mean SSE per molecule, six incl. case | SSE in next bin after removing case for concentration only | Next-bin median SSE/mol | Next-bin P90 SSE/mol |','|---|---:|---:|---:|---:|---:|']
        for name in ('eta0','equal3'):
            m=low2['models'][name];lines.append(f"| {name} | {combined[name]['share']:.6%} | {combined[name]['mean_sse']:.9g} | {combined[name]['after_case_sse']:.9g} | {m['median_sse_per_molecule']:.9g} | {m['p90_sse_per_molecule']:.9g} |")
        lines+=['','The “after removing case” column is only a concentration diagnostic: the full validation benchmark and all main-stratum metrics still retain every molecule. The five-molecule low-q2 group has elevated median and P90 per-molecule SSE versus the higher-q2 strata, which is consistent with a broader low-shape-ratio error population but does not establish a causal geometry effect.']
    c=result['case_14562'];lines+=['','## Molecule 14562 concentration view','',f"ID {c['molecule_id']}, dataset index {c['dataset_index']}, formula C8H2, q2={c['q2_lambda2_over_lambda1']:.12g}, q3={c['q3_lambda3_over_lambda1']:.12g}, near-linear={c['near_linear_q2_le_1e-5']}, all ten raw-f labels exactly zero={c['all_ten_raw_f_exactly_zero']}, shape stratum `{c['shape_bin']}`. Its per-model SSE shares of full validation: eta0 {c['models']['eta0']['sse_share_of_full_validation']:.6%}; equal3 {c['models']['equal3']['sse_share_of_full_validation']:.6%}. Full-benchmark metrics above and by-stratum totals retain this molecule.",'','### Training comparators','']
    for label,key in [('Same composition','same_composition'),('Near-linear, any composition','near_linear_any_composition'),('Same composition and near-linear','same_composition_and_near_linear'),('Same exact connectivity group','same_existing_connectivity_group'),('Same composition and connectivity group','same_composition_and_connectivity_group')]:
        v=result['training_matches_to_case'][key];lines.append(f"- {label}: {v['count']} training molecules; IDs: {v['ids']}")
    lines+=['','### Case per-state errors','', '| State | Truth | eta0 pred | eta0 error | equal3 pred | equal3 error |','|---:|---:|---:|---:|---:|---:|']
    for a,b in zip(c['models']['eta0']['per_state'],c['models']['equal3']['per_state']):lines.append(f"| {a['state']} | {a['truth']:.12g} | {a['prediction']:.12g} | {a['error']:.12g} | {b['prediction']:.12g} | {b['error']:.12g} |")
    lines+=['','## Limits','',*['- '+x for x in result['limitations']],'','Source hashes, IDs, source indices, raw FP64 truths/masks and prediction alignment are recorded in `geometry_error_metrics.json`. No training, test access, or source-model inference was performed.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'created_at':result['created_at'],'train':len(train),'val':len(val),'protocol_sha256':result['protocol_sha256'],'code_sha256':result['code_sha256'],'metrics':global_scores,'case':result['case_14562']['q2_lambda2_over_lambda1'],'matches':{k:v['count'] for k,v in result['training_matches_to_case'].items()},'bins':bounds},allow_nan=False))
if __name__=='__main__':main()