from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np,joblib
R=Path(__file__).resolve().parent;P=R/'runs/one_probe';B=R.parent
sys.path.insert(0,str(R));from array_io import member
sys.path.insert(0,str(B/'postrun_geometry'));import postrun_geometry as pg
from collections import defaultdict
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def eq(a,b):assert np.allclose(a,b,rtol=0,atol=2e-10),(a,b)
complete=js(P/'PROBE_COMPLETE.json');freeze=js(P/'FIT_FREEZE.json');val=js(P/'VALIDATION_RESULTS.json');review=js(R/'IMPLEMENTATION_REVIEW.json');auth=js(R/'FIT_AUTHORIZATION.json');seal=js(R/'SOURCE_SEAL.json')
assert complete['event']=='PROBE_COMPLETE' and complete['elapsed_seconds']<14400 and not complete['test_access']
assert not any((P/n).exists() for n in ['FAILED.json','INCOMPLETE_RESOURCE.json'])
for n,d in review['source_hashes'].items():assert sha(R/n)==d
for x in seal['files_pre_fit']+seal['files_post_fit']:assert sha(x['path'])==x['sha256']
assert auth['approved'] and auth['one_run'] and auth['review_sha256']==sha(R/'IMPLEMENTATION_REVIEW.json')
assert complete['fit_freeze_sha256']==sha(P/'FIT_FREEZE.json') and complete['validation_results_sha256']==sha(P/'VALIDATION_RESULTS.json')
assert freeze['time']<js(P/'VALIDATION_STARTED.json')['time']<val['time']
assert js(P/'VALIDATION_STARTED.json')['fit_freeze_sha256']==sha(P/'FIT_FREEZE.json')
resources={}
for phase in ['train','validation']:
 q=P/f'PHASE_{phase.upper()}_RESOURCE.json';a=js(q);assert complete['phase_resource_receipts'][phase]==sha(q)
 assert a['returncode']==0 and a['resource_failure'] is None and a['peak_tree_rss_bytes']<17179869184 and a['elapsed_seconds']<a['phase_wall_budget_seconds']<=14400
 assert a['resource_limit_bytes']==17179869184 and len(a['admission']['selected_cpus'])==8
 ph=js(P/f'SUPERVISOR_PHASE_{phase}.json');assert ph['review_sha256']==sha(R/'IMPLEMENTATION_REVIEW.json') and ph['auth_sha256']==sha(R/'FIT_AUTHORIZATION.json')
 resources[phase]=a
for key,file in [('feature_hash','TRAIN_FEATURES.npy'),('forest_hash','FOREST.joblib'),('train_predictions_hash','TRAIN_PREDICTIONS.npz')]:assert freeze[key]==sha(P/file)
for key,file in [('val_feature_hash','VAL_FEATURES.npy'),('val_predictions_hash','VAL_PREDICTIONS.npz')]:assert val[key]==sha(P/file)
X=np.load(P/'TRAIN_FEATURES.npy',mmap_mode='r');V=np.load(P/'VAL_FEATURES.npy',mmap_mode='r');assert X.shape==(120355,803) and V.shape==(6686,803) and X.dtype==V.dtype==np.float32 and np.isfinite(X).all() and np.isfinite(V).all()
pre=js(R/'IMPLEMENTATION_PREFLIGHT.json');data=Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/data');train=np.asarray(member(data/'dataset.npz','train'));rawids=member(data/'raw_labels.npz','ids')
assert sha(P/'FOREST.joblib')==freeze['forest_hash'];model=joblib.load(P/'FOREST.joblib');cfg=js(R/'IMPLEMENTATION_CONFIG.json')
assert model.get_params()==freeze['effective_estimator_params']
for k,v in cfg['learner'].items():assert model.get_params()[k]==v
assert len(model.estimators_)==256 and model.n_features_in_==803 and model.n_outputs_==10
assert all(t.tree_.n_node_samples[0]==120355 and t.tree_.max_depth<=24 and t.tree_.n_node_samples[t.tree_.children_left==-1].min()>=4 for t in model.estimators_)
assert sum(t.tree_.node_count for t in model.estimators_)==freeze['total_tree_nodes']
for row in pre['train_sample_rows']:
 loc=np.flatnonzero(train==row['dataset_index']);assert len(loc)==1 and hashlib.sha256(X[loc[0]].tobytes()).hexdigest()==row['descriptor_sha256']
zt=np.load(P/'TRAIN_PREDICTIONS.npz');assert np.array_equal(zt['indices'],train);ty=zt['truth'];tp=zt['pred'];assert ty.dtype==np.float64 and np.array_equal(ty,member(data/'raw_labels.npz','f')[train]) and member(data/'raw_labels.npz','mask_f')[train].all()
assert hashlib.sha256(train.tobytes()).hexdigest()==freeze['train_indices_sha256'] and hashlib.sha256(np.asarray(rawids[train]).tobytes()).hexdigest()==freeze['train_ids_sha256']
eq(np.sqrt(((ty-ty.mean())**2).mean()),freeze['train_target_scale']);eq(ty.mean(),freeze['train_target_mean'])
def metric(y,p):
 e=p-y;se=float((e*e).sum());st=float(((y-y.mean())**2).sum());return {'sse':se,'sst':st,'r2':1-se/st,'mae':float(abs(e).mean()),'rmse':float(np.sqrt((e*e).mean()))}
trainmet=metric(ty,tp)
for k,v in trainmet.items():eq(v,freeze['train_metrics'][k])
old,pins=pg.pinned_round();rows=pg.validation_rows(old);y=rows['truth'];z=np.load(P/'VAL_PREDICTIONS.npz')
for n,r in [('ids','ids'),('indices','indices'),('f_true','truth'),('mask_f_true','mask')]:assert np.array_equal(z[n],rows[r])
assert z['f_true'].dtype==np.float64 and rows['mask'].all();D=z['D'];blend=z['blend'];assert D.dtype==blend.dtype==np.float64 and np.isfinite(D).all() and (D>=0).all()
prov=js(B/'postrun_geometry/FIXED_SEVEN_SECONDARY_PROVENANCE.json');arr=[]
for v in prov['frozen_five_components'].values():
 assert sha(v['prediction_path'])==v['prediction_sha256'];a=np.load(v['prediction_path']);assert np.array_equal(a['f_true'],y) and np.array_equal(a['ids'],rows['ids']);arr.append(a['f'].astype(np.float64))
F5=sum(arr)/5;assert np.array_equal(blend,(D+F5)/2)
metrics={k:metric(y,a) for k,a in [('D',D),('F5',F5),('equal_blend',blend)]}
for k,m in metrics.items():
 for n,v in m.items():eq(v,val['metrics'][k][n])
ident,_=pg.stream_selected_identity_rows(pg.IDENT,rows['indices']);groups=defaultdict(list)
for i,ix in enumerate(rows['indices']):assert int(ident[int(ix)][0])==int(rows['ids'][i]);groups[ident[int(ix)][1]].append(i)
groups=[groups[k] for k in sorted(groups)]
def boot(ref,p,units):
 n=np.asarray([y[u].size for u in units]);sy=np.asarray([y[u].sum() for u in units]);sy2=np.asarray([(y[u]**2).sum() for u in units]);g=np.asarray([((ref[u]-y[u])**2).sum()-((p[u]-y[u])**2).sum() for u in units]);rng=np.random.default_rng(20260929);d=[]
 for _ in range(2000):
  ix=rng.integers(len(units),size=len(units));d.append(g[ix].sum()/(sy2[ix].sum()-sy[ix].sum()**2/n[ix].sum()))
 return np.quantile(d,[.025,.5,.975]).tolist(),float((np.asarray(d)>0).mean())
comparisons={}
for name,pred in [('D_vs_F5',D),('equal_blend_vs_F5',blend)]:
 c=val['comparisons'][name];bs={}
 for kind,units in [('molecule_bootstrap',[[i] for i in range(len(y))]),('connectivity_bootstrap',groups)]:
  ci,pos=boot(F5,pred,units);eq(ci,c[kind]['delta_r2_ci95']);eq(pos,c[kind]['positive_fraction']);bs[kind]={'ci95':ci,'positive_fraction':pos}
 er=(F5-y)**2;ep=(pred-y)**2;gain=(er-ep).sum(1);cs=c['concentration']
 eq(float(gain.sum()),cs['net_sse_gain']);eq((er-ep).sum(0),[-v['delta_sse_candidate_minus_reference'] for v in c['per_state']]);assert int((gain>0).sum())==cs['molecules_improved']
 for label,mask in [('below_q90',y<.0546),('q90_and_above',y>=.0546),('q99_and_above',y>=.2377)]:eq(float(ep[mask].sum()-er[mask].sum()),c['tails'][label]['delta_sse_candidate_minus_reference'])
 for b in cs['fixed_q2_bins']:
  sel=rows['bin2']==b['q2_bin'];assert int(sel.sum())==b['molecules'];eq(float(er[sel].sum()),b['reference_sse']);eq(float(ep[sel].sum()),b['candidate_sse'])
 eq(float(ep[rows['case']].sum()),cs['case14562']['candidate_sse'])
 comparisons[name]={'bootstrap':bs,'per_state_sse_gain':(er-ep).sum(0).tolist(),'tails':c['tails'],'concentration':cs}
for label,pred in [('D',D),('F5',F5),('equal_blend',blend)]:eq((pred-y).mean(),val['signed_error_pred_minus_truth'][label]['overall_mean']);eq((pred-y).mean(0),val['signed_error_pred_minus_truth'][label]['per_state_mean'])
files=[R/'IMPLEMENTATION_REVIEW.json',R/'FIT_AUTHORIZATION.json',P/'PROBE_COMPLETE.json',P/'FIT_FREEZE.json',P/'VALIDATION_STARTED.json',P/'VALIDATION_RESULTS.json',P/'PHASE_TRAIN_RESOURCE.json',P/'PHASE_VALIDATION_RESOURCE.json',P/'DESCRIPTOR_LAUNCH_RECEIPT.json']
out={'passed':True,'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'Saved arrays/features and forest metadata only. No fit, predict/model forward, test or active scratch outcomes.','script_sha256':sha(__file__),'artifact_sha256':{str(p):sha(p) for p in files},'train_metrics':trainmet,'validation_metrics':metrics,'comparisons':comparisons,'resources':resources,'checks':{'feature_shapes':[list(X.shape),list(V.shape)],'features_finite':True,'train_rows_all':120355,'val_rows_all':6686,'model_trees':256,'model_params_exact':True,'full_row_tree_root_counts':True,'initial_16_descriptor_hashes_match_stored_features':True,'train_raw_labels_exact':True,'val_raw_labels_exact':True,'train_scale_exact':True,'fit_before_validation':True,'two_fixed_bootstraps_reproduced':True,'launch_process_expired_before_live_check':'Expected after successful short completion; resource enforcement supported by reviewed child gates and successful phase receipts, not a retrospective /proc claim.'}}
path=R/'INDEPENDENT_COMPLETION_AUDIT.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(path),'sha256':sha(path),'metrics':metrics,'diagnostics':{k:{'state_gain':v['per_state_sse_gain'],'improved':v['concentration']['molecules_improved'],'case':v['concentration']['case14562'],'worst':v['concentration']['largest_negative_molecule'],'best':v['concentration']['largest_positive_molecule']} for k,v in comparisons.items()}}))
