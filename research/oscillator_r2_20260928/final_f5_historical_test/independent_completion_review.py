import pathlib,json,hashlib,datetime,importlib.util,numpy as np
P=pathlib.Path('/home/inspur/MTO-1/research/oscillator_r2_20260928/final_f5_historical_test');ROOT=P.parent;SRC=pathlib.Path('/home/inspur/MTO-1/experiments/qm9s_chan64_20260928')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as r:
  for x in iter(lambda:r.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def J(p):return json.loads(pathlib.Path(p).read_text())
def near(a,b):assert np.isclose(a,b,rtol=1e-12,atol=1e-12),(a,b)
f=J(P/'CANDIDATE_FREEZE.json');r=J(P/'TEST_RESULTS.json');done=J(P/'TEST_COMPLETE.json');start=J(P/'STARTED.json');auth=J(P/'EXECUTION_AUTHORIZATION.json');rev=J(P/'IMPLEMENTATION_REVIEW.json')
assert sha(P/'TEST_RESULTS.json')=='374b04f5edc19217e383068de1728c0348e63275c657eb7a4214903394e8b784'
assert sha(P/'TEST_COMPLETE.json')=='3c4cc676b3f4fb1d459ce09486e30a983682e3667a33eb6d421aaaf4e6746eee'
for file,key in [('CANDIDATE_FREEZE.json','freeze_sha256'),('PROTOCOL.md','protocol_sha256'),('evaluate.py','script_sha256'),('PREFLIGHT.json','preflight_sha256')]:
 h=sha(P/file);assert auth[key]==start[key]==rev[key]==h
assert rev['passed'] and auth['approved'] and auth['retries_authorized']==0
assert sha(P/'IMPLEMENTATION_REVIEW.json')==auth['review_sha256']==start['review_sha256']
assert sha(P/'EXECUTION_AUTHORIZATION.json')==start['authorization_sha256']
assert auth['publication_commit']==start['publication_commit']=='aba91bc88c22f280a6adde1cd3002579c7b7e410'
assert auth['archive_manifest_sha256']==start['archive_manifest_sha256']
assert datetime.datetime.fromisoformat(auth['authorized_at_utc']).timestamp()<start['time']<r['time']<=done['time']
assert done['event']=='TEST_COMPLETE' and start['event']=='TEST_STARTED'
assert done['started_sha256']==r['started_sha256']==sha(P/'STARTED.json')
assert done['result_sha256']==sha(P/'TEST_RESULTS.json') and done['report_sha256']==sha(P/'TEST_REPORT.md')
assert not any((P/n).exists() for n in ['FAILED.json','INVALID.json','PARTIAL.json'])
for table in ['source_data_scaler_hashes_verified','artifact_and_provenance_hashes_verified']:
 for path,h in f[table].items():assert sha(path)==h,path
before=start['gpu_before'];after=done['gpu_after'];assert before==r['gpu_before'] and after==r['gpu_after'];assert before['uuid']==after['uuid']=='GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184';assert before['ecc']==after['ecc'];assert not before['apps'] and before['memory_MiB']<=200;assert after['apps']==[start['pid']]
# Import only pinned geometry/memmap utility; no model or evaluator execution.
gp=ROOT/'geometry_error_audit/geometry_error_audit.py';assert sha(gp)=='cda3735529a57e83a6183ef82a85fc022022517404681bff0635a9bf91e54472'
spec=importlib.util.spec_from_file_location('saved_geometry',gp);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
mm=g.npz_member_memmap;ds=SRC/'data/dataset.npz';raw=SRC/'data/raw_labels.npz';idx=np.asarray(mm(ds,'test'));ids=np.asarray(mm(ds,'ids')[idx]);y=np.asarray(mm(raw,'f')[idx]);en=np.asarray(mm(raw,'E')[idx]);mask=np.asarray(mm(raw,'mask_f')[idx]&mm(raw,'mask_E')[idx]&mm(raw,'mask_A')[idx]);assert y.dtype==np.float64 and mask.all() and y.shape==(6686,10);assert np.array_equal(mm(raw,'ids')[idx],ids)
allidx=np.concatenate([np.asarray(mm(ds,k)) for k in ['train','val','test']]);assert np.array_equal(np.sort(allidx),np.arange(133727));assert np.unique(mm(ds,'ids')).size==133727
pred={};array_hashes={}
for n in f['component_order']:
 c=f['components'][n];assert sha(c['selected_checkpoint_path'])==c['checkpoint_sha256']
 if n.startswith('mto_'):path=pathlib.Path(c['historical_test_array']['path']);h=c['historical_test_array']['sha256_from_historical_receipt']
 else:
  path=P/'runtime'/f'{n}_selected_native_test.npz';h=done['runtime_prediction_hashes'][n];assert r['new_selected_inference'][n]['prediction_sha256']==h;assert r['new_selected_inference'][n]['checkpoint_sha256']==c['checkpoint_sha256'];assert r['new_selected_inference'][n]['selected_epoch']==c['epoch']
 assert sha(path)==h;array_hashes[n]=h
 with np.load(path,allow_pickle=False) as z:
  for key,v in [('indices',idx),('ids',ids),('f_true',y),('mask_f_true',mask)]:assert np.array_equal(z[key],v),(n,key)
  assert z['f_true'].dtype==np.float64 and z['mask_f_true'].dtype==bool
  pred[n]=z['f'].astype(np.float64);assert np.isfinite(pred[n]).all()
  if n in ['G1','G3']:
   assert np.array_equal(z['E_true'],en);assert str(z['checkpoint_sha256'])==c['checkpoint_sha256'];assert str(z['run_manifest_sha256'])==c['run_manifest_sha256'];assert np.array_equal(((2/3)/27.211386245988)*z['E_pred']*z['trace_pred'],pred[n])
preds={'eta0':pred['mto_eta0'],'old3':sum(pred[n] for n in f['component_order'][:3])/3,'F5':sum(pred[n] for n in f['component_order'])/5}
sst=float(((y-y.mean())**2).sum());errs={n:(p-y)**2 for n,p in preds.items()};metrics={}
for n,p in preds.items():
 e=p-y;v={'sse':float(errs[n].sum()),'sst':sst,'r2':1-float(errs[n].sum())/sst,'mae':float(np.abs(e).mean()),'rmse':float(np.sqrt(errs[n].mean())),'mean_signed_error':float(e.mean()),'overprediction_fraction':float((e>0).mean())};metrics[n]=v
 for k,x in v.items():near(x,r['predictors'][n]['overall'][k])
 for j,state in enumerate(r['predictors'][n]['states']):
  near(errs[n][:,j].sum(),state['sse']);near(1-errs[n][:,j].sum()/((y[:,j]-y[:,j].mean())**2).sum(),state['r2']);near(np.abs(e[:,j]).mean(),state['mae']);near(e[:,j].mean(),state['mean_signed_error'])
 for name,thr in [('q90',.0549),('q99',.2412)]:
  t=r['predictors'][n]['TRAIN_tail_slices'][name];assert t['threshold']==thr
  for side,sel in [('bright',y>=thr),('below',y<thr)]:assert int(sel.sum())==t[side]['count'];near(errs[n][sel].sum(),t[side]['sse'])
 for name,count in [('top1',1),('top10',10),('top1pct',67)]:
  v=r['predictors'][n]['molecule_error_concentration'][name];ss=np.sort(errs[n].sum(1))[-count:].sum();near(ss,v['sse']);near(ss/errs[n].sum(),v['share_of_total_sse'])
identity,_=g.stream_selected_identity_rows(SRC/'data/identity_audit_v2.json',idx);assert all(int(identity[int(i)][0])==int(mid) for i,mid in zip(idx,ids));groups=[str(identity[int(i)][1]) for i in idx]
# Independently aggregate resampling sufficient statistics; same prescribed RNG/order, no evaluator function use.
boot={}
for name,unit in [('molecule',list(range(len(y)))),('connectivity_group',groups)]:
 keys=sorted(set(unit));members=[np.flatnonzero(np.array(unit)==u) for u in keys];counts=np.array([len(a)*10 for a in members]);ys=np.array([y[a].sum() for a in members]);ys2=np.array([(y[a]**2).sum() for a in members]);gain=np.array([(errs['old3'][a]-errs['F5'][a]).sum() for a in members]);rng=np.random.default_rng(20260929);draw=[]
 for _ in range(2000):
  chosen=rng.integers(0,len(keys),len(keys));den=ys2[chosen].sum()-ys[chosen].sum()**2/counts[chosen].sum();draw.append(gain[chosen].sum()/den)
 q=np.quantile(draw,[.025,.5,.975]);pos=float((np.array(draw)>0).mean());assert np.allclose(q,r['paired_bootstrap'][name]['delta_r2_p2p5_p50_p97p5'],rtol=0,atol=1e-14);assert pos==r['paired_bootstrap'][name]['positive_fraction'];boot[name]={'units':len(keys),'quantiles':q.tolist(),'positive_fraction':pos}
z=np.asarray(mm(ds,'z')[idx]);pos=np.asarray(mm(ds,'pos')[idx]);q2=[]
for zz,pp in zip(z,pos):
 xx=pp[zz!=0].astype(np.float64);xx-=xx.mean(0);cov=xx.T@xx/len(xx);ev=np.linalg.eigvalsh(cov)[::-1];tol=64*np.finfo(float).eps*max(float(np.trace(cov)),1);q2.append(ev[1]/ev[0] if ev[0]>tol else np.nan)
bins=np.where(np.isfinite(q2),np.searchsorted(f['prospective_diagnostics']['q2_boundaries'],q2,side='left'),-1)
for n in preds:
 for b in r['predictors'][n]['TRAIN_q2_bins']:
  sel=bins==b['q2_bin'];assert int(sel.sum())==b['molecules'];near(errs[n][sel].sum(),b['sse']);near(errs[n][sel].sum()/errs[n].sum(),b['share_of_total_sse'])
gain=errs['old3'].sum(1)-errs['F5'].sum(1);change=r['primary_error_change'];near(gain.sum(),change['net_sse_gain_reference_minus_candidate']);near(gain[gain>0].sum(),change['gross_molecule_gain']);near(gain[gain<0].sum(),change['gross_molecule_loss']);assert (gain>0).sum()==change['molecules_improved'];assert (gain<0).sum()==change['molecules_worsened']
for polarity in ['positive','negative']:
 for row in change['top10_'+polarity+'_molecules']:
  i=int(np.flatnonzero(ids==row['id'])[0]);near(gain[i],row['sse_gain']);near(errs['old3'][i].sum(),row['reference_sse']);near(errs['F5'][i].sum(),row['candidate_sse'])
 for row in change['top10_'+polarity+'_groups']:
  selected=np.isin(ids,row['molecule_ids']);near(gain[selected].sum(),row['sse_gain']);assert all(hashlib.sha256(groups[i].encode()).hexdigest()==row['connectivity_key_sha256'] for i in np.flatnonzero(selected));near(errs['old3'][selected].sum(),row['reference_sse']);near(errs['F5'][selected].sum(),row['candidate_sse'])
hist=J(ROOT/'ensemble_diagnostic/EXPLORATORY_TEST.json')
near(metrics['old3']['r2'],hist['ensemble']['r2']);near(metrics['eta0']['r2'],hist['baseline']['r2'])
group_gain=np.array([gain[np.array(groups)==k].sum() for k in sorted(set(groups))]);near(group_gain[group_gain>0].sum(),change['gross_group_gain']);near(group_gain[group_gain<0].sum(),change['gross_group_loss']);assert int((group_gain>0).sum())==change['groups_improved'];assert int((group_gain<0).sum())==change['groups_worsened']
state_gain=(errs['old3']-errs['F5']).sum(0);tail_gain={name:{'bright':float((errs['old3']-errs['F5'])[y>=thr].sum()),'below':float((errs['old3']-errs['F5'])[y<thr].sum())} for name,thr in [('q90',.0549),('q99',.2412)]};geom_gain=[{'bin':b,'molecules':int((bins==b).sum()),'sse_gain':float((errs['old3']-errs['F5'])[bins==b].sum())} for b in range(6)]
assert r['extra_model_forwards']==2 and r['deployment_model_forwards']=={'F5':5,'old3':3,'eta0':1};assert len(r['new_selected_inference'])==2
out={'passed':True,'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'classification':'Independent completed saved-array and pinned-source audit; no model inference, training or evaluation rerun','result_sha256':sha(P/'TEST_RESULTS.json'),'complete_sha256':sha(P/'TEST_COMPLETE.json'),'script_sha256':sha(P/'independent_completion_review.py'),'metrics':metrics,'delta_r2':metrics['F5']['r2']-metrics['old3']['r2'],'paired_bootstrap_independent':boot,'array_hashes':array_hashes,'state_sse_gain_old3_minus_F5':state_gain.tolist(),'TRAIN_tail_sse_gain':tail_gain,'TRAIN_q2_sse_gain':geom_gain,'largest_positive_molecule_net_gain_fraction':float(gain.max()/gain.sum()),'top10_positive_net_gain_fraction':float(np.sort(gain)[-10:].sum()/gain.sum()),'molecules_improved':int((gain>0).sum()),'molecules_worsened':int((gain<0).sum()),'checks':['all125 frozen source/artifact hashes','publication-bound authorization and chronological STARTED/COMPLETE','5 checkpoint hashes/selected G epochs','5 immutable prediction hashes and exact raw FP64 truth/IDs/masks','new native-f E-times-FP64trace arithmetic','all original split indices/IDs','metrics/state/tail/geometry/concentration','independent2000 molecule/group bootstrap','GPU4 UUID/ECC unchanged and only worker before completion','no candidate change or new recipe'],'limitations':['Historical test reused and candidate selected adaptively on validation; bootstrap omits selection/adaptation uncertainty.','Two complete new checkpoint dataset passes from reviewed source and receipts; no independent re-execution.']}
with (P/'INDEPENDENT_COMPLETION_REVIEW.json').open('x',encoding='utf-8') as w:json.dump(out,w,indent=2);w.write('\n')
print(json.dumps(out,indent=2));print('REVIEW_SHA',sha(P/'INDEPENDENT_COMPLETION_REVIEW.json'))