from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np,torch
R=Path('/home/inspur/MTO-1/research/oscillator_r2_20260928/scratch_readout_comparison');P=R/'runs/mto';O=R/'completion_receipts'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def eq(a,b,tol=1e-10):assert np.allclose(a,b,rtol=0,atol=tol),(a,b)
fit=js(P/'FIT_COMPLETE.json');review=js(R/'IMPLEMENTATION_REVIEW.json');manifest=js(P/'RUN_MANIFEST.json');fixed=js(P/'FIXED_CHECKPOINT_METRICS.json');pre=js(R/'PREFLIGHT.json');init=js(R/'INITIALIZATION.json');plan=js(R/'ORDER_PLAN.json')
assert fit['event']=='FIT_COMPLETE' and fit['arm']=='mto' and fit['epochs']==100 and fit['steps']==188100 and fit['test_batches']==0
assert not any((P/n).exists() for n in ['FAILED.json','INVALID.json'])
assert sha(R/'PROTOCOL.md')==review['protocol_sha256'] and sha(R/'PREFLIGHT.json')==review['preflight_sha256']
for key in ['source_hashes','code_hashes']:
 assert fit[key]==manifest[key]==review[key]==pre[key]
 for path,d in review[key].items():assert sha(path)==d
assert fit['order_plan_sha256']==sha(R/'ORDER_PLAN.json') and fit['fixed_metrics_sha256']==sha(P/'FIXED_CHECKPOINT_METRICS.json')
hist=[json.loads(x) for x in (P/'history.jsonl').read_text().splitlines()];nh=[json.loads(x) for x in (R/'runs/native/history.jsonl').read_text().splitlines()]
assert len(hist)==100 and len(nh)==100
for row,order,nrow in zip(hist,plan['orders'],nh):
 ep=row['epoch'];assert row['order_sha256']==order['sha256']==nrow['order_sha256'] and row['updates_this_epoch']==1881 and row['steps']==1881*ep and ep==order['epoch']
 eq(row['lr'],.001 if ep<=60 else .0003 if ep<=85 else .0001)
zero=js(P/'epoch0_val.json');allrows=[{'epoch':0,'val_raw_f':zero['raw_f'],'val_normalized':zero['normalized_joint']}]+hist
assert min(allrows,key=lambda x:x['val_raw_f']['sse'])['epoch']==fit['best']['raw_f']['epoch']==16
assert min(allrows,key=lambda x:x['val_normalized'][0])['epoch']==fit['best']['joint']['epoch']==41
a=torch.load(R/'initial/native.pt',map_location='cpu',weights_only=False);b=torch.load(R/'initial/mto.pt',map_location='cpu',weights_only=False)
for native,mto in init['core_key_mapping'].items():assert torch.equal(a['model']['base.'+native],b['model']['base.core.'+mto])
assert fit['initial_hash']==b['state_sha256']==init['initialization']['mto']['state_sha256']
assert sha(R/'initial/mto.pt')==fit['checkpoint_hashes']['initial']
last=torch.load(P/'last.pt',map_location='cpu',weights_only=False);final=torch.load(P/'final100.pt',map_location='cpu',weights_only=False)
assert last['state']['epoch']==101 and last['state']['steps']==188100 and last['state']['cursor']==0 and last['state']['history']==hist and last['state']['best']==fit['best']
assert all(torch.equal(v,final['model'][k]) for k,v in last['model'].items())
for key in ['raw_f','joint']:
 ck=P/f'best_{key}.pt';z=P/f'val_best_{key}.npz';ep=fit['best'][key]['epoch'];c=torch.load(ck,map_location='cpu',weights_only=False)
 assert sha(ck)==fit['checkpoint_hashes'][key]==sha(P/f'selected/{key}_epoch{ep:03d}.pt')
 assert sha(z)==fit['prediction_hashes'][f'best_{key}']==sha(P/f'selected/{key}_epoch{ep:03d}_val.npz')
 assert c['arm']=='mto' and c['epoch']==ep and c['selection']==key and c['metric']==fit['best'][key]['metric']
 assert c['source_hashes']==review['source_hashes'] and c['code_hashes']==review['code_hashes']
for key in ['last','final100']:assert sha(P/(key+'.pt'))==fit['checkpoint_hashes'][key]
assert fit['gpu_before']['uuid']==fit['gpu_after']['uuid'] and fit['gpu_before']['ecc']==fit['gpu_after']['ecc']
ref=np.load('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/runs/mto_eta0/val_predictions.npz');truth=ref['f_true'];Etruth=ref['E_true'];assert truth.dtype==np.float64
vals={}
for key,fx in [('epoch0','initial'),('best_raw_f','raw_f'),('best_joint','joint'),('final100','final100')]:
 p=P/f'val_{key}.npz';assert sha(p)==fit['prediction_hashes'][key];z=np.load(p)
 for field in ['indices','ids','f_true','E_true']:assert np.array_equal(z[field],ref[field])
 assert ref['mask_f_true'].dtype==np.bool_ and ref['mask_f_true'].all()
 if 'mask_f_true' in z.files:assert np.array_equal(z['mask_f_true'],ref['mask_f_true'])
 assert z['f_true'].dtype==np.float64 and z['E_true'].dtype==np.float64
 f=z['f'];E=z['E'];assert np.isfinite(f).all() and (f>=0).all() and f.shape==(6686,10);er=(f-truth)**2;sst=float(((truth-truth.mean())**2).sum());sse=float(er.sum());metric={'sse':sse,'r2':1-sse/sst,'mae':float(abs(f-truth).mean()),'E_mae':float(abs(E-Etruth).mean()),'state_sse':er.sum(0).tolist(),'q90_bright_sse':float(er[truth>=.0546].sum()),'below_q90_sse':float(er[truth<.0546].sum()),'case14562_sse':float(er[ref['ids']==14562].sum())}
 vals[key]=metric
 if key=='best_raw_f':eq(sse,fit['best']['raw_f']['metric'])
 eq(sse,fixed[fx]['val']['raw_f']['sse'],1e-3)
 assert fixed[fx]['val_replay_max_abs_f']<=2e-5 and fixed[fx]['val_replay_max_abs_E']<=2e-5
 for section in ['train','val']:
  m=fixed[fx][section]['raw_f'];eq(m['r2'],1-m['sse']/m['sst']);eq(m['rmse'],np.sqrt(m['sse']/m['count']))
files=[P/'FIT_COMPLETE.json',P/'RUN_MANIFEST.json',P/'LAUNCH_RECEIPT.json',P/'history.jsonl',P/'FIXED_CHECKPOINT_METRICS.json',R/'IMPLEMENTATION_REVIEW.json',R/'PREFLIGHT.json',O/'NATIVE_INDEPENDENT_VERIFICATION.json']
out={'passed':True,'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'Completed MTO saved artifacts and CPU checkpoint comparisons only; no inference/test','script_sha256':sha(__file__),'evidence_sha256':{str(x):sha(x) for x in files},'source_count':len(review['source_hashes']),'code_count':len(review['code_hashes']),'updates':188100,'orders_exact_and_matched_native':100,'copied_initial_core_tensors_equal':len(init['core_key_mapping']),'selections':fit['best'],'saved_validation':vals,'completion_train_metrics':{k:v['train']['raw_f'] for k,v in fixed.items()},'completion_replay_maxima':{k:{'f':v['val_replay_max_abs_f'],'E':v['val_replay_max_abs_E']} for k,v in fixed.items()},'health_ecc_unchanged':True,'mask_alignment':'Scratch saved NPZ omits mask field; exact IDs/indices/FP64 f and E truth agree with pinned all-valid reference masks. No prediction-derived mask.','limits':['Full TRAIN metrics are completion-time aggregates, not independently recomputed train predictions.','Replay maxima were generated during authorized training completion; this audit performed no model forward.','Small differences between completion-time replay aggregates and selected arrays do not alter checkpoint selection; headline scores use saved selected arrays.','Temporary alias leftovers are not authoritative; final aliases match selected versioned artifacts and terminal hashes.']}
p=O/'MTO_INDEPENDENT_VERIFICATION.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'selected':fit['best'],'metrics':vals,'train':out['completion_train_metrics']}))
