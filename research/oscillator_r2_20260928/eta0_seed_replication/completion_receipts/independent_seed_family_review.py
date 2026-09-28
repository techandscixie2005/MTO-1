from pathlib import Path
import sys,json,hashlib,datetime
from collections import defaultdict
import numpy as np
import torch
R=Path('/home/inspur/MTO-1/research/oscillator_r2_20260928'); S=R/'eta0_seed_replication'; O=S/'completion_receipts'
sys.path.insert(0,str(R/'postrun_geometry'))
import postrun_geometry as pg
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def close(a,b):assert np.allclose(a,b,rtol=0,atol=2e-11),(a,b)
old,pins=pg.pinned_round(); rows=pg.validation_rows(old); y=rows['truth']; assert y.dtype==np.float64 and rows['mask'].all()
prov=js(R/'postrun_geometry/FIXED_SEVEN_SECONDARY_PROVENANCE.json')
five=[]; evidence={}
for label,v in prov['frozen_five_components'].items():
 p=Path(v['prediction_path']);assert sha(p)==v['prediction_sha256'];z=np.load(p)
 for key,rkey in [('ids','ids'),('indices','indices'),('f_true','truth'),('mask_f_true','mask')]:assert np.array_equal(z[key],rows[rkey])
 five.append(z['f'].astype(np.float64));evidence[str(p)]=sha(p)
F5=sum(five)/5;equal3=sum(five[:3])/3
result=js(S/'RESULTS.json'); f7result=js(R/'fixed_seven_sidecar/SEED_FIXED7_RESULTS.json')
review=js(S/'IMPLEMENTATION_REVIEW.json'); orders=js(S/'ORDER_PLAN.json')['seeds']; seeds={}; checks={}
for path,dig in {**review['source_hashes'],**review['code_hashes']}.items():assert sha(path)==dig
for seed in (23,37):
 base=S/f'runs/seed{seed}';fit=js(base/'FIT_COMPLETE.json');hist=[json.loads(x) for x in (base/'history.jsonl').read_text().splitlines()]
 assert fit['event']=='FIT_COMPLETE' and fit['epochs']==100 and fit['steps']==188100 and fit['test_batches']==0
 assert len(hist)==100 and [x['epoch'] for x in hist]==list(range(1,101))
 assert fit['source_hashes']==review['source_hashes'] and fit['code_hashes']==review['code_hashes']
 assert all(h['order_sha256']==o['order_sha256'] and h['updates_this_epoch']==1881 and h['steps']==h['epoch']*1881 for h,o in zip(hist,orders[str(seed)]))
 assert min(hist,key=lambda h:h['val_legacy'][0])['epoch']==fit['best_legacy_epoch']
 assert min(hist,key=lambda h:h['val_raw_f']['sse'])['epoch']==fit['best_raw_f_epoch']
 a=torch.load(base/'best_legacy.pt',map_location='cpu',weights_only=False);b=torch.load(base/'best_raw_f.pt',map_location='cpu',weights_only=False)
 assert a['model'].keys()==b['model'].keys()
 assert all(a['model'][k].dtype==b['model'][k].dtype and torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
 for k in set(a)-{'model','selection','metric'}:assert a[k]==b[k]
 for name,file in [('legacy','best_legacy.pt'),('raw_f','best_raw_f.pt')]:assert sha(base/file)==fit['checkpoint_hashes'][name]
 metrics={}
 for name in ['epoch0','best_legacy','best_raw_f','final100']:
  p=base/f'val_{name}.npz';z=np.load(p);evidence[str(p)]=sha(p)
  assert np.array_equal(z['ids'],rows['ids']) and np.array_equal(z['indices'],rows['indices']) and z['f_true'].dtype==np.float64 and np.array_equal(z['f_true'],y)
  pred=z['f'].astype(np.float64);sse=float(((pred-y)**2).sum());sst=float(((y-y.mean())**2).sum())
  metrics[name]={'sse':sse,'r2':1-sse/sst,'mae':float(abs(pred-y).mean())}
  if name=='best_legacy':seeds[seed]=pred
  if name in fit['val_prediction_hashes']:assert sha(p)==fit['val_prediction_hashes'][name]
 assert np.array_equal(np.load(base/'val_best_legacy.npz')['f'],np.load(base/'val_best_raw_f.npz')['f'])
 close(metrics['best_legacy']['sse'],result['metrics'][f'seed{seed}_legacy']['sse'])
 checks[str(seed)]={'epochs':100,'steps':188100,'orders':100,'model_tensors_equal':len(a['model']),'metadata_only_differences':{'legacy':[a['selection'],a['metric']],'raw_f':[b['selection'],b['metric']]},'validation':metrics,'terminal_sha256':sha(base/'FIT_COMPLETE.json')}
seed3=(five[0]+seeds[23]+seeds[37])/3; F7=(sum(five)+seeds[23]+seeds[37])/7
preds={'eta0':five[0],'equal3':equal3,'seed3':seed3,'F5':F5,'F7':F7}
sst=float(((y-y.mean())**2).sum()); metrics={k:{'sse':float(((p-y)**2).sum()),'r2':float(1-((p-y)**2).sum()/sst)} for k,p in preds.items()}
for k in ['F5','F7']:close(metrics[k]['r2'],f7result['metrics'][k]['r2'])
close(metrics['seed3']['r2'],result['metrics']['primary_equal_seeds_11_23_37']['r2'])
ident,_=pg.stream_selected_identity_rows(pg.IDENT,rows['indices']);groups=defaultdict(list)
for j,ix in enumerate(rows['indices']):
 entry=ident[int(ix)];assert int(entry[0])==int(rows['ids'][j]) and entry[2] is True;groups[entry[1]].append(j)
def boot(ref,pred,units,seed):
 sy=np.asarray([y[u].sum() for u in units]); sy2=np.asarray([(y[u]**2).sum() for u in units]); n=np.asarray([y[u].size for u in units]); gain=np.asarray([((ref[u]-y[u])**2).sum()-((pred[u]-y[u])**2).sum() for u in units]); rng=np.random.default_rng(seed);v=[]
 for _ in range(2000):
  ix=rng.integers(len(units),size=len(units));v.append(gain[ix].sum()/(sy2[ix].sum()-sy[ix].sum()**2/n[ix].sum()))
 return {'ci':np.quantile(v,[.025,.5,.975]).tolist(),'positive_fraction':float((np.asarray(v)>0).mean())}
bs={}
for name,ref,cand,seed,ordered in [('seed3_minus_eta0',five[0],seed3,20260928,list(groups.values())),('seed3_minus_equal3',equal3,seed3,20260928,list(groups.values())),('F7_minus_F5',F5,F7,20260929,[groups[k] for k in sorted(groups)]),('F7_minus_seed3_primary',seed3,F7,20260929,[groups[k] for k in sorted(groups)])]:
 bs[name]={}
 for mode,units in [('molecule',[[i] for i in range(len(y))]),('group',ordered)]:
  b=boot(ref,cand,units,seed);bs[name][mode]=b
  if name.startswith('F7'):
   expected=f7result['comparisons'][name]['molecule_bootstrap' if mode=='molecule' else 'connectivity_bootstrap'];close(b['ci'],expected['delta_r2_ci95']);close(b['positive_fraction'],expected['positive_fraction'])
  else:
   expected=result['bootstrap_vs_seed11' if name.endswith('eta0') else 'bootstrap_vs_fixed_equal_three'][mode if mode=='molecule' else 'connectivity_group']['primary_equal_seeds_11_23_37'];close(b['ci'],expected['delta_r2_2p5_50_97p5']);close(b['positive_fraction'],expected['positive_fraction'])
con={}
for name,ref,cand in [('seed3_vs_equal3',equal3,seed3),('F7_vs_F5',F5,F7),('F7_vs_seed3',seed3,F7)]:
 er=(ref-y)**2;ec=(cand-y)**2;gain=(er-ec).sum(1)
 con[name]={'per_state_sse_gain':(er-ec).sum(0).tolist(),'tails_sse_gain':{label:float((er-ec)[mask].sum()) for label,mask in [('belowq90',y<.0546),('q90',y>=.0546),('q99',y>=.2377)]},'improved':int((gain>0).sum()),'worsened':int((gain<0).sum()),'net_gain':float(gain.sum()),'case14562':{'ref_sse':float(er[rows['case']].sum()),'candidate_sse':float(ec[rows['case']].sum()),'gain':float(gain[rows['case']])},'max_gain_id':int(rows['ids'][gain.argmax()]),'max_gain':float(gain.max()),'max_loss_id':int(rows['ids'][gain.argmin()]),'max_loss':float(gain.min()),'q2_bins':[{'count':int((rows['bin2']==j).sum()),'ref_sse':float(er[rows['bin2']==j].sum()),'candidate_sse':float(ec[rows['bin2']==j].sum())} for j in range(6)]}
 for refname in ['F7_minus_F5' if name=='F7_vs_F5' else 'F7_minus_seed3_primary' if name=='F7_vs_seed3' else None]:
  if refname:
   c=f7result['comparisons'][refname];close(con[name]['per_state_sse_gain'],[-x['delta_sse_candidate_minus_reference'] for x in c['per_state']]);close(con[name]['net_gain'],c['concentration']['net_sse_gain'])
refs=['RESULTS.json','completion_receipts/SEED23_COMPLETION_AUDIT.json','completion_receipts/SEED37_COMPLETION_AUDIT.json','completion_receipts/SEED_FAMILY_SUMMARY_RECEIPT.json']
for f in refs:evidence[str(S/f)]=sha(S/f)
for f in ['fixed_seven_sidecar/SEED_FIXED7_RESULTS.json','fixed_seven_completion/SEED_COMPAT_EXECUTION.json','fixed_seven_completion/COMPATIBILITY_REVIEW.json','fixed_seven_completion/ORIGINAL_SCHEMA_FAILURE.json']:evidence[str(R/f)]=sha(R/f)
out={'passed':True,'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'Saved validation arrays, CPU checkpoint tensor reads and fixed source hashes only; no inference/test or active scratch results','script_sha256':sha(__file__),'sources_and_code_verified':len(set(review['source_hashes'])|set(review['code_hashes'])),'seed_checks':checks,'metrics':metrics,'bootstrap':bs,'concentration':con,'evidence_sha256':evidence,'limits':['Full-train figures are completion aggregates, not independently recomputed from train predictions.','Bootstrap is descriptive conditional on validation selection; no correction for prior adaptation.','No model forward or new predictions.']}
path=O/'SEED_FAMILY_INDEPENDENT_VERIFICATION.json'; assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(path),'sha256':sha(path),'metrics':metrics,'concentration':con}))
