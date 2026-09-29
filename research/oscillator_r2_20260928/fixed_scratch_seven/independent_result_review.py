from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np
from collections import defaultdict
R=Path('/home/inspur/MTO-1/research/oscillator_r2_20260928');S=R/'fixed_scratch_seven'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def eq(a,b):assert np.allclose(a,b,rtol=0,atol=2e-10),(a,b)
result=js(S/'F7_SCRATCH_RESULTS.json');assert sha(S/'F7_SCRATCH_RESULTS.json')=='8643e3cd9f72d27242e58a71fbd3dd9d1f8421f7f8e8f10466da3ebff0c73357'
assert result['script_sha256']==sha(S/'fixed_scratch7.py') and result['review_sha256']==sha(S/'IMPLEMENTATION_REVIEW.json')
handoff=js(S/'HANDOFF.json')
for key,path in handoff['files'].items():assert sha(path)==handoff['hashes'][key]
sys.path.insert(0,str(R/'postrun_geometry'));import postrun_geometry as pg
old,pins=pg.pinned_round();rows=pg.validation_rows(old);y=rows['truth'];prov=js(R/'postrun_geometry/FIXED_SCRATCH_SEVEN_SECONDARY_PROVENANCE.json');five=[]
for x in prov['frozen_five_components'].values():
 assert sha(x['prediction_path'])==x['prediction_sha256'];z=np.load(x['prediction_path']);assert np.array_equal(z['f_true'],y) and np.array_equal(z['ids'],rows['ids']);five.append(z['f'].astype(np.float64))
F5=sum(five)/5;scratch=[];seeds=[];artifacts={}
for family,labels in [('scratch_readout_comparison',['native','mto']),('eta0_seed_replication',['seed23','seed37'])]:
 for label in labels:
  p=R/family/'runs'/label;fit=js(p/'FIT_COMPLETE.json');assert fit['epochs']==100 and fit['steps']==188100 and fit['test_batches']==0
  typ='raw_f' if family.startswith('scratch') else 'legacy';pred=p/f'val_best_{typ}.npz';ck=p/f'best_{typ}.pt';key='prediction_hashes' if typ=='raw_f' else 'val_prediction_hashes'
  assert sha(pred)==fit[key]['best_'+typ] and sha(ck)==fit['checkpoint_hashes'][typ]
  z=np.load(pred);assert np.array_equal(z['indices'],rows['indices']) and np.array_equal(z['ids'],rows['ids']) and z['f_true'].dtype==np.float64 and np.array_equal(z['f_true'],y)
  assert rows['mask'].all();(scratch if typ=='raw_f' else seeds).append(z['f'].astype(np.float64));artifacts[str(pred)]=sha(pred);artifacts[str(ck)]=sha(ck);artifacts[str(p/'FIT_COMPLETE.json')]=sha(p/'FIT_COMPLETE.json')
F7s=(sum(five)+sum(scratch))/7;F7seed=(sum(five)+sum(seeds))/7
assert np.allclose(F7s,(5*F5+scratch[0]+scratch[1])/7,rtol=0,atol=1e-15)
metrics={}
for name,p in [('F5',F5),('F7_scratch',F7s),('F7_seed_legacy',F7seed)]:
 e=(p-y)**2;metrics[name]={'sse':float(e.sum()),'r2':float(1-e.sum()/((y-y.mean())**2).sum()),'mae':float(abs(p-y).mean())}
 for k,v in metrics[name].items():eq(v,result['metrics'][name][k])
 eq((p-y).mean(),result['signed_error_pred_minus_truth'][name]['overall_mean']);eq((p-y).mean(0),result['signed_error_pred_minus_truth'][name]['per_state_mean'])
ident,_=pg.stream_selected_identity_rows(pg.IDENT,rows['indices']);gg=defaultdict(list)
for j,ix in enumerate(rows['indices']):assert int(ident[int(ix)][0])==int(rows['ids'][j]);gg[ident[int(ix)][1]].append(j)
groups=[gg[k] for k in sorted(gg)]
def boot(ref,p,units):
 n=np.array([y[u].size for u in units]);s=np.array([y[u].sum() for u in units]);ss=np.array([(y[u]**2).sum() for u in units]);g=np.array([((ref[u]-y[u])**2).sum()-((p[u]-y[u])**2).sum() for u in units]);rng=np.random.default_rng(20260929);v=[]
 for _ in range(2000):
  ix=rng.integers(len(units),size=len(units));v.append(g[ix].sum()/(ss[ix].sum()-s[ix].sum()**2/n[ix].sum()))
 return np.quantile(v,[.025,.5,.975]).tolist(),float((np.array(v)>0).mean())
comparisons={}
for name,ref in [('F7_scratch_minus_F5',F5),('F7_scratch_minus_F7_seed_legacy',F7seed)]:
 c=result['comparisons'][name];a=(ref-y)**2;b=(F7s-y)**2;g=(a-b).sum(1)
 for kind,units in [('molecule_bootstrap',[[i] for i in range(len(y))]),('connectivity_bootstrap',groups)]:
  ci,pos=boot(ref,F7s,units);eq(ci,c[kind]['delta_r2_ci95']);eq(pos,c[kind]['positive_fraction'])
 eq((a-b).sum(0),[-x['delta_sse_candidate_minus_reference'] for x in c['per_state']]);cs=c['concentration'];eq(g.sum(),cs['net_sse_gain']);assert int((g>0).sum())==cs['molecules_improved'];assert int((g<0).sum())==cs['molecules_worsened']
 for label,m in [('below_q90',y<.0546),('q90_and_above',y>=.0546),('q99_and_above',y>=.2377)]:eq(a[m].sum(),c['tails'][label]['reference_sse']);eq(b[m].sum(),c['tails'][label]['candidate_sse'])
 for q in cs['fixed_q2_bins']:
  m=rows['bin2']==q['q2_bin'];assert m.sum()==q['molecules'];eq(a[m].sum(),q['reference_sse']);eq(b[m].sum(),q['candidate_sse'])
 for sign,ix in [('largest_positive_molecule',g.argmax()),('largest_negative_molecule',g.argmin())]:
  assert int(rows['ids'][ix])==cs[sign]['id'];eq(g[ix],cs[sign]['sse_gain_reference_minus_candidate'])
 eq(b[rows['case']].sum(),cs['case14562']['candidate_sse']);eq(a[rows['case']].sum(),cs['case14562']['reference_sse']);eq(b[rows['six']].sum(),cs['six_low_q2']['candidate_sse'])
 comparisons[name]={'delta_r2':c['delta_r2_candidate_minus_reference'],'delta_sse':c['delta_sse_candidate_minus_reference'],'group_ci':c['connectivity_bootstrap']['delta_r2_ci95'],'positive_fraction':c['connectivity_bootstrap']['positive_fraction'],'state_sse_gain':(a-b).sum(0).tolist(),'tails':c['tails'],'concentration':cs}
for f in ['F7_SCRATCH_RESULTS.json','F7_SCRATCH_REPORT.md','IMPLEMENTATION_REVIEW.json','HANDOFF.json']:artifacts[str(S/f)]=sha(S/f)
out={'passed':True,'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'exact seven saved-array arithmetic and prescribed contrasts only; no inference/test or mixture search','script_sha256':sha(__file__),'evidence_sha256':artifacts,'metrics':metrics,'comparisons':comparisons,'groups':len(groups),'weights':'each frozen5 plus native raw-selected and MTO raw-selected once, 1/7 each; eta0 once','limitations':['Pre-completed-pair-outcome but post-native-result adaptive mixture; intervals conditional on validation selection.','Seven forwards versus five/seven; no calibrated latency.']}
p=S/'INDEPENDENT_RESULT_REVIEW.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'metrics':metrics,'comparisons':{k:{x:v for x,v in c.items() if x!='concentration'} for k,c in comparisons.items()},'concentration':{k:{x:c['concentration'][x] for x in ['molecules_improved','molecules_worsened','gross_positive_gain','gross_deterioration','largest_positive_molecule','largest_negative_molecule','case14562','six_low_q2']} for k,c in comparisons.items()}}))
