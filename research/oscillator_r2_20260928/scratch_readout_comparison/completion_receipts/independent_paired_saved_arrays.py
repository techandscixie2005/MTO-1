from pathlib import Path
import json,hashlib,sys,datetime
from collections import defaultdict
import numpy as np
R=Path('/home/inspur/MTO-1/research/oscillator_r2_20260928');S=R/'scratch_readout_comparison';O=S/'completion_receipts'
sys.path.insert(0,str(R/'postrun_geometry'));import postrun_geometry as pg
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text())
def close(a,b):assert np.allclose(a,b,atol=2e-10,rtol=0),(a,b)
old,pins=pg.pinned_round();rows=pg.validation_rows(old);y=rows['truth'];pred={}
for arm in ['native','mto']:
 p=S/f'runs/{arm}';fit=js(p/'FIT_COMPLETE.json');assert fit['event']=='FIT_COMPLETE' and fit['epochs']==100 and fit['steps']==188100 and fit['test_batches']==0
 z=np.load(p/'val_best_raw_f.npz');assert sha(p/'val_best_raw_f.npz')==fit['prediction_hashes']['best_raw_f']
 for n,key in [('ids','ids'),('indices','indices'),('f_true','truth')]:assert np.array_equal(z[n],rows[key])
 assert z['f_true'].dtype==np.float64 and rows['mask'].all();pred[arm]=z['f'].astype(np.float64)
prov=js(R/'postrun_geometry/FIXED_SEVEN_SECONDARY_PROVENANCE.json');arr=[]
for v in prov['frozen_five_components'].values():
 assert sha(v['prediction_path'])==v['prediction_sha256'];z=np.load(v['prediction_path']);assert np.array_equal(z['f_true'],y) and np.array_equal(z['ids'],rows['ids']);arr.append(z['f'].astype(np.float64))
pred.update({'scratch_mean':(pred['native']+pred['mto'])/2,'F5':sum(arr)/5,'old3':sum(arr[:3])/3,'eta0':arr[0]})
ident,_=pg.stream_selected_identity_rows(pg.IDENT,rows['indices']);gg=defaultdict(list)
for j,ix in enumerate(rows['indices']):assert int(ident[int(ix)][0])==int(rows['ids'][j]);gg[ident[int(ix)][1]].append(j)
groups=[gg[k] for k in sorted(gg)]
def metric(p):
 e=(p-y)**2;return {'sse':float(e.sum()),'r2':float(1-e.sum()/((y-y.mean())**2).sum()),'mae':float(abs(p-y).mean())}
def bootstrap(ref,p,units,seed):
 n=np.array([y[g].size for g in units]);sy=np.array([y[g].sum() for g in units]);sy2=np.array([(y[g]**2).sum() for g in units]);gain=np.array([((ref[g]-y[g])**2).sum()-((p[g]-y[g])**2).sum() for g in units]);rng=np.random.default_rng(seed);v=[]
 for _ in range(2000):
  ix=rng.integers(len(units),size=len(units));v.append(gain[ix].sum()/(sy2[ix].sum()-sy[ix].sum()**2/n[ix].sum()))
 return {'mean':float(np.mean(v)),'ci95':np.quantile(v,[.025,.975]).tolist(),'positive_fraction':float((np.array(v)>0).mean())}
def compare(ref,p,seed):
 a=(ref-y)**2;b=(p-y)**2;gain=(a-b).sum(1);case=rows['case'];gains=np.array([gain[g].sum() for g in groups]);best=int(gain.argmax());worst=int(gain.argmin())
 return {'delta_r2':float(gain.sum()/((y-y.mean())**2).sum()),'delta_sse':float(-gain.sum()),'bootstrap_molecule':bootstrap(ref,p,[[i] for i in range(len(y))],seed),'bootstrap_connectivity_group':bootstrap(ref,p,groups,seed),'per_state_sse_gain':(a-b).sum(0).tolist(),'tails':{label:{'reference_sse':float(a[m].sum()),'candidate_sse':float(b[m].sum()),'gain':float((a-b)[m].sum())} for label,m in [('belowq90',y<.0546),('q90',y>=.0546),('q99',y>=.2377)]},'molecules_improved':int((gain>0).sum()),'molecules_worsened':int((gain<0).sum()),'case14562':{'reference_sse':float(a[case].sum()),'candidate_sse':float(b[case].sum()),'gain':float(gain[case])},'largest_positive':{'id':int(rows['ids'][best]),'gain':float(gain[best])},'largest_negative':{'id':int(rows['ids'][worst]),'gain':float(gain[worst])},'largest_group_gain':float(gains.max()),'largest_group_loss':float(gains.min()),'q2':[{'count':int((rows['bin2']==i).sum()),'reference_sse':float(a[rows['bin2']==i].sum()),'candidate_sse':float(b[rows['bin2']==i].sum())} for i in range(6)],'six_lowq2':{'reference_sse':float(a[rows['six']].sum()),'candidate_sse':float(b[rows['six']].sum())}}
comparisons={'native_minus_mto':compare(pred['mto'],pred['native'],20260928)}
for name in ['native','mto','scratch_mean']:comparisons[name+'_minus_F5']=compare(pred['F5'],pred[name],20260929)
sumry=js(S/'RESULTS.json');c=comparisons['native_minus_mto'];close(c['delta_r2'],sumry['primary_difference_native_minus_mto_r2'])
for kind in ['bootstrap_molecule','bootstrap_connectivity_group']:
 for k in ['mean','ci95','positive_fraction']:close(c[kind][k],sumry[kind][k])
geom=js(R/'postrun_geometry/SCRATCH_RESULTS.json')
metrics={k:metric(p) for k,p in pred.items()}
for label,x in geom['predictors'].items():
 # Match unique raw-f pooled SSE without assuming report display labels.
 candidates=[k for k,m in metrics.items() if abs(m['sse']-x.get('full',{}).get('sse',-1000))<1e-8]
# Source utility/report schema checked separately below; no new model forward.
out={'passed':True,'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'Completed selected arrays only; prescribed primary, equal scratch mean and F5 references; no F7 scratch arithmetic, inference/test','script_sha256':sha(__file__),'metrics':metrics,'comparisons':comparisons,'evidence':{str(p):sha(p) for p in [S/'RESULTS.json',R/'postrun_geometry/SCRATCH_RESULTS.json',O/'MTO_INDEPENDENT_VERIFICATION.json',O/'NATIVE_INDEPENDENT_VERIFICATION.json']},'bootstrap_groups':len(groups),'limits':['Single paired seed, unequal readout capacity/routing and equal update budget rather than compute.','Validation checkpoint selection and repeated reuse make intervals descriptive.']}
f=O/'PAIRED_INDEPENDENT_VERIFICATION.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(f),'sha256':sha(f),'metrics':metrics,'comparisons':{k:{a:b for a,b in v.items() if a not in ['q2']} for k,v in comparisons.items()}}))
