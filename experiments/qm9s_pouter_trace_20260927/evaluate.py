"""Only runs after all four fits; freezes selected checkpoints before any test inference."""
import fcntl,json,time
import numpy as np
import torch
from dataset import ROOT,Data
from model_factory import build
from trainer import setup,atomic_json,fingerprint
from prepare import sha
K=(2/3)/27.211386245988
GRID=np.arange(1051,dtype=np.float64)*.02

def regression(pred,true,mask):
    p=np.asarray(pred,dtype=np.float64)[mask];t=np.asarray(true,dtype=np.float64)[mask]
    assert np.isfinite(p).all() and np.isfinite(t).all()
    if not t.size:return dict(count=0,MAE=None,RMSE=None,R2=None)
    d=p-t;sst=np.square(t-t.mean()).sum()
    return dict(count=int(t.size),MAE=float(np.abs(d).mean()),RMSE=float(np.sqrt(np.square(d).mean())),R2=float(1-np.square(d).sum()/sst) if sst else None)
def broaden(e,f,mask):
    e=np.where(mask,e,0);f=np.where(mask,f,0)
    return (f[...,None]*np.exp(-.5*((GRID-e[...,None])/.2)**2)/(.2*np.sqrt(2*np.pi))).sum(1)
def spec_metrics(e,f,t,mask):
    mse=[];mae=[]
    for a in range(0,len(e),64):
        sl=slice(a,a+64)
        d=broaden(e[sl],f[sl],mask[sl])-broaden(t['E'][sl],t['f'][sl],mask[sl])
        mse.extend(np.square(d).mean(1));mae.extend(np.abs(d).mean(1))
    return dict(MSE=float(np.mean(mse)),MAE=float(np.mean(mae))),np.asarray(mse),np.asarray(mae)
def metrics(pred,truth,eref,supervise_E):
    trace=np.trace(pred['A'].astype(np.float64),axis1=-2,axis2=-1);target=np.trace(truth['A'],axis1=-2,axis2=-1)
    ma=truth['mask_A'];me=truth['mask_E'];mf=truth['mask_f'];mask=ma&me&mf
    out=dict(trace=regression(trace,target,ma),per_state=[dict(state=k+1,trace=regression(trace[:,k],target[:,k],ma[:,k])) for k in range(10)])
    arrays=dict(trace=trace,E_ref=eref)
    modes=dict(oracle=(truth['E'],'使用真实能量的诊断'),common=(eref,'G1 validation-selected best energy, evaluation only'))
    if supervise_E:modes['native']=(pred['E'].astype(np.float64),'own supervised energy')
    for mode,(e,label) in modes.items():
        f=K*e*trace;s,smse,smae=spec_metrics(e,f,truth,mask)
        out[mode]=dict(label=label,f=regression(f,truth['f'],mask),spectrum=s)
        arrays.update({f'f_{mode}':f,f'spectrum_{mode}_MSE_per_molecule':smse,f'spectrum_{mode}_MAE_per_molecule':smae})
        for k in range(10):out['per_state'][k][f'f_{mode}']=regression(f[:,k],truth['f'][:,k],mask[:,k])
    if supervise_E:
        out['E_eV']=regression(pred['E'],truth['E'],me)
        for k in range(10):out['per_state'][k]['E_eV']=regression(pred['E'][:,k],truth['E'][:,k],me[:,k])
    else:out['energy_head_status']='Unsupervised and frozen exclusive parameters; shared features change. No formal own-E/native-f/native-spectrum metrics.'
    return out,arrays

@torch.no_grad()
def predict(model,data,idx):
    e=[];a=[];model.eval()
    for start in range(0,len(idx),64):
        x,_=data.batch(idx[start:start+64]);ee,aa=model(**x);e.append(ee.cpu().numpy());a.append(aa.cpu().numpy())
    return dict(E=np.concatenate(e),A=np.concatenate(a))

def main():
    lock=(ROOT/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    setup(11);campaign=json.loads((ROOT/'campaign.json').read_text());names=campaign['groups']
    assert len(names)==4 and all((ROOT/'runs'/n/'FIT_COMPLETE.json').exists() and not (ROOT/'runs'/n/'FAILED.json').exists() for n in names)
    cfgs={n:json.loads((ROOT/'configs'/f'{n}.json').read_text()) for n in names};checkpoints={};fp=fingerprint()
    for n in names:
        path=ROOT/'runs'/n/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['config']==cfgs[n] and ck['fingerprint']==fp
        fit=json.loads((ROOT/'runs'/n/'FIT_COMPLETE.json').read_text());assert ck['epoch']==fit['best_epoch']
        checkpoints[n]=dict(best_pt_sha256=sha(path),epoch=ck['epoch'],own_validation_objective=ck['val'][0],config_sha256=sha(ROOT/'configs'/f'{n}.json'))
    freeze=dict(checkpoints=checkpoints,protocol_sha256=sha(ROOT/'campaign.json'),E_ref='G1',test_used_for_selection=False,
        classification=campaign['classification'],selection=campaign['selection'])
    p=ROOT/'reports/selection_before_test.json'
    if p.exists():assert json.loads(p.read_text())==freeze
    else:atomic_json(freeze,p)
    # Only after the immutable selection record has been written do we predict val/test.
    data=Data();raw=np.load(ROOT/'data/raw_labels.npz');results={n:{} for n in names}
    for part in ('val','test'):
        if part=='test':atomic_json(dict(time=time.time(),freeze_sha256=sha(p)),ROOT/'reports/test_evaluation_started.json')
        idx=data.parts[part];truth={k:raw[k][idx] for k in ('E','A','f','mask_E','mask_A','mask_f')};preds={}
        for n in names:
            path=ROOT/'runs'/n/'best.pt';assert sha(path)==checkpoints[n]['best_pt_sha256']
            ck=torch.load(path,map_location='cpu',weights_only=False);model=build(cfgs[n],data.stats).cuda();model.load_state_dict(ck['model'])
            preds[n]=predict(model,data,idx);del model;torch.cuda.empty_cache()
        eref=preds['G1']['E'].astype(np.float64)
        for n in names:
            result,arrays=metrics(preds[n],truth,eref,cfgs[n]['supervise_E']);results[n][part]=result
            out=ROOT/'runs'/n;atomic_json(result,out/f'{part}_metrics.json')
            export={'E_pred' if cfgs[n]['supervise_E'] else 'E_unsupervised_diagnostic':preds[n]['E'],'A_pred':preds[n]['A']}
            np.savez_compressed(out/f'{part}_predictions.npz',ids=data.ids[idx],indices=idx,**export,**arrays,**{k+'_true':v for k,v in truth.items()})
    atomic_json(results,ROOT/'reports/results.json')
    atomic_json(dict(time=time.time(),groups=names,selection_sha256=sha(p),results_sha256=sha(ROOT/'reports/results.json')),ROOT/'reports/ANALYSIS_COMPLETE.json')
    print('ALL_FOUR_EVALUATED',flush=True)
if __name__=='__main__':main()
