"""Exercise production checkpoint/resume and evaluation definitions without formal fitting."""
import hashlib,json,pathlib,time
import numpy as np
import torch
import trainer
from trainer import ROOT,atomic_json
from evaluate import metrics,regression,K
from supervisor import health

def main():
    report={};h,b,xml=health();assert all(h[i]['clean'] for i in (1,2,4,6))
    (ROOT/'reports/gpu_health_preflight.xml').write_text(xml)
    assert all(pathlib.Path(f'/dev/nvidia{i}').exists() for i in (1,2,4,6))
    # Config resource metadata is finalized before production-path testing.
    path=ROOT/'configs/G4.json';c=json.loads(path.read_text());c['gpu']=6;atomic_json(c,path)
    for n in ('G1','G4'):
        path=ROOT/'reports'/f'scratch_production_{n}_{time.time_ns()}'
        trainer.STOP=True;trainer.run(n,preflight_out=path)
        a=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        assert a['state']['steps']==1 and a['state']['cursor']==64
        trainer.STOP=True;trainer.run(n,preflight_out=path)
        b=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        assert b['state']['steps']==2 and b['state']['cursor']==128
        assert np.array_equal(a['state']['order'],b['state']['order'])
        assert a['rng_numpy']==b['rng_numpy'] and a['fingerprint']==b['fingerprint']
        assert all(int(v['step'])==2 for v in b['optimizer']['state'].values())
        init=torch.load(ROOT/'initial'/('original.pt' if n=='G1' else 'p_outer.pt'),weights_only=True)
        if n=='G4':
            for k in ('decoder.energy_head.weight','decoder.energy_head.bias','decoder.energy_offset'):assert torch.equal(init[k],b['model'][k])
        report[n]=dict(first_cursor=64,resumed_cursor=128,optimizer_steps=2,order_sha256=hashlib.sha256(a['state']['order'].tobytes()).hexdigest(),production_checkpoint_roundtrip=True)
        (path/'last.pt').unlink()  # Only disposable diagnostic weights; never a formal checkpoint.
    n=3;e=np.arange(30,dtype=float).reshape(n,10)/10+2;a=np.zeros((n,10,3,3));a[:,:,0,0]=np.arange(30).reshape(n,10)/30
    mask=np.ones((n,10),bool);truth=dict(E=e,A=a,f=K*e*np.trace(a,axis1=-2,axis2=-1),mask_E=mask,mask_A=mask,mask_f=mask)
    exact,arrays=metrics(dict(E=e,A=a),truth,e,True)
    assert exact['trace']['R2']==1 and exact['trace']['MAE']==0
    assert all(exact[m]['f']['MAE']==0 and exact[m]['spectrum']['MSE']==0 for m in ('oracle','common','native'))
    result,arrays=metrics(dict(E=np.full_like(e,np.nan),A=a),truth,2*e,False)
    assert 'native' not in result and 'E_eV' not in result and 'f_native' not in arrays
    assert result['oracle']['f']['MAE']==0 and np.allclose(arrays['f_common'],2*truth['f'])
    assert result['common']['spectrum']['MSE']>0 and result['trace']['count']==30
    rr=regression(np.array([1.,3.]),np.array([0.,2.]),np.array([True,True]));assert rr['R2']==0
    report['evaluation']=dict(R2_is_1_minus_SSE_over_SST=True,oracle_label=exact['oracle']['label'],true_E_used_only_for_oracle=True,G1_common_energy_scale_checked=True,no_G2_G4_native_metrics=True,zero_labels_retained=True,exact_spectrum_MSE=0)
    report['passed']=True;atomic_json(report,ROOT/'reports/system_preflight.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
