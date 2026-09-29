"""Discarded TRAIN-only cache/output/gradient/Adam tests plus identity validation."""
import fcntl
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from freeze import sha
from launch import admit,EXPECTED

def main():
    cfg=json.loads((ROOT/'config.json').read_text());pc=json.loads((PARENT/'round_config.json').read_text())
    lock=open('/tmp/mto_pouter_gpu_2.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    xml=admit(2);(ROOT/'PREFLIGHT_GPU_ADMISSION.xml').write_text(xml)
    os.environ['CUDA_VISIBLE_DEVICES']=EXPECTED[2]
    import torch
    from model_frozen import SOURCE,TYPES,build,frozen_digests,assert_optimizer,losses,AdapterMTOEA
    from cache_access import load_cache,targets
    from train import setup
    from dataset import Data
    from metrics import C_F,validate
    setup(cfg);data=Data('cuda');cache,indices,cr=load_cache(data)
    source_cfg=json.loads((SOURCE/'configs/mto_eta0.json').read_text())
    original=torch.load(pc['source_checkpoint'],map_location='cpu',weights_only=False)
    models=[build(source_cfg,data.stats,original['model'],cfg['adapter_seed']).cuda() for _ in range(2)]
    baseline_hash=frozen_digests(models[0]);assert frozen_digests(models[1])==baseline_hash
    assert baseline_hash==cr['frozen_parameter_buffer_hashes']
    chosen=np.random.default_rng(20260930).choice(indices,256,replace=False)
    inverse={int(v):i for i,v in enumerate(indices)}
    rows=np.array([inverse[int(v)] for v in chosen]);report=[];gradient_audit=[]
    tol=cfg['cache_parity']
    def compare(x,y,kind):
        aa=tol['gradient_atol'] if kind=='gradient' else tol['output_atol']
        rr=tol['gradient_rtol'] if kind=='gradient' else tol['output_rtol']
        assert torch.allclose(x,y,atol=aa,rtol=rr),(kind,float((x-y).abs().max()))
        return float((x-y).abs().max())
    for phase in ('identity','nonzero'):
        if phase=='nonzero':
            g=torch.Generator(device='cuda').manual_seed(20260930)
            with torch.no_grad():
                for t in TYPES:
                    w=models[0].right_adapter.mix[t].weight
                    w.copy_(.01*torch.randn(w.shape,generator=g,device='cuda'))
            models[1].right_adapter.load_state_dict(models[0].right_adapter.state_dict())
        for i in range(0,256,32):
            idx=chosen[i:i+32];row=torch.as_tensor(rows[i:i+32],device='cuda')
            x,y=data.batch(idx);m={t:a[row] for t,a in cache.items()}
            full=models[0](**x);cached=models[1].from_raw(m)
            with torch.no_grad():
                legacy=AdapterMTOEA.forward(models[0],**x)
                legacy_errors=[compare(a,b,'output') for a,b in zip(full,legacy)]
            output_errors=[compare(a,b,'output') for a,b in zip(full,cached)]
            f=[C_F*p[0]*p[1].diagonal(dim1=-2,dim2=-1).sum(-1) for p in (full,cached)]
            output_errors.append(compare(*f,'output'))
            for arm in cfg['arms']:
                grads=[];details=[]
                for model,pred in zip(models,(full,cached)):
                    ls=losses(pred,y,data.stats,cfg['train_f_variance'],arm)
                    gg=torch.autograd.grad(ls['total'],tuple(model.right_adapter.parameters()),retain_graph=True)
                    grads.append(torch.cat([z.flatten() for z in gg]));details.append(ls)
                grad_error=compare(*grads,'gradient')
                relative=float((grads[0]-grads[1]).norm()/grads[0].norm().clamp_min(1e-12))
                assert relative<tol['gradient_rtol']
                report.append({'phase':phase,'batch':i//32,'arm':arm,'E_A_f_max_abs':output_errors,
                    'legacy_E_A_max_abs':legacy_errors,'gradient_max_abs':grad_error,'gradient_relative_l2':relative})
                if phase=='identity':
                    gradient_audit.append({'batch':i//32,'arm':arm,'total_gradient_norm':float(grads[0].norm()),
                        **{k:float(details[0][k]) for k in ('energy','trace','raw_f','total')}})
    # Compare two next updates with nonempty Adam state, keeping both paths aligned.
    update_report=[]
    for arm in cfg['arms']:
        pair=[build(source_cfg,data.stats,original['model'],cfg['adapter_seed']).cuda() for _ in range(2)]
        opts=[torch.optim.Adam(m.right_adapter.parameters(),lr=cfg['lr'],amsgrad=True,weight_decay=0) for m in pair]
        for m,o in zip(pair,opts):assert_optimizer(m,o)
        for step in (0,1):
            row=torch.as_tensor(rows[step*32:(step+1)*32],device='cuda');idx=chosen[step*32:(step+1)*32]
            x,y=data.batch(idx);m={t:a[row] for t,a in cache.items()}
            for k,(model,opt) in enumerate(zip(pair,opts)):
                opt.zero_grad(set_to_none=True)
                pred=model(**x) if k==0 else model.from_raw(m)
                losses(pred,y,data.stats,cfg['train_f_variance'],arm)['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.right_adapter.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
                opt.step()
            delta=max(float((a-b).abs().max()) for a,b in zip(pair[0].right_adapter.parameters(),pair[1].right_adapter.parameters()))
            assert delta<=tol['next_update_atol'],('Adam update',arm,step,delta)
            update_report.append({'arm':arm,'step':step+1,'maximum_parameter_difference':delta})
        for model in pair:
            assert frozen_digests(model)==baseline_hash
            assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('right_adapter.'))
        del pair,opts
    for model in models:assert frozen_digests(model)==baseline_hash
    anchor=build(source_cfg,data.stats,original['model'],cfg['adapter_seed']).cuda()
    with np.load(SOURCE/'data/raw_labels.npz') as z:
        raw={k:z[k].copy() for k in ('E','f','mask_E','mask_f')}
    pf=json.loads((PARENT/'FROZEN_MANIFEST.json').read_text())
    anchor_metrics,_=validate(anchor,data,raw,cfg['batch_size'],pf['tail_thresholds'])
    assert abs(anchor_metrics['raw_f']['r2']-cfg['epoch_zero_anchor_r2'])<cfg['epoch_zero_anchor_r2_atol']
    assert frozen_digests(anchor)==baseline_hash
    result={'passed':True,'source_hashes':{str(ROOT/n):sha(ROOT/n) for n in ('model_frozen.py','cache_access.py','preflight.py','config.json')},
        'cache_receipt_sha256':sha(ROOT/'CACHE_COMPLETE.json'),'train_subset_indices_sha256':__import__('hashlib').sha256(chosen.tobytes()).hexdigest(),
        'train_count':256,'output_gradient_checks':report,'discarded_adam_checks':update_report,
        'fixed_objective_gradient_audit':gradient_audit,'gradient_scaling_fitted':False,
        'anchor_validation':anchor_metrics,'source_weights_and_all_buffers_unchanged':True,
        'exact_train_f_variance':cr['train_f_population_variance'],'GPU':EXPECTED[2],
        'phase_interpretation':'O3 equivariance is inherited; no guarantee of electronic phase-odd covariance',
        'production_weights_from_preflight':False,'test_inference':False}
    (ROOT/'PREFLIGHT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'anchor_r2':anchor_metrics['raw_f']['r2'],'gradient_scaling_fitted':False}))

if __name__=='__main__':main()
