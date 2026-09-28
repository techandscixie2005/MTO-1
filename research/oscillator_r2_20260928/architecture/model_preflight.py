#!/usr/bin/env python3
"""CPU model/gradient/invariance checks; no validation/test inference."""
import hashlib,json,time
import numpy as np
import torch
from model import ROOT,SRC,build,C_F
from objective import terms,native_f64
from dataset import Data
cfg=json.loads((ROOT/'config.json').read_text());stats=json.loads((ROOT/'stats.json').read_text())
torch.set_num_threads(cfg['num_threads']);data=Data(device='cpu')
x,y=data.batch(data.parts['train'][:2]);source=torch.load(SRC/'runs/mto_eta0/best.pt',map_location='cpu',weights_only=False)['model']
torch.manual_seed(20260928);R=torch.linalg.qr(torch.randn(3,3)).Q
perm=torch.randperm(len(x['z']));inverse=torch.argsort(perm)
rot=dict(x,pos=x['pos']@R.T)
relabel=dict(x,z=x['z'][perm],pos=x['pos'][perm],batch=x['batch'][perm],edge_index=inverse[x['edge_index']])
results={};base_f=None
for arm in cfg['arms']:
    m=build(arm,stats).eval()
    for k,v in m.base.state_dict().items():assert torch.equal(v,source[k]),k
    pred=m(**x);f=native_f64(pred,arm)
    assert torch.isfinite(f).all() and (f>=0).all()
    if arm=='original':
        base_f=f.detach().clone()
        with torch.no_grad():e0,a0=m.base(**x)
        assert torch.allclose(pred['E'],e0,atol=1e-7,rtol=1e-6)
        assert torch.allclose(pred['A'],a0,atol=1e-7,rtol=1e-6)
    if arm=='retained_residual':assert torch.equal(f.detach(),base_f)
    ep=list(m.base.decoder.energy_head.parameters())
    g=torch.autograd.grad(pred['f'].sum(),ep,allow_unused=True,retain_graph=True)
    gsum=sum(float(v.abs().sum()) for v in g if v is not None)
    assert (gsum==0) if arm=='direct_f' else (gsum>0)
    loss=terms(pred,y,stats)['total'];loss.backward()
    grads=[v.grad for v in m.parameters() if v.grad is not None]
    assert grads and all(torch.isfinite(v).all() for v in grads)
    with torch.no_grad():
        rf=native_f64(m(**rot),arm);pf=native_f64(m(**relabel),arm)
    rerr=float((rf-f.detach()).abs().max());perr=float((pf-f.detach()).abs().max())
    assert torch.allclose(rf,f.detach(),atol=2e-5,rtol=2e-4),(arm,rerr)
    assert torch.allclose(pf,f.detach(),atol=2e-5,rtol=2e-4),(arm,perr)
    row=dict(parameter_counts=m.parameter_counts(),loss=float(loss),f_to_energy_head_gradient_abs_sum=gsum,rotation_max_f_error=rerr,permutation_max_f_error=perr,min_f=float(f.min()),exact_zero_f_count=int((f==0).sum()),shared_pretrained_weights_exact=True)
    if pred['A'] is not None:
        eig=torch.linalg.eigvalsh(pred['A'].detach())
        assert eig.min()>-1e-6
        trace=pred['A'].double().diagonal(dim1=-2,dim2=-1).sum(-1)
        consistency=float((C_F*pred['E'].double()*trace-f).abs().max())
        assert consistency<2e-6
        row.update(min_A_eigenvalue=float(eig.min()),f_tensor_consistency_max_abs=consistency)
    if arm=='independent_trace':
        assert torch.allclose(pred['shape'].diagonal(dim1=-2,dim2=-1).sum(-1),torch.ones_like(pred['E']),atol=1e-6)
    if arm=='retained_residual':
        assert pred['tensor_reconstruction_valid'].all()
        row['exact_original_initialization']=True
    results[arm]=row
# Piecewise absolute-value correction retains nonzero slopes for both signs near zero.
z=torch.tensor([-1e-12,1e-12],dtype=torch.float64,requires_grad=True);z.abs().sum().backward();assert torch.equal(z.grad,torch.tensor([-1.,1.],dtype=torch.float64))
report=dict(passed=True,time=time.time(),scope='CPU two training molecules only; no validation/test evaluation',arms=results,residual_both_signs_weak_gradient_checked=True,source_hashes={str(ROOT/n):hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['model.py','objective.py','config.json','prepare.py','PREPARATION.json']})
(ROOT/'MODEL_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
