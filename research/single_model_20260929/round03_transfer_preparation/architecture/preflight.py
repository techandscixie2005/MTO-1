"""CPU architecture audit. Never runs fitting or reads test predictions."""
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from e3nn import o3
from model import (SOURCE, TYPES, MTOEA, build_model, load_baseline,
                   decorrelation, base_loss)
sys.path.insert(0, str(SOURCE))
from dataset import Data

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(20260929)
START = time.time()

def maxdiff(a, b):
    return float((a.detach()-b.detach()).abs().max())

def rotate(m, r):
    return {t: m[t] @ o3.Irrep(t).D_from_matrix(r).T for t in TYPES}

def norm_grad(grads):
    return math.sqrt(sum(float(g.detach().double().square().sum())
                         for g in grads if g is not None))

cfg = json.loads((SOURCE/'configs/mto_eta0.json').read_text())
stats = json.loads((SOURCE/'data/normalization.json').read_text())
ck = torch.load(SOURCE/'runs/mto_eta0/best.pt', map_location='cpu', weights_only=False)
assert ck['epoch'] == 33
report = {'source_checkpoint': str(SOURCE/'runs/mto_eta0/best.pt'),
          'source_checkpoint_sha256': hashlib.sha256((SOURCE/'runs/mto_eta0/best.pt').read_bytes()).hexdigest(),
          'adapter_equivariance': [], 'full_model_equivariance': [], 'gradient_audit': []}
assert report['source_checkpoint_sha256'] == '9f1963267cd0e5e35212baca6b4080cf986e73aed6e1662c4cb720afb9a4c136'
baseline = MTOEA(cfg, stats).eval()
baseline.load_state_dict(ck['model'])
model = load_baseline(build_model(cfg, stats, True), ck['model']).eval()
control = load_baseline(build_model(cfg, stats, False), ck['model']).eval()
data = Data(device='cpu')
x, y = data.batch(data.parts['train'][:2])
with torch.no_grad():
    bp = baseline(**x)
    mp, raw = model(**x, return_aux=True)
    cp = control(**x)
    exact = [torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(bp,mp,cp)]
    report['initial_prediction_bitwise_equal'] = exact
    report['initial_f_max_difference'] = maxdiff(bp[0]*bp[1].diagonal(dim1=-2,dim2=-1).sum(-1),
                                                 mp[0]*mp[1].diagonal(dim1=-2,dim2=-1).sum(-1))
    assert all(exact)

# Nontrivial random adapter weights ensure equivariance is not merely the
# identity map passing by construction.
active = copy.deepcopy(model).double()
with torch.no_grad():
    for layer in active.right_adapter.mix.values():
        layer.weight.normal_(std=.025)
synthetic = {t: torch.randn(3, 10, 16, d, dtype=torch.float64)
             for t,d in zip(TYPES,(1,3,5))}
valid = torch.ones(3,10,dtype=torch.bool)
raw_synthetic = {t: torch.cat((v[:,:1],v),1) for t,v in synthetic.items()}
for determinant in (1,-1):
    r = o3.rand_matrix(dtype=torch.float64) * determinant
    out = active.right_adapter(synthetic)
    changed = active.right_adapter(rotate(synthetic,r))
    err = max(maxdiff(changed[t],rotate(out,r)[t]) for t in TYPES)
    assert err < 1e-10
    d1 = decorrelation(raw_synthetic,valid)
    d2 = decorrelation(rotate(raw_synthetic,r),valid)
    assert abs(float(d1-d2)) < 1e-12
    report['adapter_equivariance'].append({'determinant':determinant,'max_abs_error':err,
                                          'decorrelation_abs_error':abs(float(d1-d2))})

# Full-model geometry transformations include a reflection and use identical
# connectivity. FP32 backbone scatter/e3nn arithmetic gets a realistic tolerance.
active = active.float()
for determinant in (1,-1):
    r = o3.rand_matrix() * determinant
    xr = dict(x)
    xr['pos'] = x['pos'] @ r.T
    with torch.no_grad():
        p, m = active(**x,return_aux=True)
        pr, mr = active(**xr,return_aux=True)
        e_error = maxdiff(p[0],pr[0])
        a_error = maxdiff(pr[1],r @ p[1] @ r.T)
        f_error = maxdiff(p[0]*p[1].diagonal(dim1=-2,dim2=-1).sum(-1),
                          pr[0]*pr[1].diagonal(dim1=-2,dim2=-1).sum(-1))
        assert torch.allclose(pr[0],p[0],rtol=2e-4,atol=2e-5)
        assert torch.allclose(pr[1],r@p[1]@r.T,rtol=5e-4,atol=5e-5)
        report['full_model_equivariance'].append({'determinant':determinant,
            'energy_max_abs_error':e_error,'A_max_abs_error':a_error,
            'energy_times_trace_max_abs_error':f_error})

# Masked NaNs, zero/one valid states, zero vectors, phase flips, and scale
# invariance above epsilon. This is not physical phase alignment of TD labels.
for count in (0,1,4,10):
    mask = torch.arange(10)[None,:].expand(3,-1) < count
    m = {t:v.clone().requires_grad_() for t,v in raw_synthetic.items()}
    with torch.no_grad():
        for v in m.values():
            v[:,1:][~mask] = float('nan')
    loss = decorrelation(m,mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in m.values())
    assert all(torch.equal(v.grad[:,0],torch.zeros_like(v.grad[:,0])) for v in m.values())
    if count<2:
        assert float(loss)==0
zero = {t:torch.zeros_like(v,requires_grad=True) for t,v in raw_synthetic.items()}
zloss = decorrelation(zero,valid)
zloss.backward()
assert float(zloss)==0 and all(torch.isfinite(v.grad).all() for v in zero.values())
sign = torch.randint(0,2,(3,11,1,1),dtype=torch.float64)*2-1
assert abs(float(decorrelation(raw_synthetic,valid)-decorrelation(
    {t:v*sign for t,v in raw_synthetic.items()},valid)))<1e-12
assert abs(float(decorrelation(raw_synthetic,valid)-decorrelation(
    {t:v*7.1 for t,v in raw_synthetic.items()},valid)))<1e-12
report['mask_zero_phase_scale_checks_passed'] = True

# Train-only fixed random sample audit; never use validation to choose lambda.
idx = np.random.default_rng(20260929).choice(data.parts['train'],32,replace=False)
report['gradient_audit_indices_sha256'] = hashlib.sha256(idx.tobytes()).hexdigest()
for start in range(0,len(idx),8):
    x,y = data.batch(idx[start:start+8])
    model.zero_grad(set_to_none=True)
    pred, m = model(**x,return_aux=True)
    base = base_loss(pred,y,stats)['total']
    mask = y['mask_E'] & y['mask_A'] & y['mask_f']
    orth = decorrelation(m,mask)
    params = [p for p in model.parameters() if p.requires_grad]
    bg = torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
    og = torch.autograd.grad(orth,params,retain_graph=True,allow_unused=True)
    assert all(torch.isfinite(g).all() for g in bg+og if g is not None)
    ratio = norm_grad(og)/max(norm_grad(bg),1e-30)
    raw_norm = torch.cat([m[t][:,1:].flatten(-2)/math.sqrt(d)
                           for t,d in zip(TYPES,(1,3,5))],-1).norm(dim=-1)
    jac = torch.autograd.grad(base,list(model.right_adapter.mix.parameters()),retain_graph=True)
    assert norm_grad(jac)>0
    (base+1e-3*orth).backward()
    report['gradient_audit'].append({'batch':start//8,'base_loss':float(base),'raw_decorrelation':float(orth),
        'base_gradient_l2':norm_grad(bg),'decorrelation_gradient_l2':norm_grad(og),
        'unweighted_gradient_ratio':ratio,'lambda_1e_minus3_gradient_ratio':1e-3*ratio,
        'identity_adapter_mixing_gradient_l2':norm_grad(jac),
        'raw_state_norm_min':float(raw_norm.min()),'raw_state_norm_max':float(raw_norm.max())})

# Exact CPU optimizer resume next update, with enabled adapter and regularizer.
x,y=data.batch(data.parts['train'][:2])
one=copy.deepcopy(model)
opt=torch.optim.Adam([p for p in one.parameters() if p.requires_grad],lr=1e-5,amsgrad=True)
def step(m,opt):
    opt.zero_grad(set_to_none=True)
    p,raw=m(**x,return_aux=True)
    loss=base_loss(p,y,stats)['total']+1e-3*decorrelation(raw,y['mask_E']&y['mask_A']&y['mask_f'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),5)
    opt.step()
step(one,opt)
two=copy.deepcopy(one)
opt2=torch.optim.Adam([p for p in two.parameters() if p.requires_grad],lr=1e-5,amsgrad=True)
opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
step(one,opt);step(two,opt2)
error=max(maxdiff(v,two.state_dict()[k]) for k,v in one.state_dict().items() if v.numel())
assert error==0
report['resume_next_update_max_abs_difference']=error
report['parameters']={'base':sum(p.numel() for p in baseline.parameters()),
    'adapter_extra':sum(p.numel() for p in model.right_adapter.parameters()),
    'adapter_enabled_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
    'control_trainable':sum(p.numel() for p in control.parameters() if p.requires_grad)}
report['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (ROOT/'model.py',ROOT/'preflight.py')}
report['duration_seconds']=time.time()-START
report['passed']=True
(ROOT/'PREFLIGHT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
