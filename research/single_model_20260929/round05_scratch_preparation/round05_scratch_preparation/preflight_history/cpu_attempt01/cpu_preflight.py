"""Fresh initialization, algebra and one-file export checks; synthetic geometry only."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('OMP_NUM_THREADS')==os.environ.get('MKL_NUM_THREADS')=='2'
import builtins,contextlib,copy,io,json,tempfile,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from e3nn import o3
from common import ROOT,read,sha,immutable_json
from fresh_model import build_fresh,build_original,base_state,tensor_hash,decorrelation,TYPES,math_module
from runtime import setup,atomic_torch
from predictor import load_predictor,export_payload

def synthetic_geometry(device='cpu'):
    z=torch.tensor([6,1,1,1,1,8,1,1],device=device)
    pos=torch.tensor([[0,0,0],[.63,.63,.63],[-.63,-.63,.63],[-.63,.63,-.63],[.63,-.63,-.63],
        [0,0,0],[.76,.0,.59],[-.76,.0,.59]],dtype=torch.float32,device=device)
    batch=torch.tensor([0,0,0,0,0,1,1,1],device=device)
    edges=[(i,j) for i in range(len(z)) for j in range(len(z)) if i!=j and batch[i]==batch[j]]
    return dict(z=z,pos=pos,batch=batch,n=2,edge_index=torch.tensor(edges,device=device).T.contiguous())

@contextlib.contextmanager
def forbid_external_data(allowed=None):
    opened=[];allowed=None if allowed is None else Path(allowed).resolve()
    real_builtin=builtins.open;real_io=io.open
    def guard(original):
        def call(path,*args,**kwargs):
            if isinstance(path,(str,os.PathLike)):
                p=Path(path).resolve()
                if p.suffix.lower() in ('.pt','.npz','.npy','.json'):
                    assert allowed is not None and p==allowed,('Unexpected data/checkpoint read',str(p))
                    opened.append(str(p))
            return original(path,*args,**kwargs)
        return call
    with patch('builtins.open',guard(real_builtin)),patch('io.open',guard(real_io)):
        yield opened

def main():
    start=time.time();cfg=read(ROOT/'config.json');mc=read(ROOT/'model_config.json');stats=read(ROOT/'TRAIN_STATISTICS.json');setup(cfg)
    assert read(ROOT/'TRAIN_STATISTICS_AUDIT.json')['passed']
    report={'synthetic_geometry_only':True,'real_geometry_or_targets_read':False,'optimizer_updates':0,
        'train_statistics_sha256':sha(ROOT/'TRAIN_STATISTICS.json')}
    # No original checkpoint, old stats or target archive may be opened during construction.
    with forbid_external_data() as opened:
        original=build_original(mc,stats).eval();control=build_fresh(mc,stats,False).eval();adapter=build_fresh(mc,stats,True).eval()
    assert opened==[]
    initial=tensor_hash(original.state_dict());assert tensor_hash(base_state(control))==tensor_hash(base_state(adapter))==initial
    assert tensor_hash(control.state_dict())==tensor_hash(adapter.state_dict())
    report['initial_base_tensor_sha256']=initial;report['initial_full_tensor_sha256']=tensor_hash(control.state_dict())
    report['initialization_external_data_reads']=opened
    x=synthetic_geometry()
    with torch.no_grad():
        a=original(**x);b=control(**x);c=adapter(**x)
    assert all(torch.equal(p,q) and torch.equal(p,r) for p,q,r in zip(a,b,c))
    report['identity_E_A_bitwise_equal']=True
    report['parameters']={'base':sum(p.numel() for p in original.parameters()),'adapter':sum(p.numel() for p in adapter.right_adapter.parameters())}
    assert report['parameters']['adapter']==4016
    active=copy.deepcopy(adapter).double()
    with torch.no_grad():
        for layer in active.right_adapter.mix.values():layer.weight.normal_(std=.025)
    synthetic={t:torch.randn(3,10,16,d,dtype=torch.float64) for t,d in zip(TYPES,(1,3,5))}
    raw={t:torch.cat((v[:,:1],v),1) for t,v in synthetic.items()};valid=torch.ones(3,10,dtype=torch.bool)
    def rotate(m,r):return {t:m[t]@o3.Irrep(t).D_from_matrix(r).T for t in TYPES}
    report['adapter_O3']=[]
    for det in (1,-1):
        r=o3.rand_matrix(dtype=torch.float64)*det;expected=rotate(active.right_adapter(synthetic),r)
        actual=active.right_adapter(rotate(synthetic,r));err=max(float((actual[t]-expected[t]).abs().max()) for t in TYPES)
        ortherr=abs(float(decorrelation(raw,valid)-decorrelation(rotate(raw,r),valid)))
        assert err<1e-10 and ortherr<1e-12
        report['adapter_O3'].append({'determinant':det,'max_abs_error':err,'orth_error':ortherr})
    for count in (0,1,4,10):
        mask=torch.arange(10)[None,:].expand(3,-1)<count;m={t:v.clone().requires_grad_() for t,v in raw.items()}
        with torch.no_grad():
            for v in m.values():v[:,1:][~mask]=float('nan')
        loss=decorrelation(m,mask);assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(v.grad).all() and not v.grad[:,0].any() for v in m.values())
        if count<2:assert float(loss)==0
    zero={t:torch.zeros_like(v,requires_grad=True) for t,v in raw.items()};loss=decorrelation(zero,valid);loss.backward()
    assert float(loss)==0 and all(torch.isfinite(v.grad).all() for v in zero.values())
    report['mask_nan_zero_one_valid_checks']=True
    active=active.float().eval();report['full_model_O3']=[]
    for det in (1,-1):
        r=o3.rand_matrix()*det;xr=dict(x,pos=x['pos']@r.T)
        with torch.no_grad():p=active(**x);pr=active(**xr)
        assert torch.allclose(pr[0],p[0],atol=2e-5,rtol=2e-4)
        assert torch.allclose(pr[1],r@p[1]@r.T,atol=5e-5,rtol=5e-4)
        report['full_model_O3'].append({'determinant':det,'E_max_abs':float((pr[0]-p[0]).abs().max()),
            'A_max_abs':float((pr[1]-r@p[1]@r.T).abs().max()),
            'native_f_max_abs':float(((pr[0].double()*pr[1].double().diagonal(dim1=-2,dim2=-1).sum(-1)
                -p[0].double()*p[1].double().diagonal(dim1=-2,dim2=-1).sum(-1))*2/(3*27.211386245988)).abs().max())})
    permutation=torch.tensor([4,2,0,3,1,7,5,6]);inverse=torch.argsort(permutation)
    transformed=dict(x,z=x['z'][permutation],pos=x['pos'][permutation]+torch.tensor([2.1,-3.7,.9]),
        batch=x['batch'][permutation],edge_index=inverse[x['edge_index']])
    with torch.no_grad():p=active(**x);pr=active(**transformed)
    assert torch.allclose(p[0],pr[0],atol=2e-5,rtol=2e-4) and torch.allclose(p[1],pr[1],atol=5e-5,rtol=5e-4)
    report['permutation_translation']={'E_max_abs':float((p[0]-pr[0]).abs().max()),
        'A_max_abs':float((p[1]-pr[1]).abs().max())}
    # Hand-modified synthetic fixture only. The final predictor opens one file
    # and embeds all base/F tensors, TRAIN stats, config and buffer fingerprints.
    with tempfile.TemporaryDirectory(prefix='round05_export_') as directory:
        path=Path(directory)/'synthetic.pt';atomic_torch(export_payload(active,mc,stats,{'purpose':'synthetic_only'}),path)
        with forbid_external_data(path) as opened:
            loaded=load_predictor(path)
            with torch.no_grad():expected=active(**x);actual=loaded(**x)
        assert opened and set(opened)=={str(path.resolve())}
        assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    report['one_checkpoint_load_and_forward_access_pass']=True;report['one_checkpoint_E_A_bitwise_equal']=True
    report['source_hashes']={n:sha(ROOT/n) for n in ('cpu_preflight.py','fresh_model.py','runtime.py','predictor.py','model_config.json','config.json')}
    report['adapter_math_sha256']=sha(math_module.__file__);report['seconds']=time.time()-start;report['passed']=True
    immutable_json(report,ROOT/'CPU_PREFLIGHT.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
