"""Unchanged Round05 bounded-preflight comparison and access helpers."""
import builtins,contextlib,io,math,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

@contextlib.contextmanager
def forbid_external_data(allowed):
    allowed=Path(allowed).resolve();opened=[]
    def guard(original):
        def call(path,*args,**kwargs):
            if isinstance(path,(str,os.PathLike)) and Path(path).suffix.lower() in ('.pt','.npz','.npy','.json'):
                assert Path(path).resolve()==allowed,('External fixture load',str(path));opened.append(str(path))
            return original(path,*args,**kwargs)
        return call
    with patch('builtins.open',guard(builtins.open)),patch('io.open',guard(io.open)):
        yield
    assert opened

def norm(grads):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))

def compare(left,right,report):
    if isinstance(left,torch.Tensor):
        assert isinstance(right,torch.Tensor) and left.shape==right.shape and left.dtype==right.dtype
        a=left.detach().cpu();b=right.detach().cpu()
        if a.is_floating_point():
            assert torch.allclose(a,b,atol=2e-6,rtol=1e-5)
            if a.numel():report.append(float((a-b).abs().max()))
        else:assert torch.equal(a,b)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:compare(left[key],right[key],report)
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):compare(a,b,report)
    else:assert left==right

def exact(left,right):
    if isinstance(left,torch.Tensor):assert left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu())
    elif isinstance(left,np.ndarray):assert left.dtype==right.dtype and np.array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:exact(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert type(left)==type(right) and len(left)==len(right)
        for a,b in zip(left,right):exact(a,b)
    else:assert left==right

