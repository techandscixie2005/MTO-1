"""Export fixed TRAIN molecules at initialization, explicitly not trained results."""
import json
import math
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import build_triplet,VARIANTS
from data_protocol import load,collate
from train import atomic_save,atomic_json

torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
data,info=load();cfg=json.loads((ROOT/'protocol.json').read_text())
models=build_triplet(cfg,info['n_ref'],info['grid'],11)
out=ROOT/'reports/initial_assembly';out.mkdir(exist_ok=True)
report={}
for label in VARIANTS[1:]:
    model=models[label].eval()
    for mid in info['oracle_ids']:
        rec=next(r for r in data['train'] if r['id']==mid)
        x,y=collate([rec])
        with torch.no_grad():s,e=model(**x,export=True)
        atomic_save(dict(status='INITIALIZATION_ONLY_NOT_A_TRAINED_RESULT',seed=11,variant=label,
                         id=mid,split='train',inputs=x,spectrum=s,target=y,**e),out/f'{label}_{mid}.pt')
        Q,A,f,E=e['Q'],e['A'],e['f'],e['E']
        sigma=cfg['sigma_eV'];grid=model.decoder.energy_grid
        cdf=lambda edge:.5*(1+torch.erf((edge-E)/(math.sqrt(2)*sigma)))
        expected=(f*(cdf(grid[-1])-cdf(grid[0]))).sum()
        numeric=torch.trapz(s,grid).sum()
        report[f'{label}/{mid}']=dict(status='INITIALIZATION_ONLY',
            shapes={k:({t:list(v.shape) for t,v in val.items()} if isinstance(val,dict) else list(val.shape)) for k,val in e.items()},
            minimum_eigenvalue_A=float(torch.linalg.eigvalsh(A).min()),
            maximum_abs_Q_trace=float(Q.diagonal(dim1=-2,dim2=-1).sum(-1).abs().max()),
            sum_f=float(f.sum()),cdf_window_area=float(expected),numeric_240_window_area=float(numeric),
            gate_saturation=float((e['c'].abs()>.99).float().mean()),
            energy_range=[float(E.min()),float(E.max())])
atomic_json(report,out/'manifest.json')
print(json.dumps(dict(exports=len(report),status='INITIALIZATION_ONLY',training_ids=info['oracle_ids'])))
