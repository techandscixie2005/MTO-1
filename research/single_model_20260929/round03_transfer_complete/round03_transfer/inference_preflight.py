"""CPU synthetic one-checkpoint export and affine failure checks, no fit."""
import builtins
import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from clean_source import ROOT,SOURCE,build_random
from predictor import export_record,load_predictor,AffineMTO
from evaluation import affine_fit
from runtime import sha,atomic_json

def main():
    torch.set_num_threads(2);sc=json.loads((ROOT/'source_config.json').read_text());stats=json.loads((ROOT/'fit_normalization.json').read_text())
    base=build_random(sc,stats).requires_grad_(False).eval()
    alpha=.7;beta=.002;record=export_record(base,sc,stats,alpha,beta,{'synthetic_preflight':True})
    direct=AffineMTO(sc,stats,alpha,beta);direct.load_state_dict(record['model'],strict=True);direct.requires_grad_(False).eval()
    z=torch.tensor([6,1,1,1,1,8,1,1]);pos=torch.tensor([[0.,0.,0.],[.6,.6,.6],[.6,-.6,-.6],[-.6,.6,-.6],[-.6,-.6,.6],[0.,0.,0.],[.8,.6,0.],[-.8,.6,0.]])
    batch=torch.tensor([0,0,0,0,0,1,1,1]);edge=torch.tensor([(i,j) for i in range(8) for j in range(8) if i!=j and batch[i]==batch[j]]).T
    geometry=dict(z=z,pos=pos,batch=batch,n=2,edge_index=edge)
    with torch.no_grad():expected=direct(**geometry)
    blob=io.BytesIO();torch.save(record,blob);blob.seek(0);opened=[]
    old_open=builtins.open;old_io_open=io.open
    def guard(function):
        def wrapped(file,*args,**kwargs):
            if isinstance(file,(str,bytes,Path)):
                path=str(Path(file).resolve());opened.append(path)
                assert '/data/' not in path and '/runs/' not in path and '/cache/' not in path
                assert not path.endswith(('.pt','.npz','.npy','.json'))
            return function(file,*args,**kwargs)
        return wrapped
    with patch('builtins.open',guard(old_open)),patch('io.open',guard(old_io_open)):
        loaded=load_predictor(blob,'cpu')
        with torch.no_grad():actual=loaded(**geometry)
    assert all(torch.equal(expected[key],actual[key]) for key in expected)
    assert torch.equal(actual['f'],torch.clamp_min(alpha*actual['native_f']+beta,0))
    y=np.array([0.,.2,.4,.8]);x=(y-.1)/2;mask=np.ones(4,dtype=bool)
    fit=affine_fit(y,x,mask);assert np.allclose([fit['alpha'],fit['beta']],[2.,.1],atol=1e-14,rtol=0)
    rejected=[]
    for bad in (np.ones(4),np.array([0.,1.,np.nan,2.])):
        try:affine_fit(y,bad,mask)
        except (AssertionError,ValueError):rejected.append(True)
    assert len(rejected)==2
    names=('inference_preflight.py','predictor.py','evaluation.py','clean_source.py','source_config.json','fit_normalization.json')
    result={'passed':True,'fixture':'two synthetic geometries and one checkpoint in memory',
        'E_A_native_f_affine_f_bitwise_equal':True,'one_uncomposed_map_on_native_f':True,
        'original_checkpoint_data_or_cache_access_during_inference':False,'opened_paths':opened,
        'zero_and_nonfinite_denominator_failures_rejected':True,'OLS_known_fixture_exact':True,
        'source_hashes':{str(ROOT/n):sha(ROOT/n) for n in names}}
    atomic_json(result,ROOT/'INFERENCE_PREFLIGHT.json');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
