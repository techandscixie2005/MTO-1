"""Discarded fit-only source smoke, original loss parity, and serialized resume."""
import argparse
import builtins
import copy
import fcntl
import io
import json
import os
import sys
from unittest.mock import patch
import numpy as np
import torch
import clean_source as cs
from clean_source import ROOT,SOURCE,FitData,build_random,source_loss,tensor_state_hash
from runtime import setup,optimizer,rng_state,restore_rng,sha,atomic_json
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT));sys.path.insert(0,str(SOURCE))
from launch import admit,EXPECTED
from objective import losses as original_losses

def compare(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype
        a=a.detach().cpu();b=b.detach().cpu()
        if a.is_floating_point():
            assert torch.allclose(a,b,atol=2e-6,rtol=1e-5)
            return float((a-b).abs().max()) if a.numel() else 0.
        assert torch.equal(a,b);return 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys();return max([compare(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b);return max([compare(x,y) for x,y in zip(a,b)]+[0.])
    assert a==b;return 0.

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=1);args=parser.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text());sc=json.loads((ROOT/'source_config.json').read_text())
    assert sc==json.loads((SOURCE/'configs/mto_eta0.json').read_text())
    split=json.loads((ROOT/'split_manifest.json').read_text());audit=json.loads((ROOT/'STATISTICS_AUDIT.json').read_text())
    assert split['passed'] and audit['passed']
    for path,value in audit['source_hashes'].items():assert sha(path)==value
    for key,value in split['arrays'].items():assert sha(value['path'])==value['sha256']
    lock=open('/tmp/mto_pouter_gpu_%d.lock'%args.gpu,'w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission=admit(args.gpu);(ROOT/'PREFLIGHT_GPU_ADMISSION.xml').write_text(admission)
    os.environ['CUDA_VISIBLE_DEVICES']=EXPECTED[args.gpu];setup(cfg)
    seen=[];reader=cs.selected_rows
    def selected(archive,name,rows):
        assert np.array_equal(rows,np.sort(np.load(ROOT/'data/fit_indices.npy')))
        assert name in ('ids','z','pos','edge','E','A','mask_E','mask_A')
        seen.append({'archive':str(archive.filename),'member':name,'rows':len(rows)})
        return reader(archive,name,rows)
    with patch.object(cs,'selected_rows',selected):data=FitData('cuda')
    opened=[];old_open=builtins.open;old_io_open=io.open
    def guard(function):
        def wrapped(file,*args,**kwargs):
            if isinstance(file,(str,bytes,os.PathLike)):
                name=str(file);opened.append(name)
                assert not name.endswith(('.pt','.pth','.npz','.npy')) and 'normalization.json' not in name
            return function(file,*args,**kwargs)
        return wrapped
    with patch('builtins.open',guard(old_open)),patch('io.open',guard(old_io_open)),patch('torch.load',side_effect=AssertionError('Initialization must not load tensors')):
        model=build_random(sc,data.stats).cuda()
    initial=tensor_state_hash(model.state_dict());repeated=build_random(sc,data.stats).cuda()
    assert initial==tensor_state_hash(repeated.state_dict());del repeated
    expected_offset=torch.tensor(data.stats['E_state_mean'],device='cuda',dtype=torch.float32)
    assert torch.allclose(torch.nn.functional.softplus(model.decoder.energy_offset),expected_offset,atol=1e-6,rtol=1e-6)
    x,y=data.batch(np.arange(cfg['batch_size']));pred=model(**x)
    ours=source_loss(pred,y,data.stats);original=original_losses(pred,y,data.stats,0)[:3]
    assert all(torch.equal(a,b) for a,b in zip(ours,original))
    params=list(model.parameters())
    ga=torch.autograd.grad(ours[0],params,retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(original[0],params,allow_unused=True)
    assert all(a is None and b is None or a is not None and b is not None and torch.equal(a,b) for a,b in zip(ga,gb))
    del pred,ours,original,ga,gb
    opt=optimizer(model,cfg);setup(cfg);generator=np.random.default_rng(cfg['order_seed'])
    def update(net,optim,rows):
        net.train();optim.zero_grad(set_to_none=True);bx,by=data.batch(rows)
        values=source_loss(net(**bx),by,data.stats);values[0].backward()
        norm=float(torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['grad_clip'],error_if_nonfinite=True));optim.step()
        return [float(v) for v in values],norm
    order=generator.permutation(len(data.global_indices));first=order[:64];second=order[64:128]
    values,norm=update(model,opt,first)
    blob=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(generator)},blob)
    uninterrupted=update(model,opt,second);final_model=copy.deepcopy(model.state_dict());final_opt=copy.deepcopy(opt.state_dict())
    expected_next=generator.permutation(len(data.global_indices))
    blob.seek(0);record=torch.load(blob,map_location='cpu',weights_only=False)
    resumed=build_random(sc,data.stats).cuda();resumed.load_state_dict(record['model'],strict=True)
    resumed_opt=optimizer(resumed,cfg);resumed_opt.load_state_dict(record['optimizer']);restore_rng(record['rng'],generator)
    replayed=update(resumed,resumed_opt,second)
    model_error=compare(final_model,resumed.state_dict());optimizer_error=compare(final_opt,resumed_opt.state_dict())
    assert np.array_equal(expected_next,generator.permutation(len(data.global_indices)))
    assert np.allclose(uninterrupted[0],replayed[0],atol=1e-6,rtol=1e-5)
    names=('clean_source.py','runtime.py','preflight.py','train_source.py','config.json','source_config.json','fit_normalization.json','split_manifest.json')
    result={'passed':True,'device':'cuda','gpu':args.gpu,'initial_model_tensor_sha256':initial,
        'fresh_random_initialization_no_checkpoint_or_full_statistics_read':True,'initialization_paths_opened':opened,
        'fit_data_access':data.decode_audit,'actual_decoded_members':seen,'original_objective_and_gradients_bitwise_equal':True,
        'first_discarded_step_losses':values,'first_preclip_gradient_norm':norm,
        'resume_next_model_max_abs':model_error,'resume_next_optimizer_max_abs':optimizer_error,
        'resume_order_exact':True,'resume_tolerance':{'atol':2e-6,'rtol':1e-5},
        'discarded_updates':True,'source_fit_started':False,'outer_or_calibration_targets_decoded':False,
        'source_hashes':{str(ROOT/name):sha(ROOT/name) for name in names}}
    atomic_json(result,ROOT/'PREFLIGHT.json');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
