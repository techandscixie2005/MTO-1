"""Allocated-GPU gate: real-data overfit plus optimizer/RNG resume checks.

Only 32 fixed historical TRAIN molecules are used. No validation/test tuning.
Failures stop the entire chain before formal 1k training.
"""
import copy,json,subprocess,sys,time
from pathlib import Path
import torch
from models import build_pair
from dataset import collate
from scale_dataset import load_scale
from protocol import verify

ROOT=Path(__file__).resolve().parents[1]

def clone_state(model):
    return {k:v.detach().clone() for k,v in model.state_dict().items()}

def validate_restored_state(actual, expected):
    assert actual.keys()==expected.keys()
    # e3nn legitimately registers empty buffers. Compare their shapes/values,
    # but do not ask max() to reduce a tensor with no elements.
    for key,value in actual.items():
        torch.testing.assert_close(value,expected[key],atol=2e-6,rtol=2e-6)
    return max((float((value-expected[key]).abs().max())
                for key,value in actual.items() if value.is_floating_point() and value.numel()>0),default=0.)

def main():
    frozen=verify()
    out=ROOT/'results/preflight.json'
    if out.exists():
        saved=json.loads(out.read_text())
        assert saved['passed'] and saved['protocol_id']==frozen['protocol_id']
        return
    assert json.loads((ROOT/'results/data_checks.json').read_text())['passed']
    subprocess.run([sys.executable,str(ROOT/'tests/test_models.py')],check=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    cfg=json.loads((ROOT/'configs/grid.json').read_text())['configs'][0]
    graphs,manifest=load_scale(ROOT,'1k',splits=('train',),limit=32)
    x,y=collate(graphs,'cuda',manifest['train_spectrum_rms'])
    pair,counts=build_pair(cfg,manifest['train_median_atoms'],spectrum_rms=manifest['train_spectrum_rms'])
    report={'passed':False,'protocol_id':frozen['protocol_id'],'gpu':torch.cuda.get_device_name(),
            'train_ids':[g['id'] for g in graphs],'parameters':counts,'arms':{}}
    for arm,model in pair.items():
        model.cuda().train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        with torch.no_grad():initial=float((model(**x)-y).square().mean())
        started=time.time();history=[];best=float('inf')
        for step in range(600):
            pred=model(**x);loss=(pred-y).square().mean()
            assert bool(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            if (step+1)%25==0:
                with torch.no_grad():value=float((model(**x)-y).square().mean())
                best=min(best,value);history.append({'step':step+1,'mse':value})
                print(arm,history[-1],flush=True)
                if step>=99 and best<.08 and best<initial*.25:break
        # Check that reloading model and AdamW state reproduces the next GPU step.
        before=clone_state(model);opt_before=copy.deepcopy(optimizer.state_dict())
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        def advance():
            optimizer.zero_grad(set_to_none=True)
            loss=(model(**x)-y).square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
        advance();expected=clone_state(model)
        model.load_state_dict(before);optimizer.load_state_dict(opt_before)
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);advance()
        max_error=validate_restored_state(model.state_dict(),expected)
        result={'initial_mse':initial,'best_mse':best,'steps':step+1,'history':history,
                'elapsed_seconds':time.time()-started,'resume_max_abs_error':max_error,
                'overfit_passed':best<.1 and best<initial*.3}
        report['arms'][arm]=result
        (ROOT/'results/preflight_progress.json').write_text(json.dumps(report,indent=2))
        assert result['overfit_passed'],f'{arm}: 32-molecule overfit gate failed: {result}'
        model.cpu();del optimizer,expected,before,opt_before;torch.cuda.empty_cache()
    # Exercise the actual epoch checkpoint format and resume path as well.
    smoke=[sys.executable,'-u',str(ROOT/'scripts/train_scale.py'),'--scale','1k','--seed','11','--smoke']
    smoke_folder=ROOT/'smoke/1k_seed11/resume_check'
    if not (smoke_folder/'baseline_last.pt').exists():
        interrupted=subprocess.run(smoke+['--tag','resume_check','--stop-after-epoch','1'])
        assert interrupted.returncode==75,interrupted.returncode
    subprocess.run(smoke+['--tag','resume_check'],check=True)
    for arm in ('baseline','mto'):
        folder=ROOT/'smoke/1k_seed11/resume_check'
        ck=torch.load(folder/f'{arm}_last.pt',map_location='cpu',weights_only=False)
        summary=json.loads((folder/f'{arm}_summary.json').read_text())
        assert ck['epoch']==3 and ck['steps']==6 and summary['epochs_run']==3
        assert [row['epoch'] for row in ck['history']]==[1,2,3]
    assert (folder/'DONE').exists()
    report['trainer_epoch_resume_passed']=True
    report['passed']=True
    out.write_text(json.dumps(report,indent=2))
    print('PREFLIGHT PASSED',flush=True)

if __name__=='__main__':main()
