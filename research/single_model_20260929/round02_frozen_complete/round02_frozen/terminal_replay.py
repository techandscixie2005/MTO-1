"""Bounded terminal geometry replay for selected and fixed20 error bins."""
import builtins
import fcntl
import io
import json
import os
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from launch import EXPECTED,admit
from freeze import sha

def main():
    cfg=json.loads((ROOT/'config.json').read_text())
    assert all((ROOT/'runs'/a/'FIT_COMPLETE.json').is_file() for a in cfg['arms'])
    mf=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text());mh=sha(ROOT/'FROZEN_MANIFEST.json')
    for path,expected in mf['source_hashes'].items():assert sha(path)==expected
    for arm in cfg['arms']:
        rd=ROOT/'runs'/arm;receipt=json.loads((rd/'FIT_COMPLETE.json').read_text())
        assert receipt['completed_epoch']==20 and receipt['manifest_sha256']==mh
        assert sha(rd/'best.pt')==receipt['best_checkpoint_sha256']
        assert sha(rd/'last.pt')==receipt['last_checkpoint_sha256']
    lock=open('/tmp/mto_pouter_gpu_1.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'TERMINAL_REPLAY_GPU_ADMISSION.xml').write_text(admit(1))
    os.environ['CUDA_VISIBLE_DEVICES']=EXPECTED[1]
    import torch
    from model_frozen import SOURCE,FrozenAdapterMTO,frozen_digests
    from predictor import load_predictor
    from train import setup
    from dataset import Data
    from metrics import validate
    from summarize_frozen import bins
    setup(cfg);data=Data('cuda')
    with np.load(SOURCE/'data/raw_labels.npz') as z:raw={k:z[k].copy() for k in ('E','f','mask_E','mask_f')}
    pc=json.loads((PARENT/'round_config.json').read_text())
    forbidden=[str(SOURCE/'data'),str(ROOT/'cache'),pc['source_checkpoint']]
    opened=[];old_open=builtins.open;old_io=io.open
    def checked(fn):
        def wrapper(file,*args,**kwargs):
            if isinstance(file,(str,bytes,Path)):
                path=str(Path(file).resolve());opened.append(path)
                assert not any(path==x or path.startswith(x+'/') for x in forbidden),('Forbidden predictor input',path)
            return fn(file,*args,**kwargs)
        return wrapper
    results={}
    try:
        builtins.open=checked(old_open);io.open=checked(old_io)
        for arm in cfg['arms']:
            rd=ROOT/'runs'/arm;complete=json.loads((rd/'FIT_COMPLETE.json').read_text())
            history=[json.loads(s) for s in (rd/'history.jsonl').read_text().splitlines()]
            results[arm]={}
            for tag in ('best','last'):
                path=rd/(tag+'.pt')
                if tag=='best':
                    model=load_predictor(path,'cuda');expected=complete['validation'];epoch=complete['selected_epoch']
                else:
                    ck=torch.load(path,map_location='cpu',weights_only=False)
                    model=FrozenAdapterMTO(ck['source_config'],ck['stats'],True)
                    model.load_state_dict(ck['model'],strict=True);model.requires_grad_(False);model.eval();model.cuda()
                    assert frozen_digests(model)==ck['frozen_parameter_buffer_hashes']
                    expected=history[-1]['validation'];epoch=ck['state']['completed_epoch'];assert epoch==20
                    del ck
                metrics,arrays=validate(model,data,raw,cfg['batch_size'],mf['tail_thresholds'])
                difference=metrics['raw_f']['r2']-expected['raw_f']['r2'];assert abs(difference)<1e-7
                results[arm][tag]={'epoch':epoch,'checkpoint_sha256':sha(path),'metrics':metrics,
                    'native_r2_replay_difference':difference,'true_pred_q99_bins':bins(arrays,mf['tail_thresholds']['q99'])}
                del model,arrays
    finally:
        builtins.open=old_open;io.open=old_io
    record={'passed':True,'manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json'),'gpu_uuid':EXPECTED[1],
        'all_arms_all_valid_validation_labels':True,'raw_arrays_written':False,'test_inference':False,
        'selected_predictor_original_base_data_cache_access':False,'evaluation_data_loaded_before_guard':True,
        'opened_paths_during_geometry_replay':opened,'source_hashes':{str(ROOT/n):sha(ROOT/n) for n in
            ('terminal_replay.py','summarize_frozen.py','predictor.py','model_frozen.py')},'arms':results}
    (ROOT/'TERMINAL_REPLAY.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({a:{t:{'epoch':x['epoch'],'r2':x['metrics']['raw_f']['r2'],'replay_delta':x['native_r2_replay_difference']}
        for t,x in tags.items()} for a,tags in results.items()},indent=2))

if __name__=='__main__':main()
