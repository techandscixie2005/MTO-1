"""Lightweight metadata proof that server checkpoints remain resumable."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT.parent))
from freeze import sha

def main():
    cfg=json.loads((ROOT/'config.json').read_text());mh=sha(ROOT/'FROZEN_MANIFEST.json');arms={}
    for arm in cfg['arms']:
        rd=ROOT/'runs'/arm;receipt=json.loads((rd/'FIT_COMPLETE.json').read_text())
        assert receipt['completed_epoch']==20
        ck=torch.load(rd/'last.pt',map_location='cpu',weights_only=False)
        assert ck['manifest_sha256']==mh and ck['state']['completed_epoch']==20
        assert len(ck['state']['history'])==21 and ck['state']['best_epoch']==receipt['selected_epoch']
        assert set(ck['rng'])=={'python','numpy','order','torch','cuda'}
        optimizer=ck['optimizer'];steps=[]
        for state in optimizer['state'].values():
            assert set(state)=={'step','exp_avg','exp_avg_sq','max_exp_avg_sq'}
            steps.append(int(state['step']))
        assert len(steps)==9 and min(steps)==max(steps)==ck['state']['steps']
        arms[arm]={'completed_epoch':20,'selected_epoch':receipt['selected_epoch'],'optimizer':'Adam_AMSGrad',
            'optimizer_parameter_tensors':len(steps),'optimizer_steps':steps[0],'all_rng_components_present':True,
            'cache_receipt_sha256':ck['cache_receipt_sha256'],'full_history_preserved':True,
            'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(rd.glob('*.pt'))]}
        del ck
    result={'passed':True,'manifest_sha256':mh,'metadata_only':True,'weights_optimizer_rng_contents_exported':False,
        'source_sha256':sha(__file__),'arms':arms}
    (ROOT/'CHECKPOINT_INVENTORY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:{k:v for k,v in d.items() if k!='files'} for a,d in arms.items()},indent=2))

if __name__=='__main__':main()
