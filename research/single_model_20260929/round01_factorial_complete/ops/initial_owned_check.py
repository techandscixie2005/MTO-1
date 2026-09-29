#!/usr/bin/env python3
"""One startup audit of the four registered arms and resumable CPU checkpoint state."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import torch

root=Path('/home/inspur/MTO-1/research/single_model_20260929')
report=json.loads((root/'monitoring/latest.json').read_text())
assert len(report['runs'])==4
expected={'control':1,'adapter':2,'decorrelation':4,'both':6}
gpus={g['index']:g for g in report['gpu_snapshot']['gpus']}
records=[]
for r in report['runs']:
    arm=Path(r['run_dir']).name
    assert r['identity_matches'] and r['state']=='RUNNING_IDENTITY_MATCH'
    assert r['observed_gpu_uuids']==[gpus[expected[arm]]['uuid']]==[r['gpu_uuid']]
    out=root/r['run_dir']
    checkpoints={}
    for name in ('best.pt','last.pt'):
        with (out/name).open('rb') as f:
            content=f.read();digest=hashlib.sha256(content).hexdigest();f.seek(0)
            ck=torch.load(f,map_location='cpu',weights_only=False)
        if name=='last.pt':
            assert all(k in ck for k in ('model','optimizer','rng','state','config','arm','manifest_sha256'))
            assert all(k in ck['rng'] for k in ('python','numpy','order','torch','cuda'))
            epoch=ck['state']['completed_epoch']
            assert ck['state']['epoch_zero_metrics'] is not None
            epoch0=ck['state']['epoch_zero_metrics']
        else:
            assert all(k in ck for k in ('model','epoch','validation','manifest_sha256'))
            epoch=ck['epoch']
        checkpoints[name]={'bytes':len(content),'sha256_at_open':digest,'epoch_at_open':epoch,
                           'manifest_sha256':ck['manifest_sha256'],'metadata_only_archived':True}
    records.append({'arm':arm,'pid':r['process_identity']['pid'],'gpu':expected[arm],
                    'uuid':r['gpu_uuid'],'state':r['state'],'checkpoints':checkpoints,
                    'epoch_zero_metrics':epoch0})
result={'passed':True,'checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'type':'Targeted startup identity/checkpoint audit; does not change four-hour cadence',
        'monitor_snapshot':report['checked_at_utc'],'runs':records,
        'unrelated_gpu0_preserved':any(p['pid']=='712998' for p in gpus[0]['compute_processes']),
        'only_gpu_1_2_4_6_used':True,'four_new_run_state_transitions_are_not_failures':True,
        'interventions':[]}
(root/'ops/INITIAL_OWNED_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'arms':[{'arm':r['arm'],'pid':r['pid'],'gpu':r['gpu'],'checkpoints':r['checkpoints']} for r in records],'unrelated_gpu0_preserved':result['unrelated_gpu0_preserved']},indent=2))
