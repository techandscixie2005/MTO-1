"""Correct one invalid diagnostic; do not rerun primary smoke or other branches."""
import json
import os
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_protocol import load,sha
from models import build_triplet,VARIANTS
from smoke_diagnostics import optimizer_diagnostics
from train import atomic_json

assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
source_dir=ROOT/'reports/instrumented_v2_1299173'
source=source_dir/VARIANTS[0]/'step600.pt'
ckpt=torch.load(source,map_location='cpu',weights_only=False)
for name in ['models.py','train.py','protocol.json','data_protocol.py','data/frozen.json']:
    assert sha(ROOT/name)==ckpt['fingerprint'][name],name
out=ROOT/f'reports/D1_correction_{os.environ["SLURM_JOB_ID"]}'
out.mkdir(exist_ok=False)
cfg=json.loads((ROOT/'protocol.json').read_text());data,info=load()
models=build_triplet(cfg,info['n_ref'],info['grid'],11)
result=optimizer_diagnostics(models[VARIANTS[0]],cfg,data['train'][:32],info,'cuda',source,out,modes=('D1',))
atomic_json(dict(scope='DIAGNOSTIC_ONLY',status='CORRECTED_D1_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],
                 node=os.uname().nodename,main_job_id=1299173,source_checkpoint_sha256=sha(source),
                 correction='immutable disk source, deep-copy optimizer state, preserved step=600, checked first/last steps 601/900',
                 result={k:v for k,v in result['D1'].items() if k not in ['evaluations','step_statistics']}),out/'CORRECTED_D1_COMPLETE.json')
print('CORRECTED_D1_COMPLETE',out,'BEST',result['D1']['best'],'LAST',result['D1']['last'],flush=True)
