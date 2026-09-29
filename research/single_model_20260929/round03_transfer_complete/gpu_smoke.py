"""One validation replay and synthetic training-step timing on an admitted GPU."""
import copy
import fcntl
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from architecture.model import SOURCE,build_model,load_baseline,training_loss
from metrics import validate
from launch import admit,EXPECTED
sys.path.insert(0,str(SOURCE))
from dataset import Data

ROOT=Path(__file__).resolve().parent
lock=open('/tmp/mto_pouter_gpu_1.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
admission=admit(1)
(ROOT/'GPU_SMOKE_ADMISSION.xml').write_text(admission)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==EXPECTED[1]
torch.set_num_threads(2)
torch.manual_seed(11)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
data=Data('cuda')
cfg=json.loads((SOURCE/'configs/mto_eta0.json').read_text())
ck=torch.load(SOURCE/'runs/mto_eta0/best.pt',map_location='cpu',weights_only=False)
model=load_baseline(build_model(cfg,data.stats,True),ck['model']).cuda()
with np.load(SOURCE/'data/raw_labels.npz') as z:
    raw={k:z[k].copy() for k in ('E','f','mask_E','mask_f')}
train=data.parts['train'];vals=raw['f'][train][raw['mask_f'][train]]
thresholds={'q90':float(np.quantile(vals,.9)),'q99':float(np.quantile(vals,.99))}
started=time.monotonic()
result,_=validate(model,data,raw,64,thresholds)
torch.cuda.synchronize()
validation_seconds=time.monotonic()-started
assert abs(result['raw_f']['r2']-.4052941183410983)<1e-5
model.train()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,amsgrad=True)
times=[]
# These eight smoke updates are discarded; no production checkpoint is written.
for i in range(8):
    x,y=data.batch(train[i*64:(i+1)*64])
    torch.cuda.synchronize();t=time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    _,loss=training_loss(model,x,y,data.stats,.001)
    loss['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    optimizer.step();torch.cuda.synchronize();times.append(time.monotonic()-t)
    assert torch.isfinite(loss['total'])
report={'passed':True,'gpu_uuid':EXPECTED[1],'epoch0_validation':result,
    'validation_seconds':validation_seconds,'step_seconds':times,
    'median_warm_step_seconds':float(np.median(times[2:])),
    'estimated_epoch_seconds':float(np.median(times[2:])*np.ceil(len(train)/64)+validation_seconds),
    'thresholds':thresholds,'smoke_optimizer_steps_discarded':8,'production_checkpoint_written':False}
(ROOT/'GPU_SMOKE.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='epoch0_validation'},indent=2))
print('epoch0_r2',result['raw_f']['r2'])
