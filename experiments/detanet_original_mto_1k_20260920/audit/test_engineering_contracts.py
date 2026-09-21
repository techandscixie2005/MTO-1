"""Independent executable checks for weighted accumulation and checkpoint contracts."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import train

torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
torch.manual_seed(901)
cfg=json.loads((ROOT/'protocol.json').read_text())


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(4,3)
    def forward(self,features):
        return self.linear(features)


original_collate=train.collate
def synthetic_collate(records,device):
    return {'features':torch.stack([r[0] for r in records])},torch.stack([r[1] for r in records])
train.collate=synthetic_collate
records=list(zip(torch.randn(7,4),torch.randn(7,3)))
a=Tiny();b=copy.deepcopy(a)
oa,sa=train.optimizer_scheduler(a,cfg);ob,sb=train.optimizer_scheduler(b,cfg)
la=train.update(a,oa,records,.37,7,'cpu')
lb=train.update(b,ob,records,.37,2,'cpu')
maximum=max(float((x-y).abs().max()) for x,y in zip(a.parameters(),b.parameters()))
assert abs(la-lb)<1e-12
for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-10)
train.collate=original_collate
generator=torch.Generator().manual_seed(11)
for val in [1.]*114:sa.step(val)
assert oa.param_groups[0]['lr']==cfg['lr']/4
fp={'model':'known','data':'known','protocol':'known'}
with tempfile.TemporaryDirectory(prefix='contract_',dir=ROOT/'reports') as temp:
    path=Path(temp)/'last.pt'
    train.atomic_save(train.checkpoint(a,oa,sa,generator,{'epoch':114,'step':1,'order':[3,1,0,2],'cursor':2},fp),path)
    draws=(random.random(),np.random.rand(),torch.rand(3),torch.randperm(7,generator=generator))
    state=train.restore(path,b,ob,sb,generator,fp,'cpu')
    assert state=={'epoch':114,'step':1,'order':[3,1,0,2],'cursor':2}
    after=(random.random(),np.random.rand(),torch.rand(3),torch.randperm(7,generator=generator))
    assert draws[:2]==after[:2]
    assert torch.equal(draws[2],after[2]) and torch.equal(draws[3],after[3])
    assert sa.state_dict()==sb.state_dict()
    for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x,y)
    try:train.restore(path,b,ob,sb,generator,dict(fp,data='CHANGED'),'cpu')
    except RuntimeError as e:assert 'fingerprint mismatch' in str(e)
    else:raise AssertionError('incorrect resume was not refused')
result=dict(weighted_uneven_accumulation=True,loss_abs_difference=abs(la-lb),parameter_max_abs=maximum,
            atomic_save_strict_load=True,scheduler_actual_reductions_and_restore=True,
            python_numpy_torch_shuffle_rng_restore=True,fingerprint_mismatch_rejected=True,success=True)
(ROOT/'reports/engineering_contracts.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
