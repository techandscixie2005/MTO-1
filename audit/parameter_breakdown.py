import json
from pathlib import Path
import torch
from models import build_pair,count_parameters
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
cfg=json.loads((ROOT/'configs/grid.json').read_text())['configs'][0]
pair,counts=build_pair(cfg,18)
out={'counts':counts,'models':{}}
for arm,model in pair.items():
    out['models'][arm]={
        'modules':{name:count_parameters(module) for name,module in model.named_children()},
        'decoder':{name:{'shape':list(p.shape),'count':p.numel()} for name,p in model.decoder.named_parameters()},
        'backbone':{name:count_parameters(module) for name,module in model.backbone.core.named_children()}}
out['mto_assembly']={name:p.numel() for name,p in pair['mto'].readout.named_parameters()}
out['backbone_blocks']=[{name:count_parameters(module) for name,module in block.named_children()} for block in pair['mto'].backbone.core.blocks]
(ROOT/'audit/parameter_breakdown.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
