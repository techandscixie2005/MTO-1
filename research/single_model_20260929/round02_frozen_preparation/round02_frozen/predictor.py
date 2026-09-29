"""Load one complete checkpoint; inference takes geometry and no labels/cache."""
import torch
from model_frozen import FrozenAdapterMTO,frozen_digests

def load_predictor(checkpoint,device='cpu'):
    record=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert record['adapter_enabled'] is True and record['geometry_only_inference'] is True
    model=FrozenAdapterMTO(record['source_config'],record['stats'],True)
    model.load_state_dict(record['model'],strict=True)
    model.requires_grad_(False);model.eval();model.to(device)
    assert frozen_digests(model)==record['frozen_parameter_buffer_hashes'], 'Frozen parameters/buffers differ'
    return model
