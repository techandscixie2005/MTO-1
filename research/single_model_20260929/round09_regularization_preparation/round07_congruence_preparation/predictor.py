"""One checkpoint, geometry-only inference; reconstructs no training readers."""
import torch
from fresh_model import build_fresh,tensor_hash
from congruence import CONTRACT

def load_predictor(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    assert checkpoint['format']=='round07_geometry_predictor_v1'
    assert checkpoint['transform_contract']==CONTRACT,'Incompatible readout construction contract'
    model=build_fresh(checkpoint['model_config'],checkpoint['stats'],checkpoint['readout_mode'])
    model.load_state_dict(checkpoint['model'],strict=True)
    assert tensor_hash(dict(model.named_buffers()))==checkpoint['buffer_tensor_sha256']
    assert tensor_hash(model.state_dict())==checkpoint['model_tensor_sha256']
    return model.to(device).eval()

def export_payload(model,model_config,stats,provenance):
    return {'format':'round07_geometry_predictor_v1','model':model.state_dict(),
        'model_config':model_config,'stats':stats,'readout_mode':model.readout_mode,
        'transform_contract':dict(CONTRACT),
        'model_tensor_sha256':tensor_hash(model.state_dict()),
        'buffer_tensor_sha256':tensor_hash(dict(model.named_buffers())),
        'provenance':provenance,'geometry_only_inference':True,'requires_qc_labels':False}
