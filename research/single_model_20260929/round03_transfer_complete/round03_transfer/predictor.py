"""One native original-MTO checkpoint and one affine map, geometry input only."""
import torch
from clean_source import MTOEA,tensor_state_hash
C_F=2.0/(3.0*27.211386245988)

class AffineMTO(torch.nn.Module):
    def __init__(self,config,stats,alpha,beta):
        super().__init__();self.base=MTOEA(config,stats)
        self.register_buffer('alpha',torch.tensor(alpha,dtype=torch.float64))
        self.register_buffer('beta',torch.tensor(beta,dtype=torch.float64))

    def forward(self,**geometry):
        energy,matrix=self.base(**geometry)
        native=C_F*energy.double()*matrix.double().diagonal(dim1=-2,dim2=-1).sum(-1)
        return {'E':energy,'A':matrix,'native_f':native,'f':torch.clamp_min(self.alpha*native+self.beta,0)}

def load_predictor(checkpoint,device='cpu'):
    record=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert record['format']=='round03_single_native_model_affine' and record['geometry_only_inference'] is True
    model=AffineMTO(record['source_config'],record['stats'],record['alpha'],record['beta'])
    model.load_state_dict(record['model'],strict=True);model.requires_grad_(False);model.eval();model.to(device)
    assert tensor_state_hash(dict(model.base.named_buffers()))==record['base_buffer_tensor_sha256']
    assert float(model.alpha)==record['alpha'] and float(model.beta)==record['beta']
    return model

def export_record(base,config,stats,alpha,beta,provenance):
    model=AffineMTO(config,stats,alpha,beta)
    model.base.load_state_dict(base.state_dict(),strict=True);model.requires_grad_(False);model.eval()
    return {'format':'round03_single_native_model_affine','model':model.state_dict(),
        'source_config':config,'stats':stats,'alpha':alpha,'beta':beta,
        'geometry_only_inference':True,'native_full_baseline_input':True,
        'base_buffer_tensor_sha256':tensor_state_hash(dict(model.base.named_buffers())),
        'provenance':provenance}
