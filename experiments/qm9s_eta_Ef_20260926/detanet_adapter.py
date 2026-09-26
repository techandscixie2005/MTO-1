"""Native scalar DetaNet; only dataset keyword and constant-device compatibility."""
import pathlib,sys,torch
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'official_detanet'))
from detanet_model.detanet import DetaNet
from detanet_model.metrics import l2loss
ARCH=dict(num_features=128,act='swish',maxl=3,num_block=3,radial_type='trainable_bessel',
    num_radial=32,attention_head=8,rc=5.0,dropout=0.0,use_cutoff=False,max_atomic_number=9,
    atom_ref=None,scale=1.0,scalar_outsize=20,irreps_out=None,summation=True,norm=False,out_type='scalar',grad_type=None)
class NativeEfDetaNet(DetaNet):
    def __init__(self):
        super().__init__(**ARCH,device=torch.device('cpu'))
        for module,name in ((self.Embedding,'elec'),(self,'mass')):
            value=getattr(module,name);delattr(module,name);module.register_buffer(name,value,persistent=False)
    def forward(self,z,pos,batch,n=None,edge_index=None):
        return super().forward(z=z,pos=pos,batch=batch,edge_index=edge_index)
def build_detanet(cfg):
    torch.manual_seed(cfg['seed'])
    return NativeEfDetaNet()
def ef_loss(output,target):
    y=torch.cat((target['E'],target['f']),dim=-1)
    mask=torch.cat((target['mask_E'],target['mask_f']),dim=-1)
    return l2loss(output[mask],y[mask])
