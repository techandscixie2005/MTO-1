"""Fresh unchanged MTO/PSD base with reviewed blocks2/3 transport only."""
import hashlib,importlib.util
from pathlib import Path
import torch
MATH=Path(__file__).resolve().parent.parent/'architecture/model.py'
spec=importlib.util.spec_from_file_location('round05_adapter_math',MATH)
math_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(math_module)
AdapterMTOEA=math_module.AdapterMTOEA
MTOEA=math_module.MTOEA
base_loss=math_module.base_loss
decorrelation=math_module.decorrelation
TYPES=math_module.TYPES
from transport import TransportBlock,MODES


class TransportMTOEA(AdapterMTOEA):
    def __init__(self,config,stats,mode):
        assert mode in MODES
        super().__init__(config,stats,False)
        for k in (1,2):self.core.blocks[k]=TransportBlock(self.core.blocks[k],mode)
        self.transport_mode=mode

    def forward(self,z,pos,batch,n,edge_index=None,return_aux=False,return_transport=False):
        for k in (1,2):self.core.blocks[k].record_diagnostics=return_transport
        pred,raw_m=super().forward(z,pos,batch,n,edge_index,return_aux=True)
        if return_transport:
            diag={f'block{k+1}_{name}':value for k in (1,2)
                  for name,value in self.core.blocks[k].last_diagnostic.items()}
            return pred,raw_m,diag
        return (pred,raw_m) if return_aux else pred


def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def transport_state(model):return {k:v for k,v in model.state_dict().items() if '.transport.' in k}
def inherited_state(model):return {k:v for k,v in model.state_dict().items() if '.transport.' not in k}
def base_state(model):return {k:v for k,v in inherited_state(model).items() if not k.startswith('right_adapter.')}


def build_fresh(config,stats,mode):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        return TransportMTOEA(config,stats,mode)


def build_original(config,stats):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        return MTOEA(config,stats)
