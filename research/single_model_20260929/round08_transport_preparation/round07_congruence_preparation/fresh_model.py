"""Fresh unchanged PSD base and isolated shared-congruence initialization."""
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
training_loss=math_module.training_loss
TYPES=math_module.TYPES
from congruence import SharedPSDCongruence,element_counts,MODES


class CongruenceMTOEA(AdapterMTOEA):
    def __init__(self, config, stats, mode):
        assert mode in MODES
        super().__init__(config, stats, False)
        # Preserve base tensors and caller/order RNG regardless of gate creation.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            self.shared_readout = SharedPSDCongruence(mode)
        self.readout_mode = mode

    def forward(self,z,pos,batch,n,edge_index=None,return_aux=False,return_readout=False):
        pred,raw_m=super().forward(z,pos,batch,n,edge_index,return_aux=True)
        counts=element_counts(z,batch,n,pred[0].dtype)
        matrix,context=self.shared_readout(pred[0],pred[1],counts)
        transformed=(pred[0],matrix)
        if return_readout:return transformed,raw_m,context
        return (transformed,raw_m) if return_aux else transformed

def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()

def inherited_state(model):return {k:v for k,v in model.state_dict().items() if not k.startswith('shared_readout.')}


def base_state(model):return {k:v for k,v in inherited_state(model).items() if not k.startswith('right_adapter.')}

def build_fresh(config,stats,mode):
    # Only the provided TRAIN statistics enter initialization. No torch.load,
    # original initial_model.pt, historical normalization or target reader.
    # Forking also makes model construction independent of loader/RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        return CongruenceMTOEA(config,stats,mode)

def build_original(config,stats):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        return MTOEA(config,stats)
