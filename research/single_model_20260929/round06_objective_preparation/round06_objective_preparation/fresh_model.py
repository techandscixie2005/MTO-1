"""Fresh original MTO plus the already reviewed right-F/raw-M mathematics."""
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

def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()

def base_state(model):return {k:v for k,v in model.state_dict().items() if not k.startswith('right_adapter.')}

def build_fresh(config,stats,enabled):
    # Only the provided TRAIN statistics enter initialization. No torch.load,
    # original initial_model.pt, historical normalization or target reader.
    # Forking also makes model construction independent of loader/RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        return AdapterMTOEA(config,stats,enabled)

def build_original(config,stats):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed'])
        return MTOEA(config,stats)
