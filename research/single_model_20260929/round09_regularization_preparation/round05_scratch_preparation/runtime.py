"""Fixed FP32 runtime and durable resumable checkpoint helpers."""
import os,random
from pathlib import Path
import numpy as np
import torch
from common import atomic_json

def setup(cfg):
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(cfg['num_threads']);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False

def optimizer(model,cfg):
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg['lr'],
        betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'],amsgrad=True)

def rng_state(generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'order':generator.bit_generator.state,
        'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}

def restore_rng(state,generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);generator.bit_generator.state=state['order']
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'].cpu())

def atomic_torch(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

class TensorData:
    def __init__(self,partition,device,rows=None):
        self.partition=partition;self.device=device
        self.global_indices=partition.indices.copy() if rows is None else rows.copy()
        data=partition.load_training_arrays(self.global_indices)
        self.arrays={k:torch.from_numpy(v).to(device) for k,v in data.items()}
        self.ids=partition.ids[np.searchsorted(partition.indices,self.global_indices)]
        self.decode_audit=partition.ledger.copy()

    def __len__(self):return len(self.global_indices)

    def batch(self,rows):
        idx=torch.as_tensor(rows,device=self.device,dtype=torch.long);n=len(rows)
        z=self.arrays['z'][idx];mask=z!=0;counts=mask.sum(1)
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
        raw=self.arrays['edge'][idx].long();em=raw[:,0]>=0
        edge=(raw+offsets[:,None,None]).transpose(1,2)[em].T.contiguous()
        x={'z':z[mask],'pos':self.arrays['pos'][idx][mask],
            'batch':torch.arange(n,device=self.device)[:,None].expand_as(z)[mask],'n':n,'edge_index':edge}
        y={key:self.arrays[key][idx] for key in ('E','A','f','raw_E','mask_E','mask_A','mask_f')}
        return x,y
