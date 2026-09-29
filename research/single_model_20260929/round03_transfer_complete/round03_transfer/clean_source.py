"""Random original MTO and a loader that decodes only source-fit labels."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
SOURCE=Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
sys.path.insert(0,str(SOURCE/'frozen_reference'))
sys.path.insert(0,str(PARENT/'reports/velocity_audit'))
from models_ea import MTOEA
from audit_velocity_labels import selected_rows

def tensor_state_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        digest.update(key.encode());digest.update(str(x.dtype).encode());digest.update(str(tuple(x.shape)).encode())
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()

def build_random(config,stats):
    torch.manual_seed(config['seed'])
    # This constructs the original architecture and random parameters directly.
    # The only label-dependent initializer is supplied through fit-only stats.
    return MTOEA(config,stats)

class FitData:
    def __init__(self,device='cpu'):
        self.device=device
        self.global_indices=np.load(ROOT/'data/fit_indices.npy')
        self.stats=json.loads((ROOT/'fit_normalization.json').read_text())
        order=np.argsort(self.global_indices);undo=np.argsort(order)
        selected=self.global_indices[order]
        with zipfile.ZipFile(SOURCE/'data/dataset.npz') as z:
            self.ids=selected_rows(z,'ids',selected)[undo]
            for key in ('z','pos','edge','E','A'):
                values=selected_rows(z,key,selected)[undo]
                setattr(self,key,torch.from_numpy(values).to(device))
        with zipfile.ZipFile(SOURCE/'data/raw_labels.npz') as z:
            assert np.array_equal(selected_rows(z,'ids',selected)[undo],self.ids)
            for key in ('mask_E','mask_A'):
                values=selected_rows(z,key,selected)[undo].astype(bool)
                setattr(self,key,torch.from_numpy(values).to(device))
        assert self.mask_E.all() and self.mask_A.all()
        assert torch.isfinite(self.E).all() and torch.isfinite(self.A).all()
        self.decode_audit={'global_rows':len(self.global_indices),'geometry_members':['ids','z','pos','edge'],
            'target_members':['E','A','mask_E','mask_A'],'raw_f_decoded':False,
            'calibration_validation_test_target_rows_decoded':0,
            'fit_index_sha256':hashlib.sha256(self.global_indices.tobytes()).hexdigest()}

    def batch(self,rows):
        idx=torch.as_tensor(rows,device=self.device,dtype=torch.long);n=len(rows)
        z=self.z[idx];mask=z!=0;counts=mask.sum(1)
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
        raw=self.edge[idx].long();em=raw[:,0]>=0
        edge=(raw+offsets[:,None,None]).transpose(1,2)[em].T.contiguous()
        x=dict(z=z[mask],pos=self.pos[idx][mask],batch=torch.arange(n,device=self.device)[:,None].expand_as(z)[mask],
            n=n,edge_index=edge)
        y={key:getattr(self,key)[idx] for key in ('E','A','mask_E','mask_A')}
        return x,y

def source_loss(pred,target,stats):
    energy,matrix=pred;me=target['mask_E'].bool();ma=target['mask_A'].bool()
    le=(energy[me]-target['E'][me]).square().mean()/stats['sE2']
    predicted=matrix[ma].diagonal(dim1=-2,dim2=-1).sum(-1)
    actual=target['A'][ma].diagonal(dim1=-2,dim2=-1).sum(-1)
    ls=(predicted-actual).square().mean()/(3*stats['sA2'])
    return le+ls,le,ls
