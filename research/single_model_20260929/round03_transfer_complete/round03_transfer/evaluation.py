"""Explicit subset-only access and all-label raw-f summaries for the affine stage."""
import json
import zipfile
import numpy as np
import torch
from clean_source import ROOT,SOURCE,FitData,selected_rows
from predictor import C_F

class SubsetData(FitData):
    def __init__(self,indices,device='cpu'):
        self.device=device;self.global_indices=np.asarray(indices,dtype=np.int64)
        order=np.argsort(self.global_indices);undo=np.argsort(order);selected=self.global_indices[order]
        with zipfile.ZipFile(SOURCE/'data/dataset.npz') as archive:
            self.ids=selected_rows(archive,'ids',selected)[undo]
            for key in ('z','pos','edge'):
                setattr(self,key,torch.from_numpy(selected_rows(archive,key,selected)[undo]).to(device))
        with zipfile.ZipFile(SOURCE/'data/raw_labels.npz') as archive:
            raw_ids=selected_rows(archive,'ids',selected)[undo];assert np.array_equal(raw_ids,self.ids)
            self.raw={key:selected_rows(archive,key,selected)[undo] for key in ('f','E','mask_f','mask_E')}

    def batch(self,rows):
        idx=torch.as_tensor(rows,device=self.device,dtype=torch.long);n=len(rows)
        z=self.z[idx];mask=z!=0;counts=mask.sum(1);offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
        raw=self.edge[idx].long();em=raw[:,0]>=0
        edge=(raw+offsets[:,None,None]).transpose(1,2)[em].T.contiguous()
        return dict(z=z[mask],pos=self.pos[idx][mask],batch=torch.arange(n,device=self.device)[:,None].expand_as(z)[mask],n=n,edge_index=edge)

@torch.no_grad()
def predict_native(model,data,batch_size):
    model.eval();energies=[];strengths=[]
    for start in range(0,len(data.global_indices),batch_size):
        x=data.batch(np.arange(start,min(start+batch_size,len(data.global_indices))))
        E,A=model(**x);e=E.cpu().numpy().astype(np.float64);a=A.cpu().numpy().astype(np.float64)
        energies.append(e);strengths.append(C_F*e*np.trace(a,axis1=-2,axis2=-1))
    e=np.concatenate(energies);f=np.concatenate(strengths);assert np.isfinite(e).all() and np.isfinite(f).all()
    return e,f

def score(truth,pred):
    t=np.asarray(truth,dtype=np.float64);p=np.asarray(pred,dtype=np.float64)
    assert t.shape==p.shape and np.isfinite(t).all() and np.isfinite(p).all()
    if not t.size:return {'count':0,'sse':0.0,'sst':0.0,'r2':None,'rmse':None,'mae':None}
    sse=float(np.square(p-t).sum());sst=float(np.square(t-t.mean()).sum())
    return {'count':int(t.size),'sse':sse,'sst':sst,'r2':1-sse/sst if sst>0 else None,
            'rmse':float(np.sqrt(sse/t.size)),'mae':float(np.abs(p-t).mean())}

def report(raw,pred,thresholds):
    truth=raw['f'];mask=raw['mask_f'].astype(bool)
    return {'raw_f':score(truth[mask],pred[mask]),
        'per_state':[score(truth[:,k][mask[:,k]],pred[:,k][mask[:,k]]) for k in range(10)],
        'bright_tail':{name:{'threshold_train_raw_f':cut,**score(truth[mask&(truth>=cut)],pred[mask&(truth>=cut)])}
                       for name,cut in thresholds.items()}}

def affine_fit(truth,pred,mask):
    # All valid rows enter once. Nonfinite values and a degenerate denominator fail.
    y=np.asarray(truth,dtype=np.float64)[mask];x=np.asarray(pred,dtype=np.float64)[mask]
    assert len(x)>1 and np.isfinite(x).all() and np.isfinite(y).all()
    dx=x-x.mean();dy=y-y.mean();denominator=float(dx@dx)
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('Nonfinite or zero affine denominator')
    alpha=float((dx@dy)/denominator);beta=float(y.mean()-alpha*x.mean())
    if not np.isfinite([alpha,beta]).all():raise ValueError('Nonfinite affine coefficients')
    return {'alpha':alpha,'beta':beta,'count':len(x),'zero_target_count':int((y==0).sum()),
            'mean_prediction':float(x.mean()),'mean_target':float(y.mean()),'centered_denominator':denominator,
            'native_calibration_subset_metrics':score(y,x),
            'in_subset_affine_fit_metrics_descriptive':score(y,np.maximum(0,alpha*x+beta))}
