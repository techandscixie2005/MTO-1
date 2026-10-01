"""Every-label raw printed-f evaluation; no test reader or calibration."""
import numpy as np
import torch
from fresh_model import base_loss
CONST=2/(3*27.211386245988)

def metric(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    assert y.shape==p.shape and y.size and np.isfinite(y).all() and np.isfinite(p).all()
    residual=p-y;sse=float(np.square(residual).sum());sst=float(np.square(y-y.mean()).sum())
    return {'count':y.size,'sse':sse,'mse':sse/y.size,'rmse':float(np.sqrt(sse/y.size)),
        'mae':float(np.abs(residual).mean()),'r2':1-sse/sst if sst>0 else None}

def score(y,p,mask,stats):
    assert y.shape==p.shape==mask.shape and mask.all(),'All frozen valid labels must be included'
    out={'pooled':metric(y[mask],p[mask]),'per_state':{str(k+1):metric(y[:,k],p[:,k]) for k in range(y.shape[1])}}
    out['bright_tail']={}
    for key in ('q90','q99'):
        selected=mask&(y>=stats[key]);assert selected.any()
        out['bright_tail'][key]={'threshold':stats[key],**metric(y[selected],p[selected])}
    bins={};threshold=stats['q99']
    for actual in (False,True):
        for predicted in (False,True):
            use=mask&((y>=threshold)==actual)&((p>=threshold)==predicted)
            bins[f'true_{int(actual)}_pred_{int(predicted)}']={'count':int(use.sum()),
                'sse':float(np.square(p[use]-y[use]).sum()),'absolute_error_sum':float(np.abs(p[use]-y[use]).sum())}
    assert sum(b['count'] for b in bins.values())==int(mask.sum())
    assert np.isclose(sum(b['sse'] for b in bins.values()),out['pooled']['sse'],rtol=1e-12,atol=1e-12)
    out['false_bright_bins']=bins
    return out

@torch.no_grad()
def evaluate(model,data,stats,batch_size=64):
    model.eval();predictions=[];truth=[];energy=[];energy_true=[];masks=[];loss_sum=np.zeros(3);count=0
    for start in range(0,len(data),batch_size):
        rows=np.arange(start,min(start+batch_size,len(data)));x,y=data.batch(rows)
        e,a=model(**x);loss=base_loss((e,a),y,stats)
        # FP64 construction of raw-f matches prior audited evaluator units.
        p=e.double()*a.double().diagonal(dim1=-2,dim2=-1).sum(-1)*CONST
        assert torch.isfinite(p).all() and (p>=0).all()
        predictions.append(p.cpu().numpy());truth.append(y['f'].double().cpu().numpy())
        energy.append(e.double().cpu().numpy());energy_true.append(y['raw_E'].double().cpu().numpy())
        masks.append(y['mask_f'].bool().cpu().numpy())
        loss_sum+=np.array([float(loss[k]) for k in ('total','energy','trace')])*len(rows);count+=len(rows)
    y=np.concatenate(truth);p=np.concatenate(predictions);m=np.concatenate(masks)
    result=score(y,p,m,stats)
    result['energy']=metric(np.concatenate(energy_true),np.concatenate(energy))
    result['base_objective']=dict(zip(('total','energy','trace'),(loss_sum/count).tolist()))
    return result,{'pred_f':p,'true_f':y,'mask':m,'pred_E':np.concatenate(energy),'true_E':np.concatenate(energy_true)}
