"""Raw-label float64 evaluation; no test-set entry point."""
import numpy as np
import torch
from architecture.model import base_loss, decorrelation

C_F = 2.0 / (3.0 * 27.211386245988)
CAL_ALPHA = 0.8511830211044088
CAL_BETA = 0.003725185373211049


def score(truth, pred):
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if truth.size == 0:
        return {'count': 0, 'sse': 0.0, 'sst': 0.0, 'r2': None, 'mae': None, 'rmse': None}
    assert np.isfinite(truth).all() and np.isfinite(pred).all()
    residual = pred - truth
    sse = float(np.square(residual).sum())
    sst = float(np.square(truth - truth.mean()).sum())
    return {'count': int(truth.size), 'sse': sse, 'sst': sst,
            'r2': 1 - sse/sst if sst > 0 else None,
            'mae': float(np.abs(residual).mean()), 'rmse': float(np.sqrt(sse/truth.size))}


@torch.no_grad()
def validate(model, data, raw, batch_size, thresholds):
    model.eval()
    indices = data.parts['val']
    ep, ap = [], []
    sums = np.zeros(3, dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    for start in range(0,len(indices),batch_size):
        x,y = data.batch(indices[start:start+batch_size])
        pred,m = model(**x,return_aux=True)
        v = base_loss(pred,y,data.stats)
        valid = y['mask_E'] & y['mask_A'] & y['mask_f']
        pair_count = int(((valid.sum(-1)*(valid.sum(-1)-1))//2).sum())
        count = np.array([int(y['mask_E'].sum()),int((y['mask_E']&y['mask_A']).sum()),pair_count])
        vals = np.array([float(v['energy']),float(v['trace']),float(decorrelation(m,valid))])
        sums += vals*count
        counts += count
        ep.append(pred[0].cpu().numpy().astype(np.float64))
        ap.append(pred[1].cpu().numpy().astype(np.float64))
    energy = np.concatenate(ep)
    matrix = np.concatenate(ap)
    strength = C_F*energy*np.trace(matrix,axis1=-2,axis2=-1)
    truth = raw['f'][indices]
    mask = raw['mask_f'][indices].astype(bool)
    result = {'raw_f': score(truth[mask],strength[mask]),
        'per_state': [score(truth[:,j][mask[:,j]],strength[:,j][mask[:,j]]) for j in range(truth.shape[1])],
        'energy': score(raw['E'][indices][raw['mask_E'][indices]],energy[raw['mask_E'][indices]]),
        'energy_loss': float(sums[0]/counts[0]),'trace_loss': float(sums[1]/counts[1]),
        'base_objective': float(sums[0]/counts[0]+sums[1]/counts[1]),
        'decorrelation': float(sums[2]/counts[2]) if counts[2] else 0.0,
        'bright_tail': {}}
    for name,threshold in thresholds.items():
        subset = mask & (truth>=threshold)
        result['bright_tail'][name] = {'threshold_train_raw_f':threshold,**score(truth[subset],strength[subset])}
    # Prespecified before fitting: the same historical scalar calibration for
    # all arms is secondary only. No refit, model averaging, or selection on it.
    calibrated=np.maximum(0,CAL_ALPHA*strength+CAL_BETA)
    result['fixed_calibrated_secondary']={
        'alpha':CAL_ALPHA,'beta':CAL_BETA,'selection_metric':False,
        'raw_f':score(truth[mask],calibrated[mask]),
        'per_state':[score(truth[:,j][mask[:,j]],calibrated[:,j][mask[:,j]]) for j in range(truth.shape[1])],
        'bright_tail':{name:{'threshold_train_raw_f':threshold,**score(
            truth[mask&(truth>=threshold)],calibrated[mask&(truth>=threshold)])}
            for name,threshold in thresholds.items()}}
    assert result['raw_f']['count']==int(mask.sum())
    assert np.isfinite(strength).all()
    arrays = {'indices':indices,'ids':data.ids[indices],'f_pred':strength,
              'f_true':truth,'mask_f':mask,'E_pred':energy,'E_true':raw['E'][indices]}
    return result,arrays
