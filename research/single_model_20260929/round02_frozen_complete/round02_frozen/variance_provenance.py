"""Independent full-TRAIN raw printed-f population variance audit; CPU only."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT/'reports/velocity_audit'))
from audit_velocity_labels import selected_rows

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    source=Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/data')
    historical=Path('/home/inspur/MTO-1/research/oscillator_r2_20260928')
    frozen=json.loads((PARENT/'FROZEN_MANIFEST.json').read_text())
    paths=[source/'dataset.npz',source/'raw_labels.npz',source/'splits.json']
    for path in paths:assert sha(path)==frozen['source_hashes'][str(path)]
    with np.load(source/'dataset.npz',allow_pickle=False) as z:
        train=z['train'].copy()
    assert train.shape==(120355,) and np.unique(train).size==120355
    order=np.argsort(train);undo=np.argsort(order)
    with zipfile.ZipFile(source/'raw_labels.npz') as z:
        y=selected_rows(z,'f',train[order])[undo]
        mask=selected_rows(z,'mask_f',train[order])[undo].astype(bool)
    values=y.astype(np.float64)[mask]
    assert values.size==1203550 and np.isfinite(values).all() and np.all(values>=0)
    variance=float(np.var(values,ddof=0,dtype=np.float64))
    mean=float(np.mean(values,dtype=np.float64));std=float(np.std(values,ddof=0,dtype=np.float64))
    hstats=json.loads((historical/'architecture/stats.json').read_text())
    hcfg=json.loads((historical/'frozen_residual/config.json').read_text())
    cfg=json.loads((ROOT/'config.json').read_text())
    assert variance==hstats['f_var_train']==hcfg['train_f_variance']==cfg['train_f_variance']
    assert mean==hstats['f_mean_train'] and std==hstats['f_std_train']==hcfg['f_std_train']
    files=paths+[historical/'architecture/prepare.py',historical/'architecture/stats.json',
        historical/'frozen_residual/config.json',historical/'frozen_residual/train.py',
        Path(__file__),PARENT/'reports/velocity_audit/audit_velocity_labels.py']
    report={'passed':True,'variance':variance,'mean':mean,'std':std,'ddof':0,
        'dtype':'float64 raw printed oscillator strength; no energy/state weighting',
        'pooling':'flatten all valid frozen TRAIN molecule-state entries in original TRAIN index order',
        'molecules':120355,'valid_labels':int(mask.sum()),'invalid_labels':int(mask.size-mask.sum()),
        'zero_labels':int(np.count_nonzero(values==0)),'all_valid_labels_retained':True,
        'coefficient':1.0,'config_value_exact_match':True,'historical_source_exact_match':True,
        'fp32_training_target_is_cast_from_same_raw_f':True,
        'float32_target_variance_for_comparison_only':float(np.var(values.astype(np.float32),dtype=np.float64)),
        'recomputed_coefficient_not_fit':True,'model_inference':False,'gpu_used':False,
        'decoded_validation_or_test_label_rows':0,'raw_arrays_written':False,
        'train_index_sha256':hashlib.sha256(train.tobytes()).hexdigest(),
        'source_hashes':{str(p):sha(p) for p in files},
        'historical_loss':'frozen_residual/train.py loss=(pred-target).square().mean()/cfg[train_f_variance]; E fixed; no LE+Ls'}
    (ROOT/'VARIANCE_PROVENANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','variance','mean','std','valid_labels','zero_labels')}))

if __name__=='__main__':main()
