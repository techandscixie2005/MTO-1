"""Descriptive raw-f error strata by nearest adjacent true energy gap."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from freeze import sha
from metrics import score

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
Q=[.1,.25,.5,.75,.9]

def nearest(energy,valid):
    delta=np.diff(energy,axis=1)
    pair=valid[:,:-1]&valid[:,1:]&np.isfinite(delta)
    assert (delta[pair]>=0).all(),'Printed state order not monotone'
    delta=np.where(pair,delta,np.inf)
    gap=np.minimum(np.pad(delta,((0,0),(0,1)),constant_values=np.inf),
                   np.pad(delta,((0,0),(1,0)),constant_values=np.inf))
    return np.where(valid & np.isfinite(gap),gap,np.nan)

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    with np.load(SOURCE/'data/dataset.npz') as z:
        train=z['train'].copy();val=z['val'].copy()
    with np.load(SOURCE/'data/raw_labels.npz') as z:
        energy=z['E'].copy();valid=z['mask_E'].copy()
    train_gap=nearest(energy[train],valid[train])
    cuts=np.unique(np.quantile(train_gap[np.isfinite(train_gap)],Q))
    definition={'definition':'minimum energy difference to adjacent printed root; eV; no relabeling',
        'training_quantiles':Q,'unique_cutoffs_eV':cuts.tolist(),'interval_convention':'lower_inclusive_upper_exclusive',
        'training_label_count':int(np.isfinite(train_gap).sum()),'source_energy':'raw_labels.npz E',
        'selection_use':False,'script_sha256':sha(__file__)}
    config=ROOT/'GAP_AUDIT_CONFIG.json'
    if config.exists():
        frozen=json.loads(config.read_text())
        assert frozen['definition']==definition,'Changed gap audit after prespecification'
    else:
        frozen={'definition':definition,'created_at_utc':datetime.now(timezone.utc).isoformat()}
        config.write_text(json.dumps(frozen,indent=2)+'\n')
    if args.freeze_only:
        print(json.dumps(frozen,indent=2));return
    round_cfg=json.loads((ROOT/'round_config.json').read_text())
    assert all((ROOT/'runs'/a/'FIT_COMPLETE.json').is_file() for a in round_cfg['arms'])
    with np.load(SOURCE/'data/raw_labels.npz') as z:
        f=z['f'][val].copy();mf=z['mask_f'][val].copy()
    gap=nearest(energy[val],valid[val])
    membership=np.searchsorted(cuts,gap,side='right')
    membership[~np.isfinite(gap)]=-1
    labels=['gap_bin_'+str(j) for j in range(len(cuts)+1)]+['unknown_gap']
    numbers=list(range(len(cuts)+1))+[-1]
    report={'configuration':frozen,'test_evaluated':False,'all_validation_f_labels_retained':True,
        'interpretation':'association only; does not establish causality, state mixing or label ambiguity',
        'arms':{}}
    for arm in round_cfg['arms']:
        with np.load(ROOT/'runs'/arm/'best_validation_predictions.npz') as z:
            assert np.array_equal(z['indices'],val)
            assert np.array_equal(z['f_true'],f) and np.array_equal(z['mask_f'],mf)
            pred=z['f_pred'].copy()
        groups={}
        for label,number in zip(labels,numbers):
            use=mf&(membership==number)
            groups[label]={'pooled':score(f[use],pred[use]),
                'per_state':[score(f[:,j][use[:,j]],pred[:,j][use[:,j]]) for j in range(10)]}
        assert sum(v['pooled']['count'] for v in groups.values())==int(mf.sum())
        whole=score(f[mf],pred[mf])
        assert abs(sum(v['pooled']['sse'] for v in groups.values())-whole['sse'])<1e-10
        report['arms'][arm]={'whole':whole,'groups':groups}
    (ROOT/'GAP_AUDIT_RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cutoffs_eV':cuts.tolist(),'complete':True,'all_valid_f_count':int(mf.sum())}))

if __name__=='__main__':
    main()
