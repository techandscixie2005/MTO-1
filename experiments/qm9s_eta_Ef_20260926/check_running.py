"""Read committed checkpoints and order hashes without interrupting workers."""
import hashlib,json
import numpy as np
import torch
from dataset import ROOT
from trainer import fingerprint
def main():
    rows={};expected=fingerprint()
    for n in ('mto_eta0','mto_eta01','mto_eta1'):
        out=ROOT/'runs'/n;ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert ck['fingerprint']==expected
        for key in ('model','optimizer','scheduler','rng_numpy','rng_numpy_global','rng_torch','rng_cuda','rng_python'):assert key in ck
        s=ck['state'];assert s['steps']>0 and len(ck['optimizer']['state'])>0
        h=s['history'];rng=np.random.default_rng(11)
        with np.load(ROOT/'data/dataset.npz') as d:train=d['train']
        for row in h:
            order=rng.permutation(train)
            assert row['order_sha256']==hashlib.sha256(order.tobytes()).hexdigest()
        best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        assert best['config']==ck['config'] and best['fingerprint']==expected
        assert np.isfinite(best['val']).all()
        rows[n]=dict(saved_epoch=s['epoch'],saved_steps=s['steps'],complete_history_epochs=len(h),best_epoch=best['epoch'],
            initial_hash=json.loads((out/'run_manifest.json').read_text())['initial_state_sha256'],
            all_saved_epoch_orders_verified=True,full_resume_state_present=True)
    assert len(set(r['initial_hash'] for r in rows.values()))==1
    (ROOT/'reports/running_checkpoint_checks.json').write_text(json.dumps(dict(passed=True,runs=rows),indent=2))
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
