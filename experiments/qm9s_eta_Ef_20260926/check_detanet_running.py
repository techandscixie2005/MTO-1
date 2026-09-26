import json,torch,numpy as np
from dataset import ROOT
from trainer_detanet import fingerprint
def main():
    out=ROOT/'runs/detanet_ef';ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    cfg=json.loads((ROOT/'configs/detanet_ef.json').read_text())
    assert ck['config']==cfg and ck['fingerprint']==fingerprint()
    assert ck['state']['steps']>=50 and ck['optimizer']['state']
    assert cfg['max_steps']==1000000 and cfg['validate_every_steps']==50 and cfg['stop_loss']==1e-5
    assert cfg['grad_clip'] is None and cfg['no_mto_early_stopping']
    hist=ck['state']['history'];assert all(r['steps']%50==0 for r in hist)
    assert all(np.isfinite(r['val']+list(r['val_spectrum'].values())).all() for r in hist)
    best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    assert best['fingerprint']==ck['fingerprint'] and best['config']==cfg
    report=dict(passed=True,steps=ck['state']['steps'],validation_records=len(hist),best_step=best['steps'],
        best_val=best['val'],all_validation_finite=True,checkpoint_optimizer_scheduler_RNG_cursor_present=True,
        four_group_campaign_confirmed=json.loads((ROOT/'campaign.json').read_text())['detanet_configuration_status']=='confirmed',
        test_not_started=not (ROOT/'reports/test_evaluation_started.json').exists())
    (ROOT/'reports/detanet_running_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
