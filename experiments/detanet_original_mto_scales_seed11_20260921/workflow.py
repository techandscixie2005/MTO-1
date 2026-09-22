"""Completion depends on real fits, test outputs and reports, never their scores."""
import json
from data_protocol import ROOT,SCALES,sha
from models import VARIANTS

def check_fits(scale):
    from train import fingerprint
    fp=fingerprint(scale);result={}
    for label in VARIANTS:
        path=ROOT/f'runs/{scale}/seed_11/{label}'
        c=json.loads((path/'FIT_COMPLETE.json').read_text())
        assert c['scale']==scale and c['seed']==11 and c['variant']==label and c['fingerprint']==fp
        assert 1<=c['best_epoch']<=c['epochs']<=1000
        assert (path/'best.pt').is_file() and (path/'last.pt').is_file()
        result[label]=c
    assert len(result)==2
    return result

def check_stage(scale):
    from train import fingerprint
    marker=json.loads((ROOT/f'reports/{scale}/STAGE_COMPLETE.json').read_text())
    assert marker['scale']==scale and marker['seed']==11 and marker['fits']==2
    assert marker['fingerprint']==fingerprint(scale)
    check_fits(scale)
    for name,digest in marker['artifacts'].items():assert sha(ROOT/name)==digest,name
    return marker

def require_previous(scale):
    index=SCALES.index(scale)
    if index:check_stage(SCALES[index-1])
