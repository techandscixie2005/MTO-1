"""Seal round02 code, fixed objective, preflight and cache provenance."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from freeze import sha,current_hashes

def main():
    cfg=json.loads((ROOT/'config.json').read_text());pc=json.loads((PARENT/'round_config.json').read_text())
    parent=json.loads((PARENT/'FROZEN_MANIFEST.json').read_text())
    hashes=current_hashes(pc);assert hashes==parent['source_hashes']
    pre=json.loads((ROOT/'PREFLIGHT.json').read_text());assert pre['passed']
    for path,expected in pre['source_hashes'].items():assert sha(path)==expected
    assert sha(ROOT/'CACHE_COMPLETE.json')==pre['cache_receipt_sha256']
    resume=json.loads((ROOT/'RESUME_PREFLIGHT.json').read_text());assert resume['passed']
    for path,expected in resume['source_hashes'].items():assert sha(path)==expected
    inference=json.loads((ROOT/'INFERENCE_PREFLIGHT.json').read_text());assert inference['passed']
    for path,expected in inference['source_hashes'].items():assert sha(path)==expected
    variance=json.loads((ROOT/'VARIANCE_PROVENANCE.json').read_text());assert variance['passed']
    for path,expected in variance['source_hashes'].items():assert sha(path)==expected
    hashes.update(variance['source_hashes'])
    names=('config.json','model_frozen.py','cache_access.py','prepare_cache.py','preflight.py','train_frozen.py',
        'freeze_round.py','launch_round.py','PROTOCOL.md','CACHE_COMPLETE.json','PREFLIGHT.json',
        'resume_preflight.py','RESUME_PREFLIGHT.json','variance_provenance.py','VARIANCE_PROVENANCE.json',
        'predictor.py','inference_preflight.py','INFERENCE_PREFLIGHT.json')
    hashes.update({str(ROOT/name):sha(ROOT/name) for name in names})
    record={'config':cfg,'source_hashes':hashes,'parent_manifest_sha256':sha(PARENT/'FROZEN_MANIFEST.json'),
        'cache_receipt_sha256':sha(ROOT/'CACHE_COMPLETE.json'),'train_count':parent['train_count'],
        'validation_count':parent['validation_count'],'tail_thresholds':parent['tail_thresholds'],
        'train_order_indices_sha256':parent['train_order_indices_sha256'],
        'validation_indices_sha256':parent['validation_indices_sha256'],
        'preflight_passed':True,'resume_preflight_passed':True,'inference_preflight_passed':True,'geometry_only_inference':True}
    path=ROOT/'FROZEN_MANIFEST.json'
    if path.exists():assert json.loads(path.read_text())==record,'Refusing to change frozen round'
    else:path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'passed':True,'manifest_sha256':sha(path),'source_file_count':len(hashes)}))

if __name__=='__main__':main()
