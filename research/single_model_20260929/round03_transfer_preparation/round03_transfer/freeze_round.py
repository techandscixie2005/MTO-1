"""Seal one fixed source and affine-transfer protocol after meaningful checks."""
import json
from clean_source import ROOT,SOURCE
from runtime import sha,atomic_json

def main():
    parent=json.loads((ROOT.parent/'FROZEN_MANIFEST.json').read_text());hashes=dict(parent['source_hashes'])
    for path,value in hashes.items():assert sha(path)==value
    for receipt_name in ('STATISTICS_AUDIT.json','PREFLIGHT.json','INFERENCE_PREFLIGHT.json'):
        receipt=json.loads((ROOT/receipt_name).read_text());assert receipt['passed']
        for path,value in receipt['source_hashes'].items():assert sha(path)==value
        hashes.update(receipt['source_hashes'])
    split=json.loads((ROOT/'split_manifest.json').read_text());assert split['passed']
    for record in split['arrays'].values():
        assert sha(record['path'])==record['sha256'];hashes[record['path']]=record['sha256']
    names=('clean_source.py','runtime.py','config.json','source_config.json','train_source.py',
        'predictor.py','evaluation.py','affine_stage.py','preflight.py','inference_preflight.py',
        'prepare_split_statistics.py','split_manifest.json','fit_normalization.json','STATISTICS_AUDIT.json',
        'PREFLIGHT.json','INFERENCE_PREFLIGHT.json','SPLIT_STATISTICS_HANDOFF.md','PROTOCOL.md',
        'freeze_round.py','launch_round.py','ROUND03_LAUNCH_AUTHORIZATION.md')
    hashes.update({str(ROOT/n):sha(ROOT/n) for n in names})
    hashes[str(SOURCE/'objective.py')]=sha(SOURCE/'objective.py')
    cfg=json.loads((ROOT/'config.json').read_text());pre=json.loads((ROOT/'PREFLIGHT.json').read_text())
    record={'config':cfg,'source_hashes':hashes,'parent_manifest_sha256':sha(ROOT.parent/'FROZEN_MANIFEST.json'),
        'initial_model_tensor_sha256':pre['initial_model_tensor_sha256'],'fixed_source_epoch':33,
        'split_manifest_sha256':sha(ROOT/'split_manifest.json'),'tail_thresholds':parent['tail_thresholds'],
        'source_fit_molecules':96284,'calibration_molecules':24071,'validation_molecules':6686,
        'no_outer_validation_during_source_fit':True,'no_model_prediction_averaging':True,
        'preflight_passed':True,'one_checkpoint_inference_passed':True}
    path=ROOT/'FROZEN_MANIFEST.json'
    if path.exists():assert json.loads(path.read_text())==record,'Refuse to change sealed protocol'
    else:atomic_json(record,path)
    print(json.dumps({'passed':True,'frozen_manifest_sha256':sha(path),'source_file_count':len(hashes)}))

if __name__=='__main__':main()
