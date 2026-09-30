"""Seal preparation sources/receipts; never fit, infer or decode any data."""
from pathlib import Path
from common import ROOT,PARENT,ROUND03,SOURCE,read_json,sha,atomic_json


def main():
    if (ROOT/'production').exists():raise RuntimeError('Unexpected production artifacts during preparation')
    names=['common.py','scalar_map.py','backbone.py','predictor.py','execution_gate.py','production_entry.py',
        'production_stages.py','stage_state.py','synthetic_checks.py','audit_design.py','inference_preflight.py',
        'production_guard_checks.py','independent_gate_checks.py','check_preflight_continuity.py','freeze_preparation.py','settings.json',
        'PROPOSED_PROTOCOL.md','RATIONALE.md','PREPARATION_AUTHORIZATION.md','PREPARATION_PHASE.md','RUNNER_HANDOFF.md',
        'INDEPENDENT_PROTOCOL_REVIEW.json','INDEPENDENT_PROTOCOL_REVIEW.md']
    closure={str(ROOT/name):sha(ROOT/name) for name in names}
    previous=read_json(ROUND03/'FROZEN_MANIFEST.json')['source_hashes']
    for path in (ROUND03/'FROZEN_MANIFEST.json',ROUND03/'affine/COEFFICIENTS_FROZEN.json',
                 ROUND03/'ROUND03_RESULTS.json',ROUND03/'split_manifest.json',
                 PARENT/'ops/ROUND03_COMPLETION_PUBLICATION_RECEIPT.json'):
        closure[str(path)]=sha(path)
    for name,digest in previous.items():
        path=Path(name)
        if '/frozen_reference/' in name or name in (str(SOURCE/'configs/mto_eta0.json'),str(SOURCE/'data/normalization.json'),
            str(PARENT/'reports/velocity_audit/audit_velocity_labels.py'),str(PARENT/'ENVIRONMENT.json')):
            assert sha(path)==digest;closure[name]=digest
    receipt_names=['SYNTHETIC_CHECKS.json','DESIGN_AUDIT.json','INFERENCE_PREFLIGHT.json',
                   'PRODUCTION_GUARD_CHECKS.json','PREFLIGHT_CONTINUITY.json','INDEPENDENT_GATE_CHECKS.json']
    preflight={}
    for name in receipt_names:
        receipt=read_json(ROOT/name);assert receipt['passed'] is True
        preflight[name]=sha(ROOT/name)
    cfg=read_json(ROOT/'settings.json')
    references={str(SOURCE/'runs/mto_eta0/best.pt'):cfg['base_checkpoint_sha256'],
        str(SOURCE/'data/dataset.npz'):previous[str(SOURCE/'data/dataset.npz')],
        str(ROUND03/'affine/calibration_predictions.npz'):cfg['calibration_array_sha256'],
        str(ROUND03/'affine/validation_predictions.npz'):cfg['validation_array_sha256']}
    # These hashes are inherited from completed audited receipts; values are not decoded here.
    result={'phase':'preparation_only','production_execution_authorized':False,'source_hashes':closure,
        'binary_input_hash_references':references,'binary_reference_scope':'No validation values decoded; actual hashes rechecked by future execution gate',
        'preflight_receipt_hashes':preflight,'previous_completed_commit':'4f9ae50647f957ac7d68ab1a20200a79fa527cf9',
        'documents_only_snapshot':{p.name:sha(p) for p in sorted((ROOT/'documents_only_snapshot').iterdir()) if p.is_file()},
        'stub_preparation_snapshot':{p.name:sha(p) for p in sorted((ROOT/'stub_preparation_snapshot').iterdir()) if p.is_file()},
        'actual_real_data_solves':0,'actual_validation_comparisons':0,'production_artifacts_present':False}
    atomic_json(result,ROOT/'FROZEN_MANIFEST.json')
    print('Source entries:',len(closure),'Manifest SHA256:',sha(ROOT/'FROZEN_MANIFEST.json'))


if __name__=='__main__':main()
