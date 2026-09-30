"""Dataset inventory and existing split overlap audit; identifiers/metadata only."""
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/home/inspur/MTO-1')
DATA = PROJECT/'experiments/qm9s_full_EA_20260925/data'
EXTRACTION = Path('/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {'split': DATA/'splits.json', 'identities': DATA/'identity_audit_v2.json',
             'extraction_metadata': EXTRACTION/'dataset_manifest.json'}
    assert sha(paths['split']) == '141d6c4ba9631de91b81076b56afc2239aa3632478895102d8024e13d79c93ca'
    assert sha(paths['identities']) == '9d384425a90dd88fbc68f8b609a303872910bb21c69e0d0a19bcccaee49cb3c4'
    split = json.loads(paths['split'].read_text())
    identities = json.loads(paths['identities'].read_text())
    metadata = json.loads(paths['extraction_metadata'].read_text())
    parts = split['indices']
    assert sorted(sum(parts.values(), [])) == list(range(len(identities)))
    ids = {k: {identities[i][0] for i in v} for k, v in parts.items()}
    groups = {k: {identities[i][1] for i in v} for k, v in parts.items()}
    overlap = {a+'__'+b: {'row_overlap': len(set(parts[a]) & set(parts[b])),
        'id_overlap': len(ids[a] & ids[b]), 'group_overlap': len(groups[a] & groups[b])}
        for a, b in itertools.combinations(parts, 2)}
    assert all(all(v == 0 for v in pair.values()) for pair in overlap.values())
    inventory = []
    for experiment in sorted((PROJECT/'experiments').iterdir()):
        if experiment.is_dir() and (experiment/'data/dataset.npz').is_file():
            file = experiment/'data/dataset.npz'
            inventory.append({'path': str(file), 'bytes': file.stat().st_size,
                              'targets_or_arrays_opened': False})
    fields = ('route_section_distribution', 'program_version_distribution',
              'rpa_flag_distribution', 'coordinate_frame_distribution',
              'charge_multiplicity_distribution', 'state_symmetry_distribution',
              'units', 'label_policy', 'coordinate_policy', 'connectivity')
    result = {'passed': True, 'scope': 'known project and known QM9S source metadata only',
        'no_target_or_prediction_array_decoded': True,
        'new_external_test_dataset_verified': False,
        'known_qm9s_area_entries': sorted(p.name for p in EXTRACTION.parent.iterdir()),
        'experiment_dataset_file_inventory': inventory,
        'same_size_is_not_asserted_as_independent_content_hash_proof': True,
        'existing_split_counts': split['counts'], 'existing_group_counts': {k: len(v) for k, v in groups.items()},
        'existing_duplicate_groups': split['duplicate_group_count'],
        'unresolved_rows_by_partition': {k: sum(not identities[i][2] for i in v) for k, v in parts.items()},
        'existing_pairwise_overlap': overlap,
        'chemical_identity_limitation': 'Zero overlap under the audited conservative connectivity keys, not proof that heuristic bond perception identifies every chemical identity.',
        'historical_exposure': {'all_existing_splits': 'historically used dataset',
            'old_validation': 'repeated tuning/calibration/selection exposure',
            'old_test': 'historical scores and diagnostics already exposed',
            'new_partition_of_this_dataset': 'cannot be called an independent external dataset'},
        'historical_qc_metadata': {k: metadata[k] for k in fields},
        'validity_metadata': {k: metadata[k] for k in ('source_log_count', 'normal_termination_count',
            'response_convergence_reported_count', 'complete_geometry_count', 'complete_successful_scalar_records')},
        'source_hashes': {str(p): sha(p) for p in [*paths.values(), Path(__file__)]}}
    (ROOT/'EXISTING_DATASET_AUDIT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'counts': split['counts'], 'pairwise_overlap': overlap,
                      'new_external_test_verified': False}, indent=2))


if __name__ == '__main__':
    main()
