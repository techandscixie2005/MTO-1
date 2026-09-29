"""Inspect existing extraction code/metadata only; no raw Gaussian logs."""
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest = json.loads((SOURCE / 'dataset_manifest.json').read_text())
    spotcheck = json.loads((SOURCE / 'source_spotcheck_report.json').read_text())
    # Extract only six metadata tokens from an existing record. Do not decode
    # its numeric target arrays, retrieve a log, or emit an ID table.
    with gzip.open(SOURCE / 'records/part-00000.jsonl.gz', 'rt') as stream:
        line = next(stream)
    sample = {}
    for name in ['basis_printed', 'coordinate_frame', 'orientation_table_count_before_labels',
                 'electric_dipole_table_count', 'unambiguous_single_calculation']:
        match = re.search('"' + name + '"\\s*:\\s*("[^"\\n]*"|[0-9]+|true|false|null)', line)
        assert match
        sample[name] = json.loads(match.group(1))
    result = {
        'status': 'shared_Input_frame_supported_by_provenance_but_no_direct_frame_alignment_test',
        'no_evidence_of_frame_mismatch': True,
        'route_distribution': manifest['route_section_distribution'],
        'coordinate_frame_distribution': manifest['coordinate_frame_distribution'],
        'unambiguous_single_calculation_count': manifest['unambiguous_single_calculation_count'],
        'extractor_geometry_policy': 'last Standard before first label if any, otherwise last pre-label orientation; unambiguous valid record requires exactly one pre-label geometry table',
        'electric_vector_policy': 'copy labeled electric XYZ; replace only with electronic-transition rows2/3/4 when every component agrees within5.1e-5; no rotation',
        'velocity_vector_policy': 'copy labeled velocity XYZ without rotation; no separate vector-frame label',
        'downstream_curation': 'prepare_round.py asserts all curated positions and A equal extracted values after FP32 conversion; dataset.py passes them through without rotation. This is reviewed historical code/provenance, not a rerun of all-data assertions.',
        'historical_spotcheck': {'files': spotcheck['checked_source_files'], 'states': spotcheck['checked_source_states'],
                               'checked': 'Input coordinates and electric table components separately against original logs',
                               'not_checked': 'rotation identity tying vector axes to geometry; velocity orientation'},
        'bounded_existing_record_metadata_sample': sample,
        'basis_sample_does_not_prove_all_record_basis_conventions': True,
        'conclusion': 'No parser rotation or mixed-geometry join is apparent. All complete records use Input coordinates and routes include nosymm, consistent with a shared input convention. Code and magnitude reconstruction alone do not independently certify vector-to-geometry alignment.',
        'interpretation': 'Do not infer a mismatch from performance or claim full frame verification. Current eta0 LE+Ls uses trace(A), which is unchanged by a pure orthogonal frame rotation; future directional A or vector objectives need stronger provenance.',
        'no_raw_log_acquisition_model_fit_or_data_edits': True,
        'source_hashes': {str(SOURCE / name): sha(SOURCE / name) for name in
                          ['dataset_manifest.json', 'extract_qm9s.py', 'verify_source_samples.py', 'source_spotcheck_report.json']},
        'script_sha256': sha(Path(__file__)),
    }
    experiment = Path('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926')
    result['source_hashes'].update({str(experiment / name): sha(experiment / name) for name in
                                    ['prepare_round.py', 'dataset.py', 'reports/data_audit.json']})
    (ROOT / 'FRAME_PROVENANCE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'metadata_sample': sample}))


if __name__ == '__main__':
    main()
