"""Verify completed analysis bindings and aggregate arithmetic; no inference/array decoding."""
from pathlib import Path
import hashlib,json,math

ROOT=Path(__file__).resolve().parent
ARMS=('control','adapter','decorrelation','both')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(a,b)
def nested(a,b):
    if isinstance(a,dict):
        for k,v in a.items():nested(v,b[k])
    elif a is None:assert b is None
    elif isinstance(a,(int,str,bool)):assert a==b
    else:close(a,b)

def main():
    out=ROOT/'completion';receipt=read(out/'ANALYSIS_RECEIPT.json');summary=read(out/'ROUND05_RESULTS.json')
    assert receipt['passed'] and receipt['no_model_inference'] and receipt['no_test_or_raw_dataset_targets_opened']
    expected=read(ROOT/'TERMINAL_ANALYSIS_SOURCE_REVIEW.json')
    assert receipt['source_hashes']==expected['source_hashes'] and expected['passed']
    checked=0
    for field in ('source_hashes','input_hashes','output_hashes'):
        for p,h in receipt[field].items():assert sha(p)==h,p;checked+=1
    old=read(ROOT/'INDEPENDENT_TERMINAL_INTEGRITY.json');assert old['passed']
    assert old['frozen_manifest_sha256']==receipt['manifest_sha256']
    control=summary['arms']['control']
    for arm in ARMS:
        result=summary['arms'][arm]
        history=[json.loads(s) for s in (ROOT/'runs'/arm/'history.jsonl').read_text().splitlines()]
        assert result['selected_epoch']==old['arms'][arm]['selected_epoch']
        for policy,ep in (('selected_best',result['selected_epoch']),('fixed60',60)):
            saved=result[policy];logged=history[ep]['validation']
            for key in ('pooled','per_state','bright_tail','false_bright_bins','energy'):nested(logged[key],saved[key])
            nested(logged['base_objective'],saved['base_objective_from_training_log'])
            assert saved['pooled']['count']==66860
            assert saved['bright_tail']['q90']['count']==6782 and saved['bright_tail']['q99']['count']==708
            assert saved['bright_tail']['q90']['threshold']==.0549 and saved['bright_tail']['q99']['threshold']==.2406
            delta=saved['pooled']['r2']-control[policy]['pooled']['r2']
            comparison=summary['paired_uncertainty']['comparisons'][policy][arm]
            close(delta,comparison['delta_r2_vs_control'])
            lo,hi=comparison['descriptive_percentile_interval'];assert math.isfinite(lo) and math.isfinite(hi) and lo<=hi
            dist=saved['error_distribution'];assert dist['top_error_count']==669
            assert 0<=dist['top_error_sse_share']<=1
        for ep,snapshot in result['trajectory_snapshots'].items():
            assert snapshot=={k:v for k,v in history[int(ep)].items() if k!='order_sha256'}
        close(result['wall_training_validation_seconds'],sum(x['seconds'] for x in history))
        verification=summary['verification'][arm]
        assert verification['completed_epoch']==60 and verification['steps']==112860 and verification['validation_labels']==66860
        assert verification['saved_arrays_recomputed'] and verification['geometry_export_tensors_exactly_equal_selected_model']
        assert verification['test_numeric_rows']==0
    gate=summary['gate'];eligible=[a for a in ARMS[1:] if summary['arms'][a]['selected_best']['pooled']['r2']-control['selected_best']['pooled']['r2']>=.003]
    assert gate=={'threshold':.003,'eligible_noncontrol_arms':eligible,'passed':bool(eligible),'automatic_extension_authorized':False}
    assert not eligible
    for policy in ('selected_best','fixed60'):
        v={a:summary['arms'][a][policy]['pooled']['r2'] for a in ARMS}
        close(v['both']-v['adapter']-v['decorrelation']+v['control'],summary['factorial_interaction_r2'][policy])
    uncertainty=summary['paired_uncertainty']
    assert uncertainty['draws']==2000 and uncertainty['seed']==20260930 and uncertainty['component_groups']==5656 and uncertainty['molecules']==6686
    assert summary['model_inference'] is False and summary['test_access'] is False
    result={'passed':True,'phase':'completed_saved_analysis_independent_aggregate_review',
        'analysis_receipt_sha256':sha(out/'ANALYSIS_RECEIPT.json'),'analysis_result_sha256':sha(out/'ROUND05_RESULTS.json'),
        'analysis_source_review_sha256':sha(ROOT/'TERMINAL_ANALYSIS_SOURCE_REVIEW.json'),
        'terminal_integrity_sha256':sha(ROOT/'INDEPENDENT_TERMINAL_INTEGRITY.json'),
        'source_input_output_hashes_verified':checked,'all8_reported_metric_groups_match_independent_history_audit':True,
        'all_trajectory_snapshots_and_gate_arithmetic_match':True,'any_noncontrol_meets_point003_gate':False,
        'checkpoint_and_prediction_files_opaque_hashed_only_by_reviewer':True,
        'bootstrap_not_recomputed_by_reviewer':True,'bootstrap_review':'Source formula/resampling-unit and aggregate receipt review; conditional reused-validation uncertainty, not training-seed or selection-corrected uncertainty.',
        'no_array_decoding_or_model_inference_or_fitting_by_reviewer':True,'reviewer_script_sha256':sha(__file__),'blocking_findings':[]}
    p=out/'INDEPENDENT_RESULT_CHECKS.json';assert not p.exists();p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':True,'review_sha256':sha(p),'hash_entries_verified':checked},sort_keys=True))

if __name__=='__main__':main()
