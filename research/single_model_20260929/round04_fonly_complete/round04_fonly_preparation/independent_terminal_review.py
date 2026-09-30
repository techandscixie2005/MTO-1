"""Read-only terminal audit: hashes and aggregates; no prediction/label decode or inference."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/inspur/MTO-1/research/single_model_20260929/round04_fonly_preparation')
P = ROOT / 'production'
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def near(a,b): assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10), (a,b)
def main():
    manifest=read(ROOT/'FROZEN_MANIFEST.json')
    assert len(manifest['source_hashes'])==51
    for name,h in manifest['source_hashes'].items(): assert sha(name)==h,name
    terminal=read(ROOT/'ROUND04_TERMINAL_MANIFEST.json')
    for name,h in terminal['input_hashes'].items(): assert sha(ROOT/name)==h,name
    v=read(P/'VALIDATION_COMPLETE.json');c=read(P/'COEFFICIENTS_FROZEN.json');e=read(P/'EXPORT_COMPLETE.json');r=read(P/'POSTFIT_GEOMETRY_REPLAY.json')
    permit=v['permit']
    bindings={'authorization_sha256':ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json','source_manifest_sha256':ROOT/'FROZEN_MANIFEST.json','independent_review_sha256':ROOT/'INDEPENDENT_PREPARATION_REVIEW.json','publication_receipt_sha256':ROOT.parent/'ops/ROUND04_PREPARATION_PUBLICATION_RECEIPT.json'}
    for key,path in bindings.items(): assert sha(path)==permit[key]
    assert c['permit']==e['permit']==r['permit']==permit
    attempts=list(P.glob('VALIDATION_ATTEMPT_*.json'));assert len(attempts)==1
    a=read(attempts[0]);assert a['permit']==permit
    assert len(list(P.glob('*_SOLVE_STARTED.json')))==len(list(P.glob('*_SOLVED.json')))==2
    assert v['coefficient_receipt_sha256']==e['coefficient_receipt_sha256']==a['coefficient_receipt_sha256']==sha(P/'COEFFICIENTS_FROZEN.json')
    assert v['export_receipt_sha256']==r['export_receipt_sha256']==sha(P/'EXPORT_COMPLETE.json')
    assert v['validation_attempt_sha256']==sha(attempts[0])
    assert r['passed'] and r['base_unchanged']
    binary_hashes={}
    for arm in ('in_sample','heldout_source'):
        start=read(P/f'{arm}_SOLVE_STARTED.json');solved=read(P/f'{arm}_SOLVED.json')
        assert start['permit']==solved['permit']==permit and solved==c['coefficients'][arm]
        assert start['started_at_unix']<solved['solved_at_unix']<c['frozen_at_unix']<a['started_at_unix']<v['completed_at_unix']
        assert solved['shape']==[4] and solved['dtype']=='float64'
        assert solved['design']['rows']==240710 and solved['design']['rank']==4 and solved['design']['condition_number']<=1e8
        assert solved['design']['rcond']==1e-12
        for entry in (solved,e['exports'][arm]):
            assert sha(entry['path'])==entry['sha256'];binary_hashes[entry['path']]=entry['sha256']
        assert e['exports'][arm]['coefficient_tensor_sha256']==solved['sha256']
        assert r['replay'][arm]['native_max_abs']<2e-6 and r['replay'][arm]['map_max_abs']<1e-12
    completed=list((ROOT/'ops/execution').glob('*/COMPLETE.json'));assert len(completed)==2
    stage_records={}
    for path in completed:
        done=read(path);launch=read(path.parent/'LAUNCH_RECEIPT.json');registration=read(path.parent/'MONITOR_REGISTRATION.json')
        assert done['exit_code']==0 and done['authorization_unchanged'] and done['permit']==launch['permit']==permit
        assert done['child_identity']==launch['child_identity'] and done['child_pid']==launch['child_pid']
        for key in ('pid','start_ticks','uid','cwd','argv_sha256','boot_id'): assert registration['identity'][key]==launch['wrapper_identity'][key]
        assert registration['resource_kind']=='cpu' and registration['expected_gpu_allocations']==0
        assert launch['environment']=={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        registered=dt.datetime.fromisoformat(registration['registered_at_utc']).timestamp()
        assert registered<launch['child_started_at_unix']
        assert sha(path.parent/'stage.log')==done['log_sha256']
        for name,h in done['output_receipt_hashes'].items(): assert sha(P/name)==h
        assert not (path.parent/'FAILED.json').exists()
        stage_records[done['stage']]={'wrapper_pid':launch['wrapper_identity']['pid'],'child_pid':done['child_pid'],'start':done['started_at_unix'],'end':done['ended_at_unix'],'exit_code':0}
    assert set(stage_records)=={'fit_export','evaluate'}
    assert stage_records['fit_export']['end']<stage_records['evaluate']['start']
    metrics=v['metrics'];assert len(metrics)==6
    for m in metrics.values():
        pooled=m['pooled']; assert pooled['count']==66860
        near(pooled['sst'],158.4062371581119)
        assert len(m['per_state'])==10 and all(x['count']==6686 for x in m['per_state'])
        assert sum(x['count'] for x in m['brightness_bins'].values())==66860
        near(sum(x['sse'] for x in m['per_state']),pooled['sse']);near(sum(x['sse'] for x in m['brightness_bins'].values()),pooled['sse'])
        for score in [pooled]+m['per_state']+list(m['true_bright_tail'].values())+list(m['brightness_bins'].values()):
            near(score['rmse']**2,score['sse']/score['count'])
            if 'r2' in score: near(score['r2'],1-score['sse']/score['sst'])
        for q,n,t in [('q90',6650,.0549),('q99',639,.2412)]:
            assert m['true_bright_tail'][q]['count']==n and m['true_bright_tail'][q]['threshold']==t
    for arm in ('in_sample','heldout_source'):
        own=metrics['spline_'+arm]['pooled']['r2']-metrics[arm]['pooled']['r2']
        incumbent=metrics['spline_'+arm]['pooled']['r2']-metrics['historical_validation_fit']['pooled']['r2']
        near(v['deltas'][arm]['vs_own_affine'],own);near(v['deltas'][arm]['vs_incumbent'],incumbent)
        assert v['confirmation_eligible'][arm]==(own>=.003 and incumbent>=.003)==False
    contrast=metrics['spline_heldout_source']['pooled']['r2']-metrics['spline_in_sample']['pooled']['r2']
    near(contrast,v['heldout_source_advantage_delta']);assert v['heldout_source_advantage_threshold_passed']==(contrast>=.003)
    near(v['within_source_increment_interaction'],v['deltas']['heldout_source']['vs_own_affine']-v['deltas']['in_sample']['vs_own_affine'])
    assert v['selected_by_validation_sse']==min(metrics,key=lambda k:metrics[k]['pooled']['sse'])=='historical_validation_fit'
    assert v['test_access'] is False and v['new_model_forward_for_validation'] is False and v['historically_reused_validation'] is True
    result={'passed':True,'reviewer':'history_baseline','scope':'Current source/terminal/private-file hashes and independent aggregate arithmetic; no target/prediction arrays decoded, no model loading, no inference or coefficient refit.', 'source_manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json'),'terminal_manifest_sha256':sha(ROOT/'ROUND04_TERMINAL_MANIFEST.json'),'report_sha256':sha(ROOT/'ROUND04_REPORT.md'),'review_script_sha256':sha(__file__),'source_entries_verified':51,'terminal_entries_verified':len(terminal['input_hashes']),'permit':permit,'private_file_hashes_only':binary_hashes,'stages':stage_records,'two_solves_one_evaluation':True,'all_66860_labels_and_state_tail_bin_arithmetic_verified':True,'confirmation_eligible':v['confirmation_eligible'],'incumbent_retained':True,'limits':['Private checkpoint schemas/base equality rely on reviewed unchanged execution assertions plus passed execution/parity receipts; this audit hashes files without reloading models.','No fresh holdout evidence; old validation reused and historical affine incumbent was fitted on it.','Current scientific conclusion is a negative result for the fixed hinge protocol, not universal failure of nonlinear models.']}
    (ROOT/'INDEPENDENT_TERMINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'receipt_sha256':sha(ROOT/'INDEPENDENT_TERMINAL_INTEGRITY.json'),'source_entries':51,'terminal_entries':len(terminal['input_hashes'])}))
if __name__=='__main__':main()
