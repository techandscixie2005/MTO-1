"""Synthetic negative-gate/recovery checks; no real target/cache/model inference."""
from common import require_cpu
require_cpu()
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from common import ROOT,atomic_json,sha,settings,source_hashes,require_preparation_authority
from production_entry import main as production_main
from execution_gate import check_bindings
from stage_state import solve_action,evaluation_action


def rejected(function,error):
    try:function()
    except error:return True
    raise AssertionError('Expected rejection')


def main():
    require_preparation_authority();torch.set_num_threads(2)
    failures={};module_before='production_stages' in sys.modules
    missing=[str(ROOT/'ABSENT_AUTHORITY.json'),str(ROOT/'ABSENT_MANIFEST.json'),str(ROOT/'ABSENT_REVIEW.json'),str(ROOT/'ABSENT_PUBLICATION.json')]
    argv=['fit_export','--authorization',missing[0],'--manifest',missing[1],'--review',missing[2],'--publication',missing[3]]
    failures['missing_authority_before_stage_import']=rejected(lambda:production_main(argv),PermissionError)
    assert ('production_stages' in sys.modules)==module_before
    with tempfile.TemporaryDirectory(prefix='mto_round04_synthetic_guards_') as directory:
        tmp=Path(directory)
        for name in ('authorization','manifest','review','publication'):
            atomic_json({'synthetic_fixture':True,'phase':'preparation_only'},tmp/(name+'.json'))
        argv=['evaluate']
        for name in ('authorization','manifest','review','publication'):argv+=['--'+name,str(tmp/(name+'.json'))]
        before=sorted(p.name for p in tmp.iterdir())
        failures['synthetic_or_preparation_authority_rejected']=rejected(lambda:production_main(argv),PermissionError)
        assert sorted(p.name for p in tmp.iterdir())==before
        assert ('production_stages' in sys.modules)==module_before
    from production_stages import fit_export,evaluate
    import production_stages as stages
    actions={'new':solve_action(False,False,False),'recover_receipt':solve_action(True,True,False),
             'retain':solve_action(True,True,True)}
    failures['uncertain_interrupted_solve_not_repeated']=rejected(lambda:solve_action(True,False,False),RuntimeError)
    failures['completed_scoring_refused']=rejected(lambda:evaluation_action(True,False),RuntimeError)
    failures['interrupted_scoring_requires_review']=rejected(lambda:evaluation_action(False,True),RuntimeError)
    # Private synthetic coefficient files only; real input access/solver/export are replaced by hard failures/spies.
    calls=[];permit={'synthetic_recovery_fixture':True}
    with tempfile.TemporaryDirectory(prefix='mto_round04_synthetic_recovery_') as directory:
        tmp=Path(directory)
        for arm in stages.ARMS:
            record={'arm':arm,'permit':permit,'coefficients':torch.tensor([.001,.05,-.01,.002],dtype=torch.float64),
                'calibration_array_sha256':settings()['calibration_array_sha256'],
                'design_diagnostics':{'synthetic':True},'fitting_diagnostics':{'synthetic':True},'solved_at_unix':1.}
            torch.save(record,tmp/('coefficients_'+arm+'.pt'))
        hashes={arm:sha(tmp/('coefficients_'+arm+'.pt')) for arm in stages.ARMS}
        with (patch.object(stages,'OUT',tmp),
              patch.object(stages,'load_calibration',side_effect=AssertionError('Real cache forbidden')),
              patch.object(stages,'solve_fixed',side_effect=AssertionError('A second solve is forbidden')),
              patch.object(stages,'export_and_replay',side_effect=lambda *args:calls.append('export_recovery'))):
            fit_export(permit)
            frozen=sha(tmp/'COEFFICIENTS_FROZEN.json')
            fit_export(permit)
            assert sha(tmp/'COEFFICIENTS_FROZEN.json')==frozen
            assert hashes=={arm:sha(tmp/('coefficients_'+arm+'.pt')) for arm in stages.ARMS}
            atomic_json({'synthetic_complete':True},tmp/'VALIDATION_COMPLETE.json')
            failures['actual_completed_evaluator_refuses_before_input_reads']=rejected(lambda:evaluate(permit),RuntimeError)
        assert calls==['export_recovery','export_recovery']
    result={'passed':True,'synthetic_fixtures_only':True,'real_data_solve':False,
            'real_cache_or_validation_values_opened':False,'model_inference':False,
            'negative_gate_and_state_checks':failures,'solve_state_decisions':actions,
            'immutable_synthetic_coefficients_preserved':True,'partial_export_resumed_without_solve':True,
            'temporary_fixture_files_removed':True,
            'source_hashes':source_hashes(['production_guard_checks.py','production_entry.py','production_stages.py',
                'stage_state.py','execution_gate.py','common.py','settings.json'])}
    atomic_json(result,ROOT/'PRODUCTION_GUARD_CHECKS.json');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
