"""Prove previously exercised design/inference mathematics unchanged; no data/model imports."""
import ast
from common import ROOT,sha,atomic_json


def function_ast(path,name):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return ast.dump(node,include_attributes=False)


def main():
    old=ROOT/'stub_preparation_snapshot'
    exact=('audit_design.py','inference_preflight.py','predictor.py','backbone.py')
    identical={name:sha(ROOT/name)==sha(old/name) for name in exact}
    assert all(identical.values())
    functions={}
    for name in ('basis','design_diagnostics','require_design','apply_map','affine_parameters'):
        functions['scalar_map.'+name]=function_ast(ROOT/'scalar_map.py',name)==function_ast(old/'scalar_map.py',name)
    for name in ('require_cpu','sha','read_json','source_hashes','settings','require_preparation_authority'):
        functions['common.'+name]=function_ast(ROOT/'common.py',name)==function_ast(old/'common.py',name)
    assert all(functions.values())
    result={'passed':True,'array_or_model_access':False,'exact_unchanged_files':identical,
        'exact_unchanged_function_AST':functions,
        'explained_changes':['atomic_json now fsyncs output file and directory',
            'solver extracted a shared function; current synthetic recovery tests exercise it',
            'production runner/settings were added behind separate external execution authority'],
        'design_receipt_sha256':sha(old/'DESIGN_AUDIT.json'),
        'inference_receipt_sha256':sha(old/'INFERENCE_PREFLIGHT.json'),
        'settings_basis_and_solver_unchanged':True,
        'source_hashes':{name:sha(ROOT/name) for name in ('check_preflight_continuity.py',*exact,'scalar_map.py','common.py','settings.json')}}
    import json
    previous=json.loads((old/'settings.json').read_text());current=json.loads((ROOT/'settings.json').read_text())
    for key in ('scale','knots','parameter_count','dtype','solver','rcond','maximum_condition_number','minimum_rank',
                'num_threads','fitting_objective','application','calibration_array_sha256','calibration_index_bytes_sha256',
                'calibration_molecules','calibration_valid_labels','historical_zero_targets_from_pinned_receipt',
                'base_checkpoint_sha256','base_tensor_sha256'):
        assert current[key]==previous[key]
    atomic_json(result,ROOT/'PREFLIGHT_CONTINUITY.json');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
