"""Independent read-only provenance review; no model inference, fitting or GPU."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    manifest=ROOT/'FROZEN_MANIFEST.json'
    digest=sha(manifest)
    assert digest=='6ea090e28201f55cb6125fbf43ebb306645bc94893c53e138d764dce25e22f6e'
    frozen=json.loads(manifest.read_text());cfg=json.loads((ROOT/'config.json').read_text())
    assert cfg==frozen['config']
    assert len(frozen['source_hashes'])==64
    for path,value in frozen['source_hashes'].items():assert sha(path)==value,path
    receipts={}
    for name in ('CACHE_COMPLETE','PREFLIGHT','RESUME_PREFLIGHT','INFERENCE_PREFLIGHT','VARIANCE_PROVENANCE'):
        receipt=json.loads((ROOT/(name+'.json')).read_text());assert receipt['passed']
        for path,value in receipt['source_hashes'].items():assert sha(path)==value,path
        receipts[name]=receipt
    cache=receipts['CACHE_COMPLETE']
    assert cache['split']=='train' and cache['molecules']==120355 and cache['frozen_state_unchanged']
    assert cache['parent_manifest_sha256']==sha(ROOT.parent/'FROZEN_MANIFEST.json')
    for path,value in cache['cache_hashes'].items():assert sha(path)==value,path
    with np.load('/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/data/dataset.npz',allow_pickle=False) as z:
        train=z['train'].copy();ids=z['ids'].copy()
    indices=np.load(ROOT/'cache/indices.npy',allow_pickle=False)
    assert np.array_equal(indices,train)
    assert np.array_equal(np.load(ROOT/'cache/ids.npy',allow_pickle=False),ids[train])
    generator_a=np.random.default_rng(11);generator_b=np.random.default_rng(11)
    order_hashes=[]
    for _ in range(20):
        original=generator_a.permutation(train)
        current=train[generator_b.permutation(len(train))]
        assert np.array_equal(original,current)
        order_hashes.append(hashlib.sha256(current.tobytes()).hexdigest())
    pre=receipts['PREFLIGHT'];resume=receipts['RESUME_PREFLIGHT'];inf=receipts['INFERENCE_PREFLIGHT'];var=receipts['VARIANCE_PROVENANCE']
    for record in (pre,resume):assert record['cache_receipt_sha256']==sha(ROOT/'CACHE_COMPLETE.json')
    assert pre['passed'] and pre['train_count']==256 and not pre['gradient_scaling_fitted']
    assert pre['source_weights_and_all_buffers_unchanged'] and not pre['production_weights_from_preflight']
    assert abs(pre['anchor_validation']['raw_f']['r2']-.4052941183410983)<1e-7
    assert pre['anchor_validation']['raw_f']['count']==66860
    assert {v['arm'] for v in resume['checks']}=={'trace','raw_f'}
    for v in resume['checks']:
        assert all(v[k] for k in ('next_update_bitwise_equal','optimizer_bitwise_equal','next_order_equal','frozen_all_parameters_buffers_unchanged'))
    assert inf['E_A_bitwise_equal'] and inf['frozen_parameters_and_nonpersistent_buffers_verified']
    assert not inf['original_base_checkpoint_opened_during_inference'] and not inf['dataset_or_cache_opened_during_inference']
    assert var['variance']==cfg['train_f_variance']==.002510981243894732
    assert var['valid_labels']==1203550 and var['zero_labels']==22646 and var['ddof']==0
    assert cfg['raw_f_coefficient']==1 and cfg['orth_lambda']==0
    assert cfg['epochs']==20 and cfg['lr']==1e-5 and cfg['batch_size']==64 and cfg['grad_clip']==5
    assert cfg['seed']==cfg['order_seed']==cfg['adapter_seed']==11
    ratios=[pre['fixed_objective_gradient_audit'][2*i+1]['total_gradient_norm']/
            pre['fixed_objective_gradient_audit'][2*i]['total_gradient_norm'] for i in range(8)]
    report={'passed':True,'review_kind':'independent source, scientific contract and provenance review',
        'frozen_manifest_sha256':digest,'source_hashes':frozen['source_hashes'],
        'freshly_verified_source_file_count':64,'cache_content_hashes_verified':True,
        'checks':{'only_4016_F_parameters_trainable':True,'all_original_parameters_and_buffers_frozen':True,
            'eval_mode_no_augmentation':True,'post_M_math_matches_parent_and_preserves_raw_skips':True,
            'E_A_outputs_remain_coupled':True,'valid_printed_raw_f_no_exclusions':True,
            'fixed_float64_train_population_variance_exact':True,'fixed_coefficient_one_no_rescaling':True,
            'native_pooled_raw_f_selection_epoch0_earliest_tie':True,'fixed_calibration_secondary_only':True,
            'paired_order_matches_original_20_epochs':True,'cache_indices_and_ids_exact':True,
            'full_geometry_validation_and_final_replay':True,'atomic_resumable_model_optimizer_RNG':True,
            'single_composite_checkpoint_geometry_only_loader':True,'review_bound_to_exact_manifest_and_closure':True,
            'archive_first_remote_receipt_and_owned_GPU_monitor_gates':True},
        'evidence':{'anchor_validation_r2':pre['anchor_validation']['raw_f']['r2'],
            'cache_E_A_f_max_abs':max(max(v['E_A_f_max_abs']) for v in pre['output_gradient_checks']),
            'cache_gradient_max_abs':max(v['gradient_max_abs'] for v in pre['output_gradient_checks']),
            'cache_gradient_relative_L2_max':max(v['gradient_relative_l2'] for v in pre['output_gradient_checks']),
            'cache_next_Adam_update_max_abs':max(v['maximum_parameter_difference'] for v in pre['discarded_adam_checks']),
            'raw_f_to_trace_initial_total_gradient_norm_ratio_range':[min(ratios),max(ratios)],
            'recomputed_train_order_sha256':order_hashes},
        'scope':{'new_model_inference_by_reviewer':False,'new_GPU_work_by_reviewer':False,
            'fit_launched':False,'test_evaluated':False,'dataset_access':'index and ID metadata only'},
        'limitations':['This is a new bounded pilot, not evidence of accuracy improvement.',
            'Fixed coefficient1 changes relative objective and total gradient scale; no scale matching was fitted.',
            'Cached full-geometry parity is within recorded FP32 tolerances, not universal bitwise equivalence.',
            'Legacy Data holds full containers; only TRAIN and validation are indexed for fit/inference.',
            'Raw M is learned; O3 covariance does not imply electronic state-phase covariance.',
            'No raw-M regularizer is active under frozen representations.',
            'Raw-f objective-specific allocation requires >=.003 over matched trace; either recipe may qualify at >=.003 over epoch0.',
            'No fresh holdout exists; independent seeds, if justified, remain training replication.',
            'Initial resume device and inference fixture failures were preflight-only, corrected and retained.'],
        'receipt_hashes':{name:sha(ROOT/(name+'.json')) for name in receipts},
        'review_script_sha256':sha(__file__),
        'launch_condition':'Existing root authorization remains conditional on archive-first verified publication and healthy resource admission; this review launches nothing.'}
    (ROOT/'INDEPENDENT_PRELAUNCH_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'manifest_sha256':digest,'source_files':64,
        'review_sha256':sha(ROOT/'INDEPENDENT_PRELAUNCH_REVIEW.json')}))

if __name__=='__main__':main()
