"""CPU-only audit/reduction of committed checkpoints and saved validation arrays."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('OMP_NUM_THREADS') == '2' and os.environ.get('MKL_NUM_THREADS') == '2'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,CAMPAIGN,read,sha,atomic_json
from execution_gate import verify
from congruence import CONTRACT  # Definitions only; no module instance or forward.
sys.path.insert(0,str(CAMPAIGN/'ops'))
from monitor import process_identity
ARMS=('original','scalar','tensor')


def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def equal_state(a,b):
    assert set(a)==set(b)
    for name in a:
        assert a[name].dtype==b[name].dtype and a[name].shape==b[name].shape
        assert torch.equal(a[name],b[name]),name


def metric(y,p):
    assert y.shape==p.shape and y.size and np.isfinite(y).all() and np.isfinite(p).all()
    d=p-y;sse=float((d*d).sum());sst=float(((y-y.mean())**2).sum())
    return {'count':int(y.size),'sse':sse,'mse':sse/y.size,'rmse':float(np.sqrt(sse/y.size)),
            'mae':float(np.abs(d).mean()),'r2':1-sse/sst if sst else None}


def score(arr,stats):
    y,p,m=arr['true_f'],arr['pred_f'],arr['mask']
    assert y.shape==p.shape==m.shape==(6686,10) and m.dtype==np.bool_ and m.all()
    assert np.isfinite(y).all() and np.isfinite(p).all() and (y>=0).all() and (p>=0).all()
    out={'pooled':metric(y[m],p[m]),'per_state':{str(k+1):metric(y[:,k],p[:,k]) for k in range(10)}}
    out['bright_tail']={q:{'threshold':stats[q],**metric(y[y>=stats[q]],p[y>=stats[q]])} for q in ('q90','q99')}
    bins={}
    for actual in (False,True):
        for pred in (False,True):
            use=((y>=stats['q99'])==actual)&((p>=stats['q99'])==pred)
            bins[f'true_{int(actual)}_pred_{int(pred)}']={'count':int(use.sum()),
                'sse':float(((p[use]-y[use])**2).sum()),'absolute_error_sum':float(np.abs(p[use]-y[use]).sum())}
    assert sum(v['count'] for v in bins.values())==66860
    assert np.isclose(sum(v['sse'] for v in bins.values()),out['pooled']['sse'],rtol=1e-12,atol=1e-12)
    out['false_bright_bins']=bins
    assert arr['true_E'].shape==arr['pred_E'].shape==(6686,10)
    out['energy']=metric(arr['true_E'],arr['pred_E'])
    return out


def close(a,b,path=''):
    if isinstance(a,dict):
        assert set(a)<=set(b),(path,set(a),set(b))
        for k in a:close(a[k],b[k],path+'/'+k)
    elif a is None:assert b is None,path
    elif isinstance(a,(int,str,bool)):assert a==b,(path,a,b)
    else:assert np.isclose(a,b,rtol=1e-11,atol=1e-12),(path,a,b)


def main():
    out=ROOT/'completion';assert not (out/'ANALYSIS_RECEIPT.json').exists(),'Completed analysis must not repeat'
    cfg=read(ROOT/'TERMINAL_ANALYSIS_CONFIG.json');stats=read(ROOT/'TRAIN_STATISTICS.json')
    training_config=read(ROOT/'config.json')
    permit=verify(ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json');manifest=permit['manifest']
    inputs={};sources={str(ROOT/n):sha(ROOT/n) for n in ('terminal_analysis.py','TERMINAL_ANALYSIS_CONFIG.json')}
    def bind(path):
        path=Path(path);inputs[str(path)]=sha(path);return path
    def record(path):return read(bind(path))
    split=record(CAMPAIGN/'dataset_audit_20260930/SPLIT_MANIFEST.json')
    bind(ROOT/'FROZEN_MANIFEST.json');bind(ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json')
    bind(ROOT/'INDEPENDENT_PREPARATION_REVIEW.json');bind(read(ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json')['publication_receipt'])
    bind(ROOT/'TRAIN_STATISTICS.json');bind(ROOT/'config.json');bind(ROOT/'model_config.json')
    retained_path=bind(cfg['retained_reference']['path'])
    assert inputs[str(retained_path)]==cfg['retained_reference']['sha256']
    retained=read(retained_path)['arms']['control']['selected_best']
    assert retained['pooled']['r2']==cfg['retained_reference']['r2']==read(ROOT/'config.json')['retained_v2_reference_r2']
    assert retained['pooled']['count']==66860
    histories={};arrays={policy:{} for policy in cfg['policies']};verification={};results={}
    reference=None;dormant_reference=None
    for arm in ARMS:
        rd=ROOT/'runs'/arm;terminal=record(rd/'FIT_COMPLETE.json')
        assert not (rd/'FAILED.json').exists() and not list(rd.glob('RESUMED_FAILURE_*'))
        assert terminal['completed_epoch']==60 and terminal['steps']==112860 and terminal['test_evaluated'] is False
        assert terminal['arm']==arm and terminal['manifest_sha256']==permit['manifest_sha256']
        assert terminal['source_split_manifest_sha256']==sha(CAMPAIGN/'dataset_audit_20260930/SPLIT_MANIFEST.json')
        attempts=list((rd/'attempts').glob('*/LAUNCH_RECEIPT.json'));assert len(attempts)==1
        launch=record(attempts[0]);observed=process_identity(launch['identity']['pid'])
        assert observed is None or observed.get('process_state')=='Z','Owned original process has not exited'
        assert launch['registration_returncode']==0
        for key in ('manifest_sha256','authorization_sha256'):
            assert launch[key]==permit[key]
        assert launch['review_sha256']==permit['review_sha256'] and launch['publication_sha256']==permit['publication_sha256']
        checkpoints={}
        for name,key in (('last.pt','last_checkpoint_sha256'),('best.pt','best_checkpoint_sha256'),('geometry_best.pt','geometry_checkpoint_sha256')):
            path=bind(rd/name);assert inputs[str(path)]==terminal[key]
            checkpoints[name]=torch.load(path,map_location='cpu',weights_only=False)
        last,best,geo=(checkpoints[n] for n in ('last.pt','best.pt','geometry_best.pt'))
        state=last['state'];assert state['completed_epoch']==60 and state['steps']==112860
        history=[json.loads(line) for line in bind(rd/'history.jsonl').read_text().splitlines()]
        assert inputs[str(rd/'history.jsonl')]==terminal['history_sha256'] and history==state['history']
        assert [r['epoch'] for r in history]==list(range(61))
        assert history[0]['order_sha256'] is None
        for row in history[1:]:
            assert row['order_sha256']==manifest['epoch_order_sha256'][row['epoch']-1]
            assert row['optimizer_batches']==1881 and row['learning_rate']==.001
            d=row['readout_diagnostics']
            assert all(np.isfinite(v) for v in d.values() if v is not None)
            assert d['strength_zero_count']+d['strength_ratio_nonzero_count']==1203550
            if arm=='original':
                assert row['gate_parameter_movement_l2']==d['gate_gradient_l2']==0
                assert d['b_min']==d['b_max']==d['b_mean']==d['A_relative_update_max']==0
        for ck in (last,best):
            assert ck['format']=='round07_resumable_v1'
            assert ck['manifest_sha256']==permit['manifest_sha256'] and ck['arm']==arm
            assert ck['initial_base_tensor_sha256']==manifest['initial_base_tensor_sha256']
            assert ck['initial_full_tensor_sha256']==manifest['initial_full_tensor_sha256']
            assert ck['config']==read(ROOT/'config.json') and ck['stats']==stats and ck['model_config']==read(ROOT/'model_config.json')
            assert ck['permit']['authorization_sha256']==permit['authorization_sha256']
            assert set(ck['rng'])=={'python','numpy','order','torch','cuda'}
            assert ck['rng']['torch'].dtype==ck['rng']['cuda'].dtype==torch.uint8
            assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={ck['state']['steps']}
            assert len(ck['optimizer']['param_groups'])==1
            group=ck['optimizer']['param_groups'][0]
            assert group['lr']==.001 and group['amsgrad'] and group['weight_decay']==0
            assert list(group['betas'])==training_config['betas'] and group['eps']==training_config['eps']
            dormant={k:v for k,v in ck['model'].items() if k.startswith('right_adapter.')}
            assert dormant
            if dormant_reference is None:dormant_reference={k:v.clone() for k,v in dormant.items()}
            equal_state(dormant,dormant_reference)
            gate_state={k.removeprefix('shared_readout.'):v for k,v in ck['model'].items() if k.startswith('shared_readout.')}
            assert sum(v.numel() for v in gate_state.values())==177
            assert all(torch.isfinite(v).all() for v in gate_state.values())
            if arm=='original':assert tensor_hash(gate_state)==manifest['initial_gate_tensor_sha256']
        selected=min(history,key=lambda r:r['validation']['pooled']['sse'])
        assert state['best']==terminal['best'] and selected['epoch']==state['best']['epoch']==best['state']['completed_epoch']
        assert state['best']['sse']==selected['validation']['pooled']['sse']
        assert sha(bind(rd/state['best']['checkpoint']))==terminal['best_checkpoint_sha256']
        assert geo['format']=='round07_geometry_predictor_v1' and geo['geometry_only_inference'] and not geo['requires_qc_labels']
        buffer_fingerprint=geo['buffer_tensor_sha256']
        assert isinstance(buffer_fingerprint,str) and len(buffer_fingerprint)==64 and all(c in '0123456789abcdef' for c in buffer_fingerprint)
        assert geo['model_config']==best['model_config'] and geo['stats']==stats
        assert geo['readout_mode']==training_config['arms'][arm]['mode']==arm
        assert geo['transform_contract']==CONTRACT
        assert geo['provenance']['source_checkpoint_sha256']==terminal['best_checkpoint_sha256']
        assert geo['provenance']['epoch']==selected['epoch'] and geo['provenance']['arm']==arm
        equal_state(best['model'],geo['model']);assert tensor_hash(geo['model'])==geo['model_tensor_sha256']
        access=record(rd/'DATA_ACCESS.json');assert access['test_numeric_rows']==0
        for label,part in (('train','train'),('validation','val')):
            assert access[label]
            for entry in access[label]:
                assert entry['partition']==part and entry['numeric_rows']==split['counts'][part]
                assert entry['numeric_global_indices_sha256']==split['arrays'][part+'_indices.npy']['content_sha256']
        metrics={}
        for policy,path,expected_sha,row in (
            ('selected_best',state['best']['predictions'],state['best']['predictions_sha256'],selected),
            ('fixed60',state['last_predictions']['path'],state['last_predictions']['sha256'],history[60])):
            path=bind(rd/path);assert inputs[str(path)]==expected_sha
            with np.load(path,allow_pickle=False) as saved:
                assert set(saved.files)=={'pred_f','true_f','mask','pred_E','true_E'}
                arr={k:saved[k] for k in saved.files}
            if reference is None:reference={k:arr[k].copy() for k in ('true_f','true_E','mask')}
            for k in reference:assert np.array_equal(reference[k],arr[k]),(arm,policy,k)
            observed_metrics=score(arr,stats);close(observed_metrics,row['validation'])
            observed_metrics['base_objective_from_training_log']=row['validation']['base_objective']
            error=np.abs(arr['pred_f']-arr['true_f']);square=error**2;n_top=int(np.ceil(error.size*cfg['top_error_fraction']))
            observed_metrics['error_distribution']={
                'prediction_quantiles':{str(q):float(np.quantile(arr['pred_f'],q)) for q in cfg['error_quantiles']},
                'absolute_error_quantiles':{str(q):float(np.quantile(error,q)) for q in cfg['error_quantiles']},
                'top_error_count':n_top,'top_error_sse_share':float(np.sort(square.ravel())[-n_top:].sum()/square.sum())}
            arrays[policy][arm]=arr;metrics[policy]=observed_metrics
        verification[arm]={'completed_epoch':60,'steps':112860,'process_exit_evidence':'absent_or_zombie; no OS exit code observed',
            'observed_process':observed,'one_original_attempt_no_resume':True,'all60_order_hashes_match':True,
            'initial_base_tensor_sha256':manifest['initial_base_tensor_sha256'],'initial_full_tensor_sha256':manifest['initial_full_tensor_sha256'],
            'optimizer_rng_and_history_checked':True,'geometry_export_tensors_exactly_equal_selected_model':True,
            'geometry_export_mode_and_transform_contract_match':True,
            'dormant_right_F_identical_across_all_selected_and_last_checkpoints':True,
            'original_gate_matches_initial_hash_if_original':arm=='original',
            'geometry_export_buffer_hash_note':'Stored fingerprint present; no new model construction or forward replay in terminal analysis.',
            'test_numeric_rows':0,'checkpoint_paths':{n:str(rd/n) for n in checkpoints},
            'checkpoint_hashes':{n:inputs[str(rd/n)] for n in checkpoints},'best_epoch':selected['epoch'],
            'validation_labels':66860,'saved_arrays_recomputed':True}
        histories[arm]=history
        results[arm]={'selected_epoch':selected['epoch'],**metrics,
            'wall_training_validation_seconds':sum(r['seconds'] for r in history),
            'trajectory_snapshots':{str(ep):{k:history[ep][k] for k in history[ep] if k!='order_sha256'} for ep in sorted({1,selected['epoch'],60})},
            'trajectory_interpretation':'TRAIN entries are weighted pre-update minibatch aggregates, not a fixed-checkpoint TRAIN evaluation.'}
        diagnostic_names=('b_mean','b_min','b_max','q_mean','q_min','q_max',
                          'A_relative_update_mean','A_relative_update_max',
                          'strength_ratio_nonzero_mean','strength_ratio_nonzero_min','strength_ratio_nonzero_max',
                          'base_gradient_l2','gate_gradient_l2')
        results[arm]['readout_trajectory_summary']={key:{
            'first':history[1]['readout_diagnostics'].get(key),
            'selected_epoch':selected['readout_diagnostics'].get(key),
            'fixed60':history[60]['readout_diagnostics'].get(key),
            'min_over_epoch_aggregates':min(r['readout_diagnostics'][key] for r in history[1:] if r['readout_diagnostics'].get(key) is not None),
            'max_over_epoch_aggregates':max(r['readout_diagnostics'][key] for r in history[1:] if r['readout_diagnostics'].get(key) is not None)} for key in diagnostic_names}
        results[arm]['readout_trajectory_summary']['interpretation']='Logged pre-update TRAIN aggregates, not validation diagnostics. gate_parameter_movement_l2 in snapshots is displacement within that epoch, not distance from initialization. Range extrema do not measure the frequency of saturation.'
        del last,best,geo,checkpoints
    # Identity metadata only. The paired unit keeps linked molecules together.
    metadata={}
    for name in ('val_indices.npy','component_index.npy'):
        entry=split['arrays'][name];path=bind(entry['path']);assert inputs[str(path)]==entry['sha256']
        metadata[name]=np.load(path,allow_pickle=False)
        assert hashlib.sha256(np.ascontiguousarray(metadata[name]).tobytes()).hexdigest()==entry['content_sha256']
    components=metadata['component_index.npy'][metadata['val_indices.npy']]
    unique,inverse=np.unique(components,return_inverse=True);groups=len(unique)
    y=reference['true_f'];n=np.bincount(inverse,weights=np.full(len(y),10.));ys=np.bincount(inverse,weights=y.sum(1));ys2=np.bincount(inverse,weights=(y*y).sum(1))
    paired={};rng=np.random.default_rng(cfg['bootstrap_seed'])
    samples=rng.integers(0,groups,size=(cfg['bootstrap_draws'],groups))
    denom=ys2[samples].sum(1)-ys[samples].sum(1)**2/n[samples].sum(1);assert (denom>0).all()
    for policy in cfg['policies']:
        errors={arm:np.bincount(inverse,weights=((arrays[policy][arm]['pred_f']-y)**2).sum(1)) for arm in ARMS}
        paired[policy]={}
        for candidate,control in cfg['paired_comparisons']:
            delta=results[candidate][policy]['pooled']['r2']-results[control][policy]['pooled']['r2']
            draws=(errors[control][samples].sum(1)-errors[candidate][samples].sum(1))/denom
            lo,hi=np.percentile(draws,cfg['bootstrap_interval_percentiles'])
            paired[policy][candidate+'_minus_'+control]={'candidate':candidate,'control':control,
                'delta_r2':delta,'descriptive_percentile_interval':[float(lo),float(hi)]}
    threshold=read(ROOT/'config.json')['promotion_delta_r2'];assert threshold==.003
    candidate_delta_original=paired['selected_best']['tensor_minus_original']['delta_r2']
    candidate_delta_scalar=paired['selected_best']['tensor_minus_scalar']['delta_r2']
    candidate_delta_retained=results['tensor']['selected_best']['pooled']['r2']-retained['pooled']['r2']
    gate_passed=all(v>=threshold for v in (candidate_delta_original,candidate_delta_scalar,candidate_delta_retained))
    reference_deltas={policy:{arm:results[arm][policy]['pooled']['r2']-retained['pooled']['r2'] for arm in ARMS} for policy in cfg['policies']}
    summary={'arms':results,'verification':verification,'paired_uncertainty':{'seed':cfg['bootstrap_seed'],
        'draws':cfg['bootstrap_draws'],'component_groups':groups,'molecules':6686,'unit':cfg['bootstrap_unit'],
        'limitations':cfg['uncertainty_limit'],'comparisons':paired},
        'retained_reference':{'source':cfg['retained_reference'],'metrics':retained,'delta_r2_vs_retained_selected_reference':reference_deltas,
            'note':'Previously published selected Round05 control on the same reused validation. No new reference inference or prediction averaging; no paired interval against this aggregate-only reference.'},
        'gate':{'threshold':threshold,'candidate':'tensor','candidate_delta_r2_vs_contemporaneous_original':candidate_delta_original,
            'candidate_delta_r2_vs_contemporaneous_scalar':candidate_delta_scalar,
            'candidate_delta_r2_vs_retained_reference':candidate_delta_retained,
            'passes_original_comparison':candidate_delta_original>=threshold,'passes_scalar_comparison':candidate_delta_scalar>=threshold,
            'passes_retained_reference_comparison':candidate_delta_retained>=threshold,'passed':gate_passed,
            'eligible_arms':['tensor'] if gate_passed else [],'automatic_extension_authorized':False,'automatic_seed_allocation_authorized':False,
            'better_control_policy':'Report the best control descriptively; no tensor claim or automatic control-specific allocation.'},
        'scope':cfg['scope'],'test_access':False,'model_inference':False,'new_data_split_not_external_confirmation':True}
    out.mkdir(exist_ok=True);atomic_json(summary,out/'ROUND07_RESULTS.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for arm in ARMS:
        h=histories[arm];axs[0,0].plot(range(61),[r['validation']['pooled']['r2'] for r in h],label=arm)
        axs[0,1].plot(range(1,61),[r['train']['raw_f_mse'] for r in h[1:]],label=arm)
        axs[0,2].plot(range(1,61),[r['train']['energy'] for r in h[1:]],label=arm)
        axs[1,0].plot(range(1,61),[r['train']['trace'] for r in h[1:]],label=arm)
        axs[1,1].plot(range(1,61),[r['gradient_clip_fraction'] for r in h[1:]],label=arm)
        axs[1,2].plot(range(61),[r['validation']['energy']['rmse'] for r in h],label=arm)
    axs[0,0].axhline(retained['pooled']['r2'],color='black',ls='--',alpha=.6,label='retained Round05 control')
    titles=['Validation pooled raw-f R²','TRAIN trajectory raw-f MSE','TRAIN trajectory energy loss',
            'TRAIN trajectory trace loss','TRAIN gradient clipping fraction','Validation energy RMSE (eV)']
    for ax,title in zip(axs.ravel(),titles):ax.set_title(title);ax.set_xlabel('Completed epoch');ax.grid(alpha=.2)
    axs[0,0].legend();fig.suptitle('Round07: original/scalar/tensor, LE+Ls, fixed 60 epochs; no TEST evaluation')
    fig.savefig(out/'ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg',metadata={'Date':None});plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    panels=[('b_mean','TRAIN mean b'),('q_mean','TRAIN mean q'),('A_relative_update_mean','TRAIN relative A change'),
            ('strength_ratio_nonzero_mean','TRAIN mean nonzero strength ratio'),('gate_gradient_l2','TRAIN mean gate gradient norm'),
            (None,'Gate displacement within each epoch')]
    for arm in ARMS:
        h=histories[arm][1:]
        for ax,(key,title) in zip(axs.ravel(),panels):
            ax.plot(range(1,61),[r['readout_diagnostics'][key] if key else r['gate_parameter_movement_l2'] for r in h],label=arm)
    for ax,(_,title) in zip(axs.ravel(),panels):ax.set_title(title);ax.set_xlabel('Completed epoch');ax.grid(alpha=.2)
    axs[0,0].legend();fig.suptitle('Round07: TRAIN trajectory diagnostics; no extra inference')
    fig.savefig(out/'GATE_AND_READOUT_TRAJECTORIES.svg',metadata={'Date':None});plt.close(fig)
    outputs={str(p):sha(p) for p in (out/'ROUND07_RESULTS.json',out/'ALIGNED_VALIDATION_AND_TRAIN_CURVES.svg',out/'GATE_AND_READOUT_TRAJECTORIES.svg')}
    for path,digest in inputs.items():assert sha(path)==digest,('Input changed during analysis',path)
    for path,digest in sources.items():assert sha(path)==digest
    atomic_json({'passed':True,'created_unix':time.time(),'manifest_sha256':permit['manifest_sha256'],
        'authorization_sha256':permit['authorization_sha256'],'source_hashes':sources,'input_hashes':inputs,
        'output_hashes':outputs,'no_model_inference':True,'no_test_or_raw_dataset_targets_opened':True,
        'checkpoint_tensors_read_on_cpu':True,'saved_validation_arrays_recomputed':True,
        'bootstrap_group_metadata_read_only':True},out/'ANALYSIS_RECEIPT.json')
    print(json.dumps({'gate':summary['gate'],'comparisons':paired,'reference_deltas':reference_deltas,'output_hashes':outputs},indent=2))


if __name__=='__main__':main()
