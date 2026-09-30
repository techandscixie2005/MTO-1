"""Reviewed future stages; never invoked without production_entry's separate exact authorization."""
import fcntl
import os
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from common import ROOT,PARENT,ROUND03,SOURCE,sha,read_json,atomic_json,settings
from backbone import MTOEA,tensor_hash
from scalar_map import solve_fixed,apply_map,basis
from predictor import export_record,load_predictor
from stage_state import solve_action,evaluation_action

OUT=ROOT/'production'
ARMS=('in_sample','heldout_source')
FIELDS={'in_sample':'full_baseline_native_f','heldout_source':'source_native_f'}


def atomic_torch(record,path):
    temporary=Path(str(path)+'.tmp')
    with open(temporary,'wb') as stream:
        torch.save(record,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    descriptor=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def score(truth,pred):
    y=np.asarray(truth,dtype=np.float64);p=np.asarray(pred,dtype=np.float64)
    if y.shape!=p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite or misaligned scoring input')
    if not y.size:return {'count':0,'sse':0.,'sst':0.,'r2':None,'rmse':None,'mae':None}
    sse=float(np.square(p-y).sum());sst=float(np.square(y-y.mean()).sum())
    return {'count':int(y.size),'sse':sse,'sst':sst,'r2':1-sse/sst if sst>0 else None,
            'rmse':float(np.sqrt(sse/y.size)),'mae':float(np.abs(p-y).mean())}


def report(y,p,mask):
    output={'pooled':score(y[mask],p[mask]),
        'per_state':[score(y[:,k][mask[:,k]],p[:,k][mask[:,k]]) for k in range(10)],
        'true_bright_tail':{},'brightness_bins':{}}
    for name,cut in (('q90',.0549),('q99',.2412)):
        selected=mask&(y>=cut);output['true_bright_tail'][name]={'threshold':cut,**score(y[selected],p[selected])}
    for true_bright in (False,True):
        for predicted_bright in (False,True):
            selected=mask&((y>=.2412)==true_bright)&((p>=.2412)==predicted_bright)
            output['brightness_bins']['true_%d_pred_%d'%(true_bright,predicted_bright)]=score(y[selected],p[selected])
    assert sum(v['count'] for v in output['brightness_bins'].values())==output['pooled']['count']
    assert np.isclose(sum(v['sse'] for v in output['brightness_bins'].values()),output['pooled']['sse'],rtol=1e-12,atol=1e-12)
    return output


def load_calibration(with_targets):
    cfg=settings();path=ROUND03/'affine/calibration_predictions.npz'
    if sha(path)!=cfg['calibration_array_sha256']:raise ValueError('Calibration array changed')
    wanted=['indices','ids','mask_f','full_baseline_native_f','source_native_f']
    if with_targets:wanted.append('f_true')
    with np.load(path,allow_pickle=False) as archive:data={key:archive[key] for key in wanted}
    mask=data['mask_f'];assert mask.dtype==np.bool_ and mask.shape==(24071,10) and mask.sum()==240710
    import hashlib
    assert hashlib.sha256(np.ascontiguousarray(data['indices']).tobytes()).hexdigest()==cfg['calibration_index_bytes_sha256']
    for key in FIELDS.values():assert data[key].shape==mask.shape and np.isfinite(data[key][mask]).all()
    if with_targets:
        assert data['f_true'].shape==mask.shape and np.isfinite(data['f_true'][mask]).all()
        assert int((data['f_true'][mask]==0).sum())==4546
    return data


def verify_coefficients(arm,permit):
    path=OUT/('coefficients_'+arm+'.pt');record=torch.load(path,map_location='cpu',weights_only=False)
    if record['arm']!=arm or record['permit']!=permit:raise ValueError('Immutable coefficient authority differs')
    if record['calibration_array_sha256']!=settings()['calibration_array_sha256']:raise ValueError('Coefficient input mismatch')
    if not torch.is_tensor(record['coefficients']) or record['coefficients'].dtype!=torch.float64:
        raise ValueError('Frozen coefficients must be an FP64 tensor')
    theta=record['coefficients'].numpy()
    if theta.shape!=(4,) or not np.isfinite(theta).all():raise ValueError('Invalid frozen coefficients')
    return record,theta


def base_model():
    cfg=settings();path=SOURCE/'runs/mto_eta0/best.pt'
    if sha(path)!=cfg['base_checkpoint_sha256']:raise ValueError('Base checkpoint changed')
    config=read_json(SOURCE/'configs/mto_eta0.json');stats=read_json(SOURCE/'data/normalization.json')
    original=torch.load(path,map_location='cpu',weights_only=False)
    model=MTOEA(config,stats);model.load_state_dict(original['model'],strict=True)
    model.requires_grad_(False).eval()
    if tensor_hash(model.state_dict())!=cfg['base_tensor_sha256']:raise ValueError('Base tensor mismatch')
    return model,config,stats


def calibration_fixture(data):
    # Only geometry/identity ZIP members; no raw label file is opened here.
    sys.path.insert(0,str(PARENT/'reports/velocity_audit'))
    from audit_velocity_labels import selected_rows
    indices=data['indices'][:64];order=np.argsort(indices);undo=np.argsort(order)
    with zipfile.ZipFile(SOURCE/'data/dataset.npz') as archive:
        ids=selected_rows(archive,'ids',indices[order])[undo]
        arrays={key:selected_rows(archive,key,indices[order])[undo] for key in ('z','pos','edge')}
    assert np.array_equal(ids,data['ids'][:64])
    z=torch.from_numpy(arrays['z']);pos=torch.from_numpy(arrays['pos']);raw=torch.from_numpy(arrays['edge']).long()
    mask=z!=0;counts=mask.sum(1);offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
    em=raw[:,0]>=0;edge=(raw+offsets[:,None,None]).transpose(1,2)[em].T.contiguous()
    return dict(z=z[mask],pos=pos[mask],batch=torch.arange(len(indices))[:,None].expand_as(z)[mask],n=len(indices),edge_index=edge)


def export_and_replay(permit,frozen):
    base,config,stats=base_model();before=tensor_hash(base.state_dict());buffers=tensor_hash(dict(base.named_buffers()))
    records={}
    for arm in ARMS:
        _,theta=verify_coefficients(arm,permit);path=OUT/(arm+'.pt')
        if not path.exists():
            record=export_record(base,config,stats,theta,{'permit':permit,'arm':arm,
                'coefficient_tensor_sha256':frozen['coefficients'][arm]['sha256'],'fitted_head':True})
            atomic_torch(record,path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['source_config']==config and saved['stats']==stats
        assert saved['provenance']=={'permit':permit,'arm':arm,
            'coefficient_tensor_sha256':frozen['coefficients'][arm]['sha256'],'fitted_head':True}
        loaded=load_predictor(path,'cpu')
        assert np.array_equal(loaded.coefficients.numpy(),theta)
        assert tensor_hash(loaded.base.state_dict())==before and tensor_hash(dict(loaded.base.named_buffers()))==buffers
        records[arm]={'path':str(path),'sha256':sha(path),'coefficient_tensor_sha256':frozen['coefficients'][arm]['sha256']}
    receipt=OUT/'EXPORT_COMPLETE.json'
    export={'permit':permit,'coefficient_receipt_sha256':sha(OUT/'COEFFICIENTS_FROZEN.json'),'exports':records}
    if receipt.exists():
        assert read_json(receipt)==export
    else:atomic_json(export,receipt)
    replay_path=OUT/'POSTFIT_GEOMETRY_REPLAY.json'
    if replay_path.exists():
        old=read_json(replay_path)
        assert old['permit']==permit and old['export_receipt_sha256']==sha(receipt) and old['passed'] is True
        return
    data=load_calibration(False);geometry=calibration_fixture(data);replays={}
    for arm in ARMS:
        loaded=load_predictor(OUT/(arm+'.pt'),'cpu')
        with torch.no_grad():actual=loaded(**geometry)
        native=actual['native_f'].numpy();_,theta=verify_coefficients(arm,permit)
        np.testing.assert_allclose(native,data['full_baseline_native_f'][:64],atol=2e-6,rtol=1e-5)
        mapped=apply_map(native,theta)
        np.testing.assert_allclose(actual['f'].numpy(),mapped,atol=1e-12,rtol=1e-12)
        replays[arm]={'native_max_abs':float(np.max(np.abs(native-data['full_baseline_native_f'][:64]))),
                     'map_max_abs':float(np.max(np.abs(actual['f'].numpy()-mapped)))}
    assert tensor_hash(base.state_dict())==before and tensor_hash(dict(base.named_buffers()))==buffers
    atomic_json({'passed':True,'permit':permit,'export_receipt_sha256':sha(receipt),
                 'fixture':'first64 calibration geometries in frozen order','replay':replays,
                 'base_unchanged':True},replay_path)


def fit_export(permit):
    frozen_path=OUT/'COEFFICIENTS_FROZEN.json'
    if frozen_path.exists():
        frozen=read_json(frozen_path);assert frozen['permit']==permit
        for arm in ARMS:
            verify_coefficients(arm,permit)
            assert sha(OUT/('coefficients_'+arm+'.pt'))==frozen['coefficients'][arm]['sha256']
        export_and_replay(permit,frozen);return
    data=None;receipts={}
    prior=read_json(ROUND03/'affine/COEFFICIENTS_FROZEN.json')
    for arm in ARMS:
        tensor_path=OUT/('coefficients_'+arm+'.pt');receipt_path=OUT/(arm+'_SOLVED.json');started_path=OUT/(arm+'_SOLVE_STARTED.json')
        action=solve_action(started_path.exists(),tensor_path.exists(),receipt_path.exists())
        if action=='solve_once':
            atomic_json({'arm':arm,'permit':permit,'started_at_unix':time.time()},started_path)
            if data is None:data=load_calibration(True)
            mask=data['mask_f'];x=data[FIELDS[arm]][mask].astype(np.float64);y=data['f_true'][mask].astype(np.float64)
            theta,diagnostics=solve_fixed(x,y)
            raw=basis(x)@theta;aff=prior['maps'][arm];affine_raw=aff['alpha']*x+aff['beta']
            assert np.square(raw-y).sum()<=np.square(affine_raw-y).sum()+1e-9
            fitting={'unclipped':score(y,raw),'clipped':score(y,np.maximum(0,raw)),
                     'affine_unclipped':score(y,affine_raw),'affine_clipped':score(y,np.maximum(0,affine_raw)),
                     'all_segment_slopes_nonnegative':bool(np.all(np.cumsum(theta[1:])>=0)),
                     'clamp_count':int((raw<0).sum()),
                     'source_input_range':[float(x.min()),float(x.max())],
                     'source_upper_knot_count':int((x>=.2412).sum())}
            record={'arm':arm,'permit':permit,'coefficients':torch.tensor(theta,dtype=torch.float64),
                    'calibration_array_sha256':settings()['calibration_array_sha256'],
                    'design_diagnostics':diagnostics,'fitting_diagnostics':fitting,'solved_at_unix':time.time()}
            atomic_torch(record,tensor_path)
        record,_=verify_coefficients(arm,permit)
        receipt={'arm':arm,'permit':permit,'path':str(tensor_path),'sha256':sha(tensor_path),
                 'shape':[4],'dtype':'float64','design':record['design_diagnostics'],
                 'fitting_diagnostics':record['fitting_diagnostics'],'solved_at_unix':record['solved_at_unix']}
        if receipt_path.exists():assert read_json(receipt_path)==receipt
        else:atomic_json(receipt,receipt_path)
        receipts[arm]=receipt
    frozen={'permit':permit,'coefficients':receipts,'frozen_at_unix':time.time(),
            'calibration_array_sha256':settings()['calibration_array_sha256'],'outer_validation_opened':False}
    atomic_json(frozen,frozen_path);export_and_replay(permit,frozen)


def bootstrap(y,predictions,mask,contrasts):
    cfg=settings();rng=np.random.default_rng(cfg['bootstrap_seed']);n=len(y)
    safe_y=np.where(mask,y,0.0)
    counts=mask.sum(1);sums=safe_y.sum(1);squares=(safe_y*safe_y).sum(1)
    errors={k:np.square(np.where(mask,p,0.0)-safe_y).sum(1) for k,p in predictions.items()}
    samples={name:[] for name in contrasts}
    for _ in range(cfg['bootstrap_draws']):
        indices=rng.integers(0,n,n);count=counts[indices].sum();sst=squares[indices].sum()-sums[indices].sum()**2/count
        for name,(candidate,control) in contrasts.items():samples[name].append(float((errors[control][indices].sum()-errors[candidate][indices].sum())/sst))
    return {'seed':cfg['bootstrap_seed'],'draws':cfg['bootstrap_draws'],'unit':'molecule',
            'interpretation':'descriptive reused-validation uncertainty, no model-selection correction',
            'delta_r2_percentile_95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()}}


def evaluate(permit):
    evaluation_action((OUT/'VALIDATION_COMPLETE.json').exists(),any(OUT.glob('VALIDATION_ATTEMPT_*.json')))
    frozen_path=OUT/'COEFFICIENTS_FROZEN.json';frozen=read_json(frozen_path)
    export=read_json(OUT/'EXPORT_COMPLETE.json');replay=read_json(OUT/'POSTFIT_GEOMETRY_REPLAY.json')
    assert frozen['permit']==export['permit']==replay['permit']==permit and replay['passed'] is True
    assert export['coefficient_receipt_sha256']==sha(frozen_path)
    assert replay['export_receipt_sha256']==sha(OUT/'EXPORT_COMPLETE.json')
    theta={};coefficient_records={}
    for arm in ARMS:
        coefficient_records[arm],theta[arm]=verify_coefficients(arm,permit)
        assert sha(OUT/('coefficients_'+arm+'.pt'))==frozen['coefficients'][arm]['sha256']
        assert sha(OUT/(arm+'.pt'))==export['exports'][arm]['sha256']
    cfg=settings();started=time.time();assert started>frozen['frozen_at_unix']
    attempt=OUT/('VALIDATION_ATTEMPT_%d.json'%time.time_ns())
    atomic_json({'permit':permit,'coefficient_receipt_sha256':sha(frozen_path),'started_at_unix':started},attempt)
    path=ROUND03/'affine/validation_predictions.npz'
    assert sha(path)==cfg['validation_array_sha256']
    with np.load(path,allow_pickle=False) as archive:
        y=archive['f_true'];mask=archive['mask_f'];native=archive['native']
        predictions={key:archive[key] for key in ('native','historical_validation_fit','in_sample','heldout_source')}
    assert mask.dtype==np.bool_ and mask.shape==(6686,10) and mask.sum()==66860
    assert y.shape==native.shape==mask.shape and np.isfinite(y[mask]).all() and np.isfinite(native[mask]).all()
    map_diagnostics={}
    for arm in ARMS:
        candidate=np.zeros_like(native,dtype=np.float64);candidate[mask]=apply_map(native[mask],theta[arm])
        predictions['spline_'+arm]=candidate
        input_min,input_max=coefficient_records[arm]['fitting_diagnostics']['source_input_range']
        raw=basis(native[mask])@theta[arm]
        map_diagnostics[arm]={'negative_unclipped_output_count':int((raw<0).sum()),
            'zero_output_count':int((candidate[mask]==0).sum()),
            'input_below_fitting_source_minimum_count':int((native[mask]<input_min).sum()),
            'input_above_fitting_source_maximum_count':int((native[mask]>input_max).sum()),
            'input_at_or_above_upper_knot_count':int((native[mask]>=.2412).sum()),
            'all_segment_slopes_nonnegative':coefficient_records[arm]['fitting_diagnostics']['all_segment_slopes_nonnegative'],
            'extrapolation_definition':'native deployed input outside this source fit input min/max; no filtering'}
    metrics={name:report(y,pred,mask) for name,pred in predictions.items()}
    assert abs(metrics['native']['pooled']['r2']-.4052941183410983)<1e-7
    assert abs(metrics['historical_validation_fit']['pooled']['r2']-.418119238799)<1e-7
    historical=read_json(ROUND03/'ROUND03_RESULTS.json')['metrics']
    for name in ('native','historical_validation_fit','in_sample','heldout_source'):
        assert abs(metrics[name]['pooled']['r2']-historical[name]['raw_f']['r2'])<1e-12
    r2={key:value['pooled']['r2'] for key,value in metrics.items()};deltas={};eligibility={};contrasts={}
    for arm in ARMS:
        key='spline_'+arm;own=r2[key]-r2[arm];incumbent=r2[key]-r2['historical_validation_fit']
        deltas[arm]={'vs_own_affine':own,'vs_incumbent':incumbent}
        eligibility[arm]=own>=.003 and incumbent>=.003
        contrasts[arm+'_vs_affine']=(key,arm);contrasts[arm+'_vs_incumbent']=(key,'historical_validation_fit')
    source_delta=r2['spline_heldout_source']-r2['spline_in_sample'];contrasts['heldout_vs_in_sample']=('spline_heldout_source','spline_in_sample')
    # Stable ordering gives the incumbent every exact SSE tie.
    selection_order=['historical_validation_fit','native','heldout_source','in_sample','spline_heldout_source','spline_in_sample']
    selected=min(selection_order,key=lambda key:metrics[key]['pooled']['sse'])
    result={'permit':permit,'coefficient_receipt_sha256':sha(frozen_path),'export_receipt_sha256':sha(OUT/'EXPORT_COMPLETE.json'),
        'validation_array_sha256':sha(path),'validation_attempt_sha256':sha(attempt),'metrics':metrics,'deltas':deltas,
        'map_diagnostics':map_diagnostics,
        'confirmation_eligible':eligibility,'heldout_source_advantage_delta':source_delta,
        'heldout_source_advantage_threshold_passed':source_delta>=.003,
        'within_source_increment_interaction':deltas['heldout_source']['vs_own_affine']-deltas['in_sample']['vs_own_affine'],
        'selected_by_validation_sse':selected,'no_automatic_seed_launch':True,'test_access':False,
        'new_model_forward_for_validation':False,'historically_reused_validation':True,
        'uncertainty':bootstrap(y,predictions,mask,contrasts),'completed_at_unix':time.time()}
    atomic_json(result,OUT/'VALIDATION_COMPLETE.json')


def run_stage(stage,permit):
    torch.set_num_threads(2)
    # This module is imported only after the complete external execution gate.
    lock=open(ROOT/'production.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if stage=='evaluate' and not OUT.exists():raise RuntimeError('No frozen fitted coefficients/exports exist')
    OUT.mkdir(exist_ok=True)
    if stage=='fit_export':fit_export(permit)
    elif stage=='evaluate':evaluate(permit)
    else:raise ValueError('Unknown stage')
