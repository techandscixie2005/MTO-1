"""Instrumented v2 smoke and explicitly isolated DIAGNOSTIC_ONLY interventions.

Uses train.update unchanged. Optimizer hooks observe gradients, parameters and
moments without mutating them. No diagnostic checkpoint can satisfy the formal gate.
"""
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from data_protocol import ROOT, collate, sha
from models import VARIANTS, UV, build_triplet
from train import atomic_json, atomic_save, evaluate, fingerprint, get_rng, restore_rng, optimizer_scheduler, update


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def tensor_hash(value):
    a = value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(a.dtype), tuple(a.shape))).encode() + a.numpy().tobytes()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(tensor_hash(value).encode())
    return h.hexdigest()


def stats(value):
    x = value.detach().double()
    return dict(shape=list(x.shape), rms=float(x.square().mean().sqrt()),
                mean=float(x.mean()), std=float(x.std(unbiased=False)),
                minimum=float(x.min()), maximum=float(x.max()), l2=float(x.norm()))


def loss_metrics(pred, target, rms):
    assert pred.shape == target.shape == (32, 240), (pred.shape, target.shape)
    p, y = pred.detach().double(), target.detach().double()
    raw = (p-y).square().mean()
    normalized = ((p-y)/rms).square().mean()
    torch.testing.assert_close(normalized, raw/rms**2, atol=1e-12, rtol=1e-12)
    return dict(raw_mse=float(raw), normalized_mse=float(normalized),
                prediction_rms=float(p.square().mean().sqrt()), target_rms=float(y.square().mean().sqrt()),
                scaling_identity_abs_error=float((normalized-raw/rms**2).abs()))


class StepObserver:
    """Read-only hooks around the actual optimizer.step used by train.update."""
    def __init__(self, model, optimizer):
        self.named = list(model.named_parameters())
        self.record = {}
        self.pre_handle = optimizer.register_step_pre_hook(self.before)
        self.post_handle = optimizer.register_step_post_hook(self.after)

    def before(self, optimizer, args, kwargs):
        self.before_vector = torch.cat([p.detach().flatten() for _, p in self.named]).clone()
        grad = torch.cat([p.grad.detach().flatten() for _, p in self.named if p.grad is not None])
        self.record = dict(gradient_l2=float(grad.norm()), gradient_rms=float(grad.square().mean().sqrt()),
                           parameter_l2_before=float(self.before_vector.norm()))

    def after(self, optimizer, args, kwargs):
        after = torch.cat([p.detach().flatten() for _, p in self.named])
        delta = after-self.before_vector
        self.record.update(update_l2=float(delta.norm()), update_rms=float(delta.square().mean().sqrt()),
                           relative_update_l2=float(delta.norm()/self.before_vector.norm().clamp_min(1e-30)))
        state = [optimizer.state[p] for _, p in self.named if 'exp_avg_sq' in optimizer.state[p]]
        for key in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
            v = torch.cat([s[key].detach().flatten() for s in state])
            self.record[key] = dict(mean=float(v.mean()), maximum=float(v.max()), rms=float(v.square().mean().sqrt()))
        current = self.record['exp_avg_sq']['mean']
        self.record['mean_max_v_over_mean_v'] = self.record['max_exp_avg_sq']['mean']/max(current, 1e-30)
        self.record['optimizer_step_range'] = [min(float(s['step']) for s in state), max(float(s['step']) for s in state)]
        del self.before_vector

    def close(self):
        self.pre_handle.remove(); self.post_handle.remove()


def initialize_with_audit(cfg, info, out):
    """Compare actual fresh A initialization with a fresh author constructor: no load."""
    interesting = {'reset_parameters', 'apply', 'xavier_uniform_', 'kaiming_uniform_',
                   'normal_', 'uniform_', 'constant_', 'zeros_'}
    events, stage, end_a_rng = [], [None], []

    def profiler(frame, event, arg):
        name = frame.f_code.co_name
        if name == '__init__' and frame.f_globals.get('__name__') == 'models' and 'variant' in frame.f_locals:
            if event == 'call': stage[0] = frame.f_locals['variant']
            elif event == 'return' and stage[0] == VARIANTS[0]: end_a_rng.append(torch.get_rng_state().clone())
        if event == 'call' and name in interesting:
            filename = frame.f_code.co_filename.replace('\\', '/')
            # Package-independent suffix allows original and vendor event sequences to be compared.
            if '/detanet_model/' in filename: filename = 'detanet_model/' + filename.split('/detanet_model/', 1)[1]
            events.append(dict(stage=stage[0], function=name, file=filename, line=frame.f_code.co_firstlineno))

    previous = sys.getprofile()
    try:
        sys.setprofile(profiler)
        models = build_triplet(cfg, info['n_ref'], info['grid'], 11)
    finally:
        sys.setprofile(previous)
    actual_events = copy.deepcopy(events)
    actual_a = [{k:v for k,v in e.items() if k != 'stage'} for e in events if e['stage'] == VARIANTS[0]]
    from reference.detanet_model.detanet import DetaNet as Original
    events.clear(); stage[0] = 'reference_A'
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        torch.manual_seed(11)
        try:
            sys.setprofile(profiler)
            reference = Original(**UV, device=torch.device('cpu'))
        finally:
            sys.setprofile(previous)
        reference_rng = torch.get_rng_state().clone()
    reference_events = [{k:v for k,v in e.items() if k != 'stage'} for e in events]
    a = models[VARIANTS[0]].core
    assert a.state_dict().keys() == reference.state_dict().keys()
    unequal = [k for k,v in a.state_dict().items() if not torch.equal(v, reference.state_dict()[k])]
    assert not unequal, unequal
    assert len(end_a_rng) == 1 and torch.equal(end_a_rng[0], reference_rng)
    assert actual_a == reference_events, 'Fresh initialization call sequence differs from author'
    # Explicitly retained seed-11 hashes can be compared across all prior audit attempts.
    result = dict(seed=11, fresh_author_constructor_no_state_dict_loading=True,
                  every_fresh_A_state_tensor_equal=True, constructor_rng_equal=True,
                  initialization_call_sequence_equal=True,
                  actual_initialization_calls=actual_events, reference_A_calls=reference_events,
                  common_backbone_equal=True, full_B_C_initial_state_equal=True,
                  model_hashes={k:state_hash(v.state_dict()) for k,v in models.items()},
                  source_hashes={n:sha(ROOT/n) for n in ['models.py','train.py','protocol.json']},
                  note='Upstream constructors already perform resets/Xavier initialization after torch defaults; these are preserved. No project-wide apply(init) is added. MTO-specific initialization is unchanged.')
    atomic_json(result, out/'initialization_audit.json')
    return models


def initial_scales(model, records, target, info, device, out):
    captured, handles = {}, []
    def save(name, value):
        if isinstance(value, (tuple, list)):
            for i,v in enumerate(value): save(name+'/'+str(i), v)
        elif torch.is_tensor(value): captured[name] = value.detach().cpu().clone()
    for name in ['Embedding', 'Radial', 'blocks.0', 'blocks.1', 'blocks.2',
                 'sout.mlp.0', 'sout.mlp.1', 'sout.mlp.2', 'sout.mlp.3', 'sout']:
        handles.append(model.core.get_submodule(name).register_forward_hook(
            lambda module, inputs, output, name=name: save(name, output)))
    final = model.core.sout.mlp[-1]
    handles.append(final.register_forward_pre_hook(lambda module, inputs: save('last_linear_input', inputs[0])))
    x,y = collate(records, device)
    with torch.no_grad(): prediction = model(**x)
    for h in handles: h.remove()
    captured['molecular_output'] = prediction.cpu(); captured['target'] = y.cpu()
    counts = torch.bincount(x['batch'], minlength=32).cpu()
    atom = captured['sout']; h = captured['last_linear_input']; batch = x['batch'].cpu()
    pooled = torch.stack([atom[batch == i].sum(0) for i in range(32)])
    phi = torch.stack([h[batch == i].sum(0) for i in range(32)])
    formula = phi @ final.weight.detach().cpu().T + counts[:,None]*final.bias.detach().cpu()
    torch.testing.assert_close(pooled, prediction.cpu(), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(formula, prediction.cpu(), atol=1e-5, rtol=1e-4)
    assert torch.equal(y.cpu(), target.cpu())
    result = dict(before_first_optimizer_step=True, tensors={k:stats(v) for k,v in captured.items()},
                  atom_counts=counts.tolist(), output_scale=model.core.scale,
                  atomic_sum_max_abs=float((pooled-prediction.cpu()).abs().max()),
                  final_linear_formula_max_abs=float((formula-prediction.cpu()).abs().max()),
                  loss=loss_metrics(prediction, y, info['train_rms']))
    atomic_json(result, out/'initial_A_scales.json')
    atomic_save(dict(scope='INITIALIZATION_AUDIT', ids=info['smoke_ids'], **captured), out/'initial_A_activations.pt')


def data_audit(records, data, info, device, out):
    assert [r['id'] for r in records] == info['smoke_ids'] and len(records) == 32
    assert all(r['split'] == 'train' for r in records)
    x,y = collate(records, device)
    expected = torch.stack([r['spectrum'] for r in records]).to(device)
    assert y.shape == (32,240) and torch.equal(y, expected)
    assert len(data['train']) == 800
    with gzip.open(ROOT/'data/targets_240.json.gz','rt') as f: raw = json.load(f)
    train = [v['spectrum'] for v in raw if v['split'] == 'train']
    recomputed = float(np.sqrt(np.mean(np.square(np.asarray(train, dtype=np.float64)))))
    assert abs(recomputed-info['train_rms']) < 1e-15
    zeros = torch.zeros_like(y); mean = y.double().mean(0, keepdim=True).expand_as(y)
    audit = dict(ids=info['smoke_ids'], target_shape=list(y.shape), target_sha256=tensor_hash(y),
                 targets_equal_ordered_stack=True, train_count=800, train_rms=info['train_rms'],
                 train_rms_recomputed_float64=recomputed,
                 zero_spectrum=loss_metrics(zeros,y,info['train_rms']),
                 pointwise_common_mean_spectrum=loss_metrics(mean,y,info['train_rms']),
                 interpretation='Both trivial baselines fit/evaluate only the same 32 TRAIN molecules; zero normalized MSE is not assumed to be 1.')
    atomic_json(audit, out/'data_loss_audit.json')
    atomic_save(dict(ids=info['smoke_ids'], target=y.cpu(), mean_spectrum=mean[0].cpu()),out/'fixed_training_targets.pt')
    return y


def snapshot(model, optimizer, generator, step, measured, ids, target_hash, scope):
    return dict(scope=scope, formal_initialization_allowed=False, model=cpu_tree(model.state_dict()),
                optimizer=cpu_tree(optimizer.state_dict()), rng=cpu_tree(get_rng(generator)),
                step=step, measured=measured, ids=ids, target_sha256=target_hash,
                fingerprint=fingerprint())


def measure(model, records, info, cfg, device):
    loss,pred = evaluate(model, records, info['train_rms'], cfg['micro_batch_size'], device)
    target = torch.stack([r['spectrum'] for r in records])
    result = loss_metrics(pred,target,info['train_rms'])
    assert abs(loss-result['normalized_mse']) <= 1e-5+1e-5*abs(loss)
    result['trainer_normalized_mse_fp32'] = loss
    return result


def run_trajectory(model, optimizer, records, info, cfg, device, directory, steps,
                   scope, early_stop, initial_step=0):
    directory.mkdir(parents=True, exist_ok=False)
    g = torch.Generator().manual_seed(11)
    model.to(device)
    target_hash = tensor_hash(torch.stack([r['spectrum'] for r in records]))
    initial = measure(model, records, info, cfg, device)
    best = initial['trainer_normalized_mse_fp32']; best_step = initial_step
    saved = snapshot(model,optimizer,g,initial_step,initial,info['smoke_ids'],target_hash,scope)
    atomic_save(saved,directory/'initial.pt'); best_snapshot = saved
    history = []; evaluations = [dict(step=initial_step, **initial)]
    observer = StepObserver(model, optimizer)
    seen = []
    def prediction_hook(module, inputs, output):
        assert output.shape == (32,240)
        seen.append(float(output.detach().square().mean().sqrt()))
    handle = model.register_forward_hook(prediction_hook)
    started = time.monotonic()
    try:
        for offset in range(1, steps+1):
            step = initial_step+offset; seen.clear()
            loss = update(model,optimizer,records,info['train_rms'],cfg['micro_batch_size'],device)
            assert len(seen) == 1, 'Unexpected multiple forwards/broadcasting in one 32-molecule update'
            row = dict(step=step, loss_before_update=loss, prediction_rms_before_update=seen[0], **observer.record)
            history.append(row)
            if offset % 10 == 0:
                values = measure(model,records,info,cfg,device)
                current = values['trainer_normalized_mse_fp32']
                evaluations.append(dict(step=step, **values))
                if current < best:
                    best=current; best_step=step
                    best_snapshot=snapshot(model,optimizer,g,step,values,info['smoke_ids'],target_hash,scope)
                if scope == 'MAIN_SMOKE_NOT_FORMAL' and step == 600:
                    atomic_save(snapshot(model,optimizer,g,step,values,info['smoke_ids'],target_hash,scope),directory/'step600.pt')
                if offset % 100 == 0 or (early_stop and best < cfg['smoke_best_mse_below']):
                    print('INSTRUMENTED',scope,directory.name,step,current,best,flush=True)
                if early_stop and best < cfg['smoke_best_mse_below'] and best < cfg['smoke_best_initial_ratio_below']*initial['trainer_normalized_mse_fp32']:
                    break
        last=measure(model,records,info,cfg,device)
        atomic_save(best_snapshot,directory/'best.pt')
        atomic_save(snapshot(model,optimizer,g,step,last,info['smoke_ids'],target_hash,scope),directory/'last.pt')
    finally:
        observer.close(); handle.remove()
    result=dict(scope=scope, initial=initial, best=best, best_step=best_step, steps=offset,
                last=last, seconds=time.monotonic()-started,
                passed=(best<cfg['smoke_best_mse_below'] and best<cfg['smoke_best_initial_ratio_below']*initial['trainer_normalized_mse_fp32']) if early_stop else None,
                evaluations=evaluations, step_statistics=history,
                target_sha256=target_hash, ids=info['smoke_ids'])
    atomic_json(result,directory/'trajectory.json')
    return result


def verify_saved_checkpoints(template, cfg, records, info, device, directory):
    verified={}
    for name in ['initial','step600','best','last']:
        p=directory/(name+'.pt')
        if not p.exists():continue
        ckpt=torch.load(p,map_location='cpu',weights_only=False)
        assert ckpt['ids']==info['smoke_ids']
        assert ckpt['target_sha256']==tensor_hash(torch.stack([r['spectrum'] for r in records]))
        model=copy.deepcopy(template).to(device);model.load_state_dict(ckpt['model'],strict=True)
        opt,_=optimizer_scheduler(model,cfg);opt.load_state_dict(ckpt['optimizer'])
        assert state_hash(model.state_dict())==state_hash(ckpt['model'])
        now=measure(model,records,info,cfg,device)
        old=ckpt['measured']['trainer_normalized_mse_fp32'];new=now['trainer_normalized_mse_fp32']
        assert abs(old-new)<=1e-5+1e-4*abs(old),(name,old,new)
        # Loading and reserializing optimizer tensors must not lose or alter moments.
        restored=cpu_tree(opt.state_dict())
        from preflight import compare_tree
        compare_tree(restored,ckpt['optimizer'])
        verified[name]=dict(step=ckpt['step'],recorded_loss=old,recomputed_loss=new,abs_difference=abs(old-new),
                            strict_model_and_optimizer_restore=True, checkpoint_sha256=sha(p),model_hash=state_hash(ckpt['model']))
        del model,opt
    atomic_json(verified,directory/'checkpoint_revalidation.json')
    return verified


def svd_solve(X,Y):
    assert X.device.type=='cpu' and X.dtype==Y.dtype==torch.float64
    U,S,Vh=torch.linalg.svd(X,full_matrices=False)
    tolerance=max(X.shape)*torch.finfo(X.dtype).eps*S[0]
    rank=int((S>tolerance).sum())
    theta=Vh[:rank].T@((U[:,:rank].T@Y)/S[:rank,None])
    return theta,dict(numerical_rank=rank,rank_tolerance=float(tolerance),singular_values=S.tolist(),
                     condition_number=float(S[0]/S[-1]) if S[-1]>0 else None,
                     retained_condition_number=float(S[0]/S[rank-1]),solution_frobenius_norm=float(theta.norm()))


def linear_diagnostic(template, cfg, records, info, checkpoint, out):
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=copy.deepcopy(template).cpu();model.load_state_dict(ckpt['model'],strict=True);model.double().eval()
    for p in model.parameters():p.requires_grad_(False)
    final=model.core.sout.mlp[-1]
    assert isinstance(final,torch.nn.Linear) and final.in_features==128 and final.out_features==240
    h=[];hook=final.register_forward_pre_hook(lambda module,inputs:h.append(inputs[0].detach().clone()))
    x,y=collate(records,'cpu',torch.float64)
    with torch.no_grad():before=model(**x)
    hook.remove();assert len(h)==1
    phi=torch.stack([h[0][x['batch']==i].sum(0) for i in range(32)])
    N=torch.bincount(x['batch'],minlength=32).double()[:,None]
    X=torch.cat([phi,N],dim=1);assert X.shape==(32,129)
    formula=phi@final.weight.T+N*final.bias
    torch.testing.assert_close(formula,before,atol=1e-8,rtol=1e-6)
    unchanged={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('core.sout.mlp.3.')}
    theta,result=svd_solve(X,y)
    with torch.no_grad():
        final.weight.copy_(theta[:128].T);final.bias.copy_(theta[128])
        fitted=model(**x)
    matrix_prediction=X@theta
    torch.testing.assert_close(fitted,matrix_prediction,atol=1e-8,rtol=1e-6)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in unchanged.items())
    result.update(scope='DIAGNOSTIC_ONLY',formal_initialization_allowed=False,source_checkpoint=str(checkpoint.relative_to(ROOT)),
                  source_checkpoint_sha256=sha(checkpoint),source_step=ckpt['step'],X_shape=list(X.shape),
                  final_column='atom_count_N_not_constant_one',atom_counts=N[:,0].tolist(),
                  frozen_earlier_parameters_unchanged=True,
                  initial_formula_max_abs=float((formula-before).abs().max()),
                  solution_weight_norm=float(theta[:128].norm()),solution_bias_norm=float(theta[128].norm()),
                  before=loss_metrics(before,y,info['train_rms']),svd_matrix=loss_metrics(matrix_prediction,y,info['train_rms']),
                  full_forward=loss_metrics(fitted,y,info['train_rms']),
                  matrix_vs_full_forward_max_abs=float((fitted-matrix_prediction).abs().max()))
    atomic_json(result,out.with_suffix('.json'))
    atomic_save(dict(scope='DIAGNOSTIC_ONLY',X=X,Y=y,theta=theta,prediction=fitted,source_checkpoint_sha256=sha(checkpoint)),out.with_suffix('.pt'))
    return result


def intervene_optimizer(opt, mode):
    assert mode in ['D0','D1','D2']
    if mode=='D1':
        for state in opt.state.values():
            if 'max_exp_avg_sq' in state:state['max_exp_avg_sq'].copy_(state['exp_avg_sq'])
    elif mode=='D2':
        assert not opt.state, 'D2 must use a newly constructed optimizer'


def optimizer_diagnostics(template,cfg,records,info,device,source,out,modes=('D0','D1','D2')):
    results={}
    from preflight import compare_tree
    for mode in modes:
        # A fresh disk load and deep state copy are both intentional: torch Adam's
        # CPU step tensor can otherwise alias the input state dictionary on CUDA.
        source_ckpt=torch.load(source,map_location='cpu',weights_only=False)
        assert source_ckpt['step']==600
        pristine_optimizer=cpu_tree(source_ckpt['optimizer'])
        model=copy.deepcopy(template).to(device);model.load_state_dict(source_ckpt['model'],strict=True)
        opt,_=optimizer_scheduler(model,cfg)
        if mode!='D2':opt.load_state_dict(cpu_tree(source_ckpt['optimizer']))
        before=cpu_tree(opt.state_dict())
        intervene_optimizer(opt,mode)
        after=cpu_tree(opt.state_dict())
        if mode=='D0':compare_tree(before,after)
        elif mode=='D1':
            compare_tree(before['param_groups'],after['param_groups'])
            for pid,s in before['state'].items():
                for key,value in s.items():compare_tree(after['state'][pid][key],s['exp_avg_sq'] if key=='max_exp_avg_sq' else value)
        else:assert not after['state']
        if mode!='D2':assert all(float(s['step'])==600 for s in after['state'].values())
        assert state_hash(model.state_dict())==state_hash(source_ckpt['model'])
        g=torch.Generator().manual_seed(11);restore_rng(source_ckpt['rng'],g)
        directory=out/mode
        result=run_trajectory(model,opt,records,info,cfg,device,directory,300,'DIAGNOSTIC_ONLY',False,initial_step=600)
        result['intervention']=dict(D0='retain all AMSGrad state',D1='NONSTANDARD: replace max_exp_avg_sq with current exp_avg_sq only',D2='new same-configuration AMSGrad: all moments and bias-correction history reset')[mode]
        result['source_step600_sha256']=sha(source)
        result['initial_model_identical_to_step600']=True
        result['optimizer_intervention_contract_passed']=True
        first_expected=1 if mode=='D2' else 601
        last_expected=300 if mode=='D2' else 900
        assert result['step_statistics'][0]['optimizer_step_range']==[float(first_expected)]*2
        assert result['step_statistics'][-1]['optimizer_step_range']==[float(last_expected)]*2
        compare_tree(source_ckpt['optimizer'],pristine_optimizer)
        result['source_optimizer_unchanged_after_training']=True
        result['directory']=str(directory.relative_to(ROOT))
        result['checkpoint_revalidation']=verify_saved_checkpoints(template,cfg,records,info,device,directory)
        atomic_json(result,directory/'trajectory.json');results[mode]=result
        del model,opt
    atomic_json({k:{a:b for a,b in v.items() if a not in ['step_statistics','evaluations']} for k,v in results.items()},out/'optimizer_interventions.json')
    return results


def run(cfg,data,info,device):
    assert cfg['smoke_max_steps']==2000 and cfg['version']=='detanet_original_mto_1k_v2_smoke2000'
    out=ROOT/f'reports/instrumented_v2_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(dict(job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,protocol=cfg,
                     note='Instrumentation and authorized DIAGNOSTIC_ONLY analyses added; main optimization and model code unchanged. The 600-step v1 cap was an engineering budget, not an author-paper rule.'),out/'attempt.json')
    records=data['train'][:32];target=data_audit(records,data,info,device,out)
    models=initialize_with_audit(cfg,info,out)
    initial_scales(models[VARIANTS[0]].to(device),records,target,info,device,out)
    main={}
    for variant,model in models.items():
        model.to(device);opt,_=optimizer_scheduler(model,cfg)
        directory=out/variant
        result=run_trajectory(model,opt,records,info,cfg,device,directory,2000,'MAIN_SMOKE_NOT_FORMAL',True)
        result['checkpoint_revalidation']=verify_saved_checkpoints(model,cfg,records,info,device,directory)
        main[variant]=result
        atomic_json({k:{a:b for a,b in v.items() if a not in ['step_statistics','evaluations']} for k,v in main.items()},out/'main_smoke_summary.json')
        model.cpu();del opt;torch.cuda.empty_cache()
    hashes={v:state_hash(m.state_dict()) for v,m in models.items()}
    aroot=out/VARIANTS[0];linear={}
    for label in ['initial','best','last']:
        linear[label]=linear_diagnostic(models[VARIANTS[0]],cfg,records,info,aroot/(label+'.pt'),out/('DIAGNOSTIC_ONLY_linear_'+label))
    interventions={}
    if not main[VARIANTS[0]]['passed']:
        interventions=optimizer_diagnostics(models[VARIANTS[0]],cfg,records,info,device,aroot/'step600.pt',out)
    assert hashes=={v:state_hash(m.state_dict()) for v,m in models.items()},'A diagnostic modified a primary model'
    summary=dict(scope='DIAGNOSTIC_ONLY plus separately labeled main smoke',job_id=os.environ['SLURM_JOB_ID'],
                 main_smoke_passed=all(v['passed'] for v in main.values()),all_diagnostics_executed=True,
                 main_models_unchanged_by_diagnostics=True,output_directory=str(out),
                 main={k:{a:b for a,b in v.items() if a not in ['step_statistics','evaluations']} for k,v in main.items()},
                 linear=linear,interventions={k:{a:b for a,b in v.items() if a not in ['step_statistics','evaluations']} for k,v in interventions.items()})
    atomic_json(summary,out/'diagnostic_summary.json')
    atomic_json(dict(status='DIAGNOSTICS_COMPLETE',formal_training_allowed=summary['main_smoke_passed'],job_id=os.environ['SLURM_JOB_ID']),out/'DIAGNOSTICS_COMPLETE.json')
    print('DIAGNOSTICS_COMPLETE',str(out),'MAIN_SMOKE_PASSED',summary['main_smoke_passed'],flush=True)
    assert summary['main_smoke_passed'],'Instrumented v2 main smoke failed; DIAGNOSTIC_ONLY successes do not authorize formal training'
    return summary
