"""Exact optimizer membership and detached coupled-decay diagnostics."""
import hashlib,json,math
import torch
from common import ROOT,read,sha

ARMS=('fixed_lr','step_lr')
WEIGHT_DECAY={'fixed_lr':0.,'step_lr':0.}
DIAGNOSTIC_FLOOR=1e-12
FROZEN_THETA={'core.blocks.1.transport.theta','core.blocks.2.transport.theta'}


def roster(model):
    included=[];excluded=[];seen=set()
    for name,p in model.named_parameters():
        assert id(p) not in seen;seen.add(id(p))
        record={'name':name,'shape':list(p.shape),'dtype':str(p.dtype),'numel':p.numel()}
        if p.requires_grad:
            assert not name.startswith('right_adapter.') and name not in FROZEN_THETA
            included.append(record)
        else:
            assert name.startswith('right_adapter.') or name in FROZEN_THETA,('Unexpected frozen parameter',name)
            excluded.append(record)
    assert len(included)==135
    assert FROZEN_THETA <= {v['name'] for v in excluded}
    assert any(v['name'].startswith('right_adapter.') for v in excluded)
    assert model.adapter_enabled is False and model.transport_mode=='original'
    return {'group_count':1,'included':included,'excluded_frozen':excluded,
            'included_tensor_count':len(included),'included_numel':sum(v['numel'] for v in included),
            'registration_order_preserved':True,'bias_norm_embedding_offset_radial_exemptions':False,
            'buffers_are_not_optimizer_parameters':True}


def verify_roster(model):
    doc=read(ROOT/'PARAMETER_ROSTER.json')
    assert doc['frozen_before_any_optimizer_update'] and doc['contract']==roster(model)
    return sha(ROOT/'PARAMETER_ROSTER.json')


def selected_decay(cfg,arm):
    assert arm in ARMS and cfg['arms'][arm]['mode']=='original'
    value=cfg['arms'][arm]['weight_decay'];assert value==WEIGHT_DECAY[arm]
    return value


def optimizer_membership(model,opt,cfg,arm,require_state=False,expected_lr=None):
    digest=verify_roster(model);assert len(opt.param_groups)==1
    group=opt.param_groups[0];params=[p for p in model.parameters() if p.requires_grad]
    assert [id(p) for p in group['params']]==[id(p) for p in params] and len(params)==135
    assert group['weight_decay']==selected_decay(cfg,arm) and group['lr']==(cfg['lr'] if expected_lr is None else expected_lr)
    assert group['betas']==tuple(cfg['betas']) and group['eps']==cfg['eps'] and group['amsgrad'] is True
    assert group['foreach'] is None and group['fused'] is None
    assert not group['capturable'] and not group['differentiable'] and not group['maximize']
    if require_state:assert {id(p) for p in opt.state}=={id(p) for p in params}
    return digest


def postclip_diagnostics(model,weight_decay):
    """No gradient mutation; same grad!=None subset that Adam actually processes."""
    assert weight_decay in WEIGHT_DECAY.values()
    with torch.no_grad():
        all_params=[p for p in model.parameters() if p.requires_grad]
        live=[p for p in all_params if p.grad is not None]
        def norm(values):return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))
        task=norm(p.grad for p in live);decay=weight_decay*norm(live)
        effective=norm(p.grad.detach().double()+weight_decay*p.detach().double() for p in live)
        radial=[p.detach().double() for name,p in model.named_parameters()
                if p.requires_grad and name=='core.Radial.radial.beta']
        assert len(radial)==1,'Unexpected trainable Bessel beta roster'
        offsets=model.decoder.energy_offset.detach().double()
        out={'postclip_task_gradient_l2':task,'coupled_term_l2':decay,'effective_gradient_l2':effective,
             'regularized_decay_task_ratio':decay/max(task,DIAGNOSTIC_FLOOR),
             'task_norm_zero':int(task==0),'task_norm_below_floor':int(task<DIAGNOSTIC_FLOOR),
             'parameters_with_grad':len(live),'parameters_grad_none':len(all_params)-len(live),
             'trainable_parameter_l2':norm(all_params),'radial_beta_min_abs':float(radial[0].abs().min()),
             'energy_offset_min':float(offsets.min()),'energy_offset_max':float(offsets.max())}
        assert all(math.isfinite(v) for v in out.values())
        return out
