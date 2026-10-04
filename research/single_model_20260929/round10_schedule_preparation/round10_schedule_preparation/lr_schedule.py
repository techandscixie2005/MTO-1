"""Absolute, prespecified epoch rates and strict complete-epoch resume metadata."""
import hashlib,json

ARMS=('fixed_lr','step_lr')
EPOCHS=60
UPDATES_PER_EPOCH=1881

def tables(cfg):
    assert list(cfg['arms'])==list(ARMS) and cfg['epochs']==EPOCHS
    assert cfg['lr']==.001 and cfg['scheduler'] is None
    assert cfg['batches_per_epoch']==UPDATES_PER_EPOCH and cfg['optimizer_updates']==112860
    expected={'fixed_lr':[.001]*60,'step_lr':[.001]*30+[.0003]*30}
    for arm in ARMS:
        assert cfg['arms'][arm]['epoch_learning_rates']==expected[arm]
        assert cfg['arms'][arm]['mode']=='original' and cfg['arms'][arm]['weight_decay']==0.
    return expected

def digest(cfg):
    return hashlib.sha256(json.dumps(tables(cfg),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def epoch_lr(cfg,arm,epoch):
    assert type(epoch) is int and 1<=epoch<=EPOCHS and arm in ARMS
    return tables(cfg)[arm][epoch-1]

def phase(arm,epoch):
    assert arm in ARMS and type(epoch) is int and 1<=epoch<=EPOCHS
    return 'fixed' if arm=='fixed_lr' else ('high' if epoch<=30 else 'low')

def metadata(cfg,arm,completed_epoch):
    assert type(completed_epoch) is int and 0<=completed_epoch<=EPOCHS and arm in ARMS
    tables(cfg)
    return {'schedule_sha256':digest(cfg),'epoch_learning_rates':tables(cfg)[arm],
        'lr_used_for_completed_epoch':None if completed_epoch==0 else epoch_lr(cfg,arm,completed_epoch),
        'schedule_phase_used':None if completed_epoch==0 else phase(arm,completed_epoch),
        'optimizer_update_start':None if completed_epoch==0 else (completed_epoch-1)*UPDATES_PER_EPOCH+1,
        'optimizer_update_end':None if completed_epoch==0 else completed_epoch*UPDATES_PER_EPOCH,
        'next_epoch_lr':None if completed_epoch==EPOCHS else epoch_lr(cfg,arm,completed_epoch+1)}

def stored_lr(cfg,arm,completed_epoch):
    metadata(cfg,arm,completed_epoch)
    return cfg['lr'] if completed_epoch==0 else epoch_lr(cfg,arm,completed_epoch)

def assign_epoch(opt,cfg,arm,epoch):
    """Set one absolute LR only. Never recreate or alter Adam state/counters."""
    value=epoch_lr(cfg,arm,epoch);assert len(opt.param_groups)==1
    opt.param_groups[0]['lr']=value
    return value

def validate_checkpoint(ck,cfg,arm):
    """Production commits have true epoch×1881 steps; fixture labels are separate."""
    assert ck['format']=='round10_resumable_v1' and ck['arm']==arm and ck['config']==cfg
    state=ck['state'];epoch=state['completed_epoch'];expected=metadata(cfg,arm,epoch)
    assert ck['schedule']==expected
    assert type(state['steps']) is int and state['steps']==epoch*UPDATES_PER_EPOCH
    groups=ck['optimizer']['param_groups'];assert len(groups)==1
    group=groups[0];assert group['lr']==stored_lr(cfg,arm,epoch)
    assert len(group['params'])==135 and len(set(group['params']))==135
    moments=ck['optimizer']['state']
    if epoch==0:assert not moments
    else:
        assert set(moments)==set(group['params'])
        assert all(float(s['step'])==state['steps'] for s in moments.values())
    assert [r['epoch'] for r in state['history']]==list(range(epoch+1))
    for row in state['history']:
        m=metadata(cfg,arm,row['epoch'])
        assert row['learning_rate']==m['lr_used_for_completed_epoch']
        assert row['schedule_phase_used']==m['schedule_phase_used'] and row['next_epoch_lr']==m['next_epoch_lr']
        assert row['optimizer_update_start']==m['optimizer_update_start'] and row['optimizer_update_end']==m['optimizer_update_end']
    return expected

def next_epoch(cfg,arm,completed_epoch):
    metadata(cfg,arm,completed_epoch)
    assert completed_epoch<EPOCHS,'Completed60 arm cannot resume'
    return completed_epoch+1
