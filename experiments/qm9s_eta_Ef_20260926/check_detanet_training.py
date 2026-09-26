"""Scratch two-step resume test; these weights are never used for the experiment."""
import copy,json,time,torch
import numpy as np
from dataset import Data,ROOT
from detanet_adapter import build_detanet,ef_loss
from trainer import setup,atomic_save,atomic_json,state_hash
from trainer_detanet import optimizer_scheduler,snapshot,restore,fingerprint,validate
def step(model,opt,x,y):
    model.train();opt.zero_grad(set_to_none=True);p=model(**x);loss=ef_loss(p,y);loss.backward();opt.step()
    assert torch.isfinite(loss)
    return float(loss)
def main():
    cfg=json.loads((ROOT/'configs/detanet_ef.json').read_text());setup(11);data=Data();m=build_detanet(cfg).cuda()
    expected=json.loads((ROOT/'reports/detanet_interface_checks.json').read_text())['initial_state_sha256']
    assert state_hash(m.state_dict())==expected
    opt,sch=optimizer_scheduler(m,cfg);rng=np.random.default_rng(11);fp=fingerprint()
    order=rng.permutation(data.parts['train']);x,y=data.batch(order[:4]);l1=step(m,opt,x,y);sch.step(l1)
    state=dict(epoch=1,cursor=4,order=order,steps=1,history=[],interval_sums=[0.,0.],interval_counts=[0,0])
    path=ROOT/'reports/detanet_scratch_resume.pt';atomic_save(snapshot(m,opt,sch,state,cfg,fp,rng),path)
    m2=build_detanet(cfg).cuda();o2,s2=optimizer_scheduler(m2,cfg);g2=np.random.default_rng(67)
    ck=torch.load(path,map_location='cuda',weights_only=False);restored=restore(ck,m2,o2,s2,g2,cfg,fp)
    assert restored['cursor']==4 and restored['steps']==1 and np.array_equal(restored['order'],order)
    assert rng.bit_generator.state==g2.bit_generator.state
    assert state_hash(m.state_dict())==state_hash(m2.state_dict()) and sch.state_dict()==s2.state_dict()
    x,y=data.batch(order[4:8]);la=step(m,opt,x,y);lb=step(m2,o2,x,y)
    max_diff=max(float((a-b).abs().max()) for a,b in zip(m.parameters(),m2.parameters()))
    assert abs(la-lb)<2e-5 and max_diff<2e-5
    start=time.monotonic();val,spec=validate(m,data,cfg);secs=time.monotonic()-start
    assert np.isfinite(val+list(spec.values())).all()
    # Check fallback scheduler semantics, with the published factor and explicit patience units.
    p=torch.nn.Parameter(torch.tensor(1.));o=torch.optim.Adam([p],lr=cfg['lr'],amsgrad=True)
    s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,factor=.5,patience=50,threshold=1e-4,threshold_mode='rel',min_lr=1e-6)
    s.step(1.)
    for _ in range(50):s.step(1.)
    assert o.param_groups[0]['lr']==.001
    s.step(1.);assert o.param_groups[0]['lr']==.0005
    report=dict(passed=True,actual_training_not_started=True,scratch_steps_only=2,initial_hash=expected,
        checkpoint_restores_model_optimizer_scheduler_RNG_cursor=True,continued_loss_abs_diff=abs(la-lb),
        continued_parameter_max_abs_diff=max_diff,full_validation_seconds=secs,full_validation_molecules=len(data.parts['val']),
        full_validation_finite=True,scheduler_fallback_test_passed=True,
        training_scope='A fresh process reconstructs seed=11; never loads scratch weights')
    atomic_json(report,ROOT/'reports/detanet_training_checks.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
