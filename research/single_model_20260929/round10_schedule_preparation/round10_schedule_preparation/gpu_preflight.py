"""Exactly six discarded updates on the first128 sealed new TRAIN rows."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
assert os.environ.get('OMP_NUM_THREADS')==os.environ.get('MKL_NUM_THREADS')=='2'
import copy,gc,json,time
import numpy as np
import torch
from common import ROOT,CAMPAIGN,sha,read,atomic_json,immutable_json
from fresh_model import build_fresh,build_original,base_state,tensor_hash
from partition_data import Partition,index_sha
from runtime import TensorData,setup,optimizer,atomic_torch,rng_state,restore_rng
from predictor import load_predictor,export_payload
from training import objective_and_diagnostics,gradient_norms
from fresh_model import base_loss
from preflight_helpers import forbid_external_data,norm,compare,exact
from lr_schedule import tables,metadata,epoch_lr,assign_epoch
from regularization import verify_roster,selected_decay,optimizer_membership,postclip_diagnostics


def output_errors(a,b):
    f=lambda p:p[0].double()*p[1].double().diagonal(dim1=-2,dim2=-1).sum(-1)*2/(3*27.211386245988)
    return {'E_max_abs':float((a[0]-b[0]).abs().max()),'A_max_abs':float((a[1]-b[1]).abs().max()),
            'native_f_max_abs':float((f(a)-f(b)).abs().max()),'E_A_bitwise_equal':all(torch.equal(x,y) for x,y in zip(a,b))}


def main():
    started_all=time.time();review=read(ROOT/'TECHNICAL_SOURCE_REVIEW.json')
    assert review['passed'] and review['scope']=='round10_128_train_6_discarded_updates'
    decision=ROOT/'ROUND10_PREPARATION_DECISION.md'
    assert sha(decision)==review['root_preparation_decision_sha256']=='4d23e6ab826dd2f4df1f4e72939529ff823fd2210a897a0ea4ba0f5dd6494890'
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
    assert not (ROOT/'GPU_PREFLIGHT.json').exists(),'Completed GPU stage must not repeat'
    assert not (ROOT/'private_preflight').exists(),'Preserve prior attempt; never repeat data reads or updates silently'
    cpu=read(ROOT/'CPU_PREFLIGHT.json');assert cpu['passed']
    roster_sha=sha(ROOT/'PARAMETER_ROSTER.json');assert cpu['parameter_roster_sha256']==roster_sha
    assert read(ROOT/'PARAMETER_ROSTER.json')['frozen_before_any_optimizer_update']
    cfg=read(ROOT/'config.json');mc=read(ROOT/'model_config.json');stats=read(ROOT/'TRAIN_STATISTICS.json')
    assert list(cfg['arms'])==['fixed_lr','step_lr'];tables(cfg)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1;setup(cfg)
    part=Partition('train');rows=part.indices[:128].copy();data=TensorData(part,'cuda',rows)
    assert len(data)==128 and all(v['numeric_rows']==128 for v in data.decode_audit)
    out=ROOT/'private_preflight';out.mkdir(exist_ok=False)
    immutable_json({'disposable':True,'production_initialization_prohibited':True,'unique_train_molecules':128,
        'index_sha256':index_sha(rows),'id_sha256':index_sha(data.ids),'planned_executed_updates':6},out/'DISPOSABLE.json')
    report={'passed':False,'train_indices_sha256':index_sha(rows),'train_ids_sha256':index_sha(data.ids),
        'unique_train_molecules':128,'actual_decode_ledger':data.decode_audit,'validation_test_numeric_rows':0,
        'executed_optimizer_updates':0,'arms':{},'parameter_roster_sha256':roster_sha,
        'device_uuid_environment':os.environ['CUDA_VISIBLE_DEVICES'],
        'resume_tolerances':{'model_optimizer_atol':2e-6,'model_optimizer_rtol':1e-5,'loss_atol':1e-6,'loss_rtol':1e-5},
        'preflight_states_are_never_production_initialization':True}
    batches=(np.arange(64),np.arange(64,128));x2,_=data.batch(np.arange(2))
    original=build_original(mc,stats).cuda().eval()
    with torch.no_grad():reference=original(**x2)
    del original;torch.cuda.empty_cache();initials=[]
    for arm,spec in cfg['arms'].items():
        assert spec['mode']=='original';setup(cfg);model=build_fresh(mc,stats,'original').cuda()
        initial=tensor_hash(base_state(model));initials.append(initial)
        assert initial==cpu['initial_base_tensor_sha256'] and tensor_hash(model.state_dict())==cpu['initial_full_tensor_sha256']
        assert not any(p.requires_grad for p in model.right_adapter.parameters())
        assert verify_roster(model)==roster_sha
        frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
        buffers={name:p.detach().clone() for name,p in model.named_buffers()}
        def unchanged_frozen():
            current=dict(model.named_parameters());current_buffers=dict(model.named_buffers())
            assert all(torch.equal(value,current[name]) for name,value in frozen.items())
            assert all(torch.equal(value,current_buffers[name]) for name,value in buffers.items())
        model.eval()
        with torch.no_grad():pred=model(**x2)
        assert all(torch.allclose(a,b,atol=2e-6,rtol=1e-5) for a,b in zip(reference,pred))
        identity_errors=output_errors(reference,pred)
        model.train();opt=optimizer(model,cfg,arm);setup(cfg);generator=np.random.default_rng(cfg['order_seed'])
        fixture_order=np.arange(128,dtype=np.int64);order_hash=index_sha(rows[fixture_order])
        audits=[]
        def step(which):
            label=30+which;rate=assign_epoch(opt,cfg,arm,label)
            model.train();x,y=data.batch(batches[which]);opt.zero_grad(set_to_none=True)
            pred,loss,diagnostic=objective_and_diagnostics(model,x,y,stats)
            assert all(torch.isfinite(v) for v in loss.values()) and all(np.isfinite(v) for v in diagnostic.values())
            params=[p for p in model.parameters() if p.requires_grad]
            ge,ga=torch.autograd.grad(loss['trace'],pred,retain_graph=True,allow_unused=True)
            term_norms={name:norm(torch.autograd.grad(loss[name],params,retain_graph=True,allow_unused=True))
                        for name in ('energy','trace')}
            direct_E=0. if ge is None else norm([ge]);direct_A=norm([ga])
            assert direct_A>0 and np.isfinite(direct_E+direct_A)
            assert direct_E==0. # E output is not an input to the unchanged A decoder
            loss['total'].backward();branch_norms=gradient_norms(model)
            assert branch_norms['base_gradient_l2']>0
            assert branch_norms['gate_gradient_l2']==0.
            clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True))
            reg=postclip_diagnostics(model,selected_decay(cfg,arm))
            assert reg['parameters_with_grad']==135 and reg['parameters_grad_none']==0
            assert reg['coupled_term_l2']==0.
            audits.append({'preupdate_losses':{k:float(v) for k,v in loss.items()},
                'direct_trace_E_gradient_l2':direct_E,'direct_trace_A_gradient_l2':direct_A,
                'term_parameter_gradient_l2':term_norms,'preclip_branch_gradient_l2':branch_norms,
                'transport_diagnostics':diagnostic,'regularization_diagnostics':reg,
                'unplanned_setting_change':False,'schedule_label':label,'learning_rate':rate,'actual_adam_step_after':1+which})
            values={k:float(v) for k,v in loss.items()};opt.step();report['executed_optimizer_updates']+=1
            optimizer_membership(model,opt,cfg,arm,require_state=True,expected_lr=rate);unchanged_frozen()
            assert all(float(v['step'])==1+which for v in opt.state.values())
            atomic_json({'executed_optimizer_updates':report['executed_optimizer_updates'],'arm':arm,
                'batch_index':which,'schedule_label':label,'actual_adam_step':1+which,'learning_rate':rate,'disposable':True,'production_initialization_prohibited':True},ROOT/'GPU_UPDATE_PROGRESS.json')
            print(json.dumps({'update':report['executed_optimizer_updates'],'arm':arm,'loss':values['total'],'preclip_norm':clip,'schedule_label':label,'actual_adam_step':1+which,'learning_rate':rate}),flush=True)
            return values,clip
        torch.cuda.reset_peak_memory_stats();start=time.time();first=step(0)
        path=out/(arm+'_after_update1.pt')
        atomic_torch({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(generator),
            'order':fixture_order,'next_cursor':64,'initial_base_tensor_sha256':initial,
            'parameter_roster_sha256':roster_sha,'weight_decay':selected_decay(cfg,arm),'disposable':True,
            'fixture_schedule_label':30,'actual_adam_step':1,'schedule':metadata(cfg,arm,30)},path)
        second=step(1);uninterrupted={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        uninterrupted_opt=copy.deepcopy(opt.state_dict());uninterrupted_rng=copy.deepcopy(rng_state(generator))
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['disposable'] and ck['next_cursor']==64 and np.array_equal(ck['order'],fixture_order)
        assert ck['parameter_roster_sha256']==roster_sha and ck['weight_decay']==selected_decay(cfg,arm)
        assert ck['fixture_schedule_label']==30 and ck['actual_adam_step']==1 and ck['schedule']==metadata(cfg,arm,30)
        assert ck['optimizer']['param_groups'][0]['lr']==.001
        assert all(float(v['step'])==1 for v in ck['optimizer']['state'].values())
        model=build_fresh(mc,stats,'original').cuda();model.load_state_dict(ck['model'],strict=True)
        opt=optimizer(model,cfg,arm);opt.load_state_dict(ck['optimizer'])
        optimizer_membership(model,opt,cfg,arm,require_state=True);unchanged_frozen();restore_rng(ck['rng'],generator)
        exact(ck['rng'],rng_state(generator))
        moment_before=copy.deepcopy(opt.state_dict()['state']);assign_epoch(opt,cfg,arm,31)
        exact(moment_before,opt.state_dict()['state']);replay=step(1)
        exact(uninterrupted_rng,rng_state(generator));assert index_sha(rows[ck['order']])==order_hash
        model_diff=[];optimizer_diff=[]
        compare(uninterrupted,model.state_dict(),model_diff);compare(uninterrupted_opt,opt.state_dict(),optimizer_diff)
        assert np.isclose(second[0]['total'],replay[0]['total'],atol=1e-6,rtol=1e-5)
        torch.cuda.synchronize();elapsed=time.time()-start
        export=out/(arm+'_geometry_fixture.pt');model.eval()
        atomic_torch(export_payload(model,mc,stats,{'disposable':True,'arm':arm}),export)
        with forbid_external_data(export):
            loaded=load_predictor(export,'cuda')
            with torch.no_grad():a=model(**x2);b=loaded(**x2)
        assert all(torch.allclose(u,v,atol=2e-6,rtol=1e-5) for u,v in zip(a,b))
        report['arms'][arm]={'initial_base_tensor_sha256':initial,'fixture_order_sha256':order_hash,'updates':3,
            'parameter_roster_sha256':roster_sha,'weight_decay':selected_decay(cfg,arm),
            'schedule_labels':[30,31,31],'actual_adam_counters':[1,2,2],
            'applied_learning_rates':[epoch_lr(cfg,arm,e) for e in (30,31,31)],
            'moments_unchanged_by_boundary_assignment':True,'fixture_is_not30_completed_epochs':True,
            'optimizer_parameter_tensor_count':135,'frozen_parameters_and_buffers_unchanged':True,
            'gradient_audits':audits,'update_losses':[first[0],second[0],replay[0]],
            'preclip_norms':[first[1],second[1],replay[1]],'clipped':[v[1]>cfg['grad_clip'] for v in (first,second,replay)],
            'model_replay_max_abs':max(model_diff),'optimizer_replay_max_abs':max(optimizer_diff),
            'loss_replay_abs':abs(second[0]['total']-replay[0]['total']),
            'rng_and_order_exact_before_and_after_replay':True,'three_updates_seconds':elapsed,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'initial_output_errors':identity_errors,
            'geometry_checkpoint_output_errors':output_errors(a,b),'geometry_checkpoint_load_forward_numerical_pass':True,
            'checkpoint_sha256':sha(path),'geometry_checkpoint_sha256':sha(export)}
        del model,opt,ck,loaded,uninterrupted,uninterrupted_opt,a,b;gc.collect();torch.cuda.empty_cache()
    assert len(set(initials))==1 and report['executed_optimizer_updates']==6
    report.update(source_hashes=review['source_hashes'],technical_review_sha256=sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json'),
        root_preparation_decision_sha256=sha(decision),seconds=time.time()-started_all,passed=True)
    immutable_json(report,ROOT/'GPU_PREFLIGHT.json');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
