"""Independent CPU-preflight metadata review; no scientific imports or rerun."""
import hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((R/n).read_text())

def main():
    out=R/'INDEPENDENT_CPU_RESULT_CHECKS.json';assert not out.exists()
    source=read('CPU_SOURCE_REVIEW.json');cpu=read('CPU_PREFLIGHT.json');roster=read('PARAMETER_ROSTER.json');receipt=read('ops/CPU_EXECUTION_RECEIPT.json')
    assert sha(R/'CPU_SOURCE_REVIEW.json')=='ca35f1c8ccfc9996032e72bab0cd7bec654290081de7e8b0951ec3973b4e21cd'
    assert sha(R/'CPU_PREFLIGHT.json')=='dfca79770fa451a64e08bd936b80294301829d527a10e60a48ccfe0e98d854c5'
    assert sha(R/'PARAMETER_ROSTER.json')=='290bd6dc4c635c90fc387202230a4af0b7758b38bad0a3570018e05510fb3943'
    assert cpu['passed'] and receipt['passed'] and receipt['exit_code']==0
    assert cpu['source_hashes']==source['source_hashes']==roster['source_hashes']
    for path,digest in source['source_hashes'].items():assert sha(path)==digest,path
    assert cpu['source_review_sha256']==roster['source_review_sha256']==receipt['source_review_sha256']==sha(R/'CPU_SOURCE_REVIEW.json')
    assert receipt['source_sha256']==sha(R/'synthetic_checks.py')
    assert receipt['cpu_preflight_sha256']==sha(R/'CPU_PREFLIGHT.json')
    assert receipt['log_sha256']==sha(R/'ops/CPU_PREFLIGHT_01.log')
    assert receipt['parameter_roster_sha256']==cpu['parameter_roster_sha256']==sha(R/'PARAMETER_ROSTER.json')
    assert receipt['command']==['/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python',str(R/'synthetic_checks.py')]
    assert receipt['cwd']==str(R) and receipt['environment']=={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
    assert receipt['full_model_optimizer_updates']==cpu['model_updates']==0
    assert receipt['planned_toy_optimizer_steps']==cpu['optimizer_arithmetic']['toy_optimizer_step_calls']==5
    assert roster['frozen_before_any_optimizer_update'] and roster['unwrapped_original_ordered_roster_equal']
    c=roster['contract'];entries=c['included'];excluded=c['excluded_frozen']
    assert c['group_count']==1 and c['included_tensor_count']==len(entries)==135
    assert c['included_numel']==sum(x['numel'] for x in entries)==1552092
    assert all(x['numel']==math.prod(x['shape']) and x['dtype']=='torch.float32' for x in entries+excluded)
    assert len({x['name'] for x in entries+excluded})==len(entries)+len(excluded)
    names={x['name'] for x in entries};frozen={x['name'] for x in excluded}
    assert {'core.Radial.radial.alpha','core.Radial.radial.beta','decoder.energy_offset','decoder.trunk.0.weight','decoder.trunk.0.bias','mto.query.weight','core.Embedding.elec_emb.weight','core.Embedding.nuclare_emb.weight'}<=names
    assert frozen=={'core.blocks.1.transport.theta','core.blocks.2.transport.theta','right_adapter.gates.0.weight','right_adapter.gates.0.bias','right_adapter.gates.1.weight','right_adapter.gates.1.bias','right_adapter.gates.3.weight','right_adapter.gates.3.bias','right_adapter.mix.0e.weight','right_adapter.mix.1o.weight','right_adapter.mix.2e.weight'}
    assert not c['bias_norm_embedding_offset_radial_exemptions'] and c['registration_order_preserved']
    for arm in ('0.0','0.0001'):
        v=cpu['optimizer_arithmetic'][arm];assert v['steps']==2 and v['task_clip_active'] and v['AMSGrad_retains_larger_past_second_moment'] and 0<=v['max_abs']<1e-10
    assert cpu['optimizer_arithmetic']['grad_none_skips_decay_and_state'] and cpu['optimizer_arithmetic']['zero_gradient_receives_decay']
    assert cpu['equal_initial_task_loss_and_gradients'] and cpu['constructor_CPU_RNG_unchanged'] and cpu['one_checkpoint_access_and_bitwise_parity']
    assert cpu['real_geometry_or_targets_read'] is False and cpu['synthetic_geometry_only']
    assert all(cpu['mask_checks'].values())
    assert cpu['full_model_symmetry']['E_max_abs']<2e-6 and cpu['full_model_symmetry']['A_max_abs']<2e-6
    for arm,decay in (('zero_decay',0.),('coupled_l2',1e-4)):
        d=cpu['postclip_diagnostics'][arm];assert d['parameters_with_grad']==135 and d['parameters_grad_none']==0
        assert math.isclose(d['coupled_term_l2'],decay*d['trainable_parameter_l2'],rel_tol=1e-12,abs_tol=1e-14)
        assert math.isclose(d['regularized_decay_task_ratio'],d['coupled_term_l2']/max(d['postclip_task_gradient_l2'],1e-12),rel_tol=1e-12,abs_tol=1e-14)
    z=cpu['zero_gradient_diagnostic'];assert z['parameters_with_grad']==1 and z['parameters_grad_none']==134 and z['task_norm_zero']==z['task_norm_below_floor']==1 and z['postclip_task_gradient_l2']==0.
    result={'passed':True,'scope':'independent_round09_cpu_metadata_and_roster_review','numerical_tests_repeated':False,'model_or_array_decoding':False,'source_pins_verified':len(source['source_hashes']),'parameter_tensors':135,'parameter_numel':1552092,'frozen_tensor_count':len(excluded),'toy_optimizer_steps':5,'full_model_updates':0,'CPU_stage_exit_code':0,'input_hashes':{str(R/n):sha(R/n) for n in ('CPU_SOURCE_REVIEW.json','CPU_PREFLIGHT.json','PARAMETER_ROSTER.json','ops/CPU_EXECUTION_RECEIPT.json','ops/CPU_PREFLIGHT_01.log')},'source_hashes':{str(Path(__file__).resolve()):sha(__file__)}}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
