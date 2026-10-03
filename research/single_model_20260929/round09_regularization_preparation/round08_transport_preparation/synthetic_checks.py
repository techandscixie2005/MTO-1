"""Reviewed synthetic-only CPU checks: zero optimizer updates, no dataset."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('OMP_NUM_THREADS')==os.environ.get('MKL_NUM_THREADS')=='2'
import copy,json,tempfile,time
from pathlib import Path
import torch
from e3nn import o3
from torch_scatter import scatter
from common import ROOT,sha,read,immutable_json
from transport import TensorTransport,TransportBlock,MODES
from fresh_model import build_fresh,build_original,base_state,inherited_state,transport_state,tensor_hash,base_loss
from preflight_helpers import forbid_external_data
from predictor import load_predictor,export_payload
from runtime import atomic_torch

DECISION='a227764a9540cfa4f89dd9ed8ea48aa923d65bb4152f2e90ff734073601379a1'
IRREPS=o3.Irreps('128x1o + 128x2e + 128x3o')
LITERAL=((0,384,3),(384,1024,5),(1024,1920,7))


def close(a,b,report,key):
    tol=(1e-10,1e-8) if a.dtype==torch.float64 else (2e-6,1e-5)
    assert torch.allclose(a,b,atol=tol[0],rtol=tol[1]),key
    report[key]=float((a-b).abs().max()) if a.numel() else 0.


def literal_H(T,a,index,mode):
    # Independent edge loop and literal irrep slices, not module indexing code.
    result=[]
    for start,end,dim in LITERAL:
        x=T[:,start:end].reshape(len(T),128,dim);out=torch.zeros_like(x)
        for j in range(len(T)):
            incoming=[k for k in range(index.shape[1]) if int(index[1,k])==j]
            for k in incoming:
                i=int(index[0,k]);out[j]=out[j]+a[k].tanh()[:,None]*x[i if mode=='neighbor' else j]
            out[j]=out[j]/max(1,len(incoming))
        result.append(out)
    return result


def literal_delta(T,a,index,mode,theta):
    H=literal_H(T,a,index,mode)
    return torch.cat([(theta[l].tanh()[None,:,None]*h).flatten(1) for l,h in enumerate(H)],1)


def rotate_T(T,O):
    return torch.cat([(T[:,start:end].reshape(len(T),128,dim)
                      @o3.Irrep(l,(-1)**l).D_from_matrix(O).T).flatten(1)
                     for l,(start,end,dim) in enumerate(LITERAL,1)],1)


def branch_checks(dtype):
    torch.manual_seed(20260930)
    T=torch.randn(4,1920,dtype=dtype)*.2;a=torch.randn(5,128,dtype=dtype)*.3
    index=torch.tensor([[0,2,1,0,2],[1,1,2,2,0]]) # receiver3 has no edges
    probe=torch.randn_like(T);report={}
    modules={mode:TensorTransport(IRREPS,mode).to(dtype=dtype) for mode in MODES}
    for mode,module in modules.items():
        assert sum(p.numel() for p in module.parameters())==384
        assert sum(p.numel() for p in module.parameters() if p.requires_grad)==(0 if mode=='original' else 384)
        assert torch.equal(module(T,a,index),torch.zeros_like(T))
        if mode=='original':continue
        inp=T.clone().requires_grad_();edge=a.clone().requires_grad_()
        value=(module(inp,edge,index)*probe).sum()
        dt,da,dtheta=torch.autograd.grad(value,(inp,edge,module.theta))
        assert torch.equal(dt,torch.zeros_like(dt)) and torch.equal(da,torch.zeros_like(da))
        H=literal_H(T,a,index,mode)
        expected=torch.stack([(h*probe[:,start:end].reshape(len(T),128,dim)).sum((0,2))
                              for h,(start,end,dim) in zip(H,LITERAL)])
        close(dtheta,expected,report,mode+'_theta0_gradient');assert dtheta.norm()>0
        with torch.no_grad():module.theta.copy_(torch.linspace(-.2,.2,384,dtype=dtype).reshape(3,128))
        out=module(T,a,index);ref=literal_delta(T,a,index,mode,module.theta)
        close(out,ref,report,mode+'_literal_edge_slice_reference')
        assert torch.equal(out[3],torch.zeros_like(out[3]))
        inp=T.clone().requires_grad_();edge=a.clone().requires_grad_()
        grads=torch.autograd.grad((module(inp,edge,index)*probe).sum(),(inp,edge,module.theta))
        assert all(torch.isfinite(g).all() and g.norm()>0 for g in grads)
        independent=torch.autograd.grad((literal_delta(inp,edge,index,mode,module.theta)*probe).sum(),
                                        (inp,edge,module.theta))
        for k,(g,h) in enumerate(zip(grads,independent)):close(g,h,report,f'{mode}_nonzero_gradient_{k}')
        for det in (1,-1):
            O=torch.linalg.qr(torch.randn(3,3,dtype=dtype))[0];O=O*torch.linalg.det(O)*det
            close(module(rotate_T(T,O),a,index),rotate_T(out,O),report,f'{mode}_O3_{det}')
        perm=torch.tensor([3,2,0,1]);inverse=torch.argsort(perm);ep=torch.tensor([4,1,3,0,2])
        close(module(T[perm],a[ep],inverse[index[:,ep]]),out[perm],report,mode+'_node_edge_permutation')
        empty=module(T,a[:0],index[:,:0]);assert torch.equal(empty,torch.zeros_like(T))
        one=torch.tensor([[0],[1]]);x=T.clone();y=T.clone();y[0]=y[0]+.3
        before=module(x,a[:1],one);after=module(y,a[:1],one)
        if mode=='neighbor':assert (after[1]-before[1]).norm()>0
        else:assert torch.equal(after[1],before[1])
        with torch.no_grad():
            module.theta.fill_(.1);positive=module(T,a,index)
            module.theta.fill_(-.1);negative=module(T,a,index)
        close(negative,-positive,report,mode+'_signed_coefficient')
        assert torch.equal(module(torch.zeros_like(T),a,index),torch.zeros_like(T))
        report[mode+'_identity_gradients_liveness_receiver_empty_nonlocal_pass']=True
    report['identity_delta_bitwise']=True
    return report


def wrapped_block_checks(old):
    torch.manual_seed(20260930);report={}
    S=torch.randn(4,128)*.1;T=torch.randn(4,1920)*.1;rbf=torch.randn(5,32)*.1
    index=torch.tensor([[0,2,1,3,2],[1,0,2,2,3]])
    sh=torch.randn(5,15)*.1
    with torch.no_grad():old_output=old(S,T,rbf,sh,index)
    for mode in MODES:
        wrapped=TransportBlock(copy.deepcopy(old),mode)
        calls=[];hook=wrapped.message.Attention.register_forward_hook(lambda *_:calls.append(1))
        with torch.no_grad():actual=wrapped(S,T,rbf,sh,index)
        assert len(calls)==1
        assert all(torch.equal(a,b) for a,b in zip(actual,old_output))
        if mode!='original':
            with torch.no_grad():wrapped.transport.theta.fill_(.1)
            calls.clear()
            with torch.no_grad():actual=wrapped(S,T,rbf,sh,index)
            assert len(calls)==1
            with torch.no_grad():
                edge=old.message.Attention(S,rbf,index);a,mijs=edge.split(128,-1)
                mijt=old.message.tp(a,sh);j=index[1];u=old.update
                ut=u.outt(scatter(mijt,j,dim=0));us=u.actu(u.outs(scatter(mijs,j,dim=0)))
                delta=literal_delta(T,a,index,mode,wrapped.transport.theta)
                wrong=literal_delta(T+ut,a,index,mode,wrapped.transport.theta)
                assert (wrong-delta).norm()>0,'Fixture must distinguish pre-block input from T+ut'
                tm=T+ut+delta;sm=S+u.drop(us);ut2,us2=u.uattn(tm,sm)
                expected=(sm+u.drop(us2),tm+ut2)
            for k,(x,y) in enumerate(zip(actual,expected)):close(x,y,report,f'{mode}_preblock_insertion_{k}')
        hook.remove();report[mode+'_one_attention_call_and_identity']=True
    return report


def mask_checks():
    torch.manual_seed(20260930);E=torch.rand(2,10,requires_grad=True)
    A=torch.randn(2,10,3,3,requires_grad=True);mask=torch.ones(2,10,dtype=torch.bool);mask[0,2]=False
    et=torch.rand(2,10);at=torch.randn(2,10,3,3);et[~mask]=float('nan');at[~mask]=float('nan')
    loss=base_loss((E,A),{'E':et,'A':at,'mask_E':mask,'mask_A':mask},{'sE2':.5,'sA2':.1})
    assert all(torch.isfinite(v) for v in loss.values());loss['total'].backward()
    assert torch.isfinite(E.grad).all() and torch.isfinite(A.grad).all()
    assert torch.equal(E.grad[~mask],torch.zeros_like(E.grad[~mask]))
    assert torch.equal(A.grad[~mask],torch.zeros_like(A.grad[~mask]))
    return {'inherited_masked_target_NaN_forward_backward_pass':True}


def geometry():
    z=torch.tensor([6,1,1,1,1,8,1,1]);batch=torch.tensor([0,0,0,0,0,1,1,1])
    pos=torch.tensor([[0,0,0],[.63,.63,.63],[-.63,-.63,.63],[-.63,.63,-.63],[.63,-.63,-.63],
                      [0,0,0],[.76,0,.59],[-.76,0,.59]],dtype=torch.float32)
    edges=[(i,j) for i in range(8) for j in range(8) if i!=j and batch[i]==batch[j]]
    return {'z':z,'pos':pos,'batch':batch,'n':2,'edge_index':torch.tensor(edges).T.contiguous()}


def main():
    start=time.time();assert not (ROOT/'CPU_PREFLIGHT.json').exists()
    review=read(ROOT/'CPU_SOURCE_REVIEW.json')
    assert review['passed'] and review['scope']=='round08_cpu_synthetic_only'
    assert sha(ROOT/'ROUND08_PREPARATION_DECISION.md')==review['root_preparation_decision_sha256']==DECISION
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
    torch.set_num_threads(2)
    report={'module_float64':branch_checks(torch.float64),'module_float32':branch_checks(torch.float32),
            'target_mask_fixture':mask_checks(),'model_updates':0,'real_geometry_or_targets_read':False,
            'synthetic_geometry_only':True}
    stats=read(ROOT/'TRAIN_STATISTICS.json');mc=read(ROOT/'model_config.json');x=geometry()
    assert sha(ROOT/'TRAIN_STATISTICS.json')=='d3d0ed5af2646959be0abcb9cebb50d7cdb2ef7fb70d7e073fe12a8b697953ae'
    before=torch.get_rng_state().clone()
    with forbid_external_data() as opened:
        base=build_original(mc,stats).eval()
        models={mode:build_fresh(mc,stats,mode).eval() for mode in MODES}
    assert not opened and torch.equal(before,torch.get_rng_state())
    basehash=tensor_hash(base.state_dict());assert basehash=='231dfaf3ffc8056e851ddac34e941fa34eee3af0f7ff9687ecab6d24d6fa1cd2'
    fullhashes=[tensor_hash(model.state_dict()) for model in models.values()];assert len(set(fullhashes))==1
    gatehashes=[tensor_hash(transport_state(model)) for model in models.values()];assert len(set(gatehashes))==1
    report.update(initial_base_tensor_sha256=basehash,initial_full_tensor_sha256=fullhashes[0],
                  initial_transport_tensor_sha256=gatehashes[0],train_statistics_sha256=sha(ROOT/'TRAIN_STATISTICS.json'))
    report['wrapped_block']=wrapped_block_checks(base.core.blocks[1])
    with torch.no_grad():reference=base(**x)
    report['full_models']={}
    for mode,model in models.items():
        assert tensor_hash(base_state(model))==basehash
        assert tensor_hash(inherited_state(model))=='3c5d20463a6aa1d4e7c25ee5e57e0ceaa53c17e5a89af35d7761b58e58d661f9'
        theta=[p for name,p in model.named_parameters() if '.transport.' in name]
        assert len(theta)==2 and sum(p.numel() for p in theta)==768
        assert sum(p.numel() for p in theta if p.requires_grad)==(0 if mode=='original' else 768)
        assert not any(p.requires_grad for p in model.right_adapter.parameters())
        assert not hasattr(model.core.blocks[0],'transport')
        with torch.no_grad():out=model(**x)
        assert all(torch.equal(a,b) for a,b in zip(out,reference))
        mr={'identity_E_A_bitwise':True,'gate_active_parameters':sum(p.numel() for p in theta if p.requires_grad)}
        if mode!='original':
            with torch.no_grad():
                for p in theta:p.fill_(.1)
        O=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        permutation=torch.tensor([4,2,0,3,1,7,5,6]);inverse=torch.argsort(permutation)
        xp=dict(x,z=x['z'][permutation],pos=x['pos'][permutation]@O.T+torch.tensor([2.,-3.,1.]),
                batch=x['batch'][permutation],edge_index=inverse[x['edge_index']])
        with torch.no_grad():out=model(**x);moved=model(**xp)
        close(moved[0],out[0],mr,'E_reflection_permutation_translation')
        close(moved[1],O@out[1]@O.T,mr,'A_reflection_permutation_translation')
        f=lambda p:p[0].double()*p[1].double().diagonal(dim1=-2,dim2=-1).sum(-1)*2/(3*27.211386245988)
        mr['native_f_symmetry_max_abs']=float((f(out)-f(moved)).abs().max())
        with tempfile.TemporaryDirectory(prefix='round08_synthetic_') as directory:
            path=Path(directory)/'synthetic.pt';atomic_torch(export_payload(model,mc,stats,{'synthetic_only':True}),path)
            with forbid_external_data(path) as opened:
                loaded=load_predictor(path)
                with torch.no_grad():actual=loaded(**x)
            assert set(opened)=={str(path.resolve())} and all(torch.equal(a,b) for a,b in zip(out,actual))
        mr['one_checkpoint_load_forward_access_and_bitwise_parity']=True
        report['full_models'][mode]=mr
    report.update(source_hashes=review['source_hashes'],source_review_sha256=sha(ROOT/'CPU_SOURCE_REVIEW.json'),
                  root_preparation_decision_sha256=DECISION,seconds=time.time()-start,passed=True)
    immutable_json(report,ROOT/'CPU_PREFLIGHT.json');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
