import copy,hashlib,importlib.util,json,random,time
import numpy as np
import torch
from e3nn import o3
from dataset import Data,ROOT
from model_factory import build,MTOEA
from objective import losses
from trainer import setup,state_hash,atomic_json,atomic_save

def close(a,b,atol=1e-7,rtol=1e-5):
    assert torch.allclose(a,b,atol=atol,rtol=rtol),float((a-b).abs().max())
def optim(m,c):return torch.optim.Adam(m.parameters(),lr=c['lr'],amsgrad=True,weight_decay=c['weight_decay'])
def scheduler(o,c):return torch.optim.lr_scheduler.ReduceLROnPlateau(o,factor=.5,patience=c['plateau_patience'],threshold=1e-4,threshold_mode='rel',min_lr=c['min_lr'])
def step(m,o,x,y,eta):
    o.zero_grad(set_to_none=True);v=losses(m(**x),y,data.stats,eta)[0];v.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);o.step();return float(v)

def main():
    global data
    setup(11);data=Data();cfgs={n:json.loads((ROOT/'configs'/f'{n}.json').read_text()) for n in ('G1','G2','G3','G4')}
    report={};x,y=data.batch(data.parts['train'][:4])
    models={n:build(c,data.stats).cuda() for n,c in cfgs.items()}
    hashes={n:state_hash(m.state_dict()) for n,m in models.items()}
    assert hashes['G1']==hashes['G2'] and hashes['G3']==hashes['G4']
    for k,v in models['G1'].state_dict().items():
        if k.startswith(('core.','mto.')):
            assert all(torch.equal(v,m.state_dict()[k]) for m in models.values()),k
    report['initialization_equal']=hashes
    expected=json.loads((ROOT/'reports/epoch_order_hashes.json').read_text())
    for c in cfgs.values():
        rng=np.random.default_rng(c['seed'])
        for row in expected:assert hashlib.sha256(rng.permutation(data.parts['train']).tobytes()).hexdigest()==row['sha256']
    report['all_1000_epoch_orders_equal']=True
    # Independent pinned forward and pinned eta=1 objective.
    spec=importlib.util.spec_from_file_location('pinned_objective',ROOT/'pinned_reference/objective.py');refobj=importlib.util.module_from_spec(spec);spec.loader.exec_module(refobj)
    torch.manual_seed(11);ref=MTOEA(cfgs['G1'],data.stats).cuda()
    assert state_hash(ref.state_dict())==hashes['G1']
    eq={}
    for dtype in (torch.float32,torch.float64):
        m=models['G1'].to(dtype);ref=ref.to(dtype)
        xx={k:v.to(dtype) if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in x.items()}
        yy={k:v.to(dtype) if v.is_floating_point() else v for k,v in y.items()}
        p=m(**xx);r=ref(**xx);tol=2e-5 if dtype==torch.float32 else 1e-10
        for u,v in zip(p,r):close(u,v,tol,tol)
        # Reference implementation of the weighted objective: the pinned decomposition with
        # only the trace term replaced by its E_true^2-weighted form.
        def reference(pred,ref_target,ref_stats):
            Er,Ar=pred;ma=ref_target['mask_A'];me=ref_target['mask_E']
            w=ref_target['E'].detach()**2/ref_stats['mean_E2_train']
            valid=ma&me
            ls=(w[valid]*(Ar.diagonal(dim1=-2,dim2=-1).sum(-1)[valid]
                          -ref_target['A'].diagonal(dim1=-2,dim2=-1).sum(-1)[valid]).square()).mean()/(3*ref_stats['sA2'])
            le=(Er[me]-ref_target['E'][me]).square().mean()/ref_stats['sE2']
            a=Ar[ma];at=ref_target['A'][ma]
            s=a.diagonal(dim1=-2,dim2=-1).sum(-1);st=at.diagonal(dim1=-2,dim2=-1).sum(-1)
            eye=torch.eye(3,device=Ar.device,dtype=Ar.dtype)
            q=a-s[:,None,None]*eye/3;qt=at-st[:,None,None]*eye/3
            lq=(q-qt).square().sum((-1,-2)).mean()/ref_stats['sA2']
            return le+ls+lq,le,ls,lq
        a=losses(p,yy,data.stats,1)[0];b=reference(r,yy,data.stats)[0]
        ga=torch.autograd.grad(a,tuple(m.parameters()),allow_unused=True);gb=torch.autograd.grad(b,tuple(ref.parameters()),allow_unused=True)
        delta=max(float((u-v).abs().max()) for u,v in zip(ga,gb) if u is not None)
        close(a,b,tol,tol)
        for u,v in zip(ga,gb):
            assert (u is None)==(v is None)
            if u is not None:close(u,v,tol,tol)
        # Decomposition identity now holds for weighted trace + weighted-free quadrupole:
        # sum_valid w*(s-st)^2 /(3 sA2) + mean_valid sum(Q-Qt)^2 / sA2.
        p=m(**xx);decomposed=losses(p,yy,data.stats,1)
        w=yy['E'].detach()**2/data.stats['mean_E2_train'];valid=yy['mask_A']&yy['mask_E']
        tr_p=p[1].diagonal(dim1=-2,dim2=-1).sum(-1);tr_t=yy['A'].diagonal(dim1=-2,dim2=-1).sum(-1)
        direct_ls=(w[valid]*(tr_p[valid]-tr_t[valid]).square()).mean()/(3*data.stats['sA2'])
        tol=2e-5 if dtype==torch.float32 else 1e-10
        close(decomposed[2],direct_ls,tol,tol)
        # "Only the trace term changed": run the pinned (unweighted) objective on the same predictions
        # and require the E and Q components to be bit-identical while the trace component differs.
        old=refobj.losses(p,yy,data.stats,1)
        close(decomposed[1],old[1],tol,tol)
        close(decomposed[3],old[3],tol,tol)
        trace_delta=float((decomposed[2]-old[2]).abs().max())
        assert trace_delta>0,'the weighted trace term must differ from the unweighted one on real data'
        eq[str(dtype)]=dict(forward_max_abs=max(float((u-v).abs().max()) for u,v in zip(p,r)),pinned_reference_gradient_max_abs=delta,weighted_trace_term_max_abs=abs(float(decomposed[2]-direct_ls)),absolute_and_relative_tolerance=tol,weighted_minus_pinned_trace_term=trace_delta,LE_and_LQ_identical_to_pinned=True)
    report['G1_reference_equivalence']=eq;models['G1'].float();del ref
    # The trace objective cannot penalize a traceless change, unlike diagonal MSE.
    tr_ones=torch.ones(2,10,dtype=torch.bool,device='cuda')
    E=torch.zeros(2,10,device='cuda',requires_grad=True)
    A=torch.eye(3,device='cuda').expand(2,10,3,3).clone().requires_grad_()   # predicted trace = 3
    # truth trace 3 and truth E nonzero: a weight is present yet the error is zero
    yy=dict(E=torch.full((2,10),3.,device='cuda'),A=A.detach().clone()+torch.diag(torch.tensor([.5,-.5,0.],device='cuda')),
            mask_E=tr_ones,mask_A=tr_ones)
    assert float(losses((E,A),yy,data.stats,0)[0])==0
    # truth trace 0 (full error of 3) with truth E=3 -> the weighted error is penalised
    yy['A'].zero_()
    close(losses((E,A),yy,data.stats,0)[0],torch.tensor((9./data.stats['mean_E2_train'])*9/(3*data.stats['sA2']),device='cuda'))
    # same full trace error but truth E=0 -> the weight zeroes the term even though the error is 3
    yy2={k:v.clone() for k,v in yy.items()};yy2['E'].zero_()
    assert float(losses((E,A),yy2,data.stats,0)[0])==0
    for eta in (0,1):
        ym={k:v.clone() for k,v in yy.items()};ym['mask_A'][0]=False;ym['mask_E'][0]=False
        before=losses((E,A),ym,data.stats,eta)[0];ym['A'][0]=float('nan');ym['E'][0]=float('nan')
        after=losses((E,A),ym,data.stats,eta)[0];close(before,after)
        gg=torch.autograd.grad(after,(E,A),retain_graph=True,allow_unused=True)
        assert all(torch.isfinite(g).all() for g in gg if g is not None)
        ym['mask_A'].zero_();ym['mask_E'].zero_();assert float(losses((E,A),ym,data.stats,eta)[0])==0
    report['trace_not_diagonal_MSE_mask_NaN_empty_and_valid_zero_labels']=True
    # --- E_true^2 trace weighting: frozen constant, no gradient path through E, exact equivalence ---
    m2E=data.stats['mean_E2_train']
    assert m2E==float(json.loads((ROOT/'data/trace_weight.json').read_text())['mean_E2_train'])
    # normalization.json must be byte-identical to the original: the new constant lives in its own file
    frozen_norm=float(json.loads((ROOT/'frozen_reference/data/normalization.json').read_text())['sA2'])
    assert json.loads((ROOT/'data/normalization.json').read_text())['sA2']==frozen_norm
    assert 'mean_E2_train' not in json.loads((ROOT/'data/normalization.json').read_text())
    # 4 molecules x 1 state, laid out exactly like the real batch: A is (B,S,3,3), E and masks are (B,S).
    N=4
    def diag_batch(tr):
        B=torch.zeros(N,1,3,3,device='cuda');B[:,0,0,0]=tr;return B
    ttr=torch.tensor([2.,2.,6.,6.],device='cuda')      # truth trace == truth energy in this fixture
    tE=ttr[:,None];A_t=diag_batch(ttr)
    E_zero=torch.zeros(N,1,device='cuda')
    ok=torch.ones(N,1,dtype=torch.bool,device='cuda')
    yv=dict(E=tE.clone(),A=A_t.clone(),mask_E=ok,mask_A=ok)
    # prediction equal to truth -> exactly zero
    close(losses((E_zero,A_t.clone()),yv,data.stats,0)[2],torch.tensor(0.,device='cuda'),1e-12,1e-12)
    # the SAME absolute trace error costs (E_bright/E_dim)^2 times more at a bright state
    dim_pred=diag_batch(ttr-torch.tensor([1.,0.,0.,0.],device='cuda'))     # error 1 at E_true=2
    bright_pred=diag_batch(ttr-torch.tensor([0.,0.,1.,0.],device='cuda'))  # error 1 at E_true=6
    ls_dim=float(losses((E_zero,dim_pred),yv,data.stats,0)[2])
    ls_bright=float(losses((E_zero,bright_pred),yv,data.stats,0)[2])
    ratio=ls_bright/ls_dim
    assert abs(ratio-(6.**2/2.**2))<1e-6,('equal absolute errors must cost (E_bright/E_dim)^2 times more',ratio)
    # exact equivalence to the closed form on a non-degenerate random batch
    gm=torch.Generator(device='cuda').manual_seed(11)
    gp=torch.randn(5,10,3,3,device='cuda',generator=gm);gt=torch.randn(5,10,3,3,device='cuda',generator=gm)
    gE=torch.rand(5,10,device='cuda',generator=gm)*8+1
    gm5=torch.ones(5,10,dtype=torch.bool,device='cuda')
    gv=dict(E=gE.clone(),A=gt.clone(),mask_E=gm5,mask_A=gm5)
    got=losses((E_zero.new_zeros(5,10),gp),gv,data.stats,0)[2]
    sp=gp.diagonal(dim1=-2,dim2=-1).sum(-1);st=gt.diagonal(dim1=-2,dim2=-1).sum(-1)
    want=((gE**2/m2E)*(sp-st).square()).mean()/(3*data.stats['sA2'])
    close(got,want,1e-10,1e-10)
    # The trace term depends on the predicted E for nothing at all: with only predicted E and predicted A
    # as inputs it must not even build a graph through the predicted E.
    Epred=torch.full((5,10),7.,device='cuda',requires_grad=True)
    tmp=losses((Epred,gp),gv,data.stats,0)[2]
    assert not tmp.requires_grad,'the weighted trace term must not depend on the predicted E at all'
    # The gradient w.r.t. the predicted A must be unchanged when the predicted E changes.
    gp2=torch.randn(5,10,3,3,device='cuda',generator=gm)
    def trace_grad_wrt_A(pred_e):
        Ap=gp2.clone().requires_grad_()
        losses((pred_e,Ap),gv,data.stats,0)[2].backward()
        return Ap.grad.detach().clone()
    zero_g=trace_grad_wrt_A(torch.zeros(5,10,device='cuda'))
    big_g=trace_grad_wrt_A(torch.full((5,10),99.,device='cuda'))
    assert zero_g.abs().sum()>0,'the trace term must actually depend on the predicted A'
    assert torch.equal(zero_g,big_g),'the trace gradient must not depend on the predicted E'
    other=losses((torch.zeros(5,10,device='cuda'),gp2),gv,data.stats,0)[2]
    same=losses((torch.full((5,10),99.,device='cuda'),gp2),gv,data.stats,0)[2]
    close(other,same,0,0)
    report['weighted_trace_loss']=dict(mean_E2_train=m2E,zero_when_correct=True,
        bright_to_dim_penalty_ratio=ratio,closed_form_max_abs=float((got-want).abs().max()),
        no_gradient_into_E_or_through_prediction=True,weight_uses_truth_E_only=True)
    # The frozen constant must equal the plain mean over exactly the valid train entries. Recompute it
    # here straight from the shipped data files, independently of prepare.py and of normalization.json.
    with np.load(ROOT/'data/dataset.npz') as dd, np.load(ROOT/'data/raw_labels.npz') as rr:
        onehot=np.zeros(dd['E'].shape[0],dtype=bool);onehot[dd['train']]=True
        sel=onehot[:,None]&rr['mask_A']&rr['mask_E']
        # the constant is defined on the float32 copy training sees, so recompute it the same way
        indep=float(np.mean(np.square(dd['E'].astype(np.float64)[sel])))
        assert not sel[~onehot].any(),'a non-train entry is in the preflight recomputation'
        assert int(sel.sum())==int(onehot.sum())*int(rr['mask_E'].shape[1]),'train entries are not mask-complete'
        # and it must NOT equal the all-split mean, which would signal leakage of val/test statistics
        allmean=float(np.mean(np.square(rr['E'][rr['mask_A']&rr['mask_E']]).astype(np.float64)))
    assert abs(m2E-indep)<1e-7,('the frozen constant disagrees with the shipped train data',m2E,indep)
    assert m2E!=allmean,'the constant must be train-only, not the all-split mean'
    report['weighted_trace_loss'].update(train_valid_entries=int(sel.sum()),
        independent_train_recomputation=indep,all_split_mean_for_contrast=allmean,
        train_only_constant_verified=True)
    for n in ('G2','G4'):
        m=models[n]
        # (a) E labels enter the eta=0 branch ONLY through the fixed weight, so neither the loss nor any
        # parameter gradient may depend on the *predicted* E. This is what "no E prediction loss" means
        # now that the truth E is read. ONE forward is reused for both variants and the gradients are
        # taken on that single graph: separate forwards would differ by CUDA index_add rounding (this run
        # is documented as not bitwise deterministic) and would mask the property under test.
        e_real,a_real=m(**x)
        params=[p for p in m.parameters() if p.requires_grad]
        def loss_and_grads(pred_e):
            v=losses((pred_e,a_real),y,data.stats,0)[0]
            g=torch.autograd.grad(v,params,retain_graph=True,allow_unused=True)
            return float(v),g
        v_real,g_real=loss_and_grads(e_real.detach())
        v_fake,g_fake=loss_and_grads(torch.full_like(e_real,99.))
        assert v_real==v_fake,'the eta=0 loss must not depend on the predicted E'
        for u,v in zip(g_real,g_fake):
            assert (u is None)==(v is None)
            if u is not None:assert torch.equal(u,v),'the predicted E must not move any eta=0 gradient'
        # (b) but the truth E is live: it sets the weight, so perturbing it must change the loss.
        yy2={k:v.clone() for k,v in y.items()};yy2['E']=yy2['E']+1.0
        v_t2=float(losses((e_real,a_real),yy2,data.stats,0)[0])
        assert v_real!=v_t2,'the eta=0 loss must read E_true through the fixed weight'
        # (c) no gradient reaches the frozen energy head, and a step leaves it bit-identical.
        frozen={k:p.clone() for k,p in m.named_parameters() if not p.requires_grad}
        assert set(frozen)=={'decoder.energy_offset','decoder.energy_head.weight','decoder.energy_head.bias'}
        o=optim(m,cfgs[n]);step(m,o,x,y,0)
        assert all(torch.equal(v,dict(m.named_parameters())[k]) for k,v in frozen.items())
        report[n+'_no_E_prediction_loss_frozen_head_weighted_trace']=dict(passed=True,
            loss_invariant_to_predicted_E=True,loss_change_from_E_true=abs(v_real-v_t2),
            frozen_parameters=list(frozen),
            note='E labels feed only the frozen E_true^2 weight; no E prediction term, no gradient into the energy head, and no dependence on the predicted E')
    # Fresh P-only model; decoder receives only the two irreps in P.
    m=build(cfgs['G3'],data.stats).cuda().double().eval()
    xx={k:v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in x.items()}
    with torch.no_grad():
        e,a,mu=m(**xx,export=True);assert float(mu.square().sum())>0
        assert not any(('beta' in k or 'tensor_gate' in k or 'cartesian_basis' in k) for k in m.decoder.state_dict())
        _c=cfgs['G3']['mto_channels']
        assert m.cg.tp.irreps_out==o3.Irreps(f'{_c}x0e + {_c}x1o')
        close(a,a.transpose(-1,-2),atol=0,rtol=0);close(a.diagonal(dim1=-2,dim2=-1).sum(-1),mu.square().sum(-1),atol=1e-12,rtol=1e-12)
        eig=torch.linalg.eigvalsh(a);assert float(eig.min())>-1e-11 and float(eig[...,:2].abs().max())<1e-11
        rows=[]
        for r in [o3.rand_matrix(dtype=torch.double,device='cuda') for _ in range(3)]+[-torch.eye(3,dtype=torch.double,device='cuda')]:
            er,ar,mur=m(**dict(xx,pos=xx['pos']@r.T),export=True)
            close(er,e,1e-8,1e-8);close(mur,mu@r.T,1e-8,1e-8);close(ar,r@a@r.T,1e-8,1e-8)
            rows.append(dict(det=float(torch.det(r)),mu_error=float((mur-mu@r.T).abs().max()),A_error=float((ar-r@a@r.T).abs().max())))
        perm=torch.randperm(len(xx['z']),device='cuda');inverse=torch.argsort(perm)
        xp=dict(xx,z=xx['z'][perm],pos=xx['pos'][perm],batch=xx['batch'][perm],edge_index=inverse[xx['edge_index']])
        ep,ap,mup=m(**xp,export=True);close(ep,e,1e-8,1e-8);close(ap,a,1e-8,1e-8);close(mup,mu,1e-8,1e-8)
        captured=[]
        hook=m.decoder.register_forward_pre_hook(lambda module,args:captured.append(args))
        m(**xx);hook.remove();assert len(captured)==1 and len(captured[0])==1 and set(captured[0][0])=={'0e','1o'}
        # Centrosymmetric synthetic octahedron, all atoms identical.
        pos=torch.cat([torch.eye(3),-torch.eye(3)]).to(device='cuda',dtype=torch.double)
        ii,jj=torch.where(~torch.eye(6,dtype=torch.bool,device='cuda'))
        xs=dict(z=torch.ones(6,dtype=torch.long,device='cuda'),pos=pos,batch=torch.zeros(6,dtype=torch.long,device='cuda'),n=1,edge_index=torch.stack([ii,jj]))
        _,ac,muc=m(**xs,export=True);assert float(muc.abs().max())<1e-10
        report['centrosymmetric_geometry']=dict(mu_max_abs=float(muc.abs().max()),A_max_abs=float(ac.abs().max()),limitation='Ordinary polar 1o head forced to zero by inversion; nonzero transition intensity cannot be represented. No samples removed.')
        report['P_only_equivariance']=dict(rotations_and_inversion=rows,permutation_mu_error=float((mup-mu).abs().max()),minimum_eigenvalue=float(eig.min()),rank_at_most_one=True,decoder_only_P=True)
    # FP32 production batch=64 and resumable Adam/AMSGrad, scheduler and RNG.
    x64,y64=data.batch(data.parts['train'][:64]);resume={}
    for n in cfgs:
        c=cfgs[n];m=build(c,data.stats).cuda();o=optim(m,c);s=scheduler(o,c)
        first=step(m,o,x64,y64,c['eta']);s.step(first)
        g=np.random.default_rng(11);order=g.permutation(data.parts['train']);cursor=64
        ck=dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),cursor=cursor,order=order,
            rng_numpy=g.bit_generator.state,rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),rng_python=random.getstate(),rng_numpy_global=np.random.get_state())
        path=ROOT/'reports'/f'scratch_resume_{n}.pt';atomic_save(ck,path)
        xn,yn=data.batch(order[cursor:cursor+64]);continued=step(m,o,xn,yn,c['eta']);s.step(continued)
        target={k:v.clone() for k,v in m.state_dict().items()}
        m2=build(c,data.stats).cuda();o2=optim(m2,c);s2=scheduler(o2,c);ck=torch.load(path,map_location='cuda',weights_only=False)
        m2.load_state_dict(ck['model']);o2.load_state_dict(ck['optimizer']);s2.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['rng_torch'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu());random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy_global']);g.bit_generator.state=ck['rng_numpy']
        xr,yr=data.batch(ck['order'][ck['cursor']:ck['cursor']+64]);restored=step(m2,o2,xr,yr,c['eta']);s2.step(restored)
        delta=max(float((v-m2.state_dict()[k]).abs().max()) for k,v in target.items() if v.numel())
        assert abs(restored-continued)<2e-5*max(1,abs(continued)) and delta<2e-6,(n,delta,continued,restored)
        for k,v in s.state_dict().items():
            if isinstance(v,float):assert v==s2.state_dict()[k] or abs(v-s2.state_dict()[k])<2e-5*max(1,abs(v))
            else:assert v==s2.state_dict()[k]
        resume[n]=dict(batch_size=64,first_loss=first,next_loss=continued,resumed_loss=restored,model_max_abs_difference=delta,optimizer_steps=int(next(iter(o2.state.values()))['step']))
        path.unlink();del m,m2,o,o2
    report['FP32_gradient_and_resume']=resume
    report.update(passed=True,time=time.time(),scratch_training_discarded=True,formal_runs_load_frozen_initial_files=True)
    atomic_json(report,ROOT/'reports/preflight.json');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
