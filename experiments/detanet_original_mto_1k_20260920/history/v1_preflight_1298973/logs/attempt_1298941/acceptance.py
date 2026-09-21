"""Blocking acceptance suite. Tolerances fixed before any numerical execution."""
import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import unittest
import torch
from e3nn import o3
from torch_geometric.nn import radius_graph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import (UV, TYPES, VARIANTS, CompatibleDetaNet, make_core, pack, unpack,
                    pool, build_triplet, MolecularTensorOrbitals, ReferenceStateCoupling,
                    PhysicalSpectrumDecoder, parameter_counts, EV_PER_HARTREE)
from data_protocol import load, collate

DEVICE = 'cpu'
MEASUREMENTS = {}
CFG = json.loads((ROOT/'protocol.json').read_text())
DTYPE = torch.float64


def close(name, x, y, atol=None, rtol=None):
    # CUDA scatter uses atomic additions. Exact CPU operator comparisons remain exact;
    # GPU comparisons use the precision-specific thresholds fixed before GPU execution.
    if DEVICE == 'cuda' and atol == 0. and rtol == 0.:
        atol, rtol = None, None
    atol = (1e-8 if x.dtype == torch.float64 else 1e-5) if atol is None else atol
    rtol = (1e-6 if x.dtype == torch.float64 else 1e-4) if rtol is None else rtol
    err = (x-y).abs()
    MEASUREMENTS[name] = dict(max_abs=float(err.max()),
        max_relative=float((err/y.abs().clamp_min(1e-12)).max()), atol=atol, rtol=rtol)
    torch.testing.assert_close(x, y, atol=atol, rtol=rtol, msg=name)


def cpu_tree(v):
    if torch.is_tensor(v): return v.detach().cpu().clone()
    if isinstance(v, (tuple,list)): return [cpu_tree(x) for x in v]
    return v


def transform(v, ir, R):
    return v @ o3.Irrep(ir).D_from_matrix(R).T


class Acceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_default_dtype(DTYPE)
        torch.manual_seed(20260920)
        cls.data, cls.info = load()
        cls.x, cls.y = collate(cls.data['train'][:2], DEVICE, DTYPE)
        cls.models = build_triplet(CFG, cls.info['n_ref'], cls.info['grid'], 11)
        for model in cls.models.values(): model.to(device=DEVICE,dtype=DTYPE)

    def test_01_reference_independent_layers_and_backward(self):
        from reference.detanet_model.detanet import DetaNet as Original
        assert Original is not CompatibleDetaNet
        assert 'reference' in importlib.import_module(Original.__module__).__file__
        assert 'vendor' in importlib.import_module(CompatibleDetaNet.__bases__[0].__module__).__file__
        for label, official in [('fixed_random', False), ('official_uv', True)]:
            ref = Original(**UV, device=torch.device(DEVICE)).to(device=DEVICE,dtype=DTYPE)
            # Reference source remains byte-for-byte untouched; manually migrate its ordinary constants.
            ref.Embedding.elec = ref.Embedding.elec.to(device=DEVICE,dtype=DTYPE)
            ref.mass = ref.mass.to(device=DEVICE,dtype=DTYPE)
            adapted = make_core().to(device=DEVICE,dtype=DTYPE)
            weights = torch.load(ROOT/'reference/trained_param/qm9spectra/borden_os.pth',
                                 map_location='cpu', weights_only=True) if official else ref.state_dict()
            ref.load_state_dict(weights, strict=True)
            adapted.load_state_dict(weights, strict=True)
            traces, handles = [{},{}], []
            for model, trace in zip((ref, adapted), traces):
                for name in ['Embedding','Radial','blocks.0','blocks.1','blocks.2','sout']:
                    def hook(module, inputs, output, name=name, trace=trace):
                        trace[name] = cpu_tree(output)
                    handles.append(model.get_submodule(name).register_forward_hook(hook))
            args = {k:v for k,v in self.x.items() if k != 'n'}
            p, q = ref(**args), adapted(**args)
            close(label+'/spectrum', p, q, atol=0., rtol=0.)
            for name in traces[0]:
                left, right = traces[0][name], traces[1][name]
                if isinstance(left,list):
                    for i,(a,b) in enumerate(zip(left,right)): close(label+'/'+name+f'/{i}',a,b,0.,0.)
                else: close(label+'/'+name,left,right,0.,0.)
            w = torch.linspace(.1,1.,240,device=DEVICE)
            (p*w).sum().backward(); (q*w).sum().backward()
            max_grad = 0.
            for (name,a),(name2,b) in zip(ref.named_parameters(),adapted.named_parameters()):
                assert name == name2 and (a.grad is None) == (b.grad is None)
                if a.grad is not None:
                    close(label+'/grad/'+name,a.grad,b.grad,0.,0.)
                    max_grad = max(max_grad,float(a.grad.abs().max()))
            MEASUREMENTS[label+'/strict_load'] = True
            MEASUREMENTS[label+'/max_gradient_magnitude'] = max_grad
            for h in handles:h.remove()

    def test_02_latent_packing_and_initialization(self):
        a,b,c = self.models.values()
        saved={}
        handle=a.core.blocks[-1].register_forward_hook(lambda m,i,o:saved.update(S=o[0],T=o[1]))
        a(**self.x)
        s,t=b.core(**{k:v for k,v in self.x.items() if k!='n'})
        close('latent/S',s,saved['S'],0.,0.);close('latent/T',t,saved['T'],0.,0.)
        assert any(str(ir)=='3o' for _,ir in b.tensor_irreps)
        close('packing/latent',pack(unpack(t,b.tensor_irreps),b.tensor_irreps),t,0.,0.)
        mixed=o3.Irreps('2x1o+3x0e+1x1o+4x2e')
        x=torch.randn(5,mixed.dim,device=DEVICE)
        close('packing/repeated_irrep',pack(unpack(x,mixed),mixed),x,0.,0.)
        for k,v in b.core.state_dict().items():assert torch.equal(v,a.core.state_dict()[k]),k
        for k,v in b.state_dict().items():assert torch.equal(v,c.state_dict()[k]),k
        assert b.state_dict().keys()==c.state_dict().keys()
        handle.remove()

    def test_03_graph_operator_then_graph_builder(self):
        a=self.models[VARIANTS[0]]
        explicit=a(**self.x)
        automatic=a(**{k:v for k,v in self.x.items() if k!='edge_index'})
        close('graph/automatic_vs_explicit',automatic,explicit,0.,0.)
        edge=self.x['edge_index'];pos=self.x['pos'];batch=self.x['batch']
        d=torch.cdist(pos,pos)
        mask=(d<5)&(d>0)&(batch[:,None]==batch[None,:])
        assert set(map(tuple,edge.T.tolist()))==set(map(tuple,mask.nonzero().tolist()))
        assert not (edge[0]==edge[1]).any()
        # Record upstream isolated-node failure separately; never remove dataset molecules.
        isolated=dict(z=torch.tensor([6],device=DEVICE),pos=torch.zeros(1,3,device=DEVICE),
                      batch=torch.zeros(1,dtype=torch.long,device=DEVICE),n=1)
        try: value=a(**isolated);MEASUREMENTS['upstream/isolated_atom']=str(value.shape)
        except (RuntimeError,IndexError) as e:MEASUREMENTS['upstream/isolated_atom']=type(e).__name__+': '+str(e)
        delta=max(1e-7,20*torch.finfo(DTYPE).eps)
        p=torch.tensor([[0.,0.,0.],[5.-delta,0.,0.],[5.,0.,0.],[5.+delta,0.,0.]],device=DEVICE)
        edges=radius_graph(p,r=5.,loop=False,max_num_neighbors=32)
        MEASUREMENTS['graph/boundary_edges']=edges.T.tolist()
        assert not ((edges[0]==0)&(edges[1]==2)).any()
        assert not ((edges[0]==0)&(edges[1]==3)).any()
        assert ((edges[0]==0)&(edges[1]==1)).any()
        dense=torch.randn(40,3,device=DEVICE)*.1
        cap=radius_graph(dense,r=5.,max_num_neighbors=32)
        degrees=torch.bincount(cap[1],minlength=40)
        MEASUREMENTS['graph/dense_degrees']=degrees.tolist()
        if DEVICE=='cuda':
            # torch_cluster 1.6.3: CUDA takes the first 33 radius matches,
            # then the Python wrapper removes self. Nodes >=33 never had self
            # in that list, and retain 33 edges. This is upstream, not repaired.
            expected=torch.tensor([32]*33+[33]*7,device=DEVICE)
            assert torch.equal(degrees,expected),(degrees,expected)
            expected_edges={(i,j) for j in range(40) for i in range(33) if i!=j}
            assert set(map(tuple,cap.T.tolist()))==expected_edges
            # Prove the actual 1k task is outside this capacity boundary.
            records=sum(self.data.values(),[]);largest_degree=0
            for start in range(0,len(records),64):
                cpu,_=collate(records[start:start+64],'cpu',torch.float32)
                gpu,_=collate(records[start:start+64],'cuda',torch.float32)
                assert set(map(tuple,cpu['edge_index'].T.tolist()))==set(map(tuple,gpu['edge_index'].T.tolist()))
                largest_degree=max(largest_degree,int(torch.bincount(cpu['edge_index'][1]).max()))
            assert largest_degree<32
            MEASUREMENTS['graph/all_1000_cpu_gpu_edges_equal']=True
            MEASUREMENTS['graph/actual_1k_max_degree']=largest_degree
        else:
            assert int(degrees.max())==32
        MEASUREMENTS['graph/requested_neighbor_cap']=32

    def test_04_assembly_and_cg_structure(self):
        for label in VARIANTS[1:]:
            model=self.models[label]
            pred,e=model(**self.x,export=True)
            for j,t in enumerate(TYPES):
                close(label+'/F/'+t,e['F'][t],e['c'][:,:,j,:,None]*e['B'][t][:,None],0.,0.)
                close(label+'/M/'+t,e['M'][t],model.mto.n_ref.rsqrt()*pool(e['F'][t],self.x['batch'],2))
                # Independent einsum verifies channel-only projection with m intact.
                close(label+'/W/'+t,e['B'][t],torch.einsum('kb,ibm->ikm',model.mto.project[t].weight,e['H'][t]))
                assert model.mto.project[t].bias is None
            gate=e['c'][self.x['batch']==0]
            atom_var=float(gate.var(0,unbiased=False).max())
            state_var=float(gate.var(1,unbiased=False).max())
            MEASUREMENTS[label+'/gate_variation']=dict(atom=atom_var,state=state_var,
                                                        saturation=float((gate.abs()>.99).double().mean()))
            assert state_var>1e-12
            if label==VARIANTS[2]:close('C/constant_over_atoms',gate,gate[:1].expand_as(gate),0.,0.)
            else:assert atom_var>1e-12
            paths={(str(model.cg.tp.irreps_in1[i.i_in1].ir),str(model.cg.tp.irreps_in2[i.i_in2].ir),
                    str(model.cg.tp.irreps_out[i.i_out].ir)) for i in model.cg.tp.instructions}
            expected={('0e','0e','0e'),('1o','1o','0e'),('2e','2e','0e'),('0e','2e','2e'),
                      ('2e','0e','2e'),('1o','1o','2e'),('2e','2e','2e')}
            assert paths==expected and model.cg.tp.weight.numel()==7*16**3
            MEASUREMENTS[label+'/CG_paths']=sorted(paths)

    def test_05_o3_translation_permutation_batch(self):
        with torch.no_grad():
            for label,model in self.models.items():
                base,be=model(**self.x,export=True)
                for iteration in range(3):
                    R=o3.rand_matrix(dtype=DTYPE,device=DEVICE)
                    if iteration%2:R=-R
                    changed=dict(self.x,pos=self.x['pos']@R.T+torch.randn(3,device=DEVICE))
                    value,e=model(**changed,export=True)
                    close(label+f'/O3/{iteration}/spectrum',value,base)
                    if be:
                        for key in ['E','f','c']:close(label+f'/O3/{iteration}/'+key,e[key],be[key])
                        for key in ['H','B','F','M','CG']:
                            for t in be[key]:close(label+f'/O3/{iteration}/{key}/{t}',e[key][t],transform(be[key][t],t,R))
                        for key in ['Q','C','A']:close(label+f'/O3/{iteration}/'+key,e[key],R@be[key]@R.T)
                reordered=[];atom_order=[];offset=0
                for rec in self.data['train'][:2]:
                    permutation=torch.randperm(len(rec['z']))
                    reordered.append(dict(rec,z=rec['z'][permutation],pos=rec['pos'][permutation]))
                    atom_order.append(permutation+offset);offset+=len(rec['z'])
                x,_=collate(reordered,DEVICE,DTYPE)
                reordered_pred,reordered_extra=model(**x,export=True)
                close(label+'/permutation',reordered_pred,base)
                if be:
                    order=torch.cat(atom_order).to(DEVICE)
                    for key in ['H','B','F']:
                        for t in TYPES:close(label+'/permutation/'+key+'/'+t,reordered_extra[key][t],be[key][t][order])
                    close(label+'/permutation/c',reordered_extra['c'],be['c'][order])
                    for key in ['M','CG']:
                        for t in be[key]:close(label+'/permutation/'+key+'/'+t,reordered_extra[key][t],be[key][t])
                for records,index in [([self.data['train'][0]],0),
                    ([self.data['train'][5],self.data['train'][0],self.data['train'][8]],1)]:
                    x,_=collate(records,DEVICE,DTYPE)
                    close(label+'/batch/'+str(index),model(**x)[index],base[0])

    def test_06_synthetic_o3_and_physics(self):
        model=self.models[VARIANTS[1]]
        for seed in [19,31,47]:
            torch.manual_seed(seed)
            h={t:torch.randn(7,128,o3.Irrep(t).dim,device=DEVICE)*.1 for t in TYPES}
            z=torch.tensor([6,8,1,1,7,1,1],device=DEVICE)
            batch=torch.tensor([0,0,0,0,1,1,1],device=DEVICE)
            m,e=model.mto(h,z,batch,2,True)
            scalar,tensors,p=model.cg(m)
            pred,physical=model.decoder(scalar,tensors,True)
            R=-o3.rand_matrix(dtype=DTYPE,device=DEVICE)
            mr,er=model.mto({t:transform(v,t,R) for t,v in h.items()},z,batch,2,True)
            sr,tr,pr=model.cg(mr)
            pred_r,phys_r=model.decoder(sr,tr,True)
            close(f'synthetic/{seed}/spectrum',pred_r,pred)
            for t in TYPES:close(f'synthetic/{seed}/M/{t}',mr[t],transform(m[t],t,R))
            Q,A,C=physical['Q'],physical['A'],physical['C']
            close(f'physics/{seed}/Qsym',Q,Q.transpose(-1,-2))
            close(f'physics/{seed}/Qtrace',Q.diagonal(dim1=-2,dim2=-1).sum(-1),torch.zeros_like(physical['E']))
            close(f'physics/{seed}/Asym',A,A.transpose(-1,-2))
            assert float(torch.linalg.eigvalsh(A).min())>=-1e-10
            close(f'physics/{seed}/trace',physical['trace_A'],physical['beta'].square()+Q.square().sum((-1,-2)))
            close(f'physics/{seed}/units',physical['f'],(2/3)*(physical['E']/EV_PER_HARTREE)*physical['trace_A'])
            close(f'physics/{seed}/Qcov',phys_r['Q'],R@Q@R.T)
            close(f'physics/{seed}/Acov',phys_r['A'],R@A@R.T)
            assert (physical['E']>0).all() and (physical['f']>=0).all()
        basis=model.decoder.cartesian_basis.flatten(1)
        close('basis/orthonormal',basis@basis.T,torch.eye(5,device=DEVICE))
        symmetric=dict(z=torch.tensor([6,6],device=DEVICE),pos=torch.tensor([[-.7,0.,0.],[.7,0.,0.]],device=DEVICE),
                       batch=torch.zeros(2,dtype=torch.long,device=DEVICE),n=1)
        _,sym=model(**symmetric,export=True)
        close('symmetry/equivalent_atom_gates',sym['c'][0],sym['c'][1])
        close('physics/factor_global_sign',(-sym['C'])@(-sym['C']).transpose(-1,-2),sym['A'])
        E=torch.tensor([[2.,5.,12.]],device=DEVICE);f=torch.tensor([[.3,.2,.7]],device=DEVICE)
        wide=torch.linspace(-5,20,10001,device=DEVICE)
        close('Gaussian/wide_area',torch.trapz(model.decoder.broaden(E,f,wide),wide),f.sum(-1))
        window=torch.linspace(1.5,13.5,10001,device=DEVICE)
        cdf=lambda x:.5*(1+torch.erf((x-E)/(math.sqrt(2)*model.decoder.sigma)))
        close('Gaussian/window_CDF',torch.trapz(model.decoder.broaden(E,f,window),window),(f*(cdf(13.5)-cdf(1.5))).sum(-1),atol=1e-7,rtol=1e-6)

    def test_07_gradients_and_gradcheck(self):
        for label,model in self.models.items():
            model.zero_grad(set_to_none=True)
            pred,extra=model(**self.x,export=True)
            loss=((pred-self.y)/self.info['train_rms']).square().mean()
            if extra:
                for key in ['E','f','Q','C','A']:extra[key].retain_grad()
            loss.backward()
            gradients={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None}
            assert all(math.isfinite(v) for v in gradients.values())
            for prefix in (['core.'] if label==VARIANTS[0] else ['core.','mto.project.','mto.router.','mto.query.','cg.','decoder.energy_head.','decoder.beta_head.','decoder.tensor_gate.']):
                assert sum(v for k,v in gradients.items() if k.startswith(prefix))>0,prefix
            for key in ['E','f','Q','C','A'] if extra else []:assert extra[key].grad.norm()>0,key
            MEASUREMENTS[label+'/gradient_norms']=gradients
            MEASUREMENTS[label+'/parameters']=parameter_counts(model)
        # Small independent smooth module exercises channel projection, routing, CG and physical decoder.
        cfg=dict(CFG,mto_channels=2,states=2,query_dim=3,router_hidden=5,head_hidden=6)
        mto=MolecularTensorOrbitals(cfg,3).to(device=DEVICE,dtype=torch.float64)
        cg=ReferenceStateCoupling(cfg).to(device=DEVICE,dtype=torch.float64)
        decoder=PhysicalSpectrumDecoder(cfg,cg.scalar_dim,cg.tensor_channels,[2.,5.,8.,11.]).to(device=DEVICE,dtype=torch.float64)
        z=torch.tensor([6,8],device=DEVICE);batch=torch.zeros(2,dtype=torch.long,device=DEVICE)
        inp=tuple((torch.randn(2,128,o3.Irrep(t).dim,device=DEVICE,dtype=torch.float64)*.05).requires_grad_() for t in TYPES)
        def fn(*v):
            m,_=mto(dict(zip(TYPES,v)),z,batch,1)
            s,t,_=cg(m)
            return decoder(s,t)[0]
        assert torch.autograd.gradcheck(fn,inp,eps=1e-6,atol=1e-5,rtol=1e-3,fast_mode=True)
        MEASUREMENTS['gradcheck/composed_mto_cg_decoder']=True


def main():
    global DEVICE,DTYPE
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--dtype',default='float64')
    args=ap.parse_args();DEVICE=args.device;DTYPE=getattr(torch,args.dtype)
    if DEVICE=='cuda':assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    started=time.time()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(success=result.wasSuccessful(),tests=result.testsRun,seconds=time.time()-started,
                device=DEVICE,dtype=args.dtype,measurements=MEASUREMENTS,
                errors=[(str(t),s) for t,s in result.errors],failures=[(str(t),s) for t,s in result.failures])
    (ROOT/f'logs/acceptance_{DEVICE}_{args.dtype}.json').write_text(json.dumps(report,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
