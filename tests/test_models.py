"""Structural and numerical tests for the full spectrum-only architecture."""
import json,math,unittest
from pathlib import Path
import torch
from e3nn import o3
from models import build_pair,EV_PER_HARTREE,TYPES
from dataset import collate

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg=json.loads((ROOT/'configs/grid.json').read_text())['configs'][0]
        cls.cfg=cfg
        cls.pair,cls.counts=build_pair(cfg,18,seed=11,spectrum_rms=.025)
        torch.manual_seed(29)
        cls.graphs=[]
        for z in ([6,8,7,1,1,1],[6,6,8,1,1]):
            pos=torch.randn(len(z),3,dtype=torch.float64)*.6
            cls.graphs.append({'z':torch.tensor(z),'pos':pos,
                'edge_index':(~torch.eye(len(z),dtype=torch.bool)).nonzero().T,
                'spectrum':torch.zeros(cfg['bins'])})
        cls.x,_=collate(cls.graphs)
        for model in cls.pair.values():model.double().eval()

    def test_parameter_match_and_initialization(self):
        self.assertLess(self.counts['relative_gap'],.01)
        for k,v in self.pair['mto'].backbone.state_dict().items():
            torch.testing.assert_close(v,self.pair['baseline'].backbone.state_dict()[k],atol=0,rtol=0)
        print('PARAMETERS',self.counts,flush=True)

    def test_full_cg_paths(self):
        tp=self.pair['mto'].coupling.tp
        paths={(tp.irreps_in1[i.i_in1].ir.l,tp.irreps_in2[i.i_in2].ir.l,tp.irreps_out[i.i_out].ir.l) for i in tp.instructions}
        self.assertEqual(paths,{(0,0,0),(1,1,0),(2,2,0),(0,2,2),(2,0,2),(1,1,2),(2,2,2)})
        self.assertTrue(all(i.has_weight for i in tp.instructions))

    def test_exact_assembly(self):
        model=self.pair['mto'];_,e=model(**self.x,export=True)
        self.assertEqual(tuple(e['gates'].shape),(11,11,3,16))
        self.assertTrue(bool((e['gates'].abs()<=1).all()))
        for i,t in enumerate(TYPES):
            f=e['bases'][t][:,None]*e['gates'][:,:,i,:,None]
            torch.testing.assert_close(e['assembly'][t],f,atol=0,rtol=0)
            explicit=torch.stack([f[self.x['batch']==b].sum(0)/math.sqrt(18) for b in range(2)])
            torch.testing.assert_close(explicit,e['mto'][t],atol=1e-12,rtol=1e-12)
        self.assertFalse(torch.allclose(e['gates'][:,:,0],e['gates'][:,:,2]))
        self.assertFalse(torch.allclose(e['gates'][:,0],e['gates'][:,1]))

    def test_rotations_reflections_covariance(self):
        old=torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            rot=o3.rand_matrix()
            for rotation in (rot,-rot,-torch.eye(3)):
                transformed={**self.x,'pos':self.x['pos']@rotation.T+torch.tensor([2.,-4.,1.])}
                for model in self.pair.values():
                    p,e=model(**self.x,export=True);p2,e2=model(**transformed,export=True)
                    torch.testing.assert_close(p,p2,atol=2e-6,rtol=2e-6)
                    for key in ('energy_eV','f','trace_A'):
                        torch.testing.assert_close(e[key],e2[key],atol=2e-6,rtol=2e-6)
                    for key in ('C','A','Q_cartesian','polarized_spectrum'):
                        torch.testing.assert_close(rotation@e[key]@rotation.T,e2[key],atol=2e-6,rtol=2e-6)
                    if model.variant=='mto':
                        torch.testing.assert_close(e['gates'],e2['gates'],atol=2e-6,rtol=2e-6)
                        for t in TYPES:
                            d=o3.Irrep(t).D_from_matrix(rotation)
                            for key in ('mto','assembly'):
                                torch.testing.assert_close(e[key][t]@d.T,e2[key][t],atol=2e-6,rtol=2e-6)
                        d=o3.Irrep('2e').D_from_matrix(rotation)
                        torch.testing.assert_close(e['relation_2e']@d.T,e2['relation_2e'],atol=2e-6,rtol=2e-6)
        finally:torch.set_default_dtype(old)

    def test_permutation_and_batching(self):
        perm=torch.randperm(len(self.x['z']));inverse=perm.argsort()
        px={**self.x,'z':self.x['z'][perm],'pos':self.x['pos'][perm],
            'batch':self.x['batch'][perm],'edge_index':inverse[self.x['edge_index']]}
        single,_=collate(self.graphs[:1])
        for model in self.pair.values():
            p=model(**self.x)
            torch.testing.assert_close(model(**px),p,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(model(**single)[0],p[0],atol=2e-6,rtol=2e-6)

    def test_psd_units_broadening_and_orientation_trace(self):
        for model in self.pair.values():
            p,e=model(**self.x,export=True)
            self.assertTrue(bool((torch.linalg.eigvalsh(e['A'])>=-1e-12).all()))
            torch.testing.assert_close(e['Q_cartesian'].diagonal(dim1=-2,dim2=-1).sum(-1),torch.zeros_like(e['beta']),atol=1e-9,rtol=1e-9)
            torch.testing.assert_close(e['f'],(2/3)*e['energy_eV']/EV_PER_HARTREE*e['trace_A'])
            torch.testing.assert_close(e['source_spectrum'],p*.025,atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(e['polarized_spectrum'].diagonal(dim1=-2,dim2=-1).sum(-1)/3,e['source_spectrum'],atol=1e-12,rtol=1e-12)
            grid=torch.linspace(-10.,30.,20001,dtype=torch.float64)
            full,_=model.decoder.broaden(e['energy_eV'],e['f'],grid)
            torch.testing.assert_close(torch.trapezoid(full,grid,dim=-1),e['f'].sum(-1),atol=1e-8,rtol=1e-8)
            order=torch.randperm(10)
            shuffled,_=model.decoder.broaden(e['energy_eV'][:,order],e['f'][:,order])
            torch.testing.assert_close(shuffled,e['source_spectrum'],atol=1e-12,rtol=1e-12)

    def test_spectrum_only_loss_reaches_every_module(self):
        for model in self.pair.values():
            model.zero_grad(set_to_none=True)
            p,e=model(**self.x,export=True)
            for key in ('energy_eV','C','A','f'):e[key].retain_grad()
            loss=(p-.13).square().mean();loss.backward()
            for key in ('energy_eV','C','A','f'):
                self.assertGreater(float(e[key].grad.abs().sum()),0.,key)
            for name,param in model.named_parameters():
                self.assertIsNotNone(param.grad,name)
                self.assertTrue(bool(torch.isfinite(param.grad).all()),name)
                self.assertGreater(float(param.grad.abs().sum()),0.,name)
            if model.variant=='mto':
                self.assertTrue(bool((model.readout.query.weight.grad.abs().sum(-1)>0).all()))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests))
    (ROOT/'results/model_tests.json').write_text(json.dumps({'passed':result.wasSuccessful(),
        'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures)},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
