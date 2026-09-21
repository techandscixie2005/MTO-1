"""Small CPU tests for the new observers, interventions and atom-count LS design."""
import copy
import json
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from smoke_diagnostics import StepObserver, intervene_optimizer, svd_solve, loss_metrics


class DiagnosticContracts(unittest.TestCase):
    def test_observer_does_not_change_update(self):
        torch.manual_seed(113)
        a=torch.nn.Sequential(torch.nn.Linear(5,7),torch.nn.SiLU(),torch.nn.Linear(7,3)).double()
        b=copy.deepcopy(a)
        oa=torch.optim.Adam(a.parameters(),lr=.001,amsgrad=True)
        ob=torch.optim.Adam(b.parameters(),lr=.001,amsgrad=True)
        x=torch.randn(11,5,dtype=torch.float64);y=torch.randn(11,3,dtype=torch.float64)
        observer=StepObserver(b,ob)
        for _ in range(3):
            for m,opt in [(a,oa),(b,ob)]:
                opt.zero_grad();(m(x)-y).square().mean().backward();opt.step()
            for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p,q))
        self.assertGreater(observer.record['update_l2'],0)
        self.assertGreater(observer.record['gradient_l2'],0)
        self.assertGreaterEqual(observer.record['mean_max_v_over_mean_v'],1)
        observer.close()

    def test_D1_changes_only_maximum(self):
        p=torch.nn.Parameter(torch.tensor([2.,-1.]))
        opt=torch.optim.Adam([p],lr=.001,amsgrad=True)
        for scale in [100.,.01,.01]:
            opt.zero_grad();(scale*p.square().sum()).backward();opt.step()
        before=copy.deepcopy(opt.state_dict());weights=p.detach().clone()
        intervene_optimizer(opt,'D1');after=opt.state_dict()
        self.assertTrue(torch.equal(p,weights))
        for key,value in before['state'][0].items():
            expected=before['state'][0]['exp_avg_sq'] if key=='max_exp_avg_sq' else value
            self.assertTrue(torch.equal(after['state'][0][key],expected))
        empty=torch.optim.Adam([p],lr=.001,amsgrad=True);intervene_optimizer(empty,'D2')
        self.assertEqual(len(empty.state),0)

    def test_atom_count_SVD(self):
        torch.manual_seed(217)
        phi=torch.randn(32,128,dtype=torch.float64)
        N=torch.arange(8,40,dtype=torch.float64)[:,None]
        X=torch.cat([phi,N],1);weight=torch.randn(128,240,dtype=torch.float64)
        bias=torch.randn(240,dtype=torch.float64)
        Y=phi@weight+N*bias
        solution,audit=svd_solve(X,Y)
        self.assertEqual(audit['numerical_rank'],32)
        torch.testing.assert_close(phi@solution[:128]+N*solution[128],Y,atol=1e-10,rtol=1e-10)
        # Replacing N by 1 in the reconstructed output is a different head.
        self.assertGreater(float((phi@solution[:128]+solution[128]-Y).norm()),1)

    def test_loss_shape_and_scaling(self):
        p=torch.arange(32*240,dtype=torch.float64).reshape(32,240)/100
        y=p*.37
        result=loss_metrics(p,y,.2)
        self.assertAlmostEqual(result['normalized_mse'],result['raw_mse']/.2**2,places=8)
        with self.assertRaises(AssertionError):loss_metrics(p,y[0],.2)


if __name__=='__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticContracts))
    report=dict(success=result.wasSuccessful(),tests=result.testsRun,
                errors=[(str(t),e) for t,e in result.errors],failures=[(str(t),e) for t,e in result.failures])
    (ROOT/'logs/diagnostic_contracts.json').write_text(json.dumps(report,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
