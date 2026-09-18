"""Regression for legitimate empty e3nn buffers in checkpoint diagnostics."""
import json,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from preflight import validate_restored_state
from models import build_pair

class RestoreTests(unittest.TestCase):
    def test_empty_and_nonempty_values(self):
        a={'empty':torch.empty(0,3),'value':torch.tensor([1.,2.]),'integer':torch.tensor([2])}
        b={k:v.clone() for k,v in a.items()}
        self.assertEqual(validate_restored_state(a,b),0.)
        self.assertEqual(validate_restored_state({'empty':torch.empty(0)},{'empty':torch.empty(0)}),0.)
        b['value'][0]+=.01
        with self.assertRaises(AssertionError):validate_restored_state(a,b)

    def test_empty_shape_is_still_checked(self):
        with self.assertRaises(AssertionError):
            validate_restored_state({'empty':torch.empty(0,3)},{'empty':torch.empty(0,4)})

    def test_actual_model_state_dicts(self):
        torch.set_num_threads(1)
        cfg=json.loads((ROOT/'configs/grid.json').read_text())['configs'][0]
        pair,_=build_pair(cfg,18,spectrum_rms=.025)
        for arm,model in pair.items():
            a=model.state_dict()
            empty=[k for k,v in a.items() if v.is_floating_point() and v.numel()==0]
            self.assertTrue(empty,arm)
            self.assertEqual(validate_restored_state(a,{k:v.clone() for k,v in a.items()}),0.)
            print(arm,'empty buffers validated:',empty,flush=True)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RestoreTests))
    (ROOT/'results/preflight_regression_tests.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
