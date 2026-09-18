"""Exercise 1k/10k/full evaluation and afterok submission in temporary fixtures.

Synthetic predictions test joins/statistics/reports, never scientific quality.
The separate model tests use the actual DetaNet and full MTO modules.
"""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_scale,report_scale,submit_ordered


class FakeModel(torch.nn.Module):
    def __init__(self,bias):
        super().__init__();self.bias=bias
    def forward(self,**kwargs):
        n=kwargs['n'];energy=torch.linspace(3,12,10)[None].expand(n,-1)
        f=torch.full((n,10),.02+self.bias)
        pred=torch.exp(-.5*((torch.linspace(1.5,13.5,601)[None,None,:]-energy[...,None])/.2)**2)
        pred=(pred*f[...,None]).sum(1)/.025
        a=torch.eye(3)[None,None].repeat(n,10,1,1)*.02
        return pred,{'energy_eV':energy,'energy_order':energy.argsort(-1),'A':a,'f':f,'trace_A':torch.full((n,10),.06)}


class WorkflowTests(unittest.TestCase):
    def test_all_scale_evaluation_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for d in ('configs','results','data','runs'): (root/d).mkdir()
            grid=json.loads((ROOT/'configs/grid.json').read_text());cfg=grid['configs'][0]
            execution=json.loads((ROOT/'configs/execution.json').read_text())
            (root/'configs/grid.json').write_text(json.dumps(grid))
            (root/'configs/execution.json').write_text(json.dumps(execution))
            (root/'results/preflight.json').write_text('{"passed":true}')
            ids=np.arange(10);split=np.array([0,0,0,0,2,2,2,2,2,2]);fresh=np.array([False]*6+[True]*4)
            spectrum=np.tile(np.linspace(.001,.05,601,dtype=np.float32),(10,1))
            np.savez(root/'data/qm9s_full.npz',ids=ids,split=split,spectra=spectrum,
                     subset_10k=np.array([True]*8+[False]*2),fresh_test=fresh,energy_eV=np.linspace(1.5,13.5,601))
            (root/'data/splits_1k.json').write_text(json.dumps({'train':[0,1],'validation':[],'test':[4,5]}))
            graphs=[{'row':j,'id':j,'identity':str(j),'spectrum':torch.from_numpy(spectrum[j])} for j in range(4,10)]
            def load(*args,**kwargs):return graphs,{}
            def collate(rows,device,norm):return {'n':len(rows)},None
            def pair(*args,**kwargs):return {'baseline':FakeModel(.001),'mto':FakeModel(0.)},{'baseline':100,'mto':100}
            with patch.object(evaluate_scale,'ROOT',root),patch.object(report_scale,'ROOT',root),patch.object(evaluate_scale,'load_scale',load),patch.object(evaluate_scale,'collate',collate),patch.object(evaluate_scale,'build_pair',pair),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.nn.Module,'cuda',lambda self:self):
                for scale,n_test in [('1k',2),('10k',4),('full',6)]:
                    manifest={'scale':scale,'count':10,'splits':{'train':4,'validation':0,'test':n_test},
                              'train_spectrum_rms':.025,'train_median_atoms':18,'sample_sha256':'fixture'}
                    (root/'data'/f'manifest_{scale}.json').write_text(json.dumps(manifest))
                    for seed in execution['seeds_by_scale'][scale]:
                        folder=root/'runs'/f'{scale}_seed{seed}';folder.mkdir()
                        (folder/'DONE').write_text('test fixture')
                        (folder/'experiment.json').write_text('{"fingerprint":"fixture"}')
                        for arm in ('baseline','mto'):
                            torch.save({'state_dict':{},'database_sha256':'fixture','fingerprint':'fixture','config':cfg,
                                        'scale_name':scale,'scale':.025,'seed':seed,'variant':arm,'best_epoch':1,
                                        'validation':{'mse':1.}},folder/f'{arm}_best.pt')
                            (folder/f'{arm}_history.json').write_text(json.dumps([{'epoch':1,'validation':{'mse':1.}}]))
                    with patch.object(sys,'argv',['evaluate_scale.py','--scale',scale]):evaluate_scale.main()
                    r=json.loads((root/'results'/scale/'comparison.json').read_text())
                    self.assertEqual(r['cohorts']['within_scale']['test_molecules'],n_test)
                    self.assertEqual(len(r['seeds']),3 if scale=='full' else 5)
                    self.assertTrue((root/'results'/scale/'COMPLETE').exists())
                    self.assertTrue((root/'results'/f'comparison_{scale}_report.md').exists())
                merged=json.loads((root/'results/comparison_all.json').read_text())
                self.assertEqual(list(merged['scales']),['1k','10k','full'])
                self.assertTrue((root/'results/comparison_report.md').exists())

    def test_submission_dependencies_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'jobs').mkdir();(root/'results').mkdir()
            for name in ('model_tests','data_checks','workflow_tests'):(root/'results'/f'{name}.json').write_text('{"passed":true}')
            with patch.object(submit_ordered,'ROOT',root),patch.object(submit_ordered,'verify',return_value={'protocol_id':'fixture'}),patch.object(submit_ordered.subprocess,'check_output',side_effect=['101\n','102\n','103\n']) as submit,patch.object(submit_ordered.subprocess,'run') as run:
                submit_ordered.main()
                self.assertNotIn('--dependency=afterok:101',submit.call_args_list[0].args[0])
                self.assertIn('--dependency=afterok:101',submit.call_args_list[1].args[0])
                self.assertIn('--dependency=afterok:102',submit.call_args_list[2].args[0])
                self.assertTrue(all('--hold' in c.args[0] for c in submit.call_args_list))
                self.assertEqual([c.args[0] for c in run.call_args_list],[['scontrol','release','101'],['scontrol','release','102'],['scontrol','release','103']])
                self.assertTrue(json.loads((root/'jobs/submission.json').read_text())['released'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WorkflowTests))
    (ROOT/'results/workflow_tests.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
