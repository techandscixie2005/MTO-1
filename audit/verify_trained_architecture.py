"""Read-only audit of completed trained checkpoints; no optimizer or model edits."""
import hashlib,json,sys,datetime
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from protocol import verify
from models import build_pair,EV_PER_HARTREE
from dataset import collate
from scale_dataset import load_scale

torch.set_num_threads(1)
frozen=verify()
grid=json.loads((ROOT/'configs/grid.json').read_text());cfg=grid['configs'][0]
execution=json.loads((ROOT/'configs/execution.json').read_text())
graphs,_=load_scale(ROOT,'1k',splits=('train',),limit=2)
records=[]
for scale in ('1k','10k'):
    marker=json.loads((ROOT/'results'/scale/'STAGE_COMPLETE.json').read_text())
    assert marker['passed'] and marker['protocol_id']==frozen['protocol_id']
    manifest=json.loads((ROOT/'data'/f'manifest_{scale}.json').read_text())
    norm=manifest['train_spectrum_rms']
    for seed in execution['seeds_by_scale'][scale]:
        folder=ROOT/'runs'/f'{scale}_seed{seed}'
        exp=json.loads((folder/'experiment.json').read_text())
        assert exp['config']==cfg and exp['training']==grid['shared_training']
        for path,digest in exp['source_hashes'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,(scale,seed,path)
        pair,counts=build_pair(cfg,manifest['train_median_atoms'],seed=seed,spectrum_rms=norm)
        for arm,model in pair.items():
            ck=torch.load(folder/f'{arm}_best.pt',map_location='cpu',weights_only=True)
            assert ck['fingerprint']==exp['fingerprint']
            assert ck['config']==cfg and ck['variant']==arm and ck['seed']==seed and ck['scale_name']==scale
            assert ck['database_sha256']==manifest['sample_sha256']
            model.load_state_dict(ck['state_dict'],strict=True)
            record={'scale':scale,'seed':seed,'arm':arm,'source_hashes_match':True,
                    'checkpoint_strict_load':True,'checkpoint_sha256':hashlib.sha256((folder/f'{arm}_best.pt').read_bytes()).hexdigest()}
            if arm=='mto':
                model.eval()
                x,y=collate(graphs,'cpu',norm)
                p,e=model(**x,export=True)
                assert e['energy_eV'].shape==(2,10) and e['A'].shape==(2,10,3,3)
                assert e['mto']['2e'].shape==(2,11,16,5)
                torch.testing.assert_close(e['A'],e['C']@e['C'].transpose(-1,-2))
                torch.testing.assert_close(e['f'],(2/3)*e['energy_eV']/EV_PER_HARTREE*e['trace_A'])
                torch.testing.assert_close(e['source_spectrum'],p*norm)
                torch.testing.assert_close(e['polarized_spectrum'].diagonal(dim1=-2,dim2=-1).sum(-1)/3,e['source_spectrum'])
                eigen=torch.linalg.eigvalsh(e['A'].double())
                assert float(eigen.min())>=-1e-6*max(1.,float(eigen.abs().max()))
                for key in ('energy_eV','C','A','f'):e[key].retain_grad()
                loss=(p-y).square().mean();loss.backward()
                grads={key:float(e[key].grad.abs().sum()) for key in ('energy_eV','C','A','f')}
                groups={name:sum(float(v.grad.abs().sum()) for v in module.parameters() if v.grad is not None)
                        for name,module in [('backbone',model.backbone),('state_MTO',model.readout),
                            ('reference_CG',model.coupling),('energy_head',model.decoder.energy_head),
                            ('beta_head',model.decoder.beta_head),('tensor_gate',model.decoder.tensor_gate)]}
                assert all(v>0 for v in [*grads.values(),*groups.values()])
                assert all(bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None)
                paths={(model.coupling.tp.irreps_in1[i.i_in1].ir.l,model.coupling.tp.irreps_in2[i.i_in2].ir.l,
                        model.coupling.tp.irreps_out[i.i_out].ir.l) for i in model.coupling.tp.instructions}
                assert paths=={(0,0,0),(1,1,0),(2,2,0),(0,2,2),(2,0,2),(1,1,2),(2,2,2)}
                record.update(physical_identities_passed=True,gradient_intermediates=grads,gradient_modules=groups,
                              min_A_eigenvalue=float(eigen.min()),cg_paths=sorted(paths))
                saved=torch.load(ROOT/'results'/scale/f'mto_assembly_examples_seed{seed}.pt',map_location='cpu',weights_only=True)
                assert saved['protocol_id']==frozen['protocol_id'] and saved['seed']==seed
                se=saved['intermediates']
                for key in ('bases','assembly','gates','mto','relation_0e','relation_2e','energy_eV','A','f','source_spectrum'):
                    assert key in se,key
                record['saved_all_intermediates']=True
            records.append(record)
            print(scale,seed,arm,'passed',flush=True)
out={'passed':True,'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'protocol_id':frozen['protocol_id'],'checkpoint_count':len(records),'mto_gradient_checks':10,
     'gradient_sample_train_ids':[g['id'] for g in graphs],'optimizer_steps_in_audit':0,
     'model_source_sha256':hashlib.sha256((ROOT/'src/models.py').read_bytes()).hexdigest(),'records':records}
(ROOT/'audit/trained_architecture_checks.json').write_text(json.dumps(out,indent=2))
print('ALL CHECKS PASSED',flush=True)
