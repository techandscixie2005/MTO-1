"""Fit both maps before a separate single outer-validation evaluation command."""
import argparse
import fcntl
import json
import os
import sys
import time
import numpy as np
import torch
from clean_source import ROOT,SOURCE,MTOEA,tensor_state_hash
from runtime import verify_manifest,sha,atomic_json,atomic_torch,setup
from predictor import export_record,load_predictor
from evaluation import SubsetData,predict_native,affine_fit,report,score
PARENT=ROOT.parent
sys.path.insert(0,str(PARENT))
from launch import admit,EXPECTED

def base_model(cfg,device):
    assert sha(cfg['source_checkpoint'])==cfg['source_checkpoint_sha256']
    record=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False)
    config=json.loads((SOURCE/'configs/mto_eta0.json').read_text())
    stats=json.loads((SOURCE/'data/normalization.json').read_text())
    model=MTOEA(config,stats);model.load_state_dict(record['model'],strict=True)
    return model.requires_grad_(False).eval().to(device),config,stats

def check_complete(mh):
    receipt=json.loads((ROOT/'runs/source33/FIT_COMPLETE.json').read_text())
    assert receipt['completed_epoch']==33 and receipt['manifest_sha256']==mh
    assert sha(receipt['checkpoint'])==receipt['checkpoint_sha256']
    return receipt

def fit_maps(cfg,mh):
    complete=check_complete(mh);out=ROOT/'affine';out.mkdir(exist_ok=True)
    if (out/'COEFFICIENTS_FROZEN.json').exists():
        frozen=json.loads((out/'COEFFICIENTS_FROZEN.json').read_text())
        assert frozen['manifest_sha256']==mh and frozen['source_checkpoint_sha256']==complete['checkpoint_sha256']
        assert frozen['full_baseline_sha256']==cfg['source_checkpoint_sha256']
        assert sha(out/'calibration_predictions.npz')==frozen['calibration_arrays_sha256']
        for path,value in frozen['source_hashes'].items():assert sha(path)==value
        export_maps(cfg,mh,frozen)
        print('Existing frozen coefficients retained; exports verified or completed',flush=True)
        return
    indices=np.load(ROOT/'data/calibration_indices.npy');assert len(indices)==cfg['calibration_molecules']
    data=SubsetData(indices,'cuda');baseline,bc,bs=base_model(cfg,'cuda')
    be,bf=predict_native(baseline,data,cfg['batch_size'])
    checkpoint=torch.load(complete['checkpoint'],map_location='cpu',weights_only=False)
    source=MTOEA(checkpoint['source_config'],checkpoint['stats'])
    source.load_state_dict(checkpoint['model'],strict=True);source.requires_grad_(False).eval().cuda()
    assert tensor_state_hash(dict(source.named_buffers()))==checkpoint['buffer_tensor_sha256']
    se,sf=predict_native(source,data,cfg['batch_size']);mask=data.raw['mask_f'].astype(bool)
    maps={'in_sample':affine_fit(data.raw['f'],bf,mask),'heldout_source':affine_fit(data.raw['f'],sf,mask)}
    # Server-only arrays preserve complete numerical inputs for independent replay.
    arrays=out/'calibration_predictions.npz'
    np.savez_compressed(arrays,indices=indices,ids=data.ids,f_true=data.raw['f'],mask_f=mask,
                        full_baseline_native_f=bf,source_native_f=sf,full_baseline_E=be,source_E=se)
    frozen={'manifest_sha256':mh,'maps':maps,'calibration_arrays_sha256':sha(arrays),
        'source_checkpoint_sha256':complete['checkpoint_sha256'],'full_baseline_sha256':cfg['source_checkpoint_sha256'],
        'calibration_indices_sha256':sha(ROOT/'data/calibration_indices.npy'),
        'fitting_source_quality_energy':score(data.raw['E'][data.raw['mask_E']],se[data.raw['mask_E']]),
        'outer_validation_evaluated_before_freeze':False,'test_evaluated':False,'frozen_at_unix':time.time(),
        'source_hashes':{str(ROOT/n):sha(ROOT/n) for n in ('affine_stage.py','evaluation.py','predictor.py')}}
    atomic_json(frozen,out/'COEFFICIENTS_FROZEN.json')
    export_maps(cfg,mh,frozen)
    print(json.dumps(frozen,indent=2))

def export_maps(cfg,mh,frozen):
    out=ROOT/'affine';baseline,bc,bs=base_model(cfg,'cpu');maps=frozen['maps']
    # Each deployment checkpoint includes the same entire original full baseline.
    for name,coeff in maps.items():
        path=out/(name+'.pt')
        if path.exists():
            existing=load_predictor(path,'cpu')
            assert tensor_state_hash(existing.base.state_dict())==tensor_state_hash(baseline.state_dict())
            assert float(existing.alpha)==coeff['alpha'] and float(existing.beta)==coeff['beta']
            continue
        atomic_torch(export_record(baseline,bc,bs,coeff['alpha'],coeff['beta'],
            {'coefficient_receipt_sha256':sha(out/'COEFFICIENTS_FROZEN.json'),'map':name,'manifest_sha256':mh,
             'full_baseline_sha256':cfg['source_checkpoint_sha256']}),path)
    atomic_json({'manifest_sha256':mh,'coefficient_receipt_sha256':sha(out/'COEFFICIENTS_FROZEN.json'),
        'checkpoints':{name:{'path':str(out/(name+'.pt')),'sha256':sha(out/(name+'.pt'))} for name in maps},
        'single_model_single_checkpoint':True},out/'EXPORT_COMPLETE.json')

def evaluate(cfg,mh):
    out=ROOT/'affine';assert not (out/'VALIDATION_COMPLETE.json').exists(),'Validation already complete'
    frozen=json.loads((out/'COEFFICIENTS_FROZEN.json').read_text());export=json.loads((out/'EXPORT_COMPLETE.json').read_text())
    assert frozen['manifest_sha256']==mh and export['manifest_sha256']==mh
    assert export['coefficient_receipt_sha256']==sha(out/'COEFFICIENTS_FROZEN.json')
    for path,expected in frozen['source_hashes'].items():assert sha(path)==expected
    for record in export['checkpoints'].values():assert sha(record['path'])==record['sha256']
    atomic_json({'manifest_sha256':mh,'coefficient_receipt_sha256':sha(out/'COEFFICIENTS_FROZEN.json'),
        'started_unix':time.time()},out/('VALIDATION_ATTEMPT_%d.json'%time.time_ns()))
    # The first outer-label access in this stage occurs only after both maps are frozen.
    with np.load(SOURCE/'data/dataset.npz') as archive:indices=archive['val'].copy()
    data=SubsetData(indices,'cuda');base,_,_=base_model(cfg,'cuda');energy,native=predict_native(base,data,cfg['batch_size'])
    thresholds=json.loads((PARENT/'FROZEN_MANIFEST.json').read_text())['tail_thresholds']
    predictions={'native':native,'historical_validation_fit':np.maximum(0,cfg['historical_alpha']*native+cfg['historical_beta'])}
    predictions.update({name:np.maximum(0,c['alpha']*native+c['beta']) for name,c in frozen['maps'].items()})
    # Geometry replay on the identical fixed first64 validation molecules verifies each standalone export.
    replay={};x=data.batch(np.arange(min(cfg['batch_size'],len(indices))))
    for name,record in export['checkpoints'].items():
        model=load_predictor(record['path'],'cuda')
        with torch.no_grad():actual=model(**x)['f'].cpu().numpy()
        expected=predictions[name][:len(actual)];delta=float(np.max(np.abs(actual-expected)))
        assert np.allclose(actual,expected,atol=2e-6,rtol=1e-5)
        replay[name]={'max_abs_f':delta,'checkpoint_sha256':record['sha256']};del model
    metrics={name:report(data.raw,pred,thresholds) for name,pred in predictions.items()}
    assert len(indices)==6686 and metrics['native']['raw_f']['count']==66860
    assert abs(metrics['native']['raw_f']['r2']-.4052941183410983)<1e-7
    assert abs(metrics['historical_validation_fit']['raw_f']['r2']-.418119238799)<1e-7
    held=metrics['heldout_source']['raw_f']['r2'];inside=metrics['in_sample']['raw_f']['r2'];anchor=metrics['native']['raw_f']['r2']
    result={'manifest_sha256':mh,'coefficient_receipt_sha256':sha(out/'COEFFICIENTS_FROZEN.json'),
        'export_receipt_sha256':sha(out/'EXPORT_COMPLETE.json'),'maps':frozen['maps'],'metrics':metrics,
        'energy':score(data.raw['E'][data.raw['mask_E']],energy[data.raw['mask_E']]),
        'heldout_minus_in_sample_r2':held-inside,'heldout_minus_native_r2':held-anchor,
        'nonlinear_preparation_gate_passed':held-inside>=cfg['advancement_minimum_delta_r2'] and held-anchor>=cfg['advancement_minimum_delta_r2'],
        'standalone_geometry_replay':replay,'validation_molecules':len(indices),'test_evaluated':False,
        'validation_reused_historically':True,'prediction_averaging':False,'validation_completed_unix':time.time()}
    np.savez_compressed(out/'validation_predictions.npz',indices=indices,ids=data.ids,f_true=data.raw['f'],
        mask_f=data.raw['mask_f'],E_true=data.raw['E'],E_pred=energy,**predictions)
    result['validation_arrays_sha256']=sha(out/'validation_predictions.npz')
    atomic_json(result,out/'VALIDATION_COMPLETE.json');print(json.dumps(result,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','evaluate']);parser.add_argument('--gpu',type=int,default=1)
    args=parser.parse_args();manifest,mh=verify_manifest();cfg=manifest['config'];check_complete(mh)
    lock=open('/tmp/mto_pouter_gpu_%d.lock'%args.gpu,'w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission=admit(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=EXPECTED[args.gpu]
    (ROOT/('AFFINE_'+args.stage.upper()+'_GPU_ADMISSION.xml')).write_text(admission)
    setup(cfg)
    if args.stage=='fit':fit_maps(cfg,mh)
    else:evaluate(cfg,mh)

if __name__=='__main__':main()
