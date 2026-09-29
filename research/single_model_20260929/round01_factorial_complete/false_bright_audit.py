"""Post-hoc aggregate error audit; no selection, fitting, or raw output export."""
import fcntl
import json
import math
import os
from pathlib import Path
import numpy as np
from freeze import sha
from launch import admit, EXPECTED

ROOT=Path(__file__).resolve().parent
QUANTILES=[0,.01,.1,.5,.9,.95,.99,.999,1]

def distribution(value):
    value=np.asarray(value,dtype=np.float64)
    return {str(q):float(v) for q,v in zip(QUANTILES,np.quantile(value,QUANTILES))}

def error_coupling(error,mask,energy,energy_mask):
    n=error.shape[1]
    cross=[];correlation=[];counts=[]
    for j in range(n):
        xc=[];cc=[];nn=[]
        for k in range(n):
            use=mask[:,j]&mask[:,k]
            x=error[:,j][use];y=error[:,k][use]
            nn.append(int(use.sum()));xc.append(float((x*y).sum()))
            denom=np.sqrt(np.square(x-x.mean()).sum()*np.square(y-y.mean()).sum())
            cc.append(float(((x-x.mean())*(y-y.mean())).sum()/denom) if denom else None)
        cross.append(xc);correlation.append(cc);counts.append(nn)
    pairs=[]
    for j in range(n-1):
        valid=mask[:,j]&mask[:,j+1]
        small=valid&energy_mask[:,j]&energy_mask[:,j+1]&((energy[:,j+1]-energy[:,j])<.0287)
        strata={}
        for name,use in (('all_valid_pairs',valid),('direct_pair_gap_below_train_q10',small)):
            x=error[:,j][use];y=error[:,j+1][use]
            individual=float((x*x+y*y).sum());combined=float(np.square(x+y).sum())
            strata[name]={'count':int(use.sum()),'individual_sse_sum':individual,
                'summed_error_sse':combined,'cross_product_sum':float((x*y).sum()),
                'summed_to_individual_sse_ratio':combined/individual if individual else None,
                'signed_mean_summed_error':float((x+y).mean()) if use.any() else None}
            assert np.isclose(combined,individual+2*float((x*y).sum()),rtol=1e-12,atol=1e-10)
        pairs.append({'states':[j+1,j+2],'strata':strata})
    return {'pairwise_valid_counts':counts,'error_cross_product_sums':cross,
        'centered_error_correlations':correlation,'adjacent_pairs':pairs,
        'small_gap_threshold_eV':.0287,'small_gap_definition':'direct adjacent true-state gap below frozen TRAIN nearest-gap q10',
        'interpretation':'summed errors diagnostic only; no relabeling, permutation, selection or alternate primary score'}

def analyze(arrays,threshold,energy,energy_mask):
    truth=arrays['f_true'];pred=arrays['f_pred'];mask=arrays['mask_f'].astype(bool)
    squared=(pred-truth)**2
    assert np.isfinite(truth[mask]).all() and np.isfinite(pred[mask]).all()
    def aggregate(use):
        return {'count':int(use.sum()),'sse':float(squared[use].sum()),
                'mae':float(np.abs(pred[use]-truth[use]).mean()) if use.any() else None,
                'per_state':[{'state':j+1,'count':int(use[:,j].sum()),
                    'sse':float(squared[:,j][use[:,j]].sum())} for j in range(truth.shape[1])]}
    whole=aggregate(mask)
    groups={}
    for tb in (False,True):
        for pb in (False,True):
            key=('true_bright' if tb else 'true_dim')+'__'+('pred_bright' if pb else 'pred_dim')
            groups[key]=aggregate(mask&((truth>=threshold)==tb)&((pred>=threshold)==pb))
    assert sum(g['count'] for g in groups.values())==whole['count']
    assert np.isclose(sum(g['sse'] for g in groups.values()),whole['sse'],rtol=1e-13,atol=1e-10)
    errors=squared[mask]
    ranking=np.argsort(-errors,kind='stable')
    flat_states=np.broadcast_to(np.arange(1,truth.shape[1]+1),truth.shape)[mask]
    top={}
    for fraction in (.001,.01,.05):
        count=math.ceil(len(errors)*fraction);indices=ranking[:count]
        top[str(fraction)]={'count':count,'sse':float(errors[indices].sum()),
            'fraction_of_all_sse':float(errors[indices].sum()/whole['sse']),
            'states_7_8':[{'state':j,'count':int((flat_states[indices]==j).sum()),
                'sse':float(errors[indices][flat_states[indices]==j].sum())} for j in (7,8)]}
    return {'whole':whole,'brightness_bins':groups,
        'quantiles':{'prediction':distribution(pred[mask]),'signed_error':distribution(pred[mask]-truth[mask]),
                     'absolute_error':distribution(np.abs(pred[mask]-truth[mask])),
                     'squared_error':distribution(errors)},'top_error_concentration':top,
        'state_error_coupling':error_coupling(pred-truth,mask,energy,energy_mask)}

def main():
    cfg=json.loads((ROOT/'round_config.json').read_text())
    frozen=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    assert all((ROOT/'runs'/a/'FIT_COMPLETE.json').is_file() for a in cfg['arms'])
    threshold=frozen['tail_thresholds']['q99']
    assert threshold==.2412
    lock=open('/tmp/mto_pouter_gpu_1.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    admission=admit(1)
    (ROOT/'FALSE_BRIGHT_GPU_ADMISSION.xml').write_text(admission)
    os.environ['CUDA_VISIBLE_DEVICES']=EXPECTED[1]
    import torch
    from architecture.model import SOURCE,build_model
    from metrics import validate
    from train import setup
    from dataset import Data
    setup(cfg)
    data=Data(device='cuda')
    with np.load(SOURCE/'data/raw_labels.npz') as z:
        raw={k:z[k].copy() for k in ('E','f','mask_E','mask_f')}
        assert np.array_equal(z['ids'],data.ids)
    source_cfg=json.loads((SOURCE/'configs/mto_eta0.json').read_text())
    result={'post_hoc':True,'selection_use':False,'test_evaluated':False,'fit_performed':False,
        'threshold':threshold,'threshold_source':'prespecified TRAIN q99',
        'bins_partition_all_valid_labels':True,'raw_arrays_written':False,
        'frozen_manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json'),'source_sha256':sha(__file__),
        'gpu_uuid':EXPECTED[1],'arms':{},'input_hashes':{}}
    for arm,ac in cfg['arms'].items():
        rd=ROOT/'runs'/arm
        receipt=json.loads((rd/'FIT_COMPLETE.json').read_text())
        assert receipt['completed_epoch']==20 and receipt['selected_epoch']==0
        with np.load(rd/'best_validation_predictions.npz') as z:
            selected={k:z[k].copy() for k in ('indices','f_true','f_pred','mask_f')}
        energy=raw['E'][selected['indices']];em=raw['mask_E'][selected['indices']].astype(bool)
        checks={'selected_epoch0':analyze(selected,threshold,energy,em)}
        assert np.isclose(checks['selected_epoch0']['whole']['sse'],receipt['validation']['raw_f']['sse'],rtol=1e-12)
        ck=torch.load(rd/'last.pt',map_location='cpu',weights_only=False)
        assert ck['state']['completed_epoch']==20 and ck['manifest_sha256']==result['frozen_manifest_sha256']
        model=build_model(source_cfg,data.stats,ac['adapter'],cfg['adapter_seed']).cuda()
        model.load_state_dict(ck['model'],strict=True)
        metrics,final=validate(model,data,raw,cfg['batch_size'],frozen['tail_thresholds'])
        expected=ck['state']['history'][-1]['validation']['raw_f']['r2']
        assert abs(metrics['raw_f']['r2']-expected)<1e-7
        for key in ('indices','f_true','mask_f'):
            assert np.array_equal(final[key],selected[key])
        checks['fixed_epoch20']=analyze(final,threshold,energy,em)
        checks['fixed_epoch20']['replayed_r2']=metrics['raw_f']['r2']
        checks['fixed_epoch20']['replay_r2_difference']=metrics['raw_f']['r2']-expected
        checks['delta_sse_epoch20_minus_epoch0']={
            'whole':checks['fixed_epoch20']['whole']['sse']-checks['selected_epoch0']['whole']['sse'],
            'per_state':[{'state':j+1,'sse_change':checks['fixed_epoch20']['whole']['per_state'][j]['sse']-
                checks['selected_epoch0']['whole']['per_state'][j]['sse']} for j in range(10)],
            'brightness_bins':{k:checks['fixed_epoch20']['brightness_bins'][k]['sse']-
                checks['selected_epoch0']['brightness_bins'][k]['sse'] for k in checks['selected_epoch0']['brightness_bins']}}
        result['arms'][arm]=checks
        result['input_hashes'][arm]={name:sha(rd/name) for name in ('FIT_COMPLETE.json','best_validation_predictions.npz','last.pt','history.jsonl')}
        del model,ck,final
    output=ROOT/'FALSE_BRIGHT_RESULTS.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    receipt={'passed':True,'source_sha256':sha(__file__),'output_sha256':sha(output),
        'frozen_manifest_sha256':result['frozen_manifest_sha256'],'all_four_arms':True,
        'no_test_no_fit_no_raw_arrays':True,'admission_sha256':sha(ROOT/'FALSE_BRIGHT_GPU_ADMISSION.xml')}
    (ROOT/'FALSE_BRIGHT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({a:r['delta_sse_epoch20_minus_epoch0'] for a,r in result['arms'].items()},indent=2))

if __name__=='__main__':
    main()
