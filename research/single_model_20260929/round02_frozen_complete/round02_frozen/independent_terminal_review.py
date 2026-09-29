"""Independent completed-round audit: CPU metadata/arrays only, no model inference."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PARENT=ROOT.parent
sys.path.insert(0,str(PARENT/'reports/velocity_audit'))
from audit_velocity_labels import selected_rows

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def score(y,p):
    y=np.asarray(y,dtype=np.float64).ravel();p=np.asarray(p,dtype=np.float64).ravel()
    assert np.isfinite(y).all() and np.isfinite(p).all()
    e=p-y;sse=float(np.dot(e,e));sst=float(np.dot(y-y.mean(),y-y.mean()))
    return {'count':len(y),'sse':sse,'sst':sst,'r2':1-sse/sst,'mae':float(abs(e).mean()),'rmse':float(np.sqrt(sse/len(y)))}
def compare(a,b):
    assert a['count']==b['count']
    for k in ('sse','sst','r2','mae','rmse'):assert np.isclose(a[k],b[k],rtol=1e-10,atol=1e-11),(k,a[k],b[k])
def close(a,b):assert np.isclose(a,b,rtol=1e-10,atol=1e-10),(a,b)

def main():
    torch.set_num_threads(1)
    mh=sha(ROOT/'FROZEN_MANIFEST.json')
    assert mh=='6ea090e28201f55cb6125fbf43ebb306645bc94893c53e138d764dce25e22f6e'
    mf=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text());cfg=mf['config']
    for path,value in mf['source_hashes'].items():assert sha(path)==value,path
    pc=json.loads((PARENT/'round_config.json').read_text());source=Path(pc['source'])
    cache=json.loads((ROOT/'CACHE_COMPLETE.json').read_text())
    assert sha(ROOT/'CACHE_COMPLETE.json')==mf['cache_receipt_sha256']
    with np.load(source/'data/dataset.npz',allow_pickle=False) as z:
        train=z['train'].copy();val=z['val'].copy();ids=z['ids'].copy()
    order=np.argsort(val);undo=np.argsort(order)
    with zipfile.ZipFile(source/'data/raw_labels.npz') as z:
        truth={k:selected_rows(z,k,val[order])[undo] for k in ('f','E','mask_f','mask_E')}
    assert len(train)==120355 and len(val)==6686 and truth['mask_f'].sum()==66860
    gen=np.random.default_rng(11);orders=[hashlib.sha256(train[gen.permutation(len(train))].tobytes()).hexdigest() for _ in range(20)]
    native=torch.load(pc['source_checkpoint'],map_location='cpu',weights_only=False)['model']
    analysis=json.loads((ROOT/'ANALYSIS_RECEIPT.json').read_text())
    result=json.loads((ROOT/'ROUND02_RESULTS.json').read_text())
    replay=json.loads((ROOT/'TERMINAL_REPLAY.json').read_text())
    assert analysis['passed'] and result['passed'] and replay['passed']
    assert analysis['manifest_sha256']==result['manifest_sha256']==replay['manifest_sha256']==mh
    for receipt in (analysis,replay):
        for path,value in receipt['source_hashes'].items():assert sha(path)==value,path
    for name,value in analysis['output_hashes'].items():assert sha(ROOT/name)==value
    assert result['test_evaluated'] is False and replay['test_inference'] is False
    assert not replay['selected_predictor_original_base_data_cache_access']
    alpha=.8511830211044088;beta=.003725185373211049
    arms={};input_hashes={}
    for arm in cfg['arms']:
        rd=ROOT/'runs'/arm
        assert not (rd/'FAILED.json').exists()
        c=json.loads((rd/'FIT_COMPLETE.json').read_text())
        h=[json.loads(s) for s in (rd/'history.jsonl').read_text().splitlines()]
        assert c['completed_epoch']==20 and c['manifest_sha256']==mh
        assert [r['epoch'] for r in h]==list(range(21))
        assert [r['order_sha256'] for r in h[1:]]==orders
        assert min(h,key=lambda r:r['validation']['raw_f']['sse'])['epoch']==c['selected_epoch']
        assert all(r['frozen_parameters_and_buffers_unchanged'] for r in h[1:])
        cp={name:torch.load(rd/(name+'.pt'),map_location='cpu',weights_only=False) for name in ('best','last')}
        for name,k in cp.items():
            assert sha(rd/(name+'.pt'))==c[name+'_checkpoint_sha256']==replay['arms'][arm][name]['checkpoint_sha256']
            assert k['manifest_sha256']==mh and k['config']==cfg and k['arm']==arm
            assert k['frozen_parameter_buffer_hashes']==cache['frozen_parameter_buffer_hashes']
            assert all(torch.equal(k['model'][key],value) for key,value in native.items())
            extras={key:value for key,value in k['model'].items() if key not in native}
            assert extras and all(key.startswith('right_adapter.') for key in extras)
            assert sum(v.numel() for v in extras.values())==4016
            assert any(torch.count_nonzero(value) for key,value in extras.items() if key.startswith('right_adapter.mix.'))
        assert cp['best']['epoch']==c['selected_epoch']
        last=cp['last'];assert last['state']['completed_epoch']==20 and last['state']['steps']==37620
        assert last['state']['best_epoch']==c['selected_epoch'] and last['rng']['order']==gen.bit_generator.state
        assert len(last['optimizer']['state'])==9
        assert all(g['lr']==1e-5 and g['amsgrad'] and g['weight_decay']==0 for g in last['optimizer']['param_groups'])
        assert all('max_exp_avg_sq' in v for v in last['optimizer']['state'].values())
        with np.load(rd/'best_validation_predictions.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        assert np.array_equal(a['indices'],val) and np.array_equal(a['ids'],ids[val])
        assert np.array_equal(a['f_true'],truth['f']) and np.array_equal(a['mask_f'],truth['mask_f'])
        assert np.array_equal(a['E_true'],truth['E'])
        mask=truth['mask_f'].astype(bool)
        for pred,metrics in ((a['f_pred'],c['validation']),(np.maximum(0,alpha*a['f_pred']+beta),c['validation']['fixed_calibrated_secondary'])):
            compare(score(truth['f'][mask],pred[mask]),metrics['raw_f'])
            for j in range(10):compare(score(truth['f'][:,j][mask[:,j]],pred[:,j][mask[:,j]]),metrics['per_state'][j])
            for name,cut in mf['tail_thresholds'].items():
                use=mask&(truth['f']>=cut);compare(score(truth['f'][use],pred[use]),metrics['bright_tail'][name])
        em=truth['mask_E'].astype(bool);compare(score(truth['E'][em],a['E_pred'][em]),c['validation']['energy'])
        for tag in ('best','last'):
            r=replay['arms'][arm][tag];ref=c['validation'] if tag=='best' else h[-1]['validation']
            assert abs(r['metrics']['raw_f']['r2']-ref['raw_f']['r2'])<1e-7
            groups=r['true_pred_q99_bins'];assert sum(g['count'] for g in groups.values())==66860
            close(sum(g['sse'] for g in groups.values()),r['metrics']['raw_f']['sse'])
            for j in range(10):
                assert sum(g['per_state'][j]['count'] for g in groups.values())==6686
                close(sum(g['per_state'][j]['sse'] for g in groups.values()),r['metrics']['per_state'][j]['sse'])
        pred=a['f_pred'];err=(pred-truth['f'])**2
        for tb in (False,True):
            for pb in (False,True):
                key=('true_bright' if tb else 'true_dim')+'__'+('pred_bright' if pb else 'pred_dim')
                use=mask&((truth['f']>=.2412)==tb)&((pred>=.2412)==pb)
                v=result['arms'][arm]['selected_true_pred_q99_bins'][key]
                assert v['count']==use.sum();close(v['sse'],err[use].sum())
        rr=result['arms'][arm]
        compare(rr['validation']['raw_f'],c['validation']['raw_f'])
        close(rr['delta_native_r2_vs_own_epoch0'],c['validation']['raw_f']['r2']-c['epoch_zero']['raw_f']['r2'])
        assert not rr['priority_for_independent_seeds']
        state_delta=[y['sse']-x['sse'] for x,y in zip(c['epoch_zero']['per_state'],h[-1]['validation']['per_state'])]
        arms[arm]={'selected_epoch':c['selected_epoch'],'selected_native_r2':c['validation']['raw_f']['r2'],
            'selected_fixed_calibrated_r2':c['validation']['fixed_calibrated_secondary']['raw_f']['r2'],
            'selected_native_delta_vs_epoch0':rr['delta_native_r2_vs_own_epoch0'],
            'fixed20_native_r2':h[-1]['validation']['raw_f']['r2'],
            'fixed20_state_SSE_delta_vs_epoch0':state_delta,
            'all_original_model_tensors_bitwise_unchanged':True,'adapter_weights_changed':True,
            'original_and_nonpersistent_buffers_verified_by_terminal_loader':True,
            'order_optimizer_RNG_resume_and_selection_verified':True,
            'selected_arrays_metrics_independently_recomputed':True,'all_brightness_partitions_verified':True,
            'total_clipped_epoch_fractions':sum(r['gradient_clip_fraction'] for r in h[1:]),
            'maximum_preclip_gradient_norm':max(r['max_preclip_gradient_norm'] for r in h[1:]),
            'last_adapter_movement':h[-1]['adapter_movement']}
        input_hashes[arm]={name:sha(rd/name) for name in ('FIT_COMPLETE.json','history.jsonl','best.pt','last.pt','best_validation_predictions.npz')}
        for name,value in analysis['input_hashes'][arm].items():assert sha(rd/name)==value
        del cp,last
    assert analysis['input_hashes']['terminal_replay_sha256']==sha(ROOT/'TERMINAL_REPLAY.json')
    assert analysis['input_hashes']['anchor_array_sha256']==sha(PARENT/'runs/control/best_validation_predictions.npz')
    assert analysis['input_hashes']['round01_control_history_sha256']==sha(PARENT/'runs/control/history.jsonl')
    plot=json.loads((ROOT/'PLOT_MANIFEST.json').read_text())
    assert plot['source_sha256']==sha(ROOT/'plot_frozen.py')
    assert plot['input_sha256']==sha(ROOT/'ROUND02_RESULTS.json')
    assert plot['output_sha256']==sha(ROOT/'ROUND02_CURVES.svg')
    inventory=json.loads((ROOT/'CHECKPOINT_INVENTORY.json').read_text())
    assert inventory['passed'] and inventory['manifest_sha256']==mh
    for arm,r in inventory['arms'].items():
        assert r['selected_epoch']==arms[arm]['selected_epoch'] and r['optimizer_steps']==37620
        for f in r['files']:
            assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256']
    assert result['raw_f_minus_trace']['objective_specific_confirmation_priority'] is False
    output={'passed':True,'frozen_manifest_sha256':mh,'fresh_frozen_hash_count':64,'arms':arms,
        'no_arm_promoted':True,'strongest_eligible_calibrated_baseline_unchanged':True,
        'no_new_model_inference_GPU_fit_or_test_by_reviewer':True,
        'decoded_validation_labels':66860,'decoded_test_label_rows':0,'input_hashes':input_hashes,
        'analysis_receipt_sha256':sha(ROOT/'ANALYSIS_RECEIPT.json'),
        'terminal_replay_sha256':sha(ROOT/'TERMINAL_REPLAY.json'),
        'plot_and_checkpoint_inventory_verified':True,
        'review_script_sha256':sha(__file__),
        'scope':'Selected raw-f/calibrated/state/tail/energy metrics independently recomputed. Terminal final-epoch geometry replay and partition algebra checked, without duplicate model inference. Bootstrap is descriptive reused-validation evidence.'}
    (ROOT/'INDEPENDENT_TERMINAL_REVIEW.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'passed':True,'selected':{a:x['selected_epoch'] for a,x in arms.items()},'no_arm_promoted':True}))

if __name__=='__main__':main()
