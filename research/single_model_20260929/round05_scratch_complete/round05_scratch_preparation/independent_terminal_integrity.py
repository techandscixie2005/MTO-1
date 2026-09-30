"""Independent terminal metadata/history/hash audit, no array or tensor decoding."""
from pathlib import Path
import hashlib,json,math

ROOT=Path(__file__).resolve().parent
ARMS=('control','adapter','decorrelation','both')
EXPECTED='66269e04d73943bec48a6331c3ca45a9efcc12045e962fe897da6196af97c5e1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10),(a,b)

def main():
    mh=sha(ROOT/'FROZEN_MANIFEST.json');assert mh==EXPECTED
    m=read(ROOT/'FROZEN_MANIFEST.json')
    for p,h in m['source_hashes'].items():assert sha(p)==h,p
    auth=read(ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json')
    ah=sha(ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json')
    assert ah=='07308b113557d2196cefc951a603f3f1a5093009eaae1576412f1e393bb8162c'
    assert auth['authorized'] and auth['scope']=='round05_four_arm_60epoch_fit'
    assert auth['epochs']==60 and tuple(auth['arms'])==ARMS and not auth['test_access']
    assert auth['frozen_manifest_sha256']==mh
    assert sha(auth['publication_receipt'])==auth['publication_receipt_sha256']
    stats=read(ROOT/'TRAIN_STATISTICS.json')
    split=read(ROOT.parent/'dataset_audit_20260930/SPLIT_MANIFEST.json')
    inputs={str(ROOT/'FROZEN_MANIFEST.json'):mh,str(ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json'):ah}
    results={};sst=None
    for arm in ARMS:
        rd=ROOT/'runs'/arm;c=read(rd/'FIT_COMPLETE.json')
        assert c['arm']==arm and c['completed_epoch']==60 and c['steps']==112860
        assert not c['test_evaluated'] and c['manifest_sha256']==mh
        assert c['initial_base_tensor_sha256']==m['initial_base_tensor_sha256']
        assert c['source_split_manifest_sha256']==m['split_manifest_sha256']
        assert not (rd/'FAILED.json').exists()
        hp=rd/'history.jsonl';assert sha(hp)==c['history_sha256']
        history=[json.loads(line) for line in hp.read_text().splitlines()]
        assert [x['epoch'] for x in history]==list(range(61))
        assert [x['order_sha256'] for x in history[1:]]==m['epoch_order_sha256']
        assert all(x['optimizer_batches']==1881 and x['learning_rate']==.001 for x in history[1:])
        assert sum(x['optimizer_batches'] for x in history[1:])==112860
        best=min(history,key=lambda x:x['validation']['pooled']['sse'])
        assert best['epoch']==c['best']['epoch']
        close(best['validation']['pooled']['sse'],c['best']['sse']);close(best['validation']['pooled']['r2'],c['best']['r2'])
        for row in history:
            v=row['validation'];p=v['pooled'];assert p['count']==66860
            assert set(v['per_state'])==set(map(str,range(1,11)))
            assert all(x['count']==6686 for x in v['per_state'].values())
            close(sum(x['sse'] for x in v['per_state'].values()),p['sse'])
            close(sum(x['mae']*x['count'] for x in v['per_state'].values())/66860,p['mae'])
            close(p['mse']*66860,p['sse']);close(p['rmse']**2,p['mse'])
            value=p['sse']/(1-p['r2'])
            if sst is None:sst=value
            close(value,sst)
            for q in ('q90','q99'):assert v['bright_tail'][q]['threshold']==stats[q]
            bins=v['false_bright_bins'];assert sum(x['count'] for x in bins.values())==66860
            close(sum(x['sse'] for x in bins.values()),p['sse'])
            close(sum(x['absolute_error_sum'] for x in bins.values())/66860,p['mae'])
        ckfiles={'best.pt':c['best_checkpoint_sha256'],'last.pt':c['last_checkpoint_sha256'],
                 'geometry_best.pt':c['geometry_checkpoint_sha256'],c['best']['checkpoint']:c['best']['checkpoint_sha256'],
                 c['best']['predictions']:c['best']['predictions_sha256']}
        for name,h in ckfiles.items():assert sha(rd/name)==h,(arm,name)
        assert c['best']['checkpoint_sha256']==c['best_checkpoint_sha256']
        assert read(rd/'BEST.json')==c['best']
        access=read(rd/'DATA_ACCESS.json');assert access['test_numeric_rows']==0
        for section,partition in (('train','train'),('validation','val')):
            ledger=access[section];assert len(ledger)==12
            count=split['counts'][partition];indexsha=split['arrays'][partition+'_indices.npy']['content_sha256']
            for e in ledger:
                assert e['partition']==partition and e['numeric_rows']==count
                assert e['numeric_global_indices_sha256']==indexsha
                assert e['archive_sha256']==m['raw_archive_opaque_sha256'][Path(e['archive']).name]
        attempts=list((rd/'attempts').iterdir());assert len(attempts)==1
        launch=read(attempts[0]/'LAUNCH_RECEIPT.json');assert launch['registration_returncode']==0
        assert launch['authorization_sha256']==ah and launch['manifest_sha256']==mh
        assert launch['publication_sha256']==auth['publication_receipt_sha256']
        assert launch['review_sha256']==auth['independent_review_sha256']
        assert launch['arm']==arm and launch['environment']['CUDA_VISIBLE_DEVICES']==launch['gpu_uuid']
        assert launch['environment']['OMP_NUM_THREADS']==launch['environment']['MKL_NUM_THREADS']=='2'
        for name in ('FIT_COMPLETE.json','history.jsonl','BEST.json','DATA_ACCESS.json','status.json'):
            inputs[str(rd/name)]=sha(rd/name)
        inputs[str(attempts[0]/'LAUNCH_RECEIPT.json')]=sha(attempts[0]/'LAUNCH_RECEIPT.json')
        results[arm]={'completed_epoch':60,'steps':112860,'history_entries':61,'attempts':1,
            'selected_epoch':best['epoch'],'selected_r2':c['best']['r2'],'fixed60_r2':history[-1]['validation']['pooled']['r2'],
            'epoch0_r2':history[0]['validation']['pooled']['r2'],'sum_recorded_epoch_seconds':sum(x['seconds'] for x in history),
            'checkpoint_and_selected_array_opaque_hashes':ckfiles,'geometry_checkpoint_path':str(rd/'geometry_best.pt')}
    control=results['control']['selected_r2']
    result={'passed':True,'phase':'independent_terminal_metadata_and_hash_review','frozen_manifest_sha256':mh,
        'current_frozen_files_verified':len(m['source_hashes']),'input_hashes':inputs,'arms':results,
        'selected_deltas_vs_control':{a:results[a]['selected_r2']-control for a in ARMS[1:]},
        'any_noncontrol_meets_point003_gate':any(results[a]['selected_r2']-control>=.003 for a in ARMS[1:]),
        'pooled_sst_from_history_consistent':sst,'all_244_validation_history_records_reconciled':True,
        'all_label_count':66860,'test_numeric_rows':0,'all_60_orders_and_steps_matched':True,
        'array_or_checkpoint_tensor_values_decoded':False,'inference_or_fit_rerun':False,
        'terminal_process_exit_evidence':'Separate owned-monitor receipt; this check does not infer OS exit status from absent PIDs.',
        'saved_array_metric_recomputation':'Reserved to science analysis; pending independent source/result review.',
        'reviewer_script_sha256':sha(__file__),'blocking_findings':[]}
    p=ROOT/'INDEPENDENT_TERMINAL_INTEGRITY.json';assert not p.exists()
    p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    brief={a:{k:v for k,v in x.items() if k in ('selected_epoch','selected_r2','fixed60_r2')} for a,x in results.items()}
    print(json.dumps({'passed':True,'review_sha256':sha(p),'arms':brief},sort_keys=True))

if __name__=='__main__':main()
