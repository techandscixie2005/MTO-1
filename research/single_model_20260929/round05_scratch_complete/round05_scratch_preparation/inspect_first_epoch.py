"""Read committed checkpoint metadata only; no model construction or inference."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('OMP_NUM_THREADS') == '2' and os.environ.get('MKL_NUM_THREADS') == '2'
import hashlib, io, json, sys, time
from pathlib import Path
import torch
from common import ROOT, CAMPAIGN, read, sha, atomic_json
from execution_gate import verify
sys.path.insert(0, str(CAMPAIGN/'ops'))
from monitor import process_identity


def main():
    output=ROOT/'ops/FIRST_EPOCH_METADATA.json'
    assert not output.exists(), 'Do not repeat completed metadata inspection'
    permit=verify(ROOT/'PRODUCTION_EXECUTION_AUTHORIZATION.json')
    manifest=permit['manifest']
    split=read(CAMPAIGN/'dataset_audit_20260930/SPLIT_MANIFEST.json')
    results={}
    for arm in ('control','adapter','decorrelation','both'):
        rd=ROOT/'runs'/arm
        assert not (rd/'FAILED.json').exists()
        attempts=list((rd/'attempts').glob('*/LAUNCH_RECEIPT.json'))
        assert len(attempts)==1, 'First-epoch inspection expects original launch only'
        launch=read(attempts[0]); observed=process_identity(launch['identity']['pid'])
        assert observed is not None
        for key in ('pid','start_ticks','uid','cwd','argv_sha256','boot_id'):
            assert observed[key]==launch['identity'][key], (arm,key)
        assert launch['registration_returncode']==0
        # Opening one atomic committed inode avoids a rename race with the worker.
        data=(rd/'last.pt').read_bytes(); checkpoint_sha=hashlib.sha256(data).hexdigest()
        ck=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
        state=ck['state']; epoch=state['completed_epoch']
        assert epoch>=1 and ck['format']=='round05_resumable_v1'
        assert ck['arm']==arm and ck['manifest_sha256']==permit['manifest_sha256']
        assert ck['initial_base_tensor_sha256']==manifest['initial_base_tensor_sha256']
        assert ck['initial_full_tensor_sha256']==manifest['initial_full_tensor_sha256']
        assert ck['permit']['authorization_sha256']==permit['authorization_sha256']
        assert state['steps']==epoch*1881
        assert [r['epoch'] for r in state['history']]==list(range(epoch+1))
        for row in state['history'][1:]:
            assert row['order_sha256']==manifest['epoch_order_sha256'][row['epoch']-1]
            assert row['optimizer_batches']==1881 and row['validation']['pooled']['count']==66860
        assert set(ck['rng'])=={'python','numpy','order','torch','cuda'}
        assert ck['rng']['torch'].dtype==torch.uint8 and ck['rng']['cuda'].dtype==torch.uint8
        opt=ck['optimizer']; assert len(opt['param_groups'])==1
        group=opt['param_groups'][0]
        assert group['lr']==.001 and group['amsgrad'] and group['weight_decay']==0
        steps={int(v['step']) for v in opt['state'].values()}
        assert steps=={state['steps']}
        for item in opt['state'].values():
            assert {'step','exp_avg','exp_avg_sq','max_exp_avg_sq'}<=set(item)
        best=state['best']; assert sha(rd/best['checkpoint'])==best['checkpoint_sha256']
        assert sha(rd/best['predictions'])==best['predictions_sha256']
        assert sha(rd/state['last_predictions']['path'])==state['last_predictions']['sha256']
        access=read(rd/'DATA_ACCESS.json'); assert access['test_numeric_rows']==0
        for partition,key in (('train','train'),('validation','val')):
            entries=access[partition]
            assert entries
            for entry in entries:
                assert entry['partition']==key
                assert entry['numeric_rows']==split['counts'][key]
                assert entry['numeric_global_indices_sha256']==split['arrays'][key+'_indices.npy']['content_sha256']
        results[arm]={'observed_identity':observed,'launch_receipt_sha256':sha(attempts[0]),
            'observed_completed_epoch':epoch,'committed_steps':state['steps'],
            'read_last_checkpoint_sha256':checkpoint_sha,'read_last_checkpoint_bytes':len(data),
            'optimizer_state_entries':len(opt['state']),'optimizer_step_values':sorted(steps),
            'rng_keys':sorted(ck['rng']),'best':best,
            'first_epoch_seconds':state['history'][1]['seconds'],
            'epoch0_r2':state['history'][0]['validation']['pooled']['r2'],
            'first_epoch_r2':state['history'][1]['validation']['pooled']['r2'],
            'all_observed_order_hashes_match':True,'data_access_sha256':sha(rd/'DATA_ACCESS.json'),
            'train_and_validation_ledgers_match_exact_partitions':True,'test_numeric_rows':0}
        del ck,data
    receipt={'passed':True,'checked_unix':time.time(),'scope':'read_only_committed_checkpoint_metadata',
        'script_sha256':sha(__file__),'manifest_sha256':permit['manifest_sha256'],
        'authorization_sha256':permit['authorization_sha256'],'arms':results,
        'model_constructed':False,'model_inference':False,'raw_target_arrays_opened':False,
        'checkpoint_tensors_loaded_on_cpu_only':True,'production_files_modified':False}
    atomic_json(receipt,output)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
