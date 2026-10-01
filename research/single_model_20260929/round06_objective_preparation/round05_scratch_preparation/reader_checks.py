"""Synthetic target-access checks; no real source, partition, label or model read."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('OMP_NUM_THREADS')=='2' and os.environ.get('MKL_NUM_THREADS')=='2'
import io,json,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import partition_data as module
from common import ROOT,sha,atomic_json

def expect(error,fn):
    try:fn()
    except error:return
    raise AssertionError('Expected '+error.__name__)

def main():
    checks={}
    with patch.object(module,'sha',side_effect=AssertionError('No real path may be inspected')):
        expect(PermissionError,lambda:module.Partition('test'))
        expect(PermissionError,lambda:module.Partition('val'))
    checks['test_and_unapproved_validation_reject_before_files']=True
    with tempfile.TemporaryDirectory(prefix='round05_synthetic_reader_') as directory:
        directory=Path(directory)
        # Forbidden rows contain invented NaN sentinels. Only rows1,3,4 are permitted.
        target=np.arange(60,dtype=np.float64).reshape(6,10);target[[0,2,5]]=np.nan
        with zipfile.ZipFile(directory/'raw_labels.npz','w',compression=zipfile.ZIP_DEFLATED) as archive:
            buffer=io.BytesIO();np.save(buffer,target,allow_pickle=False);archive.writestr('f.npy',buffer.getvalue())
        part=module.Partition.__new__(module.Partition)
        part.indices=np.array([1,3,4],dtype=np.int64);part.allowed=np.array([False,True,False,True,True,False])
        part.corpus_size=6;part.name='train';part.ledger=[];part.verified_sources={}
        calls=[];original=module.convert_block
        def inspected(raw,dtype,shape,global_rows):
            assert part.allowed[global_rows].all();calls.append(global_rows.copy())
            return original(raw,dtype,shape,global_rows)
        with patch.object(module,'DATA',directory),patch.dict(module.PINS,{'raw_labels.npz':sha(directory/'raw_labels.npz')}),patch.object(module,'convert_block',inspected),patch.object(np,'load',side_effect=AssertionError('Full np.load prohibited')),patch.object(np.lib.format,'read_array',side_effect=AssertionError('Full read_array prohibited')):
            actual=part.selected('raw_labels.npz','f')
            np.testing.assert_array_equal(actual,target[part.indices])
            assert np.array_equal(np.concatenate(calls),part.indices) and len(calls)==2
            assert part.ledger[0]['numeric_rows']==3 and part.ledger[0]['numeric_decode_blocks']==2
            assert part.ledger[0]['numeric_global_indices_sha256']==module.index_sha(part.indices)
        checks['only_selected_numeric_blocks_with_full_array_load_blocked']=True
        checks['actual_decoder_rows_and_fields_match_ledger']=True
        with patch.object(module.zipfile,'ZipFile',side_effect=AssertionError('Forbidden request opened archive')):
            for rows,error in [(np.array([0],dtype=np.int64),PermissionError),
                               (np.arange(6,dtype=np.int64),PermissionError),
                               (np.array([1,1],dtype=np.int64),ValueError),
                               (np.array([4,1],dtype=np.int64),ValueError),
                               (np.array([-1],dtype=np.int64),ValueError),
                               (np.array([6],dtype=np.int64),ValueError),
                               (np.array([1.0]),ValueError),
                               (np.array([],dtype=np.int64),ValueError)]:
                expect(error,lambda rows=rows:part.selected('raw_labels.npz','f',rows))
            expect(PermissionError,lambda:part.selected('raw_labels.npz','unknown'))
            expect(PermissionError,lambda:part.selected('other.npz','f'))
        checks['ten_invalid_requests_rejected_before_archive_open']=True
        part.verified_sources={}
        with patch.object(module,'DATA',directory),patch.dict(module.PINS,{'raw_labels.npz':'0'*64}),patch.object(module.zipfile,'ZipFile',side_effect=AssertionError('Hash mismatch opened archive')):
            expect(AssertionError,lambda:part.selected('raw_labels.npz','f'))
        checks['source_hash_mismatch_rejected_before_numeric_access']=True
    result={'passed':True,'synthetic_only':True,'real_data_or_model_read':False,'checks':checks,
            'numeric_rows_decoded_synthetic':3,'forbidden_numeric_rows_decoded':0,
            'source_hashes':{n:sha(ROOT/n) for n in ('common.py','partition_data.py','reader_checks.py','prepare_train_statistics.py')}}
    atomic_json(result,ROOT/'READER_CHECKS.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
