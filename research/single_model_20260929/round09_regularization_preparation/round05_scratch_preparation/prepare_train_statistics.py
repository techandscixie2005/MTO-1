"""New-TRAIN-only statistics; requires independent reader review before target reads."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert os.environ.get('OMP_NUM_THREADS')=='2' and os.environ.get('MKL_NUM_THREADS')=='2'
from common import ROOT,sha,read,immutable_json,PINS
import numpy as np
from partition_data import Partition,index_sha

def main():
    review=read(ROOT/'TRAIN_READER_REVIEW.json')
    assert review['passed'] and review['phase']=='train_statistics_preparation'
    assert review['split_manifest_sha256']==PINS['SPLIT_MANIFEST.json']
    for name,h in review['source_hashes'].items():assert sha(ROOT/name)==h
    assert not (ROOT/'TRAIN_STATISTICS.json').exists(),'Statistics already exist; inspect rather than refit'
    part=Partition('train');rows=part.indices
    z=part.selected('dataset.npz','z');ids=part.selected('raw_labels.npz','ids');assert np.array_equal(ids,part.ids)
    e=part.selected('raw_labels.npz','E').astype(np.float64)
    a=part.selected('raw_labels.npz','A').astype(np.float64)
    f=part.selected('raw_labels.npz','f').astype(np.float64)
    masks={k:part.selected('raw_labels.npz',k).astype(bool) for k in ('mask_E','mask_A','mask_f')}
    assert all(v.all() for v in masks.values()),'Unexpected invalid labels; no exclusions'
    assert e.shape==f.shape==(120355,10) and a.shape==(120355,10,3,3)
    assert all(np.isfinite(v).all() for v in (e,a,f)) and (f>=0).all()
    mean=e.mean(0);se2=float(np.square(e-mean).mean());sa2=float(np.square(a).sum((-2,-1)).mean())
    e_moment=float((np.square(e).mean(0)-mean**2).mean())
    a_block=float(sum(np.einsum('nsij,nsij->',b,b) for b in np.array_split(a,31))/masks['mask_A'].sum())
    assert abs(e_moment-se2)<1e-11 and abs(a_block-sa2)<1e-12
    stats={'sE2':se2,'sA2':sa2,'E_state_mean':mean.tolist(),'n_ref':float(np.median((z!=0).sum(1))),
           'f_variance':float(np.var(f,ddof=0)),'f_mean':float(f.mean()),
           'q90':float(np.quantile(f,.9)),'q99':float(np.quantile(f,.99)),
           'molecules':len(rows),'valid_E':int(masks['mask_E'].sum()),'valid_A':int(masks['mask_A'].sum()),
           'valid_f':int(masks['mask_f'].sum()),'printed_f_zeros':int((f==0).sum()),
           'train_indices_sha256':index_sha(rows),'train_ids_sha256':index_sha(part.ids),
           'split_manifest_sha256':PINS['SPLIT_MANIFEST.json'],'dtype':'float64 raw-label reduction',
           'definitions':{'sE2':'mean((E - TRAIN per-state mean)^2)','sA2':'mean(sum_ij A_ij^2)',
                          'n_ref':'median TRAIN atom count','f_variance':'pooled population ddof0',
                          'quantiles':'pooled TRAIN raw f, including zeros, NumPy linear quantile'},
           'training_target_note':'The original FP32 dataset E/A arrays are used in fitting; statistics use raw FP64 labels.'}
    assert min(se2,sa2,stats['n_ref'],stats['f_variance'])>0
    immutable_json(stats,ROOT/'TRAIN_STATISTICS.json')
    receipt={'passed':True,'purpose':'new_train_statistics_only','model_updates':0,'model_inference':False,
             'actual_decode_ledger':part.ledger,'decoded_validation_or_test_target_rows':0,
             'compressed_transport_note':'ZIP seek/decompression may pass unselected bytes; only selected contiguous TRAIN blocks reach np.frombuffer.',
             'formula_crosscheck_abs':{'sE2':abs(e_moment-se2),'sA2':abs(a_block-sa2)},
             'statistics_sha256':sha(ROOT/'TRAIN_STATISTICS.json'),'reader_review_sha256':sha(ROOT/'TRAIN_READER_REVIEW.json'),
             'source_hashes':{n:sha(ROOT/n) for n in ('common.py','partition_data.py','prepare_train_statistics.py')}}
    immutable_json(receipt,ROOT/'TRAIN_STATISTICS_AUDIT.json')
    print(__import__('json').dumps({'passed':True,'molecules':len(rows),'valid_f':stats['valid_f'],
                                   'statistics_sha256':receipt['statistics_sha256']}))
if __name__=='__main__':main()
