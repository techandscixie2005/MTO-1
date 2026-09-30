"""Audit/freeze a new partition using identity and geometry only; no label decoding."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import zipfile
import numpy as np
import scipy
import rdkit
from rdkit import RDLogger
import split_core as core

ROOT=Path(__file__).resolve().parent
SOURCES=('NEW_PARTITION_PROTOCOL.md','IMPLEMENTATION_CONTRACT.md','builder_settings.json','split_core.py','build_partition.py','synthetic_split_checks.py')
ALLOWED_MEMBERS=frozenset(('ids.npy','z.npy','pos.npy'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def canonical(obj):return (json.dumps(obj,indent=2,sort_keys=True)+'\n').encode()
def read(path):return json.loads(Path(path).read_text())
def put_immutable(path,data):
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Existing artifact differs; retain it for review: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.rename(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def array_bytes(array):
    stream=io.BytesIO();np.save(stream,array,allow_pickle=False);return stream.getvalue()

class IdentityArchive:
    def __init__(self,path):self.path=Path(path);self.decoded=[]
    def load(self,member):
        if member not in ALLOWED_MEMBERS:raise PermissionError('Target/prediction member decoding prohibited: '+member)
        with zipfile.ZipFile(self.path) as archive:
            if archive.namelist().count(member)!=1:raise ValueError('Missing/duplicate allowed member')
            with archive.open(member) as stream:array=np.lib.format.read_array(stream,allow_pickle=False)
        self.decoded.append(member);return array

def environment():
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'rdkit':rdkit.__version__}
def settings():return read(ROOT/'builder_settings.json')
def source_manifest():
    cfg=settings();assert environment()==cfg['environment']
    assert cfg['seed']==20260930 and cfg['rows']==133727 and cfg['heldout_target_rows']==6686
    assert cfg['heldout_fraction']==.05 and cfg['geometry_tolerance_angstrom']==core.GEOMETRY_TOLERANCE
    assert tuple(cfg['connectivity_factors'])==core.FACTORS and set(cfg['allowed_dataset_members'])==ALLOWED_MEMBERS
    return {'source_hashes':{name:sha(ROOT/name) for name in SOURCES},'environment':environment(),'input_pins':cfg['inputs'],'partition_name':cfg['partition_name'],'targets_used':False}
def verified_review(name,phase):
    path=ROOT/name;review=read(path)
    assert review['passed'] is True and review['phase']==phase
    assert review['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
    return review
def check_preconditions(mode):
    assert read(ROOT/'SOURCE_MANIFEST.json')==source_manifest()
    preflight=read(ROOT/'SYNTHETIC_CHECKS.json');assert preflight['passed']
    assert preflight['source_hashes']==source_manifest()['source_hashes']
    review=verified_review(settings()['independent_review_before_audit'],'pre_corpus')
    assert review['synthetic_checks_sha256']==sha(ROOT/'SYNTHETIC_CHECKS.json')
    if mode=='materialize':
        review=verified_review(settings()['independent_review_before_materialization'],'materialization')
        assert review['corpus_audit_sha256']==sha(ROOT/'CORPUS_AUDIT.json')
    return review

def prepare(mode):
    check_preconditions(mode);cfg=settings();RDLogger.DisableLog('rdApp.*')
    for value in cfg['inputs'].values():assert sha(value['path'])==value['sha256']
    archive=IdentityArchive(cfg['inputs']['dataset']['path'])
    ids=archive.load('ids.npy');z=archive.load('z.npy');pos=archive.load('pos.npy')
    assert len(ids)==cfg['rows'] and z.shape==(133727,29) and pos.shape==(133727,29,3)
    identities=read(cfg['inputs']['identities']['path'])
    labels,ambiguous,keys,pairs,summary=core.build_components(ids,z,pos,identities,lambda x:print(json.dumps(x),flush=True))
    parts,train_only,groups=core.assign_components(ids,labels,ambiguous,cfg['seed'],cfg['heldout_target_rows'])
    checks=core.verify_partitions(ids,parts,labels,ambiguous,keys,pairs,[row[1] for row in identities])
    old=read(cfg['inputs']['old_splits']['path'])['indices']
    assert sorted(sum(old.values(),[]))==list(range(len(ids)))
    exposure={'old_'+key:np.array(value,dtype=np.int64) for key,value in old.items()}
    for name in ('old_inner_fit','old_inner_calibration'):
        exposure[name]=np.load(cfg['inputs'][name]['path'],allow_pickle=False)
        assert exposure[name].dtype==np.int64 and exposure[name].ndim==1
    assert np.array_equal(np.sort(np.concatenate([exposure['old_inner_fit'],exposure['old_inner_calibration']])),np.sort(exposure['old_train']))
    exposure_counts={name:{oldname:int(np.intersect1d(rows,oldrows).size) for oldname,oldrows in exposure.items()} for name,rows in parts.items()}
    arrays={name+'_indices.npy':rows for name,rows in parts.items()}
    arrays.update({'component_index.npy':labels,'train_only_mask.npy':train_only,'ids.npy':ids})
    private=ROOT/cfg['private_output_directory']
    array_metadata={name:{'path':str(private/name),'sha256':hashlib.sha256(array_bytes(a)).hexdigest(),'content_sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),'shape':list(a.shape),'dtype':str(a.dtype),'archive_allowed':False} for name,a in arrays.items()}
    identity_signatures=json.dumps({'heavy_formula_keys':keys,'component_rows':{str(k):v for k,v in sorted(groups.items())}},sort_keys=True,separators=(',',':')).encode()+b'\n'
    audit={'passed':True,'partition_name':cfg['partition_name'],'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),'inputs':cfg['inputs'],'environment':environment(),
        'seed':cfg['seed'],'assignment':'Whole components shuffled once by PCG64; TEST then VALID until each reaches6686; remainder and ambiguous linked components TRAIN.',
        'counts':{k:len(v) for k,v in parts.items()},'heldout_target_rows':6686,'overshoot':{k:len(parts[k])-6686 for k in ('test','val')},
        'grouping':summary,'train_only_rows_after_propagation':int(train_only.sum()),'train_only_components':len(set(map(int,labels[train_only]))),
        'disjointness':checks,'old_exposure_intersections':exposure_counts,
        'array_metadata':array_metadata,'private_identity_signatures_sha256':hashlib.sha256(identity_signatures).hexdigest(),
        'decoded_dataset_members':archive.decoded,'target_prediction_members_decoded':[],
        'claim_limitations':['New partition of historically used QM9S, not independent external data.','Disjoint under audited conservative graph/formula/geometry rules; heuristic connectivity does not prove absolute chemical identity.','Fresh initialization and new TRAIN-only target statistics required; old models/calibration are ineligible.'],
        'materialized_by_this_receipt':False}
    assert archive.decoded==['ids.npy','z.npy','pos.npy']
    if mode=='audit':
        put_immutable(ROOT/'CORPUS_AUDIT.json',canonical(audit))
    else:
        assert read(ROOT/'CORPUS_AUDIT.json')==audit,'Idempotence: current audit differs from approved result'
        for name,array in arrays.items():put_immutable(private/name,array_bytes(array))
        put_immutable(private/'identity_components.json',identity_signatures)
        for name,entry in array_metadata.items():assert sha(private/name)==entry['sha256']
        manifest={'passed':True,'partition_name':cfg['partition_name'],'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),'corpus_audit_sha256':sha(ROOT/'CORPUS_AUDIT.json'),
            'materialization_review_sha256':sha(ROOT/cfg['independent_review_before_materialization']),'arrays':array_metadata,'private_identity_signatures':{'path':str(private/'identity_components.json'),'sha256':hashlib.sha256(identity_signatures).hexdigest(),'archive_allowed':False},
            'counts':audit['counts'],'disjointness':checks,'old_exposure_intersections':exposure_counts,'targets_decoded':False,'old_splits_unchanged':sha(cfg['inputs']['old_splits']['path'])==cfg['inputs']['old_splits']['sha256']}
        put_immutable(ROOT/'SPLIT_MANIFEST.json',canonical(manifest))
    print(json.dumps({'passed':True,'mode':mode,'counts':audit['counts'],'components':summary['components'],'train_only_rows':int(train_only.sum()),'corpus_audit_sha256':sha(ROOT/'CORPUS_AUDIT.json'),'private_arrays_materialized':mode=='materialize'}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','audit','materialize'));args=parser.parse_args()
    if args.mode=='freeze':put_immutable(ROOT/'SOURCE_MANIFEST.json',canonical(source_manifest()));print(sha(ROOT/'SOURCE_MANIFEST.json'))
    else:prepare(args.mode)
if __name__=='__main__':main()
