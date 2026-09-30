"""Independently reconstruct private partition identities without target access.

No builder/core import: graph, distance and component reconstruction below use
the frozen mathematical policy with a separately written implementation.
"""
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import zipfile
import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
import rdkit
from rdkit import Chem,RDLogger
from rdkit.Chem import rdDetermineBonds

ROOT=Path(__file__).resolve().parent
FACTORS=(1.2,1.15,1.1,1.25,1.3)
TOLERANCE=1e-4
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def content_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def composition(z):return tuple(sorted(Counter(int(x) for x in z).items()))
def conservative_key(m,z):
    out=Chem.RWMol();mapping={}
    for old in m.GetAtoms():
        if old.GetAtomicNum()==1:continue
        atom=Chem.Atom(int(old.GetAtomicNum()));atom.SetNoImplicit(True)
        atom.SetFormalCharge(0);atom.SetIsotope(0);atom.SetIsAromatic(False);atom.SetNumExplicitHs(0)
        mapping[old.GetIdx()]=out.AddAtom(atom)
    for bond in m.GetBonds():
        u,v=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if u in mapping and v in mapping:out.AddBond(mapping[u],mapping[v],Chem.BondType.SINGLE)
    graph=Chem.MolToSmiles(out.GetMol(),canonical=True,isomericSmiles=False,
                          allBondsExplicit=True,allHsExplicit=True) if mapping else '<NO_HEAVY_ATOMS>'
    return json.dumps([composition(z),graph],separators=(',',':'))
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert os.environ.get('OMP_NUM_THREADS')=='2' and os.environ.get('MKL_NUM_THREADS')=='2'
    start=time.time();RDLogger.DisableLog('rdApp.*')
    frozen=read(ROOT/'SOURCE_MANIFEST.json');split=read(ROOT/'SPLIT_MANIFEST.json')
    audit=read(ROOT/'CORPUS_AUDIT.json');settings=read(ROOT/'builder_settings.json')
    assert split['passed'] and audit['passed'] and split['targets_decoded'] is False
    assert split['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')
    assert split['corpus_audit_sha256']==sha(ROOT/'CORPUS_AUDIT.json')
    assert split['materialization_review_sha256']==sha(ROOT/'MATERIALIZATION_REVIEW.json')
    for name,h in frozen['source_hashes'].items():assert sha(ROOT/name)==h
    observed_environment={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'rdkit':rdkit.__version__}
    assert observed_environment==frozen['environment']==settings['environment']
    for pin in settings['inputs'].values():assert sha(pin['path'])==pin['sha256']
    metadata={}
    for name,entry in split['arrays'].items():
        assert sha(entry['path'])==entry['sha256']
        value=np.load(entry['path'],allow_pickle=False)
        assert list(value.shape)==entry['shape'] and str(value.dtype)==entry['dtype']
        assert content_sha(value)==entry['content_sha256']
        metadata[name]=value
    private=split['private_identity_signatures'];assert sha(private['path'])==private['sha256']
    saved_signatures=read(private['path'])
    decoded=[]
    def geometry_member(name):
        assert name in ('ids','z','pos'), 'No target members allowed'
        with zipfile.ZipFile(settings['inputs']['dataset']['path']) as archive:
            assert archive.namelist().count(name+'.npy')==1
            with archive.open(name+'.npy') as stream:value=np.lib.format.read_array(stream,allow_pickle=False)
        decoded.append(name+'.npy');return value
    ids=geometry_member('ids');atomic=geometry_member('z');coords=geometry_member('pos')
    n=len(ids);assert n==133727 and np.array_equal(ids,metadata['ids.npy'])
    identities=read(settings['inputs']['identities']['path'])
    assert len(set(map(int,ids)))==n and len(identities)==n
    assert [int(r[0]) for r in identities]==list(map(int,ids))
    partitions={name:metadata[name+'_indices.npy'] for name in ('train','val','test')}
    concat=np.concatenate(list(partitions.values()))
    assert np.array_equal(np.sort(concat),np.arange(n))
    assert all(np.array_equal(rows,np.sort(rows)) for rows in partitions.values())
    membership=np.full(n,-1,dtype=np.int8)
    for code,name in enumerate(('train','val','test')):membership[partitions[name]]=code
    labels=metadata['component_index.npy'];parents=list(range(n))
    def find(a):
        a=int(a)
        while parents[a]!=a:
            parents[a]=parents[parents[a]];a=parents[a]
        return a
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    seen_original={};seen_heavy={};ambiguous=np.zeros(n,dtype=bool)
    formula_rows=defaultdict(list);distance_vectors=[];candidate_failures=0
    parser=Chem.SmilesParserParams();parser.sanitize=False;parser.removeHs=False
    table=Chem.GetPeriodicTable()
    for row in range(n):
        mask=atomic[row]!=0;z=atomic[row][mask];p=np.asarray(coords[row][mask],dtype=np.float64)
        assert len(z)>0 and set(map(int,z))<={1,6,7,8,9} and np.isfinite(p).all()
        key=identities[row][1];keys=[];ambiguous[row]=not identities[row][2]
        try:
            m=Chem.MolFromSmiles(key,parser)
            if m is None:raise ValueError('parse')
            if composition([a.GetAtomicNum() for a in m.GetAtoms()])!=composition(z):ambiguous[row]=True
            if len(Chem.GetMolFrags(m,sanitizeFrags=False))!=1:ambiguous[row]=True
            if any(a.GetIsotope() or a.GetFormalCharge() for a in m.GetAtoms()):ambiguous[row]=True
            keys.append(conservative_key(m,z))
        except Exception:ambiguous[row]=True
        if ambiguous[row]:
            xyz=str(len(z))+'\nindependent identity verification\n'+'\n'.join(
                f'{table.GetElementSymbol(int(a))} {r[0]:.9f} {r[1]:.9f} {r[2]:.9f}' for a,r in zip(z,p))
            for factor in FACTORS:
                try:
                    m=Chem.MolFromXYZBlock(xyz)
                    if m is None:raise ValueError('xyz')
                    rdDetermineBonds.DetermineConnectivity(m,useVdw=True,covFactor=factor)
                    keys.append(conservative_key(m,z))
                except Exception:candidate_failures+=1
        keys=sorted(set(keys));assert keys==saved_signatures['heavy_formula_keys'][row]
        if key in seen_original:
            other=seen_original[key];assert membership[row]==membership[other];join(row,other)
        else:seen_original[key]=row
        for heavy in keys:
            if heavy in seen_heavy:
                other=seen_heavy[heavy];assert membership[row]==membership[other];join(row,other)
            else:seen_heavy[heavy]=row
        formula_rows[composition(z)].append(row)
        # Independent vectorized pair distances, rather than the builder's nested norm loop.
        ii,jj=np.triu_indices(len(z),1);distances=pdist(p,metric='euclidean')
        types=np.sort(np.stack([z[ii],z[jj]],axis=-1),axis=-1)
        blocks=[]
        for pair in sorted(set(map(tuple,types.tolist()))):
            select=(types[:,0]==pair[0])&(types[:,1]==pair[1]);blocks.append(np.sort(distances[select]))
        distance_vectors.append(np.concatenate(blocks) if blocks else np.zeros(0,dtype=np.float64))
        if (row+1)%30000==0:print(json.dumps({'independent_identity_rows':row+1}),flush=True)
    geometry_pairs=0;cross_geometry=0
    for rows in formula_rows.values():
        if len(rows)<2:continue
        matrix=np.stack([distance_vectors[i] for i in rows])
        if matrix.shape[1]:candidate=cKDTree(matrix).query_pairs(TOLERANCE,p=np.inf,eps=0,output_type='ndarray')
        else:candidate=np.array([(i,j) for i in range(len(rows)) for j in range(i+1,len(rows))],dtype=np.int64)
        for a,b in candidate:
            if matrix.shape[1] and np.max(np.abs(matrix[a]-matrix[b]))>TOLERANCE:continue
            a,b=rows[int(a)],rows[int(b)];geometry_pairs+=1;cross_geometry+=int(membership[a]!=membership[b]);join(a,b)
    assert cross_geometry==0
    rebuilt=np.array([find(i) for i in range(n)],dtype=np.int64)
    assert np.array_equal(rebuilt,labels), 'Independent complete components differ'
    groups=defaultdict(list)
    for i,label in enumerate(rebuilt):groups[int(label)].append(i)
    assert {str(k):v for k,v in groups.items()}==saved_signatures['component_rows']
    tainted={int(rebuilt[i]) for i in np.flatnonzero(ambiguous)}
    tainted_mask=np.array([int(label) in tainted for label in rebuilt],dtype=bool)
    assert np.array_equal(tainted_mask,metadata['train_only_mask.npy'])
    assert np.all(membership[tainted_mask]==0)
    components=sorted((k for k in groups if k not in tainted),key=lambda k:min(int(ids[i]) for i in groups[k]))
    permutation=np.random.Generator(np.random.PCG64(20260930)).permutation(len(components))
    sequence=[components[int(i)] for i in permutation];cursor=0;assigned={}
    for name in ('test','val'):
        rows=[]
        while len(rows)<6686:rows.extend(groups[sequence[cursor]]);cursor+=1
        assigned[name]=np.array(sorted(rows),dtype=np.int64)
    assigned['train']=np.array(sorted(i for k in sequence[cursor:]+sorted(tainted) for i in groups[k]),dtype=np.int64)
    assert all(np.array_equal(assigned[name],partitions[name]) for name in assigned)
    assert geometry_pairs==audit['grouping']['geometry_matching_pairs']
    assert candidate_failures==audit['grouping']['candidate_failures']
    counts={k:len(v) for k,v in partitions.items()};assert counts==split['counts']==audit['counts']
    old_path=settings['inputs']['old_splits']['path']
    exposure={('old_'+key):np.array(value,dtype=np.int64) for key,value in read(old_path)['indices'].items()}
    for name in ('old_inner_fit','old_inner_calibration'):
        exposure[name]=np.load(settings['inputs'][name]['path'],allow_pickle=False)
    exposure_counts={name:{oldname:int(np.intersect1d(rows,oldrows).size) for oldname,oldrows in exposure.items()}
                     for name,rows in partitions.items()}
    assert exposure_counts==audit['old_exposure_intersections']==split['old_exposure_intersections']
    assert sha(old_path)==settings['inputs']['old_splits']['sha256']
    report={'passed':True,'independent_reconstruction':True,'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),
        'split_manifest_sha256':sha(ROOT/'SPLIT_MANIFEST.json'),'corpus_audit_sha256':sha(ROOT/'CORPUS_AUDIT.json'),
        'reviewer_source_sha256':sha(__file__),'counts':counts,'components':len(groups),
        'decoded_dataset_members':decoded,'target_members_decoded':[],
        'all_rows_once':True,'private_array_and_identity_hashes_verified':True,
        'original_and_heavy_formula_keys_cross_partition':0,'geometry_pairs_reconstructed':geometry_pairs,
        'geometry_pairs_cross_partition':cross_geometry,'component_labels_exactly_reconstructed':True,
        'assignment_exactly_reconstructed':True,'ambiguity_mask_exactly_reconstructed':True,
        'ambiguous_rows':int(ambiguous.sum()),'train_only_rows':int(tainted_mask.sum()),
        'candidate_failures':candidate_failures,'old_split_hash_unchanged':True,
        'old_exposure_intersections':exposure_counts,
        'private_input_hashes':{k:v['sha256'] for k,v in split['arrays'].items()},
        'environment':observed_environment,'duration_seconds':time.time()-start,
        'claim':'Zero overlap under the fixed conservative identity rules; no absolute identity proof or external fresh holdout claim.',
        'training_authorized_by_this_verification':False}
    path=ROOT/'INDEPENDENT_SPLIT_VERIFICATION.json'
    assert not path.exists(), 'Retain existing verification rather than overwrite'
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':True,'receipt_sha256':sha(path),'counts':counts,'seconds':report['duration_seconds']}),flush=True)
if __name__=='__main__':main()
