"""Target-free conservative molecular grouping. All returned identities remain private."""
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.spatial import cKDTree

FACTORS=(1.2,1.15,1.1,1.25,1.3)
GEOMETRY_TOLERANCE=1e-4
ALLOWED_Z={1,6,7,8,9}

class UnionFind:
    def __init__(self,n): self.parent=np.arange(n,dtype=np.int64)
    def find(self,x):
        x=int(x)
        while self.parent[x]!=x:
            self.parent[x]=self.parent[self.parent[x]];x=int(self.parent[x])
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return False
        self.parent[max(a,b)]=min(a,b);return True
    def labels(self):return np.array([self.find(i) for i in range(len(self.parent))],dtype=np.int64)

class SignatureRegistry:
    def __init__(self,digest=None):
        self.values={};self.digest=digest or (lambda b:hashlib.sha256(b).hexdigest())
    def add(self,kind,signature):
        payload=(kind+'\0'+signature).encode();h=self.digest(payload)
        if h in self.values and self.values[h]!=payload:raise ValueError('Unequal signature hash collision; stop for review')
        self.values[h]=payload;return h

def formula(z):return tuple(sorted(Counter(map(int,z)).items()))

def parse_cached(key):
    params=Chem.SmilesParserParams();params.sanitize=False;params.removeHs=False
    mol=Chem.MolFromSmiles(key,params)
    if mol is None:raise ValueError('cached_key_parse_failure')
    return mol

def heavy_formula_key(mol,z):
    rw=Chem.RWMol();mapping={}
    for a in mol.GetAtoms():
        if a.GetAtomicNum()!=1:
            atom=Chem.Atom(a.GetAtomicNum());atom.SetFormalCharge(0);atom.SetIsotope(0)
            atom.SetIsAromatic(False);atom.SetNoImplicit(True);atom.SetNumExplicitHs(0)
            mapping[a.GetIdx()]=rw.AddAtom(atom)
    for bond in mol.GetBonds():
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if a in mapping and b in mapping:rw.AddBond(mapping[a],mapping[b],Chem.BondType.SINGLE)
    graph=Chem.MolToSmiles(rw.GetMol(),canonical=True,isomericSmiles=False,allBondsExplicit=True,allHsExplicit=True) if mapping else '<NO_HEAVY_ATOMS>'
    return json.dumps([formula(z),graph],separators=(',',':'))

def connectivity_candidates(z,pos):
    # Reproduce original connectivity precision/factors, without bond-order fitting.
    table=Chem.GetPeriodicTable()
    xyz=str(len(z))+'\nQM9S identity audit\n'+'\n'.join(f'{table.GetElementSymbol(int(a))} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f}' for a,p in zip(z,pos))
    keys=[];failures=[]
    for factor in FACTORS:
        try:
            mol=Chem.MolFromXYZBlock(xyz)
            if mol is None:raise ValueError('XYZ parse')
            rdDetermineBonds.DetermineConnectivity(mol,useVdw=True,covFactor=factor)
            keys.append(heavy_formula_key(mol,z))
        except Exception as exc:
            # Retain all other links and taint the whole linked component; never silently drop.
            failures.append(type(exc).__name__)
    return sorted(set(keys)),failures

def distance_signature(z,pos):
    pos=np.asarray(pos,dtype=np.float64);z=np.asarray(z)
    pair_distances=defaultdict(list)
    for i in range(len(z)):
        for j in range(i+1,len(z)):
            pair=tuple(sorted((int(z[i]),int(z[j]))))
            pair_distances[pair].append(float(np.linalg.norm(pos[i]-pos[j])))
    return np.array([d for pair in sorted(pair_distances) for d in sorted(pair_distances[pair])],dtype=np.float64)

def geometry_pairs(vectors,tolerance=GEOMETRY_TOLERANCE):
    vectors=np.asarray(vectors,dtype=np.float64)
    if len(vectors)<2:return []
    if vectors.shape[1]==0:return [(a,b) for a in range(len(vectors)) for b in range(a+1,len(vectors))]
    tree=cKDTree(vectors)
    # Complete radius pairs, including nonadjacent/non-nearest matches; no count cap.
    candidate=tree.query_pairs(r=tolerance,p=np.inf,eps=0,output_type='ndarray')
    return [(int(a),int(b)) for a,b in sorted(map(tuple,candidate.tolist()))
            if np.max(np.abs(vectors[a]-vectors[b]))<=tolerance]

def union_key_groups(uf,keys):
    groups=defaultdict(list)
    for row,rowkeys in enumerate(keys):
        for key in sorted(set(rowkeys)):groups[key].append(row)
    links=0
    for key in sorted(groups):
        rows=groups[key]
        for row in rows[1:]:links+=int(uf.union(rows[0],row))
    return groups,links

def assign_components(ids,labels,ambiguous,seed=20260930,target=None):
    groups=defaultdict(list)
    for i,label in enumerate(labels):groups[int(label)].append(i)
    train_only={label for label,rows in groups.items() if np.any(ambiguous[rows])}
    ordered=sorted((label for label in groups if label not in train_only),key=lambda x:int(np.min(ids[groups[x]])))
    shuffled=np.asarray(ordered,dtype=np.int64)
    np.random.Generator(np.random.PCG64(seed)).shuffle(shuffled)
    target=round(len(ids)*.05) if target is None else target
    parts={};cursor=0
    for name in ('test','val'):
        rows=[]
        while len(rows)<target:
            if cursor>=len(shuffled):raise ValueError('Insufficient resolved components; no assignment fallback')
            rows.extend(groups[int(shuffled[cursor])]);cursor+=1
        parts[name]=np.array(sorted(rows),dtype=np.int64)
    rows=[i for label in list(shuffled[cursor:])+sorted(train_only) for i in groups[int(label)]]
    parts['train']=np.array(sorted(rows),dtype=np.int64)
    train_only_mask=np.isin(labels,list(train_only))
    return parts,train_only_mask,groups

def build_components(ids,z_padded,pos_padded,identities,progress=lambda x:None):
    n=len(ids);assert len(identities)==n and len(set(map(int,ids)))==n
    assert np.issubdtype(ids.dtype,np.integer) and np.issubdtype(z_padded.dtype,np.integer)
    assert z_padded.ndim==2 and pos_padded.shape==z_padded.shape+(3,)
    assert np.isfinite(pos_padded).all()
    assert [int(x[0]) for x in identities]==list(map(int,ids))
    uf=UnionFind(n);registry=SignatureRegistry();ambiguous=np.zeros(n,dtype=bool)
    reasons=Counter();keys=[];formula_groups=defaultdict(list);signatures={};original=[]
    candidate_failures=0;candidate_rows=0
    for i in range(n):
        valid=z_padded[i]!=0;z=z_padded[i][valid];pos=np.asarray(pos_padded[i][valid],dtype=np.float64)
        if not len(z) or not set(map(int,z))<=ALLOWED_Z:raise ValueError('Invalid geometry atoms; stop without dropping')
        rowkeys=[];key=identities[i][1];original.append([key]);registry.add('full_key',key)
        unresolved=not identities[i][2]
        if unresolved:ambiguous[i]=True;reasons['previously_unresolved']+=1
        try:
            mol=parse_cached(key)
            if formula([a.GetAtomicNum() for a in mol.GetAtoms()])!=formula(z):
                ambiguous[i]=True;reasons['cached_composition_mismatch']+=1
            if len(Chem.GetMolFrags(mol,sanitizeFrags=False))!=1:
                ambiguous[i]=True;reasons['cached_disconnected']+=1
            if any(a.GetIsotope()!=0 or a.GetFormalCharge()!=0 for a in mol.GetAtoms()):
                ambiguous[i]=True;reasons['cached_isotope_or_charge_ambiguity']+=1
            rowkeys.append(heavy_formula_key(mol,z))
        except Exception:
            ambiguous[i]=True;reasons['cached_parse_or_graph_failure']+=1
        # Also cover newly detected metadata ambiguities; same frozen five factors, no choices.
        if ambiguous[i]:
            candidate_rows+=1;candidates,failures=connectivity_candidates(z,pos)
            rowkeys.extend(candidates)
            if failures:reasons['candidate_generation_failure']+=1;candidate_failures+=len(failures)
        for k in rowkeys:registry.add('heavy_formula',k)
        keys.append(sorted(set(rowkeys)));formula_groups[formula(z)].append(i)
        signatures[i]=distance_signature(z,pos)
        if (i+1)%20000==0:progress({'stage':'identity','rows':i+1})
    _,old_merges=union_key_groups(uf,original)
    key_groups,graph_merges=union_key_groups(uf,keys)
    geometry_matches=[];geo_merges=0
    for f in sorted(formula_groups):
        rows=formula_groups[f]
        if len(rows)>1:
            vectors=np.stack([signatures[i] for i in rows])
            for a,b in geometry_pairs(vectors):
                a,b=rows[a],rows[b];geometry_matches.append((a,b));geo_merges+=int(uf.union(a,b))
    labels=uf.labels()
    groups=defaultdict(list)
    for i,label in enumerate(labels):groups[int(label)].append(i)
    for rows in groups.values():registry.add('component',json.dumps(sorted(map(int,ids[rows])),separators=(',',':')))
    summary={'rows':n,'components':len(groups),'largest_component':max(map(len,groups.values())),
        'original_key_union_merges':old_merges,'heavy_formula_union_merges':graph_merges,'geometry_union_merges':geo_merges,
        'geometry_matching_pairs':len(geometry_matches),'formula_buckets':len(formula_groups),
        'ambiguous_rows_before_propagation':int(ambiguous.sum()),'ambiguity_reasons':dict(sorted(reasons.items())),
        'candidate_rows':candidate_rows,'candidate_failures':candidate_failures,
        'component_size_histogram':dict(sorted(Counter(map(len,groups.values())).items())),
        'hash_collision_guard_passed':True}
    progress({'stage':'components_complete',**summary})
    return labels,ambiguous,keys,geometry_matches,summary

def verify_partitions(ids,parts,labels,ambiguous,keys,geometry_matches,original_keys):
    n=len(ids);joined=np.concatenate(list(parts.values()))
    assert np.array_equal(np.sort(joined),np.arange(n)) and len(np.unique(joined))==n
    membership=np.full(n,-1,dtype=np.int8)
    for k,name in enumerate(('train','val','test')):membership[parts[name]]=k
    components=defaultdict(set)
    for i,label in enumerate(labels):components[int(label)].add(int(membership[i]))
    assert all(len(v)==1 for v in components.values())
    assert all(components[int(labels[i])]=={0} for i in np.flatnonzero(ambiguous))
    key_parts=defaultdict(set)
    for i,kk in enumerate(keys):
        for key in kk:key_parts[key].add(int(membership[i]))
    assert all(len(v)==1 for v in key_parts.values())
    original_parts=defaultdict(set)
    for i,key in enumerate(original_keys):original_parts[key].add(int(membership[i]))
    assert all(len(v)==1 for v in original_parts.values())
    assert all(membership[a]==membership[b] for a,b in geometry_matches)
    overlap={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlap[a+'__'+b]={'rows':len(set(map(int,parts[a]))&set(map(int,parts[b]))),
            'ids':len(set(map(int,ids[parts[a]]))&set(map(int,ids[parts[b]]))),
            'components':len(set(map(int,labels[parts[a]]))&set(map(int,labels[parts[b]])))}
        assert all(x==0 for x in overlap[a+'__'+b].values())
    return {'all_rows_once':True,'ambiguous_components_train_only':True,'pairwise_overlap':overlap,
            'cross_partition_original_keys':0,'cross_partition_heavy_formula_keys':0,'cross_partition_geometry_pairs':0}
