"""Synthetic target-blind grouping, safeguards, assignment and immutable-I/O checks."""
import io
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from rdkit import Chem
import split_core as core
import build_partition as builder

def raises(kind,call):
    try:call()
    except kind:return
    raise AssertionError('Expected '+kind.__name__)
def mol(smiles):return Chem.AddHs(Chem.MolFromSmiles(smiles))
def z_of(m):return np.array([a.GetAtomicNum() for a in m.GetAtoms()])

def main():
    checks={}
    z=np.array([6,1,1,8]);pos=np.array([[0.,0.,0.],[1.,.1,.2],[-.2,1.,.3],[.4,.5,1.1]])
    signature=core.distance_signature(z,pos)
    q=np.linalg.qr(np.array([[.1,.3,.9],[.4,.8,.2],[.7,.6,.5]]))[0]
    for name,p in [('rotation',pos@q),('reflection',pos*np.array([-1,1,1])),('translation',pos+np.array([12.,-4.,2.]))]:
        np.testing.assert_allclose(signature,core.distance_signature(z,p),atol=5e-15,rtol=0);checks[name]=True
    perm=np.array([3,1,0,2]);np.testing.assert_allclose(signature,core.distance_signature(z[perm],pos[perm]),atol=5e-15,rtol=0);checks['atom_permutation']=True
    a=mol('C[C@H](O)F');b=mol('C[C@@H](O)F')
    assert core.heavy_formula_key(a,z_of(a))==core.heavy_formula_key(b,z_of(b));checks['stereoisomers_conservatively_grouped']=True
    assert core.heavy_formula_key(mol('CC=O'),z_of(mol('CC=O')))==core.heavy_formula_key(mol('C=CO'),z_of(mol('C=CO')));checks['tautomer_bond_order_family_grouped']=True
    h=core.parse_cached('[H][H]');assert '<NO_HEAVY_ATOMS>' in core.heavy_formula_key(h,np.array([1,1]));checks['hydrogen_only_graph']=True
    iso=core.parse_cached('[13CH3][O-]');assert any(a.GetIsotope() or a.GetFormalCharge() for a in iso.GetAtoms());checks['unsanitized_isotope_charge_retention']=True
    explicit=Chem.MolToSmiles(mol('C'),allHsExplicit=True)
    assert core.parse_cached(explicit).GetNumAtoms()==5;checks['explicit_hydrogens_preserved']=True
    vectors=np.array([[0.,0.],[1.,1.],[.00005,.00005],[-1.,-1.],[.00009,.00001]])
    pairs=core.geometry_pairs(vectors)
    assert set(pairs)=={(0,2),(0,4),(2,4)};checks['complete_nonadjacent_radius_pairs']=True
    edge=np.array([[0.],[1e-4],[np.nextafter(1e-4,np.inf)]])
    assert (0,1) in core.geometry_pairs(edge) and (0,2) not in core.geometry_pairs(edge);checks['exact_fp64_geometry_boundary']=True
    uf=core.UnionFind(4);core.union_key_groups(uf,[['A'],['B'],['A','C'],['B','C']]);assert uf.labels().tolist()==[0,0,0,0]
    _,train_only,_=core.assign_components(np.array([10,20,30,40]),uf.labels(),np.array([False,False,True,True]),target=0)
    assert train_only.all();checks['all_candidate_links_and_ambiguous_component_propagation']=True
    reg=core.SignatureRegistry(lambda b:'collision');reg.add('x','a');raises(ValueError,lambda:reg.add('x','b'));checks['unequal_hash_collision_stops']=True
    ids=np.arange(100,112,dtype=np.int64);labels=np.repeat([0,3,6,9],3).astype(np.int64);ambiguous=np.zeros(12,dtype=bool);ambiguous[9]=True
    p,t,g=core.assign_components(ids,labels,ambiguous,target=2);p2,t2,g2=core.assign_components(ids,labels,ambiguous,target=2)
    assert all(np.array_equal(p[k],p2[k]) for k in p) and {k:len(v) for k,v in p.items()}=={'test':3,'val':3,'train':6}
    assert set(range(9,12))<=set(p['train']);checks['deterministic_whole_group_overshoot']=True
    keys=[[str(x)] for x in labels]
    proof=core.verify_partitions(ids,p,labels,ambiguous,keys,[],[x[0] for x in keys]);assert proof['all_rows_once'];checks['all_rows_and_original_keys_verified']=True
    raises(ValueError,lambda:core.assign_components(ids,labels,np.ones(12,dtype=bool),target=2));checks['insufficient_eligible_rows_stop']=True
    # Known graph identity joins conformers even when geometry signatures differ.
    methane=mol('C');zz=np.tile(z_of(methane),(2,1));pp=np.stack([np.arange(15).reshape(5,3)*.1,np.arange(15).reshape(5,3)*.2])
    fake_candidates=core.connectivity_candidates
    try:
        core.connectivity_candidates=lambda z,p:([core.heavy_formula_key(methane,z)],['SyntheticCandidateFailure'])
        ll,aa,kk,gg,ss=core.build_components(np.array([1,2]),zz,pp,[[1,explicit,True,'synthetic'],[2,explicit,False,'synthetic']])
        assert ll.tolist()==[0,0] and aa.tolist()==[False,True] and ss['candidate_failures']==1
    finally:core.connectivity_candidates=fake_candidates
    checks['conformer_group_and_candidate_failure_retains_known_links']=True
    cands,failures=core.connectivity_candidates(np.array([1,1]),np.array([[0.,0.,0.],[.74,0.,0.]]));assert cands and not failures;checks['all_five_connectivity_factors_exercised']=True
    with tempfile.TemporaryDirectory(prefix='qm9s_synthetic_only_') as temp:
        temp=Path(temp);archive=temp/'synthetic.npz'
        with zipfile.ZipFile(archive,'w') as f:
            f.writestr('ids.npy',builder.array_bytes(np.array([1,2],dtype=np.int64)))
            f.writestr('E.npy',b'NEVER DECODE THIS SYNTHETIC SENTINEL')
        guarded=builder.IdentityArchive(archive)
        assert guarded.load('ids.npy').tolist()==[1,2]
        for forbidden in ('E.npy','A.npy','f.npy','mask_f.npy','predictions.npy','train.npy'):
            raises(PermissionError,lambda member=forbidden:guarded.load(member))
        assert guarded.decoded==['ids.npy'];checks['target_prediction_mask_members_rejected_before_open']=True
        pth=temp/'immutable.txt';builder.put_immutable(pth,b'a');builder.put_immutable(pth,b'a')
        raises(ValueError,lambda:builder.put_immutable(pth,b'b'));assert pth.read_bytes()==b'a';checks['immutable_write_idempotence']=True
        oldroot=builder.ROOT
        try:
            builder.ROOT=temp;raises(FileNotFoundError,lambda:builder.verified_review('missing.json','pre_corpus'))
        finally:builder.ROOT=oldroot
        checks['missing_independent_review_fails']=True
    result={'passed':True,'checks':checks,'synthetic_only':True,'real_dataset_or_labels_read':False,'source_hashes':builder.source_manifest()['source_hashes'],'environment':builder.environment()}
    builder.put_immutable(builder.ROOT/'SYNTHETIC_CHECKS.json',builder.canonical(result))
    print(json.dumps({'passed':True,'checks':len(checks),'receipt_sha256':builder.sha(builder.ROOT/'SYNTHETIC_CHECKS.json')}))
if __name__=='__main__':main()
