"""Review aggregate target-blind audit before private split materialization."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((ROOT/n).read_text())
a=read('CORPUS_AUDIT.json');m=read('SOURCE_MANIFEST.json');terminal=read('ops/audit_attempt_repair01/COMPLETE.json')
assert sha(ROOT/'CORPUS_AUDIT.json')=='644b6e6d5e2c7af1b2a6f234715a8d5acdaed9d596f746eba9f897c539f45a1c'
assert a['source_manifest_sha256']==sha(ROOT/'SOURCE_MANIFEST.json')=='2bcd4d256179ad99f6b88b34f50bba1a99facc9389a6438535662410c6783a38'
for name,h in m['source_hashes'].items():assert sha(ROOT/name)==h
assert terminal['exit_code']==0 and terminal['review_unchanged'] and terminal['source_manifest_unchanged']
assert terminal['output_sha256']==sha(ROOT/'CORPUS_AUDIT.json')
old=read('repair01_original/CORPUS_AUDIT.json')
assert {k:v for k,v in a.items() if k!='source_manifest_sha256'}=={k:v for k,v in old.items() if k!='source_manifest_sha256'}
assert not (ROOT/'private_partition').exists()
assert a['passed'] and not a['materialized_by_this_receipt']
assert a['decoded_dataset_members']==['ids.npy','z.npy','pos.npy'] and not a['target_prediction_members_decoded']
assert a['counts']=={'train':120355,'val':6686,'test':6686} and sum(a['counts'].values())==133727
assert a['overshoot']=={'val':0,'test':0}
g=a['grouping'];assert sum(int(k)*v for k,v in g['component_size_histogram'].items())==133727
assert sum(g['component_size_histogram'].values())==g['components']==114572
assert g['original_key_union_merges']+g['heavy_formula_union_merges']+g['geometry_union_merges']==133727-g['components']
assert g['candidate_failures']==0 and g['ambiguous_rows_before_propagation']==a['train_only_rows_after_propagation']==278
assert a['train_only_components']==263 and g['geometry_matching_pairs']==83 and g['largest_component']==14
d=a['disjointness'];assert d['all_rows_once'] and d['ambiguous_components_train_only']
assert all(v==0 for v in d['pairwise_overlap'].values() for v in v.values())
assert all(d[k]==0 for k in ['cross_partition_geometry_pairs','cross_partition_heavy_formula_keys','cross_partition_original_keys'])
e=a['old_exposure_intersections']
for name,row in e.items():
    assert row['old_train']+row['old_val']+row['old_test']==a['counts'][name]
    assert row['old_inner_fit']+row['old_inner_calibration']==row['old_train']
for name,expected in [('old_train',120355),('old_val',6686),('old_test',6686),('old_inner_fit',96284),('old_inner_calibration',24071)]:
    assert sum(row[name] for row in e.values())==expected
result={'passed':True,'phase':'materialization','reviewer':'science_implementation, independent of builder author',
    'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),'corpus_audit_sha256':sha(ROOT/'CORPUS_AUDIT.json'),
    'pre_corpus_review_sha256':sha(ROOT/'PRE_CORPUS_REVIEW.json'),'reviewer_source_sha256':sha(__file__),
    'audit_terminal_receipt_sha256':sha(ROOT/'ops/audit_attempt_repair01/COMPLETE.json'),
    'repair_old_new_scientific_aggregates_and_planned_array_hashes_exactly_equal':True,
    'old_failed_materialization_preserved':(ROOT/'ops/materialize_attempt/FAILED.json').exists(),
    'counts':a['counts'],'components':g['components'],'aggregate_arithmetic_passed':True,'no_findings':True,
    'materialization_scope':'Write the exact private arrays already hashed in the approved aggregate after deterministic rebuild equality. No target decoding, statistics, model fitting or scoring.',
    'independent_saved_array_reconstruction_still_required':True,
    'historical_exposure':'New test contains5989 oldTRAIN,361 oldvalidation,336 oldtest rows. No historical predictor is eligible; scratch initialization/newTRAINstats required.',
    'limitations':'Component-disjoint under declared conservative rules, not absolute identity proof or external fresh data.'}
raw=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode();path=ROOT/'MATERIALIZATION_REVIEW.json'
if path.exists():assert path.read_bytes()==raw
else:path.write_bytes(raw)
print(json.dumps({'passed':True,'review_sha256':sha(path)}))
