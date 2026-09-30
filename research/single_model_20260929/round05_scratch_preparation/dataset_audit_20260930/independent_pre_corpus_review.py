"""Bind independent source/protocol review to the exact synthetic-only receipt."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
synthetic=json.loads((ROOT/'SYNTHETIC_CHECKS.json').read_text())
assert sha(ROOT/'SOURCE_MANIFEST.json')=='2bcd4d256179ad99f6b88b34f50bba1a99facc9389a6438535662410c6783a38'
assert sha(ROOT/'SYNTHETIC_CHECKS.json')=='a8668972b37dff51ebb0d293f04d888e5b151b506c8e89e836670ad8ccd7d851'
assert synthetic['passed'] and synthetic['synthetic_only'] and not synthetic['real_dataset_or_labels_read']
assert len(synthetic['checks'])==22 and all(synthetic['checks'].values())
assert synthetic['source_hashes']==manifest['source_hashes']
assert synthetic['environment']==manifest['environment']
for name,h in manifest['source_hashes'].items(): assert sha(ROOT/name)==h,name
snapshot=ROOT/'repair01_original'
snapshot_manifest=json.loads((snapshot/'SNAPSHOT_MANIFEST.json').read_text())
for name,h in snapshot_manifest['files'].items():assert sha(snapshot/name)==h,name
for name in ['split_core.py','builder_settings.json','NEW_PARTITION_PROTOCOL.md','IMPLEMENTATION_CONTRACT.md']:
    assert sha(ROOT/name)==sha(snapshot/name),'Scientific grouping/assignment changed'
from build_partition import canonical
fixture={'histogram':{2:7,10:3,1:12,14:1}}
saved=canonical(fixture);reloaded=json.loads(saved)
assert saved==canonical(reloaded)
changed=json.loads(saved);changed['histogram']['10']+=1
assert saved!=canonical(changed)
assert not (ROOT/'private_partition').exists() and not (ROOT/'SPLIT_MANIFEST.json').exists()
failed=json.loads((ROOT/'ops/materialize_attempt/FAILED.json').read_text())
assert failed['exit_code']!=0 and failed['output_sha256'] is None
result={
 'passed':True,'phase':'pre_corpus','reviewer':'science_implementation, independent of builder author',
 'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),
 'synthetic_checks_sha256':sha(ROOT/'SYNTHETIC_CHECKS.json'),
 'source_hashes':manifest['source_hashes'],
 'independent_review_source_sha256':sha(__file__),
 'review_basis':'Read all six frozen source/protocol/settings files and all 22 synthetic check definitions; verified exact closure and receipts. Independently executed canonical integer/string histogram round-trip and changed-count rejection. No corpus or target decoding by this review.',
 'repair':{'scope':'Canonical JSON comparison and regression only; grouping/settings/protocol unchanged.',
           'old_records_byte_verified':len(snapshot_manifest['files']),
           'snapshot_manifest_sha256':sha(snapshot/'SNAPSHOT_MANIFEST.json'),
           'failed_attempt_preserved':True,'no_private_arrays_or_split_manifest':True,
           'independent_multidigit_roundtrip_and_count_mutation_checks':True},
 'findings':[],
 'verified_design':[
  'IdentityArchive allows only ids/z/pos and rejects other members before ZIP open.',
  'Existing full keys, conservative heavy/formula keys, and all five ambiguous-row candidate keys are unioned, with complete-component TRAIN propagation.',
  'Complete cKDTree query_pairs p=inf eps=0 and direct FP64 bound implement the fixed geometry safeguard; no nearest-only truncation.',
  'Smallest-row DSU representatives, ascending minimum-ID component ordering and one PCG64 shuffle implement deterministic TEST then validation assignment with retained overshoot.',
  'Every valid-corpus row is retained once; original-key, heavy-key, component, row, ID and geometry safeguards are explicitly checked.',
  'Audit and materialization are separated by exact source/synthetic and source/corpus review gates; different existing outputs fail instead of being replaced.',
  'Old split and inner-split arrays enter exposure counts only; no target statistics, prediction or checkpoint accesses exist in the builder.'
 ],
 'scope':'Allows target-blind corpus aggregate audit only under existing root authority. Does not authorize private split materialization, target statistics, model preparation, training or test scoring.',
 'limitations':['Bond perception is heuristic; overlap absence is under the documented rules, not proof of absolute identity.','New partition contains historically exposed records and is not independent external data.','The approved single executor must avoid concurrent writes; immutable output behavior is not a multi-writer scheduler.']
}
encoded=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
path=ROOT/'PRE_CORPUS_REVIEW.json'
if path.exists(): assert path.read_bytes()==encoded
else:path.write_bytes(encoded)
print(json.dumps({'passed':True,'review_sha256':sha(path),'source_manifest_sha256':result['source_manifest_sha256'],'source_hashes_verified':len(manifest['source_hashes'])}))
