"""Bind independent source/protocol review to the exact synthetic-only receipt."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
synthetic=json.loads((ROOT/'SYNTHETIC_CHECKS.json').read_text())
assert sha(ROOT/'SOURCE_MANIFEST.json')=='1315f7fcb6112c893ad0855b2144af7cf4711baadd7e6438ae5f96075d8dd042'
assert sha(ROOT/'SYNTHETIC_CHECKS.json')=='1ca088f8db35f63c4d46eb3cd98e6fc86bfb6bc46cae548f7cd6cf11856de9a9'
assert synthetic['passed'] and synthetic['synthetic_only'] and not synthetic['real_dataset_or_labels_read']
assert len(synthetic['checks'])==21 and all(synthetic['checks'].values())
assert synthetic['source_hashes']==manifest['source_hashes']
assert synthetic['environment']==manifest['environment']
for name,h in manifest['source_hashes'].items(): assert sha(ROOT/name)==h,name
result={
 'passed':True,'phase':'pre_corpus','reviewer':'science_implementation, independent of builder author',
 'source_manifest_sha256':sha(ROOT/'SOURCE_MANIFEST.json'),
 'synthetic_checks_sha256':sha(ROOT/'SYNTHETIC_CHECKS.json'),
 'source_hashes':manifest['source_hashes'],
 'independent_review_source_sha256':sha(__file__),
 'review_basis':'Read all six frozen source/protocol/settings files and all 21 synthetic check definitions; verified actual closure and receipt bytes. No corpus or target decoding by this review.',
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
