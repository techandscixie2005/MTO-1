"""Exact text/AST and hash review of one metadata-only freezer repair."""
import ast
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

manifest=R/'freeze_attempt01_preserved/MANIFEST.json'
assert sha(manifest)=='6b8b365847f7d6da9d5ed4fdefe9f741ea657f0b8f9332b2076db2e425a54bde'
saved=json.loads(manifest.read_text());assert saved['optimizer_updates']==0 and saved['frozen_manifest_absent']
for entry in saved['files']:
    p=Path(entry['snapshot_path']);assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
old=R/'freeze_attempt01_preserved/freeze_preparation.py';new=R/'freeze_preparation.py'
assert sha(old)=='c9b38ac7b0d6cdc1956a584b166768d56ea07a449cfa4562258648cecaaaf05a'
assert sha(new)=='c63773b4241b07a61eec0e2ede7c9594b34603b61b7f9c114016086a055d1ed4'
text=old.read_text();assert text.count('from lr_schedule import tables,digest,epoch_lr')==1 and text.count('digest(cfg)')==2
expected=text.replace('from lr_schedule import tables,digest,epoch_lr','from lr_schedule import tables,digest as schedule_digest,epoch_lr').replace('digest(cfg)','schedule_digest(cfg)')
assert new.read_text()==expected
ast.parse(new.read_text())
assert sha(R/'SOURCE_LINEAGE.json')==sha(R/'freeze_attempt01_preserved/SOURCE_LINEAGE.json')
assert not (R/'FROZEN_MANIFEST.json').exists()
tech=json.loads((R/'TECHNICAL_SOURCE_REVIEW.json').read_text())
for p,h in tech['source_hashes'].items():assert sha(p)==h,p
assert sha(R/'ROUND10_PREPARATION_DECISION.md')=='4d23e6ab826dd2f4df1f4e72939529ff823fd2210a897a0ea4ba0f5dd6494890'
inputs={str(manifest):sha(manifest),str(new):sha(new),str(R/'SOURCE_LINEAGE.json'):sha(R/'SOURCE_LINEAGE.json'),str(R/'TECHNICAL_SOURCE_REVIEW.json'):sha(R/'TECHNICAL_SOURCE_REVIEW.json')}
inputs.update({e['snapshot_path']:e['sha256'] for e in saved['files']})
out=R/'METADATA_REPAIR_REVIEW.json';assert not out.exists()
record={'passed':True,'scope':'round10_one_metadata_freeze_retry_only','root_preparation_decision_sha256':sha(R/'ROUND10_PREPARATION_DECISION.md'),
        'diagnosis':'Later loop digest string shadowed schedule helper; initial seal raised before manifest write.',
        'old_freezer_sha256':sha(old),'corrected_freezer_sha256':sha(new),'exact_change':'One import alias and two calls only; AST parse passed.',
        'input_hashes':inputs,'source_hashes':{str(Path(__file__).resolve()):sha(__file__)},
        'numerical_source_pins_unchanged':len(tech['source_hashes']),'existing_lineage_unchanged':True,
        'one_distinct_metadata_retry_permitted':True,'numerical_reexecution_permitted':False,'production_authorized':False,
        'preserve_failure_and_distinct_retry_receipt_required':True,'new_scientific_setting_or_tolerance_change':False,
        'no_model_target_or_array_decoding_by_reviewer':True}
out.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({'passed':True,'review_sha256':sha(out),'unchanged_pins':len(tech['source_hashes'])}))
