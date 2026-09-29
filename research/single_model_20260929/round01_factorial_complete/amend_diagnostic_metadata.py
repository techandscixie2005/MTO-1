"""One reviewed metadata/source amendment; frozen subsets/bins stay identical."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

records=[]
for config,script,reason in (
    ('GAP_AUDIT_CONFIG.json','gap_audit.py','Move validation f/mask reads after terminal gate; thresholds unchanged'),
    ('MECHANISM_CONFIG.json','mechanism_audit.py','Label drift R2 as baseline-output agreement rather than target predictive accuracy; subset/statistics unchanged')):
    path=ROOT/config;original=ROOT/(path.stem+'_ORIGINAL.json')
    before=json.loads(path.read_text())
    if not original.exists():
        original.write_bytes(path.read_bytes())
    else:
        assert json.loads(original.read_text())==before,'Amendment already applied'
    after=json.loads(json.dumps(before))
    target=after['definition'] if config.startswith('GAP') else after
    old_script=target['script_sha256'];target['script_sha256']=sha(ROOT/script)
    path.write_text(json.dumps(after,indent=2)+'\n')
    records.append({'config':config,'original_config_sha256':sha(original),'updated_config_sha256':sha(path),
        'original_script_sha256':old_script,'updated_script_sha256':target['script_sha256'],'reason':reason})
report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'independent_reviewer':'history_baseline',
    'fit_sources_changed':False,'subset_or_bin_definitions_changed':False,'amendments':records}
(ROOT/'DIAGNOSTIC_METADATA_AMENDMENT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
