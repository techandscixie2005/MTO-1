import json,os,pathlib,subprocess,sys,time
from dataset import ROOT
from prepare import sha
from trainer import atomic_json,fingerprint
from supervisor import health
def main():
    assert json.loads((ROOT/'reports/preflight.json').read_text())['passed']
    assert json.loads((ROOT/'reports/system_preflight.json').read_text())['passed']
    h,b,xml=health();(ROOT/'reports/gpu_health_immediately_before_submit.xml').write_text(xml)
    permissions={i:os.access(f'/dev/nvidia{i}',os.R_OK|os.W_OK) for i in (1,2,4,6)};assert all(permissions.values())
    for p in (ROOT/'data').iterdir():
        if p.is_file():p.chmod(0o444)
    for p in (ROOT/'initial').glob('*.pt'):p.chmod(0o444)
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.json'))+list(ROOT.glob('*.md'))+list((ROOT/'configs').glob('*.json'))
    files+=list((ROOT/'frozen_reference').rglob('*.py'))+list((ROOT/'pinned_reference').rglob('*.py'))
    atomic_json(dict(time=time.time(),files={str(p.relative_to(ROOT)):sha(p) for p in files},training_fingerprint=fingerprint(),GPU_device_permissions=permissions,
        health=h,currently_busy_uuids=sorted(b),old_experiment_access='read only; no shared environment changes; no GPU reset; no GitHub push'),ROOT/'reports/submission_manifest.json')
    subprocess.check_call([sys.executable,'control.py','submit'],cwd=ROOT)
if __name__=='__main__':main()
