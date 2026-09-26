"""Save immutable implementation and runtime evidence while workers continue."""
import hashlib,json,pathlib,tarfile,time,sys
ROOT=pathlib.Path(__file__).resolve().parent
def main():
    label=sys.argv[1] if len(sys.argv)>1 else 'mto_launch'
    assert label in ('mto_launch','detanet_launch')
    files=sorted(set(list(ROOT.glob('*.py'))+[ROOT/'campaign.json',ROOT/'README.md']+list((ROOT/'configs').glob('*.json'))+list((ROOT/'frozen_reference').rglob('*.py'))+list((ROOT/'official_detanet').rglob('*.py'))+list((ROOT/'reports').glob('*.json'))+[ROOT/'reports/detanet_protocol.md',ROOT/'reports/environment_freeze.txt',ROOT/'reports/nvidia_before_launch.txt']))
    target=ROOT/f'execution_snapshot_{label}.tar.gz';assert not target.exists()
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with tarfile.open(target,'w:gz') as tf:
        for p in files:tf.add(p,arcname=str(p.relative_to(ROOT)))
    report_name='execution_snapshot.json' if label=='mto_launch' else 'execution_snapshot_detanet.json'
    (ROOT/'reports'/report_name).write_text(json.dumps(dict(time=time.time(),archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=manifest,scope=label),indent=2))
    old=ROOT/'reports/SUPERVISOR_FAILED.json'
    if old.exists():
        data=json.loads(old.read_text())
        if 'KeyboardInterrupt' in data.get('traceback',''):
            old.rename(ROOT/'reports/SUPERVISOR_intentional_restart_attempt_1.json')
    print('Saved immutable execution snapshot:',target)
if __name__=='__main__':main()
