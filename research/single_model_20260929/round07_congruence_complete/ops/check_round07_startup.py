"""Scheduled registry observation plus explicit Round07 startup verification; no scientific imports."""
import argparse,collections,hashlib,json,os,subprocess
from pathlib import Path
from monitor import ROOT,atomic_json,file_info,process_identity,stamp
AUTH='dfb6a5fe356e30da37260830f8032fe7892f2749bb0afc30b6daf277652afaaa'
SOURCE='68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203'
PUBLICATION='ce3f342916bb6ecb6a76516bc8da8b64fc1c9839f406dcd9e852c6c12185d6c0'
ASSIGN={'original':(1,'GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800'),'scalar':(2,'GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa'),'tensor':(4,'GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lock_evidence(pid,target):
    found=[]
    for fd in (Path('/proc')/str(pid)/'fd').iterdir():
        try:
            if os.readlink(fd)!=str(target):continue
            text=(Path('/proc')/str(pid)/'fdinfo'/fd.name).read_text()
            locks=[x for x in text.splitlines() if x.startswith('lock:')]
            found.append(dict(fd=int(fd.name),target=str(target),lock_records=locks))
        except FileNotFoundError:continue
    assert found and any('FLOCK' in x and 'WRITE' in x for e in found for x in e['lock_records']),str(target)
    return found
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',action='append',required=True);args=p.parse_args();assert len(set(args.run_id))==3
    rd=ROOT/'round07_congruence_preparation';assert sha(rd/'PRODUCTION_EXECUTION_AUTHORIZATION.json')==AUTH
    assert sha(rd/'FROZEN_MANIFEST.json')==SOURCE
    assert sha(ROOT/'ops/ROUND07_PREPARATION_PUBLICATION_RECEIPT.json')==PUBLICATION
    result=subprocess.run(['/usr/bin/python3',str(ROOT/'ops/monitor.py'),'--root',str(ROOT)],capture_output=True,text=True,check=True)
    name=json.loads(result.stdout)['record'];record=ROOT/'monitoring'/name;obs=json.loads(record.read_text())
    reg=json.loads((ROOT/'ops/registry.json').read_text());assert len(obs['runs'])==29
    gpu={x['index']:x for x in obs['gpu_snapshot']['gpus']};runs=[]
    for runid in args.run_id:
        registered=next(x for x in reg['runs'] if x['id']==runid);observed=next(x for x in obs['runs'] if x['id']==runid)
        arm=Path(registered['run_dir']).name;index,uuid=ASSIGN[arm];pid=registered['identity']['pid']
        live=process_identity(pid);keys=('pid','start_ticks','uid','cwd','argv_sha256','boot_id')
        assert live and all(live[k]==registered['identity'][k] for k in keys)
        assert observed['state']=='RUNNING_IDENTITY_MATCH' and observed['observed_gpu_uuids']==[uuid]
        assert registered['gpu_uuid']==uuid and live['cwd']==str(rd)
        attempt=rd/'runs'/arm/'attempts'/runid;launch=json.loads((attempt/'LAUNCH_RECEIPT.json').read_text())
        assert launch['identity']==registered['identity'] and launch['registration_returncode']==0
        assert launch['authorization_sha256']==AUTH and launch['manifest_sha256']==SOURCE and launch['publication_sha256']==PUBLICATION
        assert launch['gpu']==index and launch['gpu_uuid']==uuid
        values=dict(x.split('=',1) for x in (Path('/proc')/str(pid)/'environ').read_bytes().decode().split('\x00') if '=' in x)
        env={k:values.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONUNBUFFERED')}
        assert env=={'CUDA_VISIBLE_DEVICES':uuid,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
        locks=lock_evidence(pid,Path('/tmp/mto_pouter_gpu_%d.lock'%index))
        worker=lock_evidence(pid,rd/'runs'/arm/'worker.lock')
        for period in ('ecc_volatile','ecc_aggregate'):
            values=[v for k,v in gpu[index][period].items() if 'uncorrectable' in k];assert values and all(v=='0' for v in values)
        assert gpu[index]['remapped_rows']['remapped_row_pending']=='No' and gpu[index]['remapped_rows']['remapped_row_failure']=='No'
        assert not observed['files']['failure_path']['exists'] and not observed['files']['completion_path']['exists']
        runs.append(dict(id=runid,arm=arm,identity=live,physical_gpu=index,gpu_uuid=uuid,observed_gpu_uuids=observed['observed_gpu_uuids'],environment=env,gpu_lock=locks,worker_lock=worker,launch_receipt_sha256=sha(attempt/'LAUNCH_RECEIPT.json'),admission_sha256=sha(attempt/'GPU_ADMISSION.xml'),status=observed['files'].get('status_path'),checkpoint_file_metadata_only=observed['checkpoints'],log_metadata=observed['files']['log_path']))
    others=[x for x in obs['runs'] if x['id'] not in args.run_id];assert all(x['process_identity'] is None and not x['observed_gpu_uuids'] for x in others)
    assert dict(collections.Counter(x['state'] for x in others))=={'COMPLETE_MARKER_PRESENT':24,'FAILURE_MARKER_REVIEW_REQUIRED':2}
    prior=json.loads((ROOT/'monitoring/CHECK_20261001T200037Z.json').read_text())
    oldfail={x['id']:x['files']['failure_path'].get('content') for x in prior['runs'] if x['state']=='FAILURE_MARKER_REVIEW_REQUIRED'}
    for x in others:
        if x['id'] in oldfail:assert x['files']['failure_path'].get('content')==oldfail[x['id']]
    assert not obs['interventions'];assert all(x.get('run_id') in args.run_id and x['state']=='RUNNING_IDENTITY_MATCH' for x in obs['incidents'])
    cronsha=hashlib.sha256(subprocess.run(['crontab','-l'],capture_output=True,check=True).stdout).hexdigest();assert cronsha=='13caacb74408db396e9eae0d0929aec97d102c41c00c2f5b138da9d0417b330e'
    receipt=dict(passed=True,checked_at_utc=obs['checked_at_utc'],receipt_at_utc=stamp(),scheduled_wakeup='2026-10-02T00:01Z',canonical_record=name,canonical_sha256=sha(record),registry_sha256=obs['registry_sha256'],registered_runs=29,active_owned=3,runs=runs,other26_workers_absent=True,historical_failures_unchanged=True,incidents=obs['incidents'],incident_interpretation='Newly observed authorized startup transitions; no new failure',gpu_snapshot=obs['gpu_snapshot'],authorization_sha256=AUTH,frozen_manifest_sha256=SOURCE,publication_receipt_sha256=PUBLICATION,cron_sha256=cronsha,cron_unchanged=True,lock_note='Inherited GPU lock remains held by child fd; fdinfo may report launcher PID, so lock owner PID is not required to equal child PID.',checkpoint_or_dataset_decoding=False,scientific_execution_by_monitor=False,interventions=[])
    target=ROOT/'monitoring/ROUND07_STARTUP_CONFIRMATION_20261002T0001.json';assert not target.exists();atomic_json(target,receipt)
    print(json.dumps(dict(path=str(target),sha256=sha(target),canonical_record=name,canonical_sha256=sha(record),active_owned=3,runs=[dict(arm=x['arm'],pid=x['identity']['pid'],gpu=x['physical_gpu'],status=x['status']) for x in runs])))
if __name__=='__main__':main()
