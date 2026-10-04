"""Scheduled registry observation plus explicit Round09 startup verification; no scientific imports."""
import argparse,collections,hashlib,json,os,subprocess
from pathlib import Path
from monitor import ROOT,atomic_json,file_info,process_identity,stamp
AUTH='fc16c206e829c345098fcd853febcd2f83c88a56c3ea92f5506c0edd67678765'
SOURCE='52e815958178f2eeb691a2f185eb37e4b54276d867c08061c58ef1eff653c930'
PUBLICATION='440d25926c543a7a03089f8515135a124f56851ecf84e3464b20cd07561efd79'
ASSIGN={'zero_decay':(1,'GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800'),'coupled_l2':(2,'GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa')}
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
    p=argparse.ArgumentParser();p.add_argument('--run-id',action='append',required=True);args=p.parse_args();assert len(set(args.run_id))==2
    rd=ROOT/'round09_regularization_preparation';assert sha(rd/'PRODUCTION_EXECUTION_AUTHORIZATION.json')==AUTH
    assert sha(rd/'FROZEN_MANIFEST.json')==SOURCE
    assert sha(ROOT/'ops/ROUND09_PREPARATION_PUBLICATION_RECEIPT.json')==PUBLICATION
    result=subprocess.run(['/usr/bin/python3',str(ROOT/'ops/monitor.py'),'--root',str(ROOT)],capture_output=True,text=True,check=True)
    name=json.loads(result.stdout)['record'];record=ROOT/'monitoring'/name;obs=json.loads(record.read_text())
    reg=json.loads((ROOT/'ops/registry.json').read_text());assert len(obs['runs'])==36
    gpu={x['index']:x for x in obs['gpu_snapshot']['gpus']};runs=[]
    for runid in args.run_id:
        registered=next(x for x in reg['runs'] if x['id']==runid);observed=next(x for x in obs['runs'] if x['id']==runid)
        arm=Path(registered['run_dir']).name;index,uuid=ASSIGN[arm];pid=registered['identity']['pid']
        live=process_identity(pid);keys=('pid','start_ticks','uid','cwd','argv_sha256','boot_id')
        assert live and all(live[k]==registered['identity'][k] for k in keys)
        assert observed['state']=='RUNNING_IDENTITY_MATCH' and observed['observed_gpu_uuids']==[uuid]
        assert registered['gpu_uuid']==uuid and live['cwd']==str(rd)
        attempt=rd/'runs'/arm/'attempts'/runid;launch=json.loads((attempt/'LAUNCH_RECEIPT.json').read_text())
        assert all(launch['identity'][k]==registered['identity'][k] for k in keys) and launch['registration_returncode']==0
        assert launch['registration_barrier_released'] is True
        with (attempt/'train.log').open() as stream:barrier_line=stream.readline(4097)
        assert len(barrier_line)<=4096
        assert json.loads(barrier_line)=={'registration_barrier_passed':{'registered':True,'pid':pid,'binding_sha256':AUTH}}
        assert Path(launch['command'][1])==rd/'registered_entry.py' and Path(launch['command'][2])==rd/'train.py'
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
        runs.append(dict(id=runid,arm=arm,identity=live,physical_gpu=index,gpu_uuid=uuid,observed_gpu_uuids=observed['observed_gpu_uuids'],environment=env,registration_barrier_released=True,barrier_log_line_sha256=hashlib.sha256(barrier_line.encode()).hexdigest(),stable_identity_fields=list(keys),gpu_lock=locks,worker_lock=worker,launch_receipt_sha256=sha(attempt/'LAUNCH_RECEIPT.json'),admission_sha256=sha(attempt/'GPU_ADMISSION.xml'),status=observed['files'].get('status_path'),checkpoint_file_metadata_only=observed['checkpoints'],log_metadata=observed['files']['log_path']))
    others=[x for x in obs['runs'] if x['id'] not in args.run_id];assert all(x['process_identity'] is None and not x['observed_gpu_uuids'] for x in others)
    assert dict(collections.Counter(x['state'] for x in others))=={'COMPLETE_MARKER_PRESENT':32,'FAILURE_MARKER_REVIEW_REQUIRED':2}
    prior=json.loads((ROOT/'monitoring/CHECK_20261003T201107Z.json').read_text())
    oldfail={x['id']:x['files']['failure_path'].get('content') for x in prior['runs'] if x['state']=='FAILURE_MARKER_REVIEW_REQUIRED'}
    for x in others:
        if x['id'] in oldfail:assert x['files']['failure_path'].get('content')==oldfail[x['id']]
    assert not obs['interventions'];assert all(x.get('run_id') in args.run_id and x['state']=='RUNNING_IDENTITY_MATCH' for x in obs['incidents'])
    cronsha=hashlib.sha256(subprocess.run(['crontab','-l'],capture_output=True,check=True).stdout).hexdigest();assert cronsha=='13caacb74408db396e9eae0d0929aec97d102c41c00c2f5b138da9d0417b330e'
    receipt=dict(passed=True,checked_at_utc=obs['checked_at_utc'],receipt_at_utc=stamp(),scheduled_wakeup='2026-10-04T00:09Z',canonical_record=name,canonical_sha256=sha(record),registry_sha256=obs['registry_sha256'],registered_runs=36,active_owned=2,runs=runs,other34_workers_absent=True,historical_failures_unchanged=True,incidents=obs['incidents'],incident_interpretation='Newly observed authorized startup transitions; no new failure',gpu_snapshot=obs['gpu_snapshot'],authorization_sha256=AUTH,frozen_manifest_sha256=SOURCE,publication_receipt_sha256=PUBLICATION,cron_sha256=cronsha,cron_unchanged=True,lock_note='Inherited GPU lock remains held by child fd; fdinfo may report launcher PID, so lock owner PID is not required to equal child PID.',checkpoint_or_dataset_decoding=False,scientific_execution_by_monitor=False,interventions=[])
    target=ROOT/'monitoring/ROUND09_STARTUP_CONFIRMATION_20261004T0009.json';assert not target.exists();atomic_json(target,receipt)
    print(json.dumps(dict(path=str(target),sha256=sha(target),canonical_record=name,canonical_sha256=sha(record),active_owned=2,runs=[dict(arm=x['arm'],pid=x['identity']['pid'],gpu=x['physical_gpu'],status=x['status']) for x in runs])))
if __name__=='__main__':main()
