"""Launch the unchanged one-time evaluator with a UUID mask before imports."""
import datetime
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python'
UUID='GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def save(record,path):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');os.replace(tmp,path)

def main():
    assert not (ROOT/'affine/VALIDATION_COMPLETE.json').exists(),'Completed validation must not repeat'
    manifest=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    for path,value in manifest['source_hashes'].items():assert sha(path)==value
    gate=json.loads((ROOT/'TERMINAL_SOURCE_GATE.json').read_text());assert gate['passed'] and gate['source_process_exited']
    pinned={str(ROOT/'affine'/name):sha(ROOT/'affine'/name) for name in ('COEFFICIENTS_FROZEN.json','EXPORT_COMPLETE.json','in_sample.pt','heldout_source.pt')}
    attempt=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt_path=ROOT/('VALIDATION_LAUNCH_'+attempt+'.json');log_path=ROOT/('AFFINE_VALIDATION_EXECUTION_'+attempt+'.log')
    command=[PYTHON,str(ROOT/'affine_stage.py'),'evaluate','--gpu','1']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,PYTHONUNBUFFERED='1')
    record={'attempt':attempt,'command':command,'cwd':str(ROOT),'environment_overrides':{'CUDA_VISIBLE_DEVICES':UUID,'PYTHONUNBUFFERED':'1'},
        'intended_physical_gpu':1,'intended_visible_logical_device':0,'uuid_mask_set_before_interpreter':True,
        'launch_source_sha256':sha(__file__),'manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json'),
        'pinned_affine_artifact_hashes':pinned,'log_path':str(log_path),'started_unix':time.time(),
        'gpu_observations':[],'reviewed_scientific_command_unchanged':True}
    with log_path.open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record['pid']=process.pid;record['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        stat=Path('/proc/%d/stat'%process.pid).read_text();record['start_ticks']=int(stat.rsplit(')',1)[1].split()[19])
        record['uid']=Path('/proc/%d'%process.pid).stat().st_uid
        save(record,receipt_path)
        seen=set()
        while process.poll() is None:
            query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if query.returncode==0:
                for line in query.stdout.splitlines():
                    fields=[x.strip() for x in line.split(',')]
                    if len(fields)==3 and fields[0]==str(process.pid):
                        key=tuple(fields)
                        if key not in seen:
                            seen.add(key);record['gpu_observations'].append({'unix':time.time(),'pid':process.pid,'gpu_uuid':fields[1],'memory_MiB':fields[2]})
            proc=Path('/proc/%d/cmdline'%process.pid)
            if proc.exists():
                try:
                    argv=proc.read_bytes()
                    if b'affine_stage.py' in argv:record['observed_argv_sha256']=hashlib.sha256(argv).hexdigest()
                except FileNotFoundError:pass
            save(record,receipt_path);time.sleep(.5)
        record['exit_code']=process.returncode
    record['completed_unix']=time.time()
    record['observed_gpu_uuids']=sorted({x['gpu_uuid'] for x in record['gpu_observations']})
    record['actual_gpu1_placement_observed']=bool(record['gpu_observations']) and record['observed_gpu_uuids']==[UUID]
    record['pinned_affine_artifacts_unchanged']=all(sha(path)==value for path,value in pinned.items())
    if (ROOT/'affine/VALIDATION_COMPLETE.json').exists():record['validation_receipt_sha256']=sha(ROOT/'affine/VALIDATION_COMPLETE.json')
    save(record,receipt_path)
    print(json.dumps({k:v for k,v in record.items() if k!='gpu_observations'},indent=2))
    assert process.returncode==0 and record['pinned_affine_artifacts_unchanged']
    assert all(x['gpu_uuid']==UUID for x in record['gpu_observations']),'Observed resource-contract mismatch; retain outputs and report'

if __name__=='__main__':main()
