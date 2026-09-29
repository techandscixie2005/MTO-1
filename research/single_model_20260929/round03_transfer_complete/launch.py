"""Admit and launch owned factorial arms without touching unrelated processes."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from freeze import sha,current_hashes

ROOT=Path(__file__).resolve().parent
PYTHON='/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python'
EXPECTED={1:'GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800',
          2:'GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa',
          4:'GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184',
          6:'GPU-2431a641-8045-a10b-4aa0-2769010e0708'}

def admit(index):
    assert index in EXPECTED
    xml=subprocess.check_output(['nvidia-smi','-q','-x','-i',str(index)],text=True)
    gpu=ET.fromstring(xml).find('gpu')
    assert gpu.findtext('uuid')==EXPECTED[index]
    memory=int(gpu.findtext('fb_memory_usage/used').split()[0])
    assert memory<1000,('GPU memory occupied',index,memory)
    for node in gpu.findall('processes/process_info'):
        assert 'C' not in node.findtext('type',''),('Compute process present',index,node.findtext('pid'))
    for period in ('volatile','aggregate'):
        for node in gpu.findall('ecc_errors/'+period+'/*'):
            if 'uncorrectable' in node.tag:
                assert node.text=='0',('Uncorrectable ECC',index,node.tag,node.text)
    for tag in ('remapped_row_pending','remapped_row_failure'):
        assert gpu.findtext('remapped_rows/'+tag)=='No',(index,tag)
    for tag in ('channel_repair_pending','tpc_repair_pending'):
        assert gpu.findtext('ecc_errors/'+tag) in ('No','N/A'),(index,tag)
    return xml

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arms',nargs='+',default=['control','adapter','decorrelation','both'])
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--publication-receipt',type=Path,required=True)
    args=p.parse_args()
    assert args.publication_receipt.is_file(),'Lightweight source archive/push receipt required'
    review=json.loads(args.review.read_text())
    assert review['passed'],'Independent implementation review must pass'
    for path,expected in review['source_hashes'].items():
        assert sha(path)==expected,('Changed after independent review',path)
    cfg=json.loads((ROOT/'round_config.json').read_text())
    frozen=json.loads((ROOT/'FROZEN_MANIFEST.json').read_text())
    assert current_hashes(cfg)==frozen['source_hashes']
    publication=json.loads(args.publication_receipt.read_text())
    assert publication['download_before_stage_before_commit_push'] is True
    assert publication['remote_verified'] is True
    assert publication['frozen_manifest_sha256']==sha(ROOT/'FROZEN_MANIFEST.json')
    launch_lock=(ROOT/'launch.lock').open('w')
    fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipts=[]
    for arm in args.arms:
        assert arm in cfg['arms']
        gpu=cfg['arms'][arm]['gpu']
        out=ROOT/'runs'/arm;out.mkdir(parents=True,exist_ok=True)
        assert not (out/'FIT_COMPLETE.json').exists(),('Arm already complete',arm)
        glock=open('/tmp/mto_pouter_gpu_%d.lock'%gpu,'w')
        fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        xml=admit(gpu)
        attempt='%s_%d'%(arm,time.time_ns())
        (out/(attempt+'_gpu_admission.xml')).write_text(xml)
        if (out/'FAILED.json').exists():
            os.replace(out/'FAILED.json',out/(attempt+'_previous_failure.json'))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=EXPECTED[gpu],PYTHONUNBUFFERED='1',
                 PYTHONWARNINGS='ignore::FutureWarning,ignore::UserWarning',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        log=(out/'train.log').open('a')
        process=subprocess.Popen([PYTHON,str(ROOT/'train.py'),'--arm',arm],cwd=ROOT,
            env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(glock.fileno(),),start_new_session=True)
        receipt={'arm':arm,'attempt':attempt,'pid':process.pid,'gpu':gpu,'gpu_uuid':EXPECTED[gpu],
            'manifest_sha256':sha(ROOT/'FROZEN_MANIFEST.json'),'review_sha256':sha(args.review),
            'publication_receipt':str(args.publication_receipt),'publication_receipt_sha256':sha(args.publication_receipt),
            'launched_at_unix':time.time(),'command':[PYTHON,str(ROOT/'train.py'),'--arm',arm]}
        (out/'LAUNCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
        register=['/usr/bin/python3',str(ROOT/'ops/register_run.py'),'--id',attempt,'--pid',str(process.pid),
            '--run-dir','runs/'+arm,'--gpu-uuid',EXPECTED[gpu]]
        for flag,name in (('status','status.json'),('log','train.log'),('completion','FIT_COMPLETE.json'),('failure','FAILED.json')):
            register+=['--'+flag+'-path','runs/'+arm+'/'+name]
        for name in ('last.pt','best.pt'):
            register+=['--checkpoint-path','runs/'+arm+'/'+name]
        registration=subprocess.run(register,text=True,capture_output=True)
        receipt['monitor_registration_returncode']=registration.returncode
        (out/'LAUNCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if registration.returncode:
            # Do not signal the process: its durable outputs and exact receipt
            # allow the coordinator to repair registration without losing work.
            raise RuntimeError('Registration failed for owned PID %d: %s'%(process.pid,registration.stderr))
        receipts.append(receipt)
        glock.close();log.close()
    print(json.dumps(receipts,indent=2))

if __name__=='__main__':
    main()
