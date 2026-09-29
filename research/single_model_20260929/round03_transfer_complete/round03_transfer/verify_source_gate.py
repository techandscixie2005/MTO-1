"""Read-only terminal source/hash/process gate before the frozen affine stage."""
import json
import os
import time
from runtime import ROOT,sha,atomic_json,verify_manifest

def main():
    manifest,mh=verify_manifest();rd=ROOT/'runs/source33'
    assert mh=='93674b785f0f93fb2671fca83941e9af2961ab665c3814a027791fbb3109ed37'
    review=json.loads((ROOT/'INDEPENDENT_PRELAUNCH_REVIEW.json').read_text())
    publication=json.loads((ROOT.parent/'ops/ROUND03_PUBLICATION_RECEIPT.json').read_text())
    assert review['passed'] and review['frozen_manifest_sha256']==mh and review['source_hashes']==manifest['source_hashes']
    assert publication['remote_verified'] and publication['download_before_stage_before_commit_push']
    assert publication['frozen_manifest_sha256']==mh and publication['independent_review_sha256']==sha(ROOT/'INDEPENDENT_PRELAUNCH_REVIEW.json')
    complete=json.loads((rd/'FIT_COMPLETE.json').read_text());launch=json.loads((rd/'LAUNCH_RECEIPT.json').read_text())
    assert launch['manifest_sha256']==mh
    assert launch['review_sha256']==sha(ROOT/'INDEPENDENT_PRELAUNCH_REVIEW.json')
    assert launch['publication_receipt_sha256']==sha(ROOT.parent/'ops/ROUND03_PUBLICATION_RECEIPT.json')
    assert complete['completed_epoch']==33 and complete['manifest_sha256']==mh and complete['steps']==49665
    assert not (rd/'FAILED.json').exists()
    assert sha(complete['checkpoint'])==complete['checkpoint_sha256']
    assert sha(rd/'last.pt')==complete['last_checkpoint_sha256']
    proc='/proc/%d/stat'%launch['pid'];state='absent'
    if os.path.exists(proc):state=open(proc).read().rsplit(')',1)[1].split()[0]
    assert state in ('absent','Z'),('Source process remains active',launch['pid'],state)
    result={'passed':True,'manifest_sha256':mh,'all_frozen_hashes_match':True,'completed_epoch':33,
        'steps':49665,'source_pid':launch['pid'],'source_process_state':state,'source_process_exited':True,
        'OS_exit_code_observed':False,
        'source_final_sha256':complete['checkpoint_sha256'],'last_checkpoint_sha256':complete['last_checkpoint_sha256'],
        'review_sha256':sha(ROOT/'INDEPENDENT_PRELAUNCH_REVIEW.json'),
        'publication_receipt_sha256':sha(ROOT.parent/'ops/ROUND03_PUBLICATION_RECEIPT.json'),
        'checked_unix':time.time(),'source_hashes':{str(ROOT/'verify_source_gate.py'):sha(ROOT/'verify_source_gate.py')},
        'GPU_admission_and_lock_deferred_to_each_reviewed_stage_command':True}
    atomic_json(result,ROOT/'TERMINAL_SOURCE_GATE.json');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
