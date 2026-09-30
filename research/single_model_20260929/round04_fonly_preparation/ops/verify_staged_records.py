#!/usr/bin/env python3
"""Verify every staged new blob against downloaded bytes before publication."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--archive',type=Path,required=True)
a=p.parse_args()
root=a.archive.resolve()
assert root.is_relative_to(Path('D:/MTO/archives/single_model_20260929').resolve())
review=json.loads((root/'STAGED_REVIEW.json').read_text())
env=os.environ.copy()
env['GIT_INDEX_FILE']=str(root/'.publication-index')
env['GIT_NO_LAZY_FETCH']='1'
repo='D:/MTO/publication/MTO-1'
def git(*args):
    return subprocess.run(['git','-C',repo,*args],capture_output=True,check=True,env=env,timeout=120).stdout
secrets=re.compile(rb'(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
for f in review['files']:
    data=(root/f['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==f['sha256'],f['path']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+bytes([0])+data).hexdigest()==f['blob'],f['path']
    assert git('cat-file','blob',f['blob'])==data,f['path']
    assert bytes([0]) not in data and not secrets.search(data),f['path']
    data.decode('utf-8-sig')
    assert f['git_path'].startswith('research/single_model_20260929/'),f['git_path']
changes=git('diff','--cached','--name-status',review['parent']).decode().splitlines()
assert all(line.startswith('A\t') for line in changes),changes
assert {line.split('\t',1)[1] for line in changes}=={f['git_path'] for f in review['files']}
assert git('write-tree','--missing-ok').decode().strip()==review['tree']
result={'reviewed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'passed':True,
    'new_blobs_present_and_byte_verified':len(review['files']),
    'all_changes_allowlisted_additions':True,'binary_and_credential_patterns_absent':True,
    'new_files_max_bytes':max(f['bytes'] for f in review['files']),
    'total_bytes':sum(f['bytes'] for f in review['files']),
    'tree':review['tree'],'parent':review['parent'],
    'staged_diff_sha256':hashlib.sha256((root/'STAGED_DIFF.patch').read_bytes()).hexdigest()}
destination=root/'CONTENT_REVIEW.json'
if destination.exists():
    destination=root/('CONTENT_REVIEW_RECHECK_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
destination.write_text(json.dumps(result,indent=2)+'\n')
result['review_record']=str(destination)
print(json.dumps(result,indent=2))
