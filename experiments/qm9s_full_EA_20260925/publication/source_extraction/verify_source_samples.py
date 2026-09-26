"""Independent line-based spot-checks against source logs, without importing parser."""
import gzip
import hashlib
import json
import pathlib
import re
import sys

root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'dataset_manifest.json').read_text(encoding='utf-8'))
source=pathlib.Path(manifest['source_directory'])
checks=[]
for shard in manifest['shards']:
    with gzip.open(root/'records'/(shard['name']+'.jsonl.gz'),'rt',encoding='utf-8') as f:
        records=[json.loads(line) for line in f]
    indices={0,len(records)//2,len(records)-1}
    failures=[i for i,r in enumerate(records) if not r.get('normal_termination',False)]
    if failures:indices.add(failures[0])
    for i in sorted(indices):
        r=records[i];raw=(source/r['source_file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==r['source_sha256']
        lines=raw.decode('utf-8',errors='replace').splitlines()
        states=[];dip={};coords=[];atoms=[]
        for n,line in enumerate(lines):
            if re.match(r'^\s*Excited State\s+\d+:',line):
                p=line.split()
                states.append((int(p[2][:-1]),float(p[4]),float(p[6]),float(p[8].split('=')[1])))
            if 'Ground to excited state transition electric dipole moments (Au):' in line:
                for row in lines[n+2:]:
                    p=row.split()
                    if len(p)!=6 or not p[0].isdigit():break
                    dip[int(p[0])]=[float(v) for v in p[1:4]]
            if 'Input orientation:' in line:
                aa=[];xx=[]
                for row in lines[n+5:]:
                    p=row.split()
                    if len(p)!=6 or not p[0].isdigit():break
                    aa.append(int(p[1]));xx.append([float(v) for v in p[3:6]])
                atoms=aa;coords=xx
        assert len(states)==len(r['states'])
        for observed,stored in zip(states,r['states']):
            assert observed==(stored['state_index'],stored['energy_eV'],stored['wavelength_nm'],stored['oscillator_strength'])
            if stored['state_index'] in dip:
                assert stored['transition_dipole_table_au']==dip[stored['state_index']]
        if r.get('coordinate_frame')=='Input':
            assert atoms==r['atomic_numbers']
            assert coords==r['positions_angstrom']
        checks.append(dict(molecule_id=r['molecule_id'],source_hash_verified=True,labels_verified=len(states),geometry_verified=r.get('coordinate_frame')=='Input'))
report=dict(checked_source_files=len(checks),checked_source_states=sum(c['labels_verified'] for c in checks),method='Independent line-based reads; evenly distributed shard samples plus failed records',samples=checks)
(root/'source_spotcheck_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
