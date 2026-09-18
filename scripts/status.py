import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
execution=json.loads((ROOT/'configs/execution.json').read_text())
for scale,seeds in execution['seeds_by_scale'].items():
    done=[s for s in seeds if (ROOT/'runs'/f'{scale}_seed{s}'/'DONE').exists()]
    print(scale,'pairs',f'{len(done)}/{len(seeds)}','stage_complete',(ROOT/'results'/scale/'STAGE_COMPLETE.json').exists())
print('preflight',(ROOT/'results/preflight.json').exists())
registry=ROOT/'jobs/submission.json'
if registry.exists():
    r=json.loads(registry.read_text())
    subprocess.run(['squeue','-j',','.join(j['job_id'] for j in r['stages'].values()),'-o','%i %j %T %R'])
