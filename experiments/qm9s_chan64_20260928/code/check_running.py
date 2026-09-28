import json,pathlib,time
import torch
from dataset import ROOT
from trainer import atomic_json,state_hash,fingerprint
from supervisor import alive,health
def main():
    expected=json.loads((ROOT/'reports/initialization.json').read_text())['groups'];orders=json.loads((ROOT/'reports/epoch_order_hashes.json').read_text())
    rows={};fp=fingerprint();h,b,xml=health()
    for n in ('G1','G2','G3','G4'):
        out=ROOT/'runs'/n;receipt=json.loads((out/'launch_receipt.json').read_text());manifest=json.loads((out/'run_manifest.json').read_text())
        assert alive(receipt['pid'],'trainer.py',n) and h[receipt['gpu']]['clean']
        assert receipt['resume'] is False and manifest['initial_state_sha256']==expected[n]
        assert not (out/'FAILED.json').exists()
        cfg=json.loads((ROOT/'configs'/f'{n}.json').read_text());assert cfg==manifest['config']
        order=json.loads((out/'current_order.json').read_text());assert order['sha256']==orders[order['epoch']-1]['sha256']
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert ck['fingerprint']==fp and ck['config']==cfg
        if ck['state']['steps']==0:assert state_hash(ck['model'])==expected[n]
        if n in ('G2','G4'):
            initial=torch.load(ROOT/'initial'/('original.pt' if n=='G2' else 'p_outer.pt'),weights_only=True)
            for k in ('decoder.energy_head.weight','decoder.energy_head.bias','decoder.energy_offset'):assert torch.equal(initial[k],ck['model'][k])
        status=json.loads((out/'status.json').read_text());assert status.get('steps',0)>0
        rows[n]=dict(state='RUNNING',pid=receipt['pid'],gpu=receipt['gpu'],gpu_uuid=receipt['gpu_uuid'],status=status,
            last_checkpoint_epoch=ck['state']['epoch'],last_checkpoint_cursor=ck['state']['cursor'],last_checkpoint_steps=ck['state']['steps'],
            checkpoint_age_seconds=time.time()-(out/'last.pt').stat().st_mtime,initial_state_sha256=expected[n],order=order,
            log=str(ROOT/'logs'/f'{n}.log'),best_exists=(out/'best.pt').exists())
    workers=[]
    for proc in pathlib.Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                if (proc/'cwd').resolve()==ROOT:
                    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                    if 'trainer.py' in cmd:workers.append(dict(pid=int(proc.name),cmd=cmd))
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert len(workers)==4
    assert not (ROOT/'reports/test_evaluation_started.json').exists()
    report=dict(time=time.time(),passed=True,formal_worker_count=4,runs=rows,workers=workers,supervisor_pid=json.loads((ROOT/'supervisor_receipt.json').read_text())['pid'],test_not_started=True)
    atomic_json(report,ROOT/'reports/running_checks.json');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
