import hashlib,json,os,pathlib,time
from supervisor import gpu_health,busy_uuids
ROOT=pathlib.Path(__file__).resolve().parent
def main():
    nxt=ROOT/'campaign.next.json';campaign=json.loads(nxt.read_text())
    assert json.loads((ROOT/'reports/detanet_confirmation.json').read_text())['status']=='approved'
    assert json.loads((ROOT/'reports/detanet_training_checks.json').read_text())['passed']
    assert len(campaign['runs'])==4 and len(set(s['name'] for s in campaign['runs']))==4
    cfg=json.loads((ROOT/'configs/detanet_ef.json').read_text());gpu=cfg['gpu'];health=gpu_health()[gpu]
    assert not health['volatile'] and not health['aggregate'] and health['memory_MiB']<200 and health['uuid'] not in busy_uuids()
    assert not (ROOT/'runs/detanet_ef/last.pt').exists()
    for s in campaign['runs']:
        assert (ROOT/s['trainer']).exists()
        assert json.loads((ROOT/'configs'/f"{s['name']}.json").read_text())['gpu']==s['gpu']
    report=dict(time=time.time(),gpu=health,configuration_sha256=hashlib.sha256((ROOT/'configs/detanet_ef.json').read_bytes()).hexdigest(),
        reason_for_gpu_6='GPU5 recorded one corrected DRAM ECC during preflight, no uncorrectable errors; use idle GPU6 with zero ECC history for the formal run',
        interpretation='Paper settings take precedence; see detanet_protocol.md',user_confirmation='detanet_confirmation.json')
    (ROOT/'reports/detanet_activation.json').write_text(json.dumps(report,indent=2))
    os.replace(nxt,ROOT/'campaign.json')
    print('Fourth group activated; existing supervisor will start it when GPU6 is idle.')
if __name__=='__main__':main()
