"""Submit all stages held, wire afterok, persist registry, then release."""
import datetime,json,subprocess
from pathlib import Path
from protocol import verify
ROOT=Path(__file__).resolve().parents[1]

def main():
    frozen=verify()
    registry=ROOT/'jobs/submission.json'
    assert not registry.exists(),'Already submitted; inspect registry before changing jobs'
    assert json.loads((ROOT/'results/model_tests.json').read_text())['passed']
    assert json.loads((ROOT/'results/data_checks.json').read_text())['passed']
    assert json.loads((ROOT/'results/workflow_tests.json').read_text())['passed']
    jobs={};previous=None
    try:
        for scale,wall in [('1k','12:00:00'),('10k','2-00:00:00'),('full','7-00:00:00')]:
            cmd=['sbatch','--hold','--parsable',f'--job-name=MTOvT_{scale}',f'--time={wall}']
            if previous:cmd.append(f'--dependency=afterok:{previous}')
            cmd.extend([str(ROOT/'scripts/stage.slurm'),scale])
            job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
            assert job.isdigit(),job
            jobs[scale]={'job_id':job,'afterok':previous,'includes':['all paired training','test evaluation','tensor diagnostics','reports'],
                         'gpus':2,'time_limit':wall}
            previous=job
        payload={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'remote_root':str(ROOT),'protocol_id':frozen['protocol_id'],'stages':jobs,'released':False}
        registry.write_text(json.dumps(payload,indent=2))
    except Exception:
        for job in jobs.values():subprocess.run(['scancel',job['job_id']],check=False)
        raise
    for job in jobs.values():subprocess.run(['scontrol','release',job['job_id']],check=True)
    payload['released']=True;registry.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
