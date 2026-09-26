import json,pathlib,shutil,time
ROOT=pathlib.Path(__file__).resolve().parent
shutil.copy2(ROOT/'frozen_reference/data/hashes.json',ROOT/'data/hashes.json')
for name in ('mto_eta0','mto_eta01','mto_eta1'):
    out=ROOT/'runs'/name
    assert not (out/'last.pt').exists(), 'Unexpected training checkpoint: do not overwrite'
    error=json.loads((out/'FAILED.json').read_text())
    assert error['type']=='FileNotFoundError' and 'data/hashes.json' in error['error']
    (out/'FAILED.json').rename(out/'FAILED_attempt_1.json')
    if (out/'PROCESS_EXIT.json').exists():(out/'PROCESS_EXIT.json').rename(out/'PROCESS_EXIT_attempt_1.json')
(ROOT/'reports/launch_repair.json').write_text(json.dumps(dict(time=time.time(),failure='Missing copied data/hashes.json in the new directory; all three workers failed before optimizer updates',repair='Copied the verified frozen hash manifest; preserved failure records and full append-only logs; relaunch same configurations and seeds'),indent=2))
