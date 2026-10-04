"""Stdlib-only registration barrier before importing the scientific child."""
import json,os,select,stat,sys,time
from pathlib import Path


def wait_registered(timeout_seconds=90.):
    raw=os.environ.pop('MTO_REGISTRATION_FD')
    expected=os.environ.pop('MTO_REGISTRATION_BINDING')
    assert len(expected)==64 and all(c in '0123456789abcdef' for c in expected)
    fd=int(raw);assert fd>=3 and stat.S_ISFIFO(os.fstat(fd).st_mode)
    deadline=time.monotonic()+timeout_seconds;payload=b''
    try:
        while b'\n' not in payload:
            remaining=deadline-time.monotonic()
            assert remaining>0,'Registration barrier timed out before scientific imports'
            ready,_,_=select.select([fd],[],[],remaining)
            assert ready,'Registration barrier timed out before scientific imports'
            chunk=os.read(fd,4096)
            assert chunk,'Registration failed or parent exited before release'
            payload+=chunk;assert len(payload)<=4096
    finally:os.close(fd)
    value=json.loads(payload)
    assert value=={'registered':True,'pid':os.getpid(),'binding_sha256':expected}
    return value


def main():
    assert len(sys.argv)>=2
    target=Path(sys.argv[1]).resolve();root=Path(__file__).resolve().parent
    assert target.parent==root and target.name in ('gpu_preflight.py','train.py')
    receipt=wait_registered()
    print(json.dumps({'registration_barrier_passed':receipt}),flush=True)
    # run_path stays in this registered PID; /proc argv/identity never changes.
    import runpy
    sys.argv=sys.argv[1:]
    runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__':main()
