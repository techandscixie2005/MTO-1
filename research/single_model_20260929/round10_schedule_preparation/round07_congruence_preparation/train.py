"""Reviewed production entry: authorization first, then one resumable arm."""
import argparse,os

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['original','scalar','tensor'])
    p.add_argument('--authorization',required=True);args=p.parse_args()
    from execution_gate import verify
    permit=verify(args.authorization)
    assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
    assert os.environ.get('OMP_NUM_THREADS')=='2' and os.environ.get('MKL_NUM_THREADS')=='2'
    # No target/model/torch import occurs until the distinct production gate.
    from training import run
    run(args.arm,permit)
if __name__=='__main__':main()
