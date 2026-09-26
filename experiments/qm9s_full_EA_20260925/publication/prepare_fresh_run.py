"""Copy frozen code/configs and restored data to a NEW run directory. Does not train."""
import argparse,pathlib,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',required=True);args=p.parse_args();dest=pathlib.Path(args.destination).resolve()
    if dest.exists():raise SystemExit('Destination must not exist; archived runs will never be overwritten')
    source=ROOT/'data/dataset.npz'
    if not source.exists():raise SystemExit('Run publication/verify_archive.py --restore-data first')
    dest.mkdir(parents=True)
    for name in ['train_ea.py','models_ea.py','data_ea.py','preflight.py','analyze.py','campaign.json','analyze_final_independent.py']:
        shutil.copy2(ROOT/name,dest/name)
    shutil.copytree(ROOT/'upstream',dest/'upstream',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'configs',dest/'configs')
    (dest/'data').mkdir()
    for name in ['dataset.npz','normalization.json','splits.json','hashes.json','identity_audit_v2.json']:shutil.copy2(ROOT/'data'/name,dest/'data'/name)
    for name in ['runs','logs','reports']:(dest/name).mkdir()
    print('Prepared fresh workspace:',dest)
    print('Use a compatible environment, then: CUDA_VISIBLE_DEVICES=<healthy GPU> python train_ea.py <config name>')
    print('All runs should finish before test evaluation. Do not reuse the historical incomplete-fit exception.')
if __name__=='__main__':main()
