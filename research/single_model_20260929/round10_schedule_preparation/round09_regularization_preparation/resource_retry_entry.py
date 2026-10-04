"""Distinct resource-only entry; original scientific child/barrier are unchanged."""
import os,sys
from pathlib import Path
from common import ROOT,read,sha

def main():
    assert len(sys.argv)==2 and Path(sys.argv[1]).resolve()==ROOT/'gpu_preflight.py'
    review=read(ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json')
    assert review['passed'] and review['scope']=='round09_resource_retry01_6_discarded_updates'
    assert os.environ['MTO_REGISTRATION_BINDING']==sha(ROOT/'RESOURCE_RETRY_SOURCE_REVIEW.json')
    assert sha(ROOT/'ROUND09_RESOURCE_RETRY_DECISION.md')==review['root_resource_retry_decision_sha256']=='673bcc503705fe6063e14f02d3573738d82a12907f18e730d41d760c7ae277e8'
    assert sha(ROOT/'TECHNICAL_SOURCE_REVIEW.json')==review['original_technical_review_sha256']=='5c4b82c015ec2b5abc475fbaa846757f7eb5b3fc9e0bc00991bacef0175ed457'
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
    selection=read(ROOT/'RESOURCE_RETRY_SELECTION.json')
    assert os.environ['CUDA_VISIBLE_DEVICES']==selection['selected_gpu_uuid']
    assert os.environ['OMP_NUM_THREADS']==os.environ['MKL_NUM_THREADS']=='2'
    assert not (ROOT/'GPU_PREFLIGHT.json').exists() and not (ROOT/'private_preflight').exists()
    # The reviewed original stdlib barrier releases the same registered PID only
    # after exact registry success, then runpy enters unchanged gpu_preflight.py.
    from registered_entry import main as enter_original
    enter_original()

if __name__=='__main__':main()
