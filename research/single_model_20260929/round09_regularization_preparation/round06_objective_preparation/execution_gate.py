"""Stdlib-only exact production permission gate, before data/model imports."""
from pathlib import Path
from common import ROOT,PINS,read,sha

def verify(authorization):
    authorization=Path(authorization)
    assert authorization.is_file(),'Distinct root production authorization is absent'
    auth=read(authorization)
    assert auth['authorized'] is True and auth['scope']=='round06_two_arm_60epoch_fit'
    manifest_path=ROOT/'FROZEN_MANIFEST.json';mh=sha(manifest_path);manifest=read(manifest_path)
    assert auth['frozen_manifest_sha256']==mh
    assert auth['split_manifest_sha256']==PINS['SPLIT_MANIFEST.json']
    assert auth['independent_split_verification_sha256']==PINS['INDEPENDENT_SPLIT_VERIFICATION.json']
    assert auth['epochs']==60 and auth['arms']==['trace_control','raw_f']
    assert auth['test_access'] is False and auth['historical_weights'] is False
    review_path=ROOT/'INDEPENDENT_PREPARATION_REVIEW.json';review=read(review_path)
    assert auth['independent_review_sha256']==sha(review_path)
    assert review['passed'] and review['frozen_manifest_sha256']==mh
    assert review['source_hashes']==manifest['source_hashes']
    publication_path=Path(auth['publication_receipt']);publication=read(publication_path)
    assert sha(publication_path)==auth['publication_receipt_sha256']
    assert publication['remote_verified'] and publication['download_before_stage_before_commit_push']
    assert publication['frozen_manifest_sha256']==mh and publication['independent_review_sha256']==sha(review_path)
    for path,digest in manifest['source_hashes'].items():assert sha(path)==digest,('Changed frozen input',path)
    assert manifest['split_manifest_sha256']==PINS['SPLIT_MANIFEST.json']
    assert manifest['independent_split_verification_sha256']==PINS['INDEPENDENT_SPLIT_VERIFICATION.json']
    return {'authorization_sha256':sha(authorization),'manifest_sha256':mh,'manifest':manifest,
        'review_sha256':sha(review_path),'publication_sha256':sha(publication_path)}
