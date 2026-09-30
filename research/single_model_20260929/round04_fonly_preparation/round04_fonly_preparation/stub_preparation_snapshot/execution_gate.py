"""Fail-closed future execution contract; this preparation release cannot run a fit."""
import argparse
from pathlib import Path
from common import ROOT, read_json, sha, settings


def check_bindings(authority, manifest_sha, review_sha, publication_sha, review, publication):
    expected = {'phase': 'production_execution', 'real_target_fit_authorized': True,
                'fixed_validation_authorized': True, 'test_access_authorized': False,
                'source_manifest_sha256': manifest_sha,
                'independent_review_sha256': review_sha,
                'publication_receipt_sha256': publication_sha,
                'maximum_real_data_solves': 2, 'maximum_validation_comparisons': 1}
    if any(authority.get(key) != value for key, value in expected.items()):
        raise PermissionError('Missing exact, separate production execution authorization')
    if review.get('passed') is not True or review.get('source_manifest_sha256') != manifest_sha:
        raise PermissionError('Independent production source review is not bound')
    if publication.get('remote_verified') is not True or publication.get('download_before_stage_before_commit_push') is not True:
        raise PermissionError('Archive-first remote publication is not verified')
    if publication.get('source_manifest_sha256') != manifest_sha or publication.get('independent_review_sha256') != review_sha:
        raise PermissionError('Publication does not bind exact manifest/review')


def verify_execution(authorization, manifest, review, publication):
    paths = [Path(path) for path in (authorization, manifest, review, publication)]
    if any(not path.is_file() for path in paths):
        raise PermissionError('Production requires separate authorization, manifest, review and publication files')
    auth, frozen, reviewed, published = [read_json(path) for path in paths]
    check_bindings(auth, sha(paths[1]), sha(paths[2]), sha(paths[3]), reviewed, published)
    if reviewed.get('source_hashes') != frozen.get('source_hashes'):
        raise PermissionError('Reviewed source closure is not exact')
    for path, expected in frozen['source_hashes'].items():
        if sha(path) != expected:
            raise PermissionError('Source closure changed')
    cfg = settings()
    if cfg['fit_or_evaluation_runner_released'] is not True:
        raise PermissionError('This preparation release contains no production fit/evaluation runner')
    raise PermissionError('Production execution entry point is not implemented in this preparation release')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--publication', required=True)
    args = parser.parse_args()
    verify_execution(args.authorization, args.manifest, args.review, args.publication)


if __name__ == '__main__':
    main()
