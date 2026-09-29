"""Promotion-gated confirmation specifications. No runner is produced or called."""
import argparse
import json
from pathlib import Path
from contracts import make_config, pilot_manifest, require, sha, validate_promotion, write_json_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--promotion', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    promotion = json.loads(args.promotion.read_text())
    validate_promotion(promotion)
    evidence = promotion['pilot_final_analysis']
    require(sha(evidence['path']) == evidence['sha256'], 'Promotion analysis does not match its pin')
    root = Path(__file__).resolve().parent
    contracts = json.loads((root / 'SOURCE_CONTRACTS.json').read_text())
    pilot = pilot_manifest()
    require(pilot == contracts['pilot_snapshot'], 'Pilot snapshot changed')
    require(args.output.resolve().is_relative_to(root.resolve()), 'Write only inside confirmation preparation')
    for seed in (23, 37):
        cfg = make_config(pilot, contracts, seed, promotion)
        cfg['promotion_record_sha256'] = sha(args.promotion)
        cfg['source_contracts_sha256'] = sha(root / 'SOURCE_CONTRACTS.json')
        write_json_once(cfg, args.output / ('seed%d.CONFIRMATION_SPEC.json' % seed))
    print(json.dumps({'specifications_written': 2, 'launch_enabled': False, 'runner_written': False}))


if __name__ == '__main__':
    main()
