"""Evaluate one completed scale; stage completion also requires assembly exports."""
import argparse, csv, hashlib, itertools, json, math, statistics
from pathlib import Path
import numpy as np
import torch
from models import build_pair
from dataset import collate, metrics
from scale_dataset import load_scale, chosen_mask

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('baseline', 'mto')

def bootstrap(diffs):
    # Bounded batches avoid allocating 10,000 x 13,000 indices at once.
    rng = np.random.default_rng(20260914)
    boot = []
    for start in range(0, 10000, 32):
        indices = rng.integers(len(diffs), size=(min(32, 10000-start), len(diffs)))
        boot.extend(diffs[indices].mean(1).tolist())
    return np.quantile(boot, [.025, .975]).tolist()

def aggregate(per_seed, molecule_mse, mask, seeds, norm):
    result = {'test_molecules': int(mask.sum()), 'aggregate': {}, 'per_seed': per_seed}
    for arm in ARMS:
        rows = [r for r in per_seed if r['arm'] == arm]
        result['aggregate'][arm] = {k: {'mean': statistics.mean(r[k] for r in rows),
                                      'std_across_seeds': statistics.stdev(r[k] for r in rows)}
                                   for k in ('mse', 'global_rmse', 'mean_molecule_rmse', 'mae', 'cosine', 'pearson',
                                             'source_mse', 'source_global_rmse', 'source_mae')}
    b, m = (result['aggregate'][arm]['mse']['mean'] for arm in ARMS)
    seed_deltas = np.asarray([molecule_mse['mto'][j, mask].mean()-molecule_mse['baseline'][j, mask].mean()
                             for j in range(len(seeds))])
    molecule_deltas = (molecule_mse['mto'][:, mask]-molecule_mse['baseline'][:, mask]).mean(0)
    ci = bootstrap(molecule_deltas)
    null = [abs((seed_deltas*np.asarray(signs)).mean()) for signs in itertools.product((-1, 1), repeat=len(seeds))]
    result.update(mto_mse_relative_improvement_percent=100*(b-m)/b,
                  paired_seed_mse_deltas=seed_deltas.tolist(),
                  mto_better_test_mse_seed_count=int((seed_deltas < 0).sum()),
                  paired_mto_minus_baseline_mse=float(molecule_deltas.mean()),
                  paired_molecule_bootstrap_95ci=ci,
                  source_paired_molecule_bootstrap_95ci=[v*norm**2 for v in ci],
                  paired_seed_sign_flip_two_sided_p=float(np.mean(np.asarray(null) >= abs(seed_deltas.mean())-1e-15)),
                  ci_scope='Conditional on fixed split and fitted models; molecules resampled after averaging paired seed differences. Not independent training-set uncertainty.',
                  seed_test_scope=f'Exact two-sided paired sign-flip test across {len(seeds)} fitted seed pairs; minimum possible p={2/2**len(seeds):g}. Descriptive secondary analysis; no significance claim from conditional molecule CI alone.')
    return result

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=['1k', '10k', 'full'], required=True)
    args = parser.parse_args()
    scales = [args.scale]
    previous_scale = {'1k':None, '10k':'1k', 'full':'10k'}[args.scale]
    if previous_scale:
        assert (ROOT / 'results' / previous_scale / 'COMPLETE').exists(), 'Previous scale must fully finish first'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    grid = json.loads((ROOT / 'configs/grid.json').read_text())
    cfg = grid['configs'][0]
    execution = json.loads((ROOT / 'configs/execution.json').read_text())
    seeds = execution['seeds_by_scale'][args.scale]
    for scale in scales:
        for seed in seeds:
            assert (ROOT / 'runs' / f'{scale}_seed{seed}' / 'DONE').exists(), (scale, seed)
    assert json.loads((ROOT / 'results/preflight.json').read_text())['passed']
    graphs, _ = load_scale(ROOT, 'full', splits=('test',))
    rows = np.asarray([g['row'] for g in graphs]); ids = np.asarray([g['id'] for g in graphs])
    target_source = torch.stack([g['spectrum'] for g in graphs])
    with np.load(ROOT / 'data/qm9s_full.npz', allow_pickle=False) as a:
        subset, fresh, energy = chosen_mask(ROOT,args.scale,a)[rows], a['fresh_test'][rows], a['energy_eV']
        train_mean = {}
        for scale in scales:
            chosen = chosen_mask(ROOT,scale,a)
            train_mean[scale] = a['spectra'][chosen & (a['split'] == 0)].mean(0)
    examples = np.flatnonzero(fresh)[:4]
    all_results = {'config': cfg, 'seeds': seeds, 'seeds_by_scale': execution['seeds_by_scale'], 'execution_amendment': execution, 'scales': {}, 'test_used_for_tuning': False,
                   'protocol_sha256': hashlib.sha256((ROOT / 'configs/grid.json').read_bytes()).hexdigest(),
                   'scope': 'Fixed-capacity scaling experiment; one nested random identity split, from scratch. Does not test scaffold extrapolation or optimal architecture capacity.'}
    for scale in scales:
        manifest = json.loads((ROOT / 'data' / f'manifest_{scale}.json').read_text())
        norm = manifest['train_spectrum_rms']
        target = target_source/norm
        out = ROOT / 'results' / scale; out.mkdir(exist_ok=True)
        cohorts = {'within_scale': subset, 'common_fresh_test': fresh}
        per_seed = {k: [] for k in cohorts}
        molecule_mse = {arm: np.empty((len(seeds), len(ids))) for arm in ARMS}
        examples_predictions = {arm: np.zeros((len(examples), len(energy)), dtype=np.float64) for arm in ARMS}
        with (out / 'per_molecule.csv').open('w', newline='') as f:
            fields = ['id', 'identity', 'seed', 'arm', 'within_scale_test', 'common_fresh_test', 'mse', 'rmse', 'mae', 'cosine', 'pearson', 'source_mse', 'source_mae']
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for si, seed in enumerate(seeds):
                pair, counts = build_pair(cfg, manifest['train_median_atoms'], seed=seed, spectrum_rms=norm)
                folder = ROOT / 'runs' / f'{scale}_seed{seed}'
                experiment = json.loads((folder / 'experiment.json').read_text())
                for arm, model in pair.items():
                    ck = torch.load(folder / f'{arm}_best.pt', map_location='cpu', weights_only=True)
                    assert ck['database_sha256'] == manifest['sample_sha256']
                    assert ck['fingerprint'] == experiment['fingerprint'] and ck['config'] == cfg and ck['scale_name'] == scale
                    assert ck['scale'] == norm and ck['seed'] == seed and ck['variant'] == arm
                    model.load_state_dict(ck['state_dict']); model.cuda().eval()
                    pred = []
                    physical = {k:[] for k in ('energy_eV','energy_order','A','f','trace_A')}
                    for start in range(0, len(graphs), 32):
                        x, _ = collate(graphs[start:start+32], 'cuda', norm)
                        prediction, exported = model(**x,export=True)
                        pred.append(prediction.cpu())
                        for key in physical:
                            physical[key].append(exported[key].cpu())
                    pred = torch.cat(pred)
                    assert bool(torch.isfinite(pred).all()),(scale,seed,arm,'nonfinite predictions')
                    physical = {k:torch.cat(v).numpy() for k,v in physical.items()}
                    assert all(np.isfinite(v).all() for v in physical.values())
                    np.savez_compressed(out / f'states_{arm}_seed{seed}.npz',ids=ids,**physical)
                    eigenvalues=np.linalg.eigvalsh(physical['A'])
                    trace=physical['trace_A']
                    effective_rank=trace**2 / np.maximum(np.sum(eigenvalues**2,axis=-1),1e-30)
                    bright=trace>1e-8
                    diagnostics={'min_eigenvalue':float(eigenvalues.min()),
                                 'mean_effective_rank_bright':float(effective_rank[bright].mean()),
                                 'bright_state_count':int(bright.sum()),
                                 'interpretation':'Spectrum-only latent decomposition; tensor directions and individual transitions are not identified by isotropic spectrum labels.'}
                    (out / f'tensor_diagnostics_{arm}_seed{seed}.json').write_text(json.dumps(diagnostics,indent=2))
                    vals = {k: v.double().numpy() for k, v in metrics(pred, target).items()}
                    molecule_mse[arm][si] = vals['mse']
                    examples_predictions[arm] += pred[examples].numpy()*norm/len(seeds)
                    np.savez_compressed(out / f'predictions_{arm}_seed{seed}.npz', ids=ids,
                                        source_predictions=pred.numpy()*norm, energy_eV=energy,
                                        within_scale_test=cohorts['within_scale'], common_fresh_test=fresh)
                    for name, mask in cohorts.items():
                        summary = {'scale': scale, 'cohort': name, 'seed': seed, 'arm': arm,
                                   'parameters': counts[arm], 'best_epoch': ck['best_epoch'],
                                   'validation_mse': ck['validation']['mse'],
                                   **{k: float(v[mask].mean()) for k,v in vals.items() if k != 'rmse'}}
                        summary.update(global_rmse=math.sqrt(summary['mse']), mean_molecule_rmse=float(vals['rmse'][mask].mean()),
                                       source_mse=summary['mse']*norm**2, source_global_rmse=math.sqrt(summary['mse'])*norm,
                                       source_mae=summary['mae']*norm)
                        per_seed[name].append(summary)
                    for j, g in enumerate(graphs):
                        writer.writerow({'id': g['id'], 'identity': g['identity'], 'seed': seed, 'arm': arm,
                            'within_scale_test': bool(cohorts['within_scale'][j]), 'common_fresh_test': bool(fresh[j]),
                            **{k: float(v[j]) for k,v in vals.items()},
                            'source_mse': float(vals['mse'][j]*norm**2), 'source_mae': float(vals['mae'][j]*norm)})
                    model.cpu(); torch.cuda.empty_cache()
                    print('Evaluated', scale, seed, arm, flush=True)
        results = {'manifest': manifest, 'seeds': seeds, 'cohorts': {name: aggregate(per_seed[name], molecule_mse, mask, seeds, norm) for name, mask in cohorts.items()}}
        naive = metrics(torch.from_numpy(train_mean[scale])[None].expand(len(ids), -1)/norm, target)
        results['train_mean_spectrum_baseline'] = {name: {k: float(v[mask].double().mean()) for k,v in naive.items()} for name,mask in cohorts.items()}
        (out / 'comparison.json').write_text(json.dumps(results, indent=2))
        records = per_seed['within_scale'] + per_seed['common_fresh_test']
        with (out / 'per_seed.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        np.savez_compressed(out / 'example_spectra.npz', ids=ids[examples], target=target_source[examples].numpy(),
                            energy_eV=energy, **examples_predictions)
        all_results['scales'][scale] = results
    (ROOT / 'results' / f'evaluation_{args.scale}.json').write_text(json.dumps(all_results, indent=2))
    from report_scale import make_report
    make_report(all_results)
    for previous_name in ('1k','10k','full'):
        if previous_name == args.scale: break
        previous = json.loads((ROOT / 'results' / f'evaluation_{previous_name}.json').read_text())
        for key in ('config','protocol_sha256'):
            assert previous[key] == all_results[key], ('Frozen protocol changed', key)
        all_results['scales'].update(previous['scales'])
    all_results['scales']={s:all_results['scales'][s] for s in ('1k','10k','full') if s in all_results['scales']}
    (ROOT / 'results/comparison_all.json').write_text(json.dumps(all_results,indent=2))
    if len(all_results['scales'])>1: make_report(all_results)
    (ROOT / 'results' / args.scale / 'COMPLETE').write_text(f'All {len(seeds)} paired fits, test cohorts, statistics and figures for this scale completed.\n')

if __name__ == '__main__':
    main()
