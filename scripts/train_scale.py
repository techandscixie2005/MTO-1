"""Paired fits with atomic epoch checkpoints and exact optimizer/RNG resume."""
import argparse, hashlib, json, math, os, random, signal, time
from pathlib import Path
import torch
from models import build_pair
from dataset import collate, metrics
from scale_dataset import load_scale

ROOT = Path(__file__).resolve().parents[1]
STOP = False

def stop(signum, frame):
    global STOP
    STOP = True

def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)

def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, tmp)
    tmp.replace(path)

@torch.no_grad()
def evaluate(model, graphs, scale, device):
    model.eval(); values = {}
    for start in range(0, len(graphs), 32):
        x, y = collate(graphs[start:start+32], device, scale)
        for key, value in metrics(model(**x), y).items():
            values[key] = values.get(key, 0.) + float(value.double().sum())
    return {key: value / len(graphs) for key, value in values.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', choices=['1k', '10k', 'full'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tag', default='smoke')
    ap.add_argument('--stop-after-epoch', type=int)
    args = ap.parse_args()
    signal.signal(signal.SIGUSR1, stop)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available(), 'Training requires an allocated GPU'
    grid = json.loads((ROOT / 'configs/grid.json').read_text())
    cfg = grid['configs'][0]; tc = grid['shared_training'].copy()
    if args.smoke:
        tc.update(max_epochs=3, min_epochs=3, patience=3)
    graphs, manifest = load_scale(ROOT, args.scale, limit=64 if args.smoke else None)
    train = [g for g in graphs if g['split'] == 'train']
    val = [g for g in graphs if g['split'] == 'validation']
    del graphs
    scale = manifest['train_spectrum_rms']
    pair, counts = build_pair(cfg, manifest['train_median_atoms'], seed=args.seed, spectrum_rms=scale)
    for key, value in pair['baseline'].backbone.state_dict().items():
        assert torch.equal(value, pair['mto'].backbone.state_dict()[key])
    folder = ROOT / ('smoke' if args.smoke else 'runs') / f'{args.scale}_seed{args.seed}'
    if args.smoke:
        folder = folder / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__)]}
    contract = {'config': cfg, 'training': tc, 'seed': args.seed, 'scale': args.scale,
                'database_sha256': manifest['sample_sha256'], 'manifest': manifest,
                'source_hashes': hashes, 'counts': counts, 'train_count': len(train),
                'validation_count': len(val), 'smoke_only': args.smoke}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (folder / 'experiment.json').exists():
        assert json.loads((folder / 'experiment.json').read_text())['fingerprint'] == fingerprint, 'Resume contract changed'
    else:
        write_json(folder / 'experiment.json', {**contract, 'fingerprint': fingerprint,
            'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
            'job_id': os.environ.get('SLURM_JOB_ID'), 'test_evaluated': False})
    if (folder / 'DONE').exists():
        print('Already completed', folder, flush=True); return
    for arm in (['baseline', 'mto'] if args.seed % 4 == 3 else ['mto', 'baseline']):
        if (folder / f'{arm}_summary.json').exists():
            assert (folder / f'{arm}_best.pt').exists()
            continue
        torch.manual_seed(args.seed); random.seed(args.seed)
        model = pair[arm].cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=tc['weight_decay'])
        best, stale, best_epoch, start_epoch, history, elapsed, steps = float('inf'), 0, 0, 0, [], 0., 0
        last = folder / f'{arm}_last.pt'
        if last.exists():
            ck = torch.load(last, map_location='cpu', weights_only=False)
            assert ck['fingerprint'] == fingerprint
            model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
            torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
            random.setstate(ck['python_rng'])
            best, stale, best_epoch = ck['best'], ck['stale'], ck['best_epoch']
            start_epoch, history, elapsed, steps = ck['epoch'], ck['history'], ck['elapsed'], ck['steps']
            del ck
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        reason = 'max_epochs'
        for epoch in range(start_epoch, tc['max_epochs']):
            if epoch >= tc['min_epochs'] and stale >= tc['patience']:
                reason = 'validation_early_stopping'; break
            model.train()
            warm, total = tc['warmup_epochs'], tc['max_epochs']
            mult = (epoch+1)/warm if epoch < warm else .05+.95*(1+math.cos(math.pi*(epoch-warm)/max(1,total-warm)))/2
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr'] * mult
            order = list(range(len(train))); random.Random(args.seed*10000+epoch).shuffle(order)
            total_loss = 0.
            for offset in range(0, len(train), tc['batch_size']):
                rows = [train[i] for i in order[offset:offset+tc['batch_size']]]
                x, y = collate(rows, 'cuda', scale)
                loss = (model(**x)-y).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f'{arm}: nonfinite loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); total_loss += float(loss) * len(rows); steps += 1
            validation = evaluate(model, val, scale, 'cuda')
            row = {'epoch': epoch+1, 'train_mse': total_loss/len(train), 'validation': validation,
                   'elapsed_seconds': elapsed+time.time()-started, 'lr': cfg['lr']*mult, 'optimizer_steps': steps}
            history.append(row)
            if validation['mse'] < best:
                best, stale, best_epoch = validation['mse'], 0, epoch+1
                save(folder / f'{arm}_best.pt', {'state_dict': {k: v.detach().cpu() for k,v in model.state_dict().items()},
                    'variant': arm, 'config': cfg, 'counts': counts, 'seed': args.seed, 'scale_name': args.scale,
                    'n_ref': manifest['train_median_atoms'], 'scale': scale, 'best_epoch': best_epoch,
                    'validation': validation, 'database_sha256': manifest['sample_sha256'], 'fingerprint': fingerprint})
            else:
                stale += 1
            write_json(folder / f'{arm}_history.json', history)
            save(last, {'fingerprint': fingerprint, 'epoch': epoch+1, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'best': best, 'stale': stale, 'best_epoch': best_epoch,
                        'history': history, 'elapsed': row['elapsed_seconds'], 'steps': steps,
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                        'python_rng': random.getstate()})
            print(json.dumps({'scale': args.scale, 'seed': args.seed, 'arm': arm, **row}), flush=True)
            if STOP or (args.stop_after_epoch and epoch+1 == args.stop_after_epoch):
                print('Checkpoint saved; restart identical command to resume', flush=True)
                raise SystemExit(75)
        if len(history) >= tc['min_epochs'] and stale >= tc['patience']:
            reason = 'validation_early_stopping'
        write_json(folder / f'{arm}_summary.json', {'variant': arm, 'best_validation_mse': best,
            'best_epoch': best_epoch, 'epochs_run': len(history), 'parameters': counts[arm],
            'elapsed_seconds': elapsed+time.time()-started, 'optimizer_steps': steps, 'stop_reason': reason,
            'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(), 'test_evaluated': False,
            'fingerprint': fingerprint})
        pair[arm] = model.cpu(); del optimizer
        torch.cuda.empty_cache()
    (folder / 'DONE').write_text('Both paired fits complete; test not evaluated.\n')

if __name__ == '__main__':
    main()
