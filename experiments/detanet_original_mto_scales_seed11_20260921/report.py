"""Test only after both scale fits, then publish checked artifacts and completion."""
import argparse
import csv
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data_protocol import ROOT,SCALES,load,sha
from models import VARIANTS,build_pair
from train import fingerprint,atomic_json,adaptive_evaluate

def table(rows):
    text=['| Model | MSE (source) | MSE / train RMS² | MAE | Cosine | Epochs | Best epoch | Training hours |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['variant']} | {r['source_mse']:.8g} | {r['normalized_mse']:.8g} | {r['source_mae']:.8g} | {r['cosine_mean']:.6g} | {r['epochs']} | {r['best_epoch']} | {r['seconds']/3600:.3f} |")
    return text

def report(scale):
    from workflow import require_previous,check_stage,check_fits
    require_previous(scale)
    out=ROOT/'reports'/scale;out.mkdir(parents=True,exist_ok=True)
    if (out/'STAGE_COMPLETE.json').exists():check_stage(scale);return
    cfg=json.loads((ROOT/'protocol.json').read_text());fp=fingerprint(scale)
    completions=check_fits(scale)
    data,info=load(scale);test=data['test']
    target=np.stack([r['spectrum'].numpy() for r in test]).astype(np.float64)
    ids=np.asarray([r['id'] for r in test]);grid=np.asarray(info['grid'])
    device='cuda';assert torch.cuda.is_available() and os.getenv('SLURM_JOB_ID')
    predictions={};rows=[];artifacts=[]
    for label in VARIANTS:
        models=build_pair(cfg,info['n_ref'],info['grid'],11)
        model=models[label];del models
        run=ROOT/f'runs/{scale}/seed_11/{label}'
        ckpt=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        assert ckpt['fingerprint']==fp
        assert ckpt['progress']['best_epoch']+1==completions[label]['best_epoch']
        model.load_state_dict(ckpt['model'],strict=True);del ckpt
        model.to(device).eval()
        state=dict(micro_batch_size=completions[label]['micro_batch_size'],oom_adjustments=[])
        _,pred=adaptive_evaluate(model,test,info['train_rms'],state,device,out)
        pred=pred.numpy().astype(np.float64);assert pred.shape==target.shape and np.isfinite(pred).all()
        err=pred-target
        denominator=np.linalg.norm(pred,axis=-1)*np.linalg.norm(target,axis=-1)
        cosine=np.divide((pred*target).sum(-1),denominator,out=np.zeros(len(pred)),where=denominator>0)
        mse=float(np.mean(err**2))
        row=dict(scale=scale,seed=11,variant=label,source_mse=mse,normalized_mse=mse/info['train_rms']**2,
            source_mae=float(np.mean(np.abs(err))),cosine_mean=float(cosine.mean()),
            cosine_zero_norm_count=int((denominator==0).sum()),
            **{k:completions[label][k] for k in ('epochs','best_epoch','seconds','micro_batch_size')})
        rows.append(row);predictions[label]=pred
        path=out/f'test_{label}_seed11.npz'
        np.savez_compressed(path,ids=ids,grid=grid,target=target,prediction=pred,cosine=cosine)
        artifacts.append(path)
        hist=json.loads((run/'history.json').read_text())
        assert len(hist)==row['epochs'] and hist[-1]['epoch']==row['epochs']
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for ax,key in zip(axes,('train_mse','validation_mse','lr')):
            ax.plot([h['epoch'] for h in hist],[h[key] for h in hist]);ax.set(xlabel='Epoch',ylabel=key,yscale='log')
        fig.suptitle(f'{scale} / {label} / seed=11')
        path=out/f'curves_{label}.png';fig.savefig(path,dpi=150);plt.close(fig);artifacts.append(path)
        del model
        torch.cuda.empty_cache()
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,mid in zip(axes.flat,info['examples']):
        index=list(ids).index(mid);ax.plot(grid,target[index],color='black',label='target')
        for label in VARIANTS:ax.plot(grid,predictions[label][index],label=label)
        ax.set(title=f'{scale}, ID {mid}, seed=11',xlabel='Energy (eV)',ylabel='Source intensity')
    axes.flat[0].legend(fontsize=7)
    path=out/'examples_seed11.png';fig.savefig(path,dpi=150);plt.close(fig);artifacts.append(path)
    improvement=100*(rows[0]['source_mse']-rows[1]['source_mse'])/rows[0]['source_mse'] if rows[0]['source_mse'] else None
    winner=min(rows,key=lambda r:r['source_mse'])['variant'] if rows[0]['source_mse']!=rows[1]['source_mse'] else 'tie'
    metrics=dict(scale=scale,seed=11,counts=info['splits'],train_rms=info['train_rms'],rows=rows,
        mto_relative_mse_improvement_percent=improvement,winner=winner,
        total_training_gpu_hours=sum(r['seconds'] for r in rows)/3600,fingerprint=fp)
    atomic_json(metrics,out/'test_metrics.json');artifacts.append(out/'test_metrics.json')
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    artifacts.append(out/'comparison.csv')
    text=[f'# {scale}: original DetaNet vs planned MTO, seed=11','',
        f"Split counts: {info['splits']}. Train-only RMS: {info['train_rms']:.12g}.",
        'Both complete DetaNet backbones start from the same fresh backbone initialization; each model uses its own best validation checkpoint.',
        'Single-seed preliminary comparison; no cross-seed standard deviation or significance claims. Existing historical test identities are reused.',
        'Source spectra are linearly interpolated to the same author 240-point grid. Source broadening is unknown; MTO sigma=0.2 eV is the retained assumption.',
        'Training hours measure cumulative training/validation/checkpoint wall time on one GPU per model; queue and short-check time are excluded.',
        'Cosine is averaged per molecule; zero-norm pairs are assigned zero and their count is recorded.','']+table(rows)
    text+=['',f'Lower test MSE: {winner}.',f'MTO relative MSE improvement = 100 × (A − B) / A: {improvement}% (negative means MTO is worse).']
    (out/'comparison.md').write_text('\n'.join(text)+'\n');artifacts.append(out/'comparison.md')
    # Required checkpoints and histories are part of the stage's reviewable evidence.
    for label in VARIANTS:
        for name in ('FIT_COMPLETE.json','best.pt','last.pt','history.json','short_check.json'):
            artifacts.append(ROOT/f'runs/{scale}/seed_11/{label}/{name}')
    assert len(rows)==2 and all(p.is_file() and p.stat().st_size>0 for p in artifacts)
    atomic_json(dict(scale=scale,seed=11,fits=2,test_molecules=len(test),evaluated_predictions=2*len(test),
        fingerprint=fp,artifacts={str(p.relative_to(ROOT)):sha(p) for p in artifacts},status='STAGE_COMPLETE'),out/'STAGE_COMPLETE.json')
    check_stage(scale)
    if scale=='full': final_report()

def final_report():
    from workflow import check_stage
    text=['# DetaNet / MTO: single-seed scale comparison','',
          'Only this run, seed=11. Six fresh fits across existing 1k, 10k, full splits. No significance claims or cross-seed statistics.','']
    all_metrics=[]
    for scale in SCALES:
        check_stage(scale);m=json.loads((ROOT/f'reports/{scale}/test_metrics.json').read_text());all_metrics.append(m)
        text += [f'## {scale}',f"Counts: {m['counts']}",'']+table(m['rows'])
        text += ['',f"Lower MSE: {m['winner']}; MTO relative MSE improvement: {m['mto_relative_mse_improvement_percent']}%; cumulative training GPU hours: {m['total_training_gpu_hours']:.3f}.",'']
    (ROOT/'reports/summary.md').write_text('\n'.join(text)+'\n')
    atomic_json(all_metrics,ROOT/'reports/summary.json')
    atomic_json(dict(seed=11,fits=6,scales=list(SCALES),artifacts={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'reports/summary.md',ROOT/'reports/summary.json']}),ROOT/'reports/WORKFLOW_COMPLETE.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scale',choices=SCALES,required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report(args.scale)
    if args.scale=='full': final_report()
