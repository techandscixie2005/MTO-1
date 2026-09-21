"""One unified test opening after all 15 independent validation-selected fits."""
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data_protocol import ROOT,load,collate
from models import build_triplet,VARIANTS
from train import fingerprint,atomic_json,atomic_save


def metrics(pred,target,rms):
    err=pred-target
    mse=float(np.mean(err**2));mae=float(np.mean(np.abs(err)))
    denominator=np.sum((target-target.mean())**2)
    cosine=np.sum(pred*target,axis=-1)/(np.linalg.norm(pred,axis=-1)*np.linalg.norm(target,axis=-1))
    pc=pred-pred.mean(-1,keepdims=True);tc=target-target.mean(-1,keepdims=True)
    with np.errstate(invalid='ignore',divide='ignore'):
        pearson=np.sum(pc*tc,axis=-1)/(np.linalg.norm(pc,axis=-1)*np.linalg.norm(tc,axis=-1))
    return dict(source_mse=mse,source_mae=mae,source_rmse=math.sqrt(mse),normalized_mse=mse/rms**2,
                normalized_mae=mae/rms,normalized_rmse=math.sqrt(mse)/rms,
                global_r2=float(1-np.sum(err**2)/denominator),cosine_mean=float(np.nanmean(cosine)),
                pearson_mean=float(np.nanmean(pearson)),pearson_undefined=int(np.isnan(pearson).sum())),cosine,pearson


def detach_tree(v):
    if torch.is_tensor(v):return v.detach().cpu()
    if isinstance(v,dict):return {k:detach_tree(x) for k,x in v.items()}
    return v


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((ROOT/'protocol.json').read_text());fp=fingerprint()
    assert json.loads((ROOT/'reports/PREFLIGHT_PASSED.json').read_text())['fingerprint']==fp
    completions={}
    for seed in cfg['seeds']:
        for label in VARIANTS:
            path=ROOT/f'runs/seed_{seed}/{label}/FIT_COMPLETE.json'
            c=json.loads(path.read_text());assert c['fingerprint']==fp
            completions[(seed,label)]=c
    # The gate above precedes every test label/forward below.
    data,info=load();test=data['test'];target=np.stack([r['spectrum'].numpy() for r in test])
    grid=np.asarray(info['grid']);ids=np.asarray([r['id'] for r in test])
    example_indices=[list(ids).index(mid) for mid in info['examples']]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    predictions={};rows=[];exports={}
    for seed in cfg['seeds']:
        models=build_triplet(cfg,info['n_ref'],info['grid'],seed)
        for label,model in models.items():
            path=ROOT/f'runs/seed_{seed}/{label}/best.pt'
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            assert ckpt['fingerprint']==fp
            model.load_state_dict(ckpt['model'],strict=True);model.to(device).eval()
            values=[]
            with torch.no_grad():
                for start in range(0,len(test),cfg['micro_batch_size']):
                    x,_=collate(test[start:start+cfg['micro_batch_size']],device)
                    values.append(model(**x).cpu().numpy())
            pred=np.concatenate(values);assert np.isfinite(pred).all();predictions[(seed,label)]=pred
            scores,cosine,pearson=metrics(pred,target,info['train_rms'])
            rows.append(dict(seed=seed,variant=label,**scores,**{k:completions[(seed,label)][k] for k in ['best_epoch','epochs','seconds']}))
            np.savez_compressed(ROOT/f'reports/test_{label}_seed{seed}.npz',ids=ids,grid=grid,
                                target=target,prediction=pred,cosine=cosine,pearson=pearson)
            if seed==11 and label!=VARIANTS[0]:
                for mid in info['examples']:
                    rec=next(r for r in test if r['id']==mid);x,_=collate([rec],device)
                    with torch.no_grad():spectrum,e=model(**x,export=True)
                    e=detach_tree(e);atomic_save(dict(id=mid,seed=seed,variant=label,grid=grid,**e),ROOT/f'reports/assembly_{label}_{mid}_seed11.pt')
                    E,f=e['E'],e['f'];sigma=cfg['sigma_eV']
                    cdf=lambda edge:.5*(1+torch.erf((edge-E)/(math.sqrt(2)*sigma)))
                    expected=float((f*(cdf(grid[-1])-cdf(grid[0]))).sum())
                    actual=float(np.trapz(spectrum.detach().cpu().numpy()[0],grid))
                    exports[f'{label}/{mid}']=dict(shapes={k:{t:list(v.shape) for t,v in e[k].items()} for k in ['H','B','F','M','CG']},
                        c_shape=list(e['c'].shape),min_eigenvalue_A=float(torch.linalg.eigvalsh(e['A']).min()),
                        Q_trace_max=float(e['Q'].diagonal(dim1=-2,dim2=-1).sum(-1).abs().max()),
                        gate_saturation=float((e['c'].abs()>.99).float().mean()),energy_range=[float(E.min()),float(E.max())],
                        effective_states=int((f>f.max()*.01).sum()),sum_f=float(f.sum()),
                        analytic_window_area=expected,numeric_240_window_area=actual,window_quadrature_error=actual-expected,
                        tensor_rms=float(e['Q'].square().mean().sqrt()),beta_rms=float(e['beta'].square().mean().sqrt()))
            model.cpu()
        del models
        # Every seed has its own raw prediction overlay; no smoothing or sample selection.
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,index in zip(axes.flat,example_indices):
            ax.plot(grid,target[index],color='black',label='target')
            for label in VARIANTS:ax.plot(grid,predictions[(seed,label)][index],label=label)
            ax.set(title=f'ID {ids[index]}, seed {seed}',xlabel='Energy (eV)',ylabel='Source intensity')
        axes.flat[0].legend(fontsize=7);fig.savefig(ROOT/f'reports/examples_seed{seed}.png',dpi=150);plt.close(fig)
    for label in VARIANTS:
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for seed in cfg['seeds']:
            h=json.loads((ROOT/f'runs/seed_{seed}/{label}/history.json').read_text())
            for ax,key in zip(axes,['train_mse','validation_mse','lr']):
                ax.plot([r['epoch'] for r in h],[r[key] for r in h],label=f'seed {seed}')
                ax.set(xlabel='Epoch',ylabel=key,yscale='log')
        axes[0].legend();fig.suptitle(label);fig.savefig(ROOT/f'reports/curves_{label}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,index in zip(axes.flat,example_indices):
        ax.plot(grid,target[index],color='black',label='target')
        for label in VARIANTS:
            values=np.stack([predictions[(seed,label)][index] for seed in cfg['seeds']])
            mean,std=values.mean(0),values.std(0,ddof=1);line=ax.plot(grid,mean,label=label)[0]
            ax.fill_between(grid,mean-std,mean+std,color=line.get_color(),alpha=.15)
        ax.set(title=f'ID {ids[index]}, seed mean +/- SD',xlabel='Energy (eV)',ylabel='Source intensity')
    axes.flat[0].legend(fontsize=7);fig.savefig(ROOT/'reports/examples_seed_mean.png',dpi=150);plt.close(fig)
    summary={}
    score_keys=['source_mse','source_mae','source_rmse','normalized_mse','normalized_mae','normalized_rmse','global_r2','cosine_mean','pearson_mean']
    for label in VARIANTS:
        subset=[r for r in rows if r['variant']==label]
        summary[label]={k:dict(mean=float(np.mean([r[k] for r in subset])),std=float(np.std([r[k] for r in subset],ddof=1))) for k in score_keys}
    differences={}
    for left,right in [(VARIANTS[0],VARIANTS[1]),(VARIANTS[1],VARIANTS[2])]:
        values=[next(r['normalized_mse'] for r in rows if r['seed']==s and r['variant']==right)-next(r['normalized_mse'] for r in rows if r['seed']==s and r['variant']==left) for s in cfg['seeds']]
        differences[f'{right} minus {left}']=dict(per_seed=values,mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
    atomic_json(dict(per_seed=rows,summary=summary,paired_differences=differences),ROOT/'reports/test_metrics.json')
    atomic_json(exports,ROOT/'reports/assembly_diagnostics.json')
    text=['# New 1k comparison','',
          'Original model code/architecture equivalence; target data underwent grid adaptation. This is not a complete reproduction of the paper training experiment.',
          'Source broadening is unknown. Unit-integral Gaussian sigma=0.2 eV is the disclosed initial assumption. Absolute physical intensity units are not asserted.',
          'All fits start from scratch; no author or historical checkpoints initialize training. The historical 1k test identities have been seen before.',
          '', 'Global R² = 1 - sum((prediction-target)^2) / sum((target-mean(target over all molecules and bins))^2).',
          'Normalized metrics divide source errors by the single training RMS. Cosine and Pearson are computed per molecule; undefined Pearson values are counted.',
          '', '| Seed | Model | MSE (source) | MSE (normalized) | MAE | RMSE | Global R² | Cosine | Pearson | Best epoch |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:text.append(f"| {r['seed']} | {r['variant']} | {r['source_mse']:.8g} | {r['normalized_mse']:.8g} | {r['source_mae']:.8g} | {r['source_rmse']:.8g} | {r['global_r2']:.6g} | {r['cosine_mean']:.6g} | {r['pearson_mean']:.6g} | {r['best_epoch']} |")
    text += ['', 'Mean and sample SD across five seeds:','```json',json.dumps(summary,indent=2),'```','',
             'A versus B compares the original direct-spectrum network with the complete planned MTO physical model. Different readouts prevent attributing the entire difference to assembly.',
             'B versus C diagnoses state-conditioned atom selection with identical CG/decoder and initial parameters. It does not identify the whole contribution of MTO+CG.',
             'The isotropic spectrum does not uniquely identify real excited states, orbitals, or tensor directions.',
             '', 'Paired normalized MSE differences:','```json',json.dumps(differences,indent=2),'```']
    for seed in cfg['seeds']:text+=['',f'![Fixed examples seed {seed}](examples_seed{seed}.png)']
    text+=['','![Separate seed mean](examples_seed_mean.png)']
    for label in VARIANTS:text+=['',f'![Training/validation/LR {label}](curves_{label}.png)']
    (ROOT/'reports/comparison.md').write_text('\n'.join(text))
    required=[ROOT/'reports/comparison.md',ROOT/'reports/test_metrics.json',ROOT/'reports/assembly_diagnostics.json']
    assert all(p.is_file() and p.stat().st_size>0 for p in required)
    atomic_json(dict(fingerprint=fp,fits=15,tests=1500,report='reports/comparison.md',status='STAGE_COMPLETE'),ROOT/'STAGE_COMPLETE.json')


if __name__=='__main__':main()
