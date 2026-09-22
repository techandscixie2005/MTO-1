import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
root=Path(__file__).resolve().parent
metrics=json.loads((root/'test_metrics.json').read_text())
ha=json.loads((root/'history_A.json').read_text());hb=json.loads((root/'history_B.json').read_text())
a=np.load(root/'test_detanet_original_uv_seed11.npz');b=np.load(root/'test_detanet_mto_planned_seed11.npz')
assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['target'],b['target'])
y=a['target'];pa=a['prediction'];pb=b['prediction'];grid=a['grid'];ids=a['ids']
ea=((pa-y)**2).mean(1);eb=((pb-y)**2).mean(1)
def q(x):return dict(zip(('min','q25','median','q75','q90','q95','max'),np.quantile(x,[0,.25,.5,.75,.9,.95,1]).tolist()))
summary={'metrics':{k:v for k,v in metrics.items() if k!='fingerprint'},'history':{},'per_molecule':{}}
for label,h,row in zip(('A','B'),(ha,hb),metrics['rows']):
    best=h[row['best_epoch']-1]
    reductions=[r for i,r in enumerate(h) if i and r['lr']<h[i-1]['lr']]
    summary['history'][label]=dict(best=best,last=h[-1],lr_reductions=reductions,
        seconds_per_epoch=row['seconds']/row['epochs'],
        snapshots=[h[i-1] for i in (1,10,25,50,79,100,150,200,231,250,500,750,958,1000) if i<=len(h)])
for label,p,e,h in [('A',pa,ea,ha),('B',pb,eb,hb)]:
    area=np.trapezoid(p,grid,axis=1);target_area=np.trapezoid(y,grid,axis=1)
    peak=np.argmax(p,axis=1);target_peak=np.argmax(y,axis=1)
    energy=((p-y)**2).mean(0)
    summary['per_molecule'][label]=dict(mse_quantiles=q(e),cosine_quantiles=q(a['cosine'] if label=='A' else b['cosine']),
        negative_bin_fraction=float((p<0).mean()),negative_spectral_area=float(np.trapezoid(np.minimum(p,0),grid,axis=1).mean()),
        peak_energy_absolute_error_mean=float(np.abs(grid[peak]-grid[target_peak]).mean()),
        peak_energy_absolute_error_median=float(np.median(np.abs(grid[peak]-grid[target_peak]))),
        integrated_intensity_mae=float(np.abs(area-target_area).mean()),
        integrated_intensity_mean_bias=float((area-target_area).mean()),
        largest_10_error_fraction=float(np.sort(e)[-10:].sum()/e.sum()),
        worst=[dict(id=int(ids[i]),mse=float(e[i]),A=float(ea[i]),B=float(eb[i])) for i in np.argsort(e)[-5:][::-1]])
summary['paired']=dict(B_mse_wins=int((eb<ea).sum()),A_mse_wins=int((ea<eb).sum()),
    B_mae_wins=int((np.abs(pb-y).mean(1)<np.abs(pa-y).mean(1)).sum()),
    B_cosine_wins=int((b['cosine']>a['cosine']).sum()),
    relative_mse_improvement_quantiles=q(100*(ea-eb)/ea),
    largest_improvements=[dict(id=int(ids[i]),A=float(ea[i]),B=float(eb[i]),difference=float(ea[i]-eb[i])) for i in np.argsort(ea-eb)[-5:][::-1]],
    largest_regressions=[dict(id=int(ids[i]),A=float(ea[i]),B=float(eb[i]),difference=float(ea[i]-eb[i])) for i in np.argsort(ea-eb)[:5]])
summary['bands']=[]
for low,high in [(1.5,5),(5,8),(8,10.5),(10.5,13.51)]:
    mask=(grid>=low)&(grid<high);x=float(((pa[:,mask]-y[:,mask])**2).mean());z=float(((pb[:,mask]-y[:,mask])**2).mean())
    summary['bands'].append(dict(low=low,high=high,bins=int(mask.sum()),A_mse=x,B_mse=z,improvement_percent=100*(x-z)/x,
        contribution_to_total_improvement=float((((pa[:,mask]-y[:,mask])**2)-((pb[:,mask]-y[:,mask])**2)).sum()/((pa-y)**2-(pb-y)**2).sum())))
summary['fixed_examples']=[dict(id=int(mid),A_mse=float(ea[np.where(ids==mid)[0][0]]),B_mse=float(eb[np.where(ids==mid)[0][0]])) for mid in ids[:4]]
summary['target_energy_summary']=dict(mean_integrated_intensity=float(np.trapezoid(y,grid,axis=1).mean()))
(root/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (root/'per_molecule.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(['id','A_mse','B_mse','MTO_relative_improvement_percent','A_cosine','B_cosine'])
    writer.writerows(zip(ids,ea,eb,100*(ea-eb)/ea,a['cosine'],b['cosine']))
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
for ax,limit in zip(axes[0],(1000,250)):
    for label,h,color in [('A: original DetaNet',ha,'#2364AA'),('B: planned MTO',hb,'#E67E22')]:
        ax.plot([r['epoch'] for r in h],[r['validation_mse'] for r in h],color=color,label=label+' validation')
        ax.plot([r['epoch'] for r in h],[r['train_mse'] for r in h],color=color,ls='--',alpha=.8,label=label+' train')
    for h,color in [(ha,'#2364AA'),(hb,'#E67E22')]:
        best=min(h,key=lambda r:r['validation_mse']);ax.scatter(best['epoch'],best['validation_mse'],color=color,s=60,marker='*',zorder=5)
    ax.set(xlim=(1,limit),ylim=(.01,10000 if limit==1000 else 3),yscale='log',xlabel='Epoch',ylabel='Train-RMS normalized MSE',title='Learning curves (full error range)' if limit==1000 else 'First 250 epochs (zoom: MSE <= 3)')
    ax.grid(alpha=.2)
axes[0,0].legend(fontsize=7)
ax=axes[1,0];ax.scatter(ea,eb,c=np.where(eb<ea,'#2A9D8F','#C54B4B'),alpha=.8,s=22)
low=min(ea.min(),eb.min())*.7;high=max(ea.max(),eb.max())*1.5
ax.plot([low,high],[low,high],'k--',lw=1)
ax.set(xscale='log',yscale='log',xlim=(low,high),ylim=(low,high),xlabel='A test MSE per molecule',ylabel='B test MSE per molecule',title=f'MTO lower MSE: {(eb<ea).sum()}/100 molecules');ax.grid(alpha=.2)
ax=axes[1,1]
ax.plot(grid,((pa-y)**2).mean(0),label='A: original DetaNet',color='#2364AA')
ax.plot(grid,((pb-y)**2).mean(0),label='B: planned MTO',color='#E67E22')
ax.set(xlabel='Energy (eV)',ylabel='Mean squared error across test molecules',title='Error by spectral energy');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('1k scale (800 / 100 / 100), seed=11; best-validation checkpoints',fontsize=14)
fig.savefig(root/'analysis_overview.png',dpi=170);plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='history'},indent=2))
print('HISTORY',json.dumps(summary['history'],indent=2))
