"""Plot only actually executed preflight histories and train-only oracle fits."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_protocol import load
data,info=load()
smoke=json.loads((ROOT/'reports/smoke_seed11.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for name,row in smoke.items():
    steps=[0]+[v['step'] for v in row['history']]
    mse=[row['initial']]+[v['mse'] for v in row['history']]
    for ax in axes:ax.plot(steps,mse,label=name)
for ax in axes:
    ax.axhline(.1,ls='--',color='black',label='required MSE < 0.1')
    ax.set(xlabel='Optimizer step',ylabel='Train-RMS normalized MSE')
axes[0].set_yscale('log');axes[1].set_ylim(0,1.5);axes[0].legend(fontsize=7)
fig.suptitle('Seed 11, fixed 32 TRAIN molecules; smoke diagnostics, not generalization')
fig.savefig(ROOT/'reports/smoke_curves.png',dpi=160);plt.close(fig)
oracle=torch.load(ROOT/'reports/decoder_oracle.pt',map_location='cpu',weights_only=False)
grid=np.asarray(info['grid']);fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for index,(ax,mid) in enumerate(zip(axes.flat,info['oracle_ids'])):
    rec=next(r for r in data['train'] if r['id']==mid)
    ax.plot(grid,rec['spectrum'].numpy(),label='target',color='black')
    for result in oracle['real']:ax.plot(grid,result['prediction'][index].numpy(),label=f"start {result['seed']}",alpha=.8)
    ax.set(title=f'TRAIN ID {mid}',xlabel='Energy (eV)',ylabel='Source intensity')
axes[0,0].legend(fontsize=8);fig.suptitle('Free ten-peak decoder oracle, sigma=0.2 eV; local solutions only')
fig.savefig(ROOT/'reports/decoder_oracle_train.png',dpi=160);plt.close(fig)
print('Created smoke_curves.png and decoder_oracle_train.png from real logs and stored predictions.')
