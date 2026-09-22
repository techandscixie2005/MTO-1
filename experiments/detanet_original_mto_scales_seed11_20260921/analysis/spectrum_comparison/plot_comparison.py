import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parent
d=json.loads((root/'common_spectra.json').read_text(encoding='utf-8'));grid=np.array(d['grid'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'Target':'#18232F','DetaNet':'#2877BD','MTO':'#E47726'}
scales=['1k','10k','full'];titles=['1k: 800 training molecules','10k: 8,000 training molecules','Full: 103,785 training molecules']
for page,indices in enumerate(([0,1],[2,3]),1):
    fig,axes=plt.subplots(2,3,figsize=(14.5,7.5))
    fig.subplots_adjust(left=.073,right=.987,top=.82,bottom=.09,wspace=.12,hspace=.40)
    fig.suptitle('Same test molecules across data scales | seed = 11',fontsize=17,y=.982,fontweight='bold')
    fig.text(.5,.931,'Best-validation checkpoints; original 240-point spectra; no smoothing or per-spectrum normalization',ha='center',fontsize=10,color='#455466')
    fig.legend(handles=[Line2D([0],[0],color=colors[k],lw=2.3,label=k) for k in colors],loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,frameon=False,fontsize=11)
    for row,i in enumerate(indices):
        target=np.array(d['scales']['1k']['DetaNet']['target'][i])
        values=[target]+[np.array(d['scales'][s][name]['prediction'][i]) for s in scales for name in ('DetaNet','MTO')]
        lower=min(float(v.min()) for v in values);upper=max(float(v.max()) for v in values)
        pad=.09*(upper-lower)
        for col,scale in enumerate(scales):
            ax=axes[row,col]
            ax.plot(grid,target,color=colors['Target'],lw=2.25,label='Target',zorder=4)
            for name in ('DetaNet','MTO'):
                ax.plot(grid,d['scales'][scale][name]['prediction'][i],color=colors[name],lw=1.65,label=name,zorder=3)
            ax.axhline(0,color='#B8C0C8',lw=.7,zorder=1)
            ax.set(xlim=(1.5,13.5),ylim=(lower-pad,upper+pad),xlabel='Energy (eV)',title=f'ID {d["ids"][i]}  |  {titles[col]}')
            ax.title.set_fontsize(10)
            if col==0:ax.set_ylabel('Intensity (source units)')
            else:ax.tick_params(labelleft=False)
            ax.grid(alpha=.14)
            a=d['scales'][scale]['DetaNet']['mse'][i];b=d['scales'][scale]['MTO']['mse'][i]
            ax.text(.035,.96,f'MSE  A: {a:.2e}\n         B: {b:.2e}',transform=ax.transAxes,va='top',fontsize=8.5,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':2})
    fig.savefig(root/f'same_molecules_comparison_{page}.png',dpi=170,facecolor='white')
    fig.savefig(root/f'same_molecules_comparison_{page}.svg',facecolor='white')
    plt.close(fig)
for i,mid in enumerate(d['ids']):
    print(mid,{s:{name:d['scales'][s][name]['mse'][i] for name in ('DetaNet','MTO')} for s in scales})
