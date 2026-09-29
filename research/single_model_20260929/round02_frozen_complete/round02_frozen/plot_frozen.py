"""Aggregate terminal learning curves, without raw data or predictions."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    result=json.loads((ROOT/'ROUND02_RESULTS.json').read_text());assert result['passed']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'mto-round02-frozen-f'})
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.7),layout='constrained')
    for arm,color,label in [('trace','#2764a5','F: LE + Ls'),('raw_f','#c46a16','F: LE + Lf')]:
        rows=result['aligned'];train=result['arms'][arm]['train_dynamics']
        axes[0].plot([x['epoch'] for x in rows],[x[arm]['native_r2'] for x in rows],label=label,color=color,lw=1.7)
        axes[1].plot([x['epoch'] for x in train],[x['train']['raw_f'] for x in train],color=color,lw=1.7)
        axes[2].plot([0]+[x['epoch'] for x in train],[0]+[100*x['adapter_movement']['mean_relative'] for x in train],color=color,lw=1.7)
    anchor=result['arms']['trace']['epoch_zero']['raw_f']['r2']
    axes[0].axhline(anchor,color='#555555',ls='--',lw=1,label='Zero-update reference')
    for ax in axes:
        ax.set_xlabel('Epoch');ax.set_xlim(0,20);ax.set_xticks([0,5,10,15,20]);ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Pooled validation raw-f R²');axes[0].legend(frameon=False,fontsize=8)
    axes[1].set_ylabel('TRAIN normalized raw-f MSE (online)')
    axes[2].set_ylabel('Mean relative F change on fixed TRAIN (%)')
    fig.suptitle('Frozen original MTO: shared pre-CG adapter, two fixed objectives',fontsize=12)
    path=ROOT/'ROUND02_CURVES.svg';fig.savefig(path,metadata={'Date':None,'Description':'Aggregate metrics only. Validation reused; no prediction averaging.'});plt.close(fig)
    (ROOT/'PLOT_MANIFEST.json').write_text(json.dumps({'source_sha256':sha(__file__),
        'input_sha256':sha(ROOT/'ROUND02_RESULTS.json'),'output_sha256':sha(path),
        'matplotlib_version':matplotlib.__version__,'aggregate_only':True},indent=2)+'\n')
    print(str(path))

if __name__=='__main__':main()
