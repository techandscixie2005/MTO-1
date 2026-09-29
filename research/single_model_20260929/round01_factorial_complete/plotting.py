"""Reproduce aggregate aligned learning curves after all four arms finish."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
ARMS=['control','adapter','decorrelation','both']
LABELS={'control':'Control','adapter':'Shared right adapter','decorrelation':'Raw-state decorrelation','both':'Both'}
STYLE={'control':('#2764a5','-'),'adapter':('#d67b1e','--'),
       'decorrelation':('#2a8753',':'),'both':('#993c99','-.')}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    cfg=json.loads((ROOT/'round_config.json').read_text())
    paths={a:ROOT/'runs'/a/'history.jsonl' for a in ARMS}
    assert all((ROOT/'runs'/a/'FIT_COMPLETE.json').is_file() for a in ARMS)
    history={a:[json.loads(line) for line in path.read_text().splitlines()] for a,path in paths.items()}
    epochs=list(range(cfg['epochs']+1))
    assert all([row['epoch'] for row in h]==epochs for h in history.values())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'svg.hashsalt':'mto-single-model-factorial-20260929'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),sharey=True,layout='constrained')
    for ax,secondary in zip(axes,(False,True)):
        for arm in ARMS:
            curve=[(row['validation']['fixed_calibrated_secondary'] if secondary else row['validation'])['raw_f']['r2']
                   for row in history[arm]]
            color,style=STYLE[arm]
            ax.plot(epochs,curve,color=color,linestyle=style,linewidth=1.7,label=LABELS[arm])
        base=history['control'][0]['validation']
        reference=(base['fixed_calibrated_secondary'] if secondary else base)['raw_f']['r2']
        ax.axhline(reference,color='#444444',linewidth=1.1,linestyle=(0,(5,3)),label='Epoch 0 reference')
        ax.set(title='Fixed historical calibration (secondary)' if secondary else 'Native prediction (primary)',
               xlabel='Continuation epoch',xlim=(0,cfg['epochs']))
        ax.set_xticks([0,5,10,15,20])
        ax.grid(axis='y',alpha=.22)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Pooled validation raw-f R²')
    axes[1].legend(frameon=False,loc='lower left',fontsize=8.8)
    fig.suptitle('Matched single-model continuation: aligned validation trajectories',fontsize=12)
    out=ROOT/'ALIGNED_VALIDATION_CURVES.svg'
    fig.savefig(out,metadata={'Date':None,'Description':'Aggregate validation metrics only; one model per arm; fixed calibration never selects checkpoints.'})
    plt.close(fig)
    receipt={'source_sha256':sha(__file__),'matplotlib_version':matplotlib.__version__,
        'input_history_sha256':{a:sha(p) for a,p in paths.items()},'output_sha256':sha(out),
        'epochs':epochs,'aggregate_metrics_only':True,'predictions_averaged':False,
        'reference':'control epoch0 from the same runtime evaluator','output':str(out)}
    (ROOT/'PLOT_MANIFEST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
