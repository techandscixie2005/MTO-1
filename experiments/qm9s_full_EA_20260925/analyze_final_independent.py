import pathlib,json,hashlib,math,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(os.environ['RESULTS_ROOT']) if 'RESULTS_ROOT' in os.environ else pathlib.Path(__file__).resolve().parents[1]/'outputs/MTO1_full_EA_results_20260925'
res=json.loads((ROOT/'reports/results.json').read_text())
stats=json.loads((ROOT/'data/normalization.json').read_text());split=json.loads((ROOT/'data/splits.json').read_text())
aud={};preds={};hist={};base=None
for name,m in res['test'].items():
    p=ROOT/'runs'/name/'test_predictions.npz'
    expected=json.loads((ROOT/'reports/checkpoint_verification.json').read_text())[name]['prediction_sha256']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    with np.load(p) as f:d={k:f[k] for k in f.files}
    assert np.array_equal(d['molecule_id'],split['ids']['test'])
    if base is None:base=d
    else:
        for k in ['molecule_id','E_true','A_true']:assert np.array_equal(d[k],base[k]),(name,k)
    assert all(np.isfinite(d[k]).all() for k in ['E_pred','A_pred','E_true','A_true'])
    de=d['E_pred'].astype('float64')-d['E_true'];da=d['A_pred'].astype('float64')-d['A_true']
    le=np.mean(de**2,axis=1)/stats['sE2'];la=np.mean(np.sum(da**2,axis=(-2,-1)),axis=1)/stats['sA2'];L=le+la
    e_mae=np.mean(abs(de));a_rmse=np.sqrt(np.mean(np.sum(da**2,axis=(-2,-1))))
    assert abs(L.mean()-m['L'])<1e-12 and abs(e_mae-m['E_MAE_eV'])<1e-12 and abs(a_rmse-m['A_Frobenius_RMSE_au2'])<1e-12
    grid=np.linspace(0,21,1051);spmse=[];spmae=[]
    for i in range(0,len(L),128):
        def spectrum(E,A):
            f=2/3*(E/27.211386245988)*np.trace(A,axis1=-2,axis2=-1)
            return np.sum(f[...,None]*np.exp(-.5*((grid-E[...,None])/.2)**2)/(.2*np.sqrt(2*np.pi)),axis=1)
        df=spectrum(d['E_pred'][i:i+128].astype('float64'),d['A_pred'][i:i+128].astype('float64'))-spectrum(d['E_true'][i:i+128],d['A_true'][i:i+128])
        spmse.extend(np.mean(df**2,1));spmae.extend(np.mean(abs(df),1))
    spmse=np.array(spmse);spmae=np.array(spmae)
    assert abs(spmse.mean()-m['spectrum_MSE'])<1e-12 and abs(spmae.mean()-m['spectrum_MAE'])<1e-12
    n1=int(np.ceil(len(L)*.01));top=np.argsort(L)[-10:][::-1]
    hist[name]=[json.loads(s) for s in (ROOT/'runs'/name/'history.jsonl').read_text().splitlines()]
    best=next(r for r in hist[name] if r['epoch']==res['fits'][name]['best_epoch']);last=hist[name][-1]
    assert last['epoch']==res['fits'][name]['epochs']
    aud[name]=dict(verified=True,train_L_at_best=best['train'][0],val_L_at_best=best['val'][0],
        last_train_L=last['train'][0],last_val_L=last['val'][0],last_lr=last['lr'],
        tensor_fraction_of_loss=float(la.mean()/L.mean()),loss_quantiles=dict(zip(['p50','p90','p95','p99','max'],np.quantile(L,[.5,.9,.95,.99,1]).tolist())),
        top_1pct_molecule_loss_share=float(np.sort(L)[-n1:].sum()/L.sum()),
        spectrum_top_1pct_error_share=float(np.sort(spmse)[-n1:].sum()/spmse.sum()),
        worst_ids=[dict(id=int(d['molecule_id'][i]),L=float(L[i]),worst_state=int(np.argmax(np.sum(da[i]**2,(-2,-1))))+1) for i in top],
        normalized_A_error_by_state=(np.mean(np.sum(da**2,(-2,-1)),0)/stats['sA2']).tolist())
    preds[name]=dict(d=d,L=L,LE=le,LA=la,spmse=spmse)
baseline=preds['detanet_reference'];reference=preds['mto_reference'];difference=reference['L']-baseline['L']
comparison=dict(mto_molecule_win_fraction=float(np.mean(difference<0)),median_loss_difference=float(np.median(difference)))
for label,values in [('mto_minus_detanet',difference),('spectral_MSE_mto_minus_detanet',reference['spmse']-baseline['spmse'])]:
    order=np.argsort(values)[::-1][:10]
    comparison[label+'_largest_regressions']=[dict(id=int(base['molecule_id'][i]),delta=float(values[i])) for i in order]
# Repeat the exact reported molecular-group bootstrap with independent implementation.
ident=json.loads((ROOT/'data/identity_audit_v2.json').read_text());keys={}
for j,i in enumerate(split['indices']['test']):keys.setdefault(ident[i][1],[]).append(j)
groups=list(keys.values());sums=np.array([difference[g].sum() for g in groups]);counts=np.array([len(g) for g in groups]);rng=np.random.default_rng(20260925)
boot=[]
for _ in range(2000):
    ix=rng.integers(len(groups),size=len(groups));boot.append(sums[ix].sum()/counts[ix].sum())
comparison['independent_group_bootstrap_CI']=np.quantile(boot,[.025,.975]).tolist()
# Existing report loops over reference and selected; they are the same model here,
# so its dictionary retains the second (equally valid) Monte Carlo draw of 2000.
boot2=[]
for _ in range(2000):
    ix=rng.integers(len(groups),size=len(groups));boot2.append(sums[ix].sum()/counts[ix].sum())
comparison['reported_CI_reproduced_second_draw']=np.quantile(boot2,[.025,.975]).tolist()
assert np.allclose(comparison['reported_CI_reproduced_second_draw'],res['comparisons']['mto_reference']['group_bootstrap_95pct_CI'],atol=1e-12)
result=dict(status='PASS',N_test=len(base['molecule_id']),verified_metrics=['L','E_MAE','A_Frobenius_RMSE','spectrum_MSE','spectrum_MAE','paired_group_bootstrap_CI'],runs=aud,comparison=comparison)
(ROOT/'reports/independent_analysis.json').write_text(json.dumps(result,indent=2))
labels={'detanet_reference':'DetaNet','mto_reference':'MTO reference','mto_lr3e4':'MTO lr=3e-4','mto_lr3e3':'MTO lr=3e-3','mto_channels32':'MTO channels=32','mto_batch128':'MTO batch=128'}
fig,ax=plt.subplots(2,2,figsize=(13,9))
for name in ['detanet_reference','mto_reference','mto_channels32']:
    h=hist[name];c=None
    line=ax[0,0].plot([r['epoch'] for r in h],[r['val'][0] for r in h],label=labels[name])[0];c=line.get_color()
    ax[0,0].plot([r['epoch'] for r in h],[r['train'][0] for r in h],color=c,ls='--',alpha=.65)
    ax[0,1].plot(np.arange(1,11),res['test'][name]['E_MAE_by_state_eV'],marker='o',label=labels[name])
    ax[1,0].plot(np.arange(1,11),res['test'][name]['A_Frobenius_RMSE_by_state_au2'],marker='o',label=labels[name])
ax[0,0].set(title='Training vs validation (solid: val, dashed: train)',xlabel='Epoch',ylabel='Normalized E+A loss',ylim=(.05,1.3));ax[0,0].legend(fontsize=8)
ax[0,1].set(title='Test energy error by state',xlabel='State index',ylabel='MAE (eV)');ax[0,1].legend(fontsize=8)
ax[1,0].set(title='Test tensor error by state',xlabel='State index',ylabel='Frobenius RMSE (a.u. squared)');ax[1,0].legend(fontsize=8)
names=list(res['test']);x=np.arange(len(names));v=[res['fits'][n]['best_val'] for n in names];t=[res['test'][n]['L'] for n in names]
ax[1,1].bar(x-.18,v,width=.36,label='Validation');ax[1,1].bar(x+.18,t,width=.36,label='Test')
ax[1,1].set(title='Frozen best checkpoint: validation and test',ylabel='Normalized E+A loss',xticks=x,xticklabels=[labels[n] for n in names]);ax[1,1].tick_params(axis='x',rotation=25);ax[1,1].legend()
for a in ax.flat:a.grid(alpha=.2)
fig.tight_layout();fig.savefig(ROOT/'reports/results_diagnostics.png',dpi=180);plt.close(fig)
print(json.dumps(dict(runs=aud,comparison=comparison),indent=2))
