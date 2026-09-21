"""Build actual review bundles from finished bounded diagnostics, never training."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=int,required=True);args=ap.parse_args()
    out=ROOT/f'reports/instrumented_v2_{args.job}'
    summary=json.loads((out/'diagnostic_summary.json').read_text())
    variants=['detanet_original_uv','detanet_mto_planned','detanet_mto_global_gate']
    main={v:json.loads((out/v/'trajectory.json').read_text()) for v in variants}
    diag_dirs={v:ROOT/summary['interventions'][v].get('directory',str((out/v).relative_to(ROOT))) for v in summary['interventions']}
    diag={v:json.loads((p/'trajectory.json').read_text()) for v,p in diag_dirs.items()}
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for name,data in main.items():
        ev=data['evaluations'];axes[0].plot([p['step'] for p in ev],[p['trainer_normalized_mse_fp32'] for p in ev],label=name)
    axes[0].axhline(.1,color='black',ls='--');axes[0].set(title='MAIN smoke: fixed 32 TRAIN molecules',xlabel='Optimizer step',ylabel='Normalized MSE',yscale='log')
    for name,data in diag.items():
        rows=data['step_statistics']
        axes[1].plot([p['step']-601 for p in rows]+[300],
                     [p['loss_before_update'] for p in rows]+[data['last']['trainer_normalized_mse_fp32']],label=name)
    axes[1].set(title='DIAGNOSTIC_ONLY: copies of A step600',xlabel='Additional optimizer steps',ylabel='Normalized MSE',yscale='log')
    for ax in axes:ax.legend(fontsize=7)
    fig.savefig(out/'main_and_diagnostic_losses.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,data in diag.items():
        rows=data['step_statistics'];steps=[r['step']-600 for r in rows]
        for ax,key in zip(axes.flat,['gradient_l2','update_l2','prediction_rms_before_update','mean_max_v_over_mean_v']):
            ax.plot(steps,[r[key] for r in rows],label=name);ax.set(xlabel='Additional steps',ylabel=key,yscale='log')
    axes.flat[0].legend();fig.suptitle('DIAGNOSTIC_ONLY: actual gradients, updates and optimizer moments')
    fig.savefig(out/'optimizer_diagnostic_traces.png',dpi=160);plt.close(fig)
    audit=json.loads((out/'data_loss_audit.json').read_text())
    scales=json.loads((out/'initial_A_scales.json').read_text())
    init=json.loads((out/'initialization_audit.json').read_text())
    correction=summary.get('D1_correction',{})
    lines=['# Instrumented v2 smoke and bounded diagnostics','',f'Actual main Slurm job: {args.job}. Correction metadata: {json.dumps(correction)}.',
           'Main policy: fixed seed 11, the same 32 TRAIN IDs and [32,240] targets, common original 800-training RMS, 2000-step maximum and original early-success thresholds. The v1 600-step cap was a human engineering budget, not an author-paper rule.',
           'models.py, train.py and protocol.json are byte-identical to the preceding authorized v2. preflight.py now invokes observation/checkpoint/diagnostic code; this instrumentation is documented separately from the earlier budget change.',
           '', '## Data and actual fresh initialization',
           f"Training RMS: {audit['train_rms']:.16g}; recomputed from the original 800 training targets: {audit['train_rms_recomputed_float64']:.16g}.",
           f"Zero-spectrum normalized MSE on these 32 molecules: {audit['zero_spectrum']['normalized_mse']:.10g}. Pointwise common-mean spectrum MSE: {audit['pointwise_common_mean_spectrum']['normalized_mse']:.10g}.",
           f"Actual fresh A tensors equal fresh independent author constructor: {init['every_fresh_A_state_tensor_equal']}; initialization call sequence equal: {init['initialization_call_sequence_equal']}; constructor RNG equal: {init['constructor_rng_equal']}.",
           'No same-state_dict loading is used for that fresh-constructor audit. Original author constructor resets are retained. All arrays assert exact [32,240] shape; ordered labels are checked; normalized_MSE=raw_MSE/train_RMS² is independently verified.',
           '', '| Initial A quantity | RMS | Minimum | Maximum |','|---|---:|---:|---:|']
    for key,v in scales['tensors'].items():lines.append(f"| {key} | {v['rms']:.8g} | {v['minimum']:.8g} | {v['maximum']:.8g} |")
    lines+=['','## Main smoke — these alone determine the smoke gate',
            '| Branch | Initial MSE | Best MSE | Best step | Steps executed | Passed |','|---|---:|---:|---:|---:|---|']
    for name,data in main.items():lines.append(f"| {name} | {data['initial']['trainer_normalized_mse_fp32']:.9g} | {data['best']:.9g} | {data['best_step']} | {data['steps']} | {data['passed']} |")
    lines+=['','All existing initial, step600, best and last checkpoints were strictly reloaded, including optimizer state; their losses were recomputed from the real saved weights. CPU/GPU roundoff differences are retained in checkpoint_revalidation.json. Smoke/diagnostic checkpoints have formal_initialization_allowed=false.',
            '', '## DIAGNOSTIC_ONLY: original last-linear-layer SVD',
            'Earlier layers are frozen in independent CPU float64 copies. Phi=sum_i h_i; X=[Phi,N] has shape [32,129]. The last column is atom count N. Minimum-norm rank-truncated SVD is used, with tolerance max(m,n)*eps*smax. Solutions are written into the original final per-atom linear layer and checked by full forward.',
            '| Source A checkpoint | Rank | Condition | Solution norm | SVD normalized MSE | Full-forward normalized MSE |','|---|---:|---:|---:|---:|---:|']
    for name,v in summary['linear'].items():lines.append(f"| {name} | {v['numerical_rank']} | {v['condition_number']:.9g} | {v['solution_frobenius_norm']:.9g} | {v['svd_matrix']['normalized_mse']:.9g} | {v['full_forward']['normalized_mse']:.9g} |")
    lines+=['','## DIAGNOSTIC_ONLY: AMSGrad interventions',
            'The first D1 run was invalidated: loading a reused optimizer dictionary shared a CPU step tensor, which D0 incremented from 600 to 900. The main smoke, SVD, D0 and D2 were unaffected. D1 alone was rerun from the immutable disk checkpoint with a deep copy, first/last optimizer counters 601/900, and explicit source-state immutability checks. Invalid records and both executed source versions are preserved; this is a diagnostic engineering fix, not a mere budget extension.',
            'All copies begin with identical A step600 model tensors and the same training batch. D0 retains every optimizer state. D1 changes only max_exp_avg_sq to exp_avg_sq and is a nonstandard intervention. D2 starts a new same-configuration AMSGrad; it resets first/second moments and bias-correction history, so its effect cannot be attributed solely to the historical maximum.',
            '| Diagnostic | Starting MSE | Best MSE | Last MSE | Additional steps | First update L2 | Last update L2 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,data in diag.items():
        rows=data['step_statistics'];lines.append(f"| {name} | {data['initial']['trainer_normalized_mse_fp32']:.9g} | {data['best']:.9g} | {data['last']['trainer_normalized_mse_fp32']:.9g} | {data['steps']} | {rows[0]['update_l2']:.9g} | {rows[-1]['update_l2']:.9g} |")
    lines+=['','![Main and isolated diagnostic losses](main_and_diagnostic_losses.png)',
            '', '![Actual optimizer traces](optimizer_diagnostic_traces.png)',
            '', '## Boundaries',
            'These diagnostics use no validation/test examples and cannot substitute for a passed main smoke gate. Neither an overparameterized 32-row linear fit nor a state-reset intervention establishes generalization. There is no change to original DetaNet structure/initialization/output scale or planned MTO. No formal fits or test predictions are implied by DIAGNOSTICS_COMPLETE.',
            'Source broadening remains unknown; sigma=0.2 eV remains a disclosed assumption. Existing 601-point targets were commonly adapted to the author 240-point grid. This is not the paper full-data training protocol.',
            '', '## Bundle contents',
            'The evidence ZIP contains actual executed source/configuration, smoke and diagnostic JSON, all acceptance JSON and logs, plots, initialization activations, SVD matrices/solutions, and SHA256 manifests. A separate checkpoint ZIP contains the real initial/step600/best/last optimizer checkpoints for primary smoke and D0/D1/D2. The historical failed records are preserved separately in history and summarized in the evidence bundle.']
    (out/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n')
    source=[]
    for pattern in ['*.py','protocol.json','configs/*.json','planning/*.md','SOURCE_AND_MODEL_AUDIT.md','PLAN_IMPLEMENTATION_TEST_MAP.md','jobs/*.slurm','jobs/submission.json','tests/*.py','vendor/**/*.py','reference/**/*.py','reference/**/LICENSE*','reference/LICENSE*']:
        source.extend(ROOT.glob(pattern))
    source += [ROOT/'audit/frozen_code_sha256.json',ROOT/'audit/instrumentation_change_record.json',ROOT/'audit/reference_sha256.json',ROOT/'audit/D1_isolation_bug.json',ROOT/'audit/observer_empty_state_bug.json',ROOT/'audit/rerun_D1_isolated.py',ROOT/'RUN_STATUS.md']
    source += list((ROOT/'logs').glob('acceptance*'))+list((ROOT/'logs').glob('diagnostic_contracts*'))
    source += [ROOT/f'logs/instrumented-{args.job}.log',ROOT/f'logs/environment-{args.job}.json',ROOT/f'logs/gpu-{args.job}.csv',ROOT/'reports/resume_cpu_gpu.json',ROOT/'reports/decoder_oracle.json']
    for old in ['v1_preflight_1298973','v2_preflight_1299137']:
        source += [ROOT/f'history/{old}/RUN_STATUS.md',ROOT/f'history/{old}/protocol.json',ROOT/f'history/{old}/reports/smoke_seed11.json']
    source += list((ROOT/'data').glob('*'))
    source += [ROOT/'reference/trained_param/qm9spectra/borden_os.pth']
    checkpoint_paths=[]
    extra_roots={p.parent for p in diag_dirs.values() if p.parent != out}
    extra_roots.update((ROOT/'reports').glob('D1_correction_*'))
    for p in list(out.rglob('*'))+[q for extra in extra_roots for q in extra.rglob('*')]:
        if not p.is_file():continue
        if p.suffix=='.pt' and p.parent.name in variants+['D0','D1','D2']:
            if p.parent in [out/v for v in variants]+list(diag_dirs.values()):checkpoint_paths.append(p)
        else:source.append(p)
    if correction:
        cj=correction['job_id']
        source += [ROOT/f'logs/D1-correction-{cj}.log',ROOT/f'logs/environment-{cj}.json']
    source += list((ROOT/'logs').glob('D1-correction-*.log'))
    source.append(Path(__file__))
    source=sorted(set(p for p in source if p.is_file()))
    delivery=ROOT/'deliverables';delivery.mkdir(exist_ok=True)
    for name,paths in [('evidence',source),('checkpoints',checkpoint_paths)]:
        target=delivery/f'detanet_mto_v2_diagnostics_{args.job}_{name}.zip'
        assert not target.exists(),'Do not overwrite an earlier delivered bundle'
        manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in paths:z.write(p,str(p.relative_to(ROOT)))
            z.writestr('BUNDLE_SHA256.json',json.dumps(manifest,indent=2))
            z.writestr('BUNDLE_SCOPE.txt',f'Job {args.job}; {name}. Training/smoke diagnostics only. No formal-test success is asserted.\n')
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
            for name_in_zip,digest in manifest.items():assert hashlib.sha256(z.read(name_in_zip)).hexdigest()==digest
        print(json.dumps(dict(bundle=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(paths))),flush=True)


if __name__=='__main__':main()
