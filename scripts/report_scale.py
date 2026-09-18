import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
COLORS = {'baseline': '#476a93', 'mto': '#c96532'}

def make_report(r):
    multi = len(r['scales']) > 1
    title = '、'.join(r['scales']) if multi else next(iter(r['scales']))
    lines = [f'# QM9S：{title} DetaNet / MTO 对比结果', '',
             '两组从头训练，共享主干初始化、数据顺序与训练规则。新版 MTO 保留态条件装配、参考 CG 关系、E／A／f 与可微展宽，仅以光谱 MSE 监督。对照为参数量匹配的 DetaNet 池化加相同物理解码器。配置在新版测试前冻结；历史 1k／10k 测试曾用于旧版结果报告，本次不能视为从未接触的测试集。', '',
             '这里的全量是源 QM9S 经逐行对齐、有效光谱检查及 canonical identity 去重后符合原实验中性闭壳层范围的全部分子。实际纳入数和排除明细见 data/audit.json。', '',
             '| 规模 | Train / Validation / Test | DetaNet 归一化 MSE | MTO 归一化 MSE | 相对改善 | 较好种子数 |',
             '|---|---|---:|---:|---:|---:|']
    for scale, result in r['scales'].items():
        seeds = result.get('seeds', r['seeds'])
        m = result['manifest']; c = result['cohorts']['within_scale']
        cells = [f"{c['aggregate'][arm]['mse']['mean']:.6f} ± {c['aggregate'][arm]['mse']['std_across_seeds']:.6f}" for arm in COLORS]
        lines.append(f"| {scale} ({m['count']}) | " + '/'.join(str(m['splits'][s]) for s in ('train','validation','test')) + ' | ' + ' | '.join(cells) + f" | {c['mto_mse_relative_improvement_percent']:.2f}% | {c['mto_better_test_mse_seed_count']}/{len(seeds)} |")
        out = ROOT / 'results' / scale
        for cohort in ('within_scale', 'common_fresh_test'):
            q = result['cohorts'][cohort]
            keys = ['mse', 'cosine', 'pearson'] if cohort == 'within_scale' else ['source_mse', 'cosine', 'pearson']
            fig, axes = plt.subplots(1,3,figsize=(12,4))
            for ax, key in zip(axes, keys):
                means = [q['aggregate'][arm][key]['mean'] for arm in COLORS]
                errors = [q['aggregate'][arm][key]['std_across_seeds'] for arm in COLORS]
                ax.bar(['DetaNet', 'DetaNet + MTO'], means, yerr=errors, capsize=4, color=list(COLORS.values()))
                ax.set_title(key); ax.spines[['top','right']].set_visible(False)
                ax.margins(y=.16)
                for j,v in enumerate(means):
                    ax.text(j,v+errors[j],f'{v:.5g}',ha='center',va='bottom',fontsize=9)
            fig.suptitle(f"QM9S {scale} | {cohort} N={q['test_molecules']} | {len(seeds)} paired seeds (SD)")
            fig.tight_layout(); fig.savefig(out / f'comparison_{cohort}.png',dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9,5))
        for arm,color in COLORS.items():
            for seed in seeds:
                h = json.loads((ROOT / 'runs' / f'{scale}_seed{seed}' / f'{arm}_history.json').read_text())
                ax.plot([x['epoch'] for x in h], [x['validation']['mse'] for x in h], color=color, alpha=.65, label=arm if seed==seeds[0] else None)
        ax.set(xlabel='Epoch',ylabel='Validation MSE / train RMS squared',title=f'QM9S {scale}: all {len(seeds)} paired seeds')
        ax.legend(); ax.spines[['top','right']].set_visible(False)
        fig.tight_layout(); fig.savefig(out / 'validation_curves.png',dpi=180); plt.close(fig)
        with np.load(out / 'example_spectra.npz') as a:
            fig, axes = plt.subplots(2,2,figsize=(11,7))
            for j,ax in enumerate(axes.flat):
                ax.plot(a['energy_eV'],a['target'][j],color='#333333',label='Source spectrum')
                for arm,color in COLORS.items():
                    ax.plot(a['energy_eV'],a[arm][j],color=color,label=arm)
                ax.set(title=f"Fixed fresh test molecule ID {a['ids'][j]}",xlabel='Energy / eV',ylabel='Source spectral intensity')
                ax.spines[['top','right']].set_visible(False)
            axes[0,0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / 'example_spectra.png',dpi=180); plt.close(fig)
    lines += ['', '± 为跨训练种子的样本标准差。不同规模的训练 RMS 不同，因此跨规模比较应使用下表同一留出集上的源单位误差，不能直接把上表归一化 MSE 当作学习曲线。', '',
              '| 规模 | 共同新留出分子数 | DetaNet 源单位 MSE | MTO 源单位 MSE | MTO 相对改善 |',
              '|---|---:|---:|---:|---:|']
    for scale,result in r['scales'].items():
        c=result['cohorts']['common_fresh_test']
        cells=[f"{c['aggregate'][arm]['source_mse']['mean']:.8g} ± {c['aggregate'][arm]['source_mse']['std_across_seeds']:.3g}" for arm in COLORS]
        lines.append(f"| {scale} | {c['test_molecules']} | " + ' | '.join(cells) + f" | {c['mto_mse_relative_improvement_percent']:.2f}% |")
    lines += ['', '共同留出集排除了原 1k 实验已查看的 100 个测试分子；其与各规模测试集部分重合，不能当作独立重复实验。', '',
              '配对不确定性（MTO−DetaNet，负值表示更好）：']
    for scale,result in r['scales'].items():
        c=result['cohorts']['within_scale']; lo,hi=c['paired_molecule_bootstrap_95ci']
        count = len(result.get('seeds', r['seeds']))
        lines += [f"- {scale}：按分子配对、先平均种子的条件 bootstrap 95% CI = [{lo:.6f}, {hi:.6f}]；{count} 个种子的精确双侧 sign-flip p = {c['paired_seed_sign_flip_two_sided_p']:.4f}（最小可能 p={2/2**count:g}）。"]
    lines += ['', '分子 bootstrap 区间以当前划分和已拟合模型为条件，不能代替训练数据集重采样的不确定性。不能仅凭分子 CI 声称稳定统计显著性。', '',
              '该实验检验固定完整架构随数据量扩展的行为，不代表最优容量或充分调参，也不检验 scaffold 外推。各向同性光谱损失不能唯一确定候选态分解或张量方向；潜在 MTO 不解释为真实轨道。固定 Gaussian σ=0.2 eV 是模型约定，不是对源 CSV 展宽宽度的确认。每个种子的 E／A／f、有效秩及 PSD 诊断随结果导出。', '',
              '全局 RMSE 为整体 MSE 的平方根；逐分子 RMSE 均值另列在 per_seed.csv。训练耗时、优化步数、峰值显存和停止原因保存在 runs/*/*_summary.json。阶段完成时还导出四个预先固定示例的原子基 B、系数 c、装配 F、MTO 张量、CG 关系及物理解码中间量：mto_assembly_examples_seed*.pt。']
    for scale in r['scales']:
        lines += ['', f'![{scale} 对比]({scale}/comparison_within_scale.png)', '',
                  f'![{scale} 验证曲线]({scale}/validation_curves.png)', '',
                  f'![{scale} 固定测试光谱]({scale}/example_spectra.png)']
    if not multi:
        (ROOT / 'results' / f'comparison_{title}_report.md').write_text('\n'.join(lines), encoding='utf-8')
        return
    lines += ['', '![共同留出集规模比较](scaling_common_test.png)']
    fig, axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key in zip(axes,('source_mse','cosine','pearson')):
        for arm,color in COLORS.items():
            x=[result['manifest']['splits']['train'] for result in r['scales'].values()]
            y=[result['cohorts']['common_fresh_test']['aggregate'][arm][key]['mean'] for result in r['scales'].values()]
            e=[result['cohorts']['common_fresh_test']['aggregate'][arm][key]['std_across_seeds'] for result in r['scales'].values()]
            ax.errorbar(x,y,yerr=e,marker='o',capsize=4,color=color,label=arm)
        ax.set(xscale='log',xlabel='Training molecules',title=key); ax.spines[['top','right']].set_visible(False)
    axes[0].legend(); fig.suptitle('Identical fresh holdout | fixed c32 capacity | mean and SD over paired seeds')
    fig.tight_layout(); fig.savefig(ROOT / 'results/scaling_common_test.png',dpi=180); plt.close(fig)
    (ROOT / 'results/comparison_report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__ == '__main__':
    make_report(json.loads((ROOT / 'results/comparison_all.json').read_text()))
