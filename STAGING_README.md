# MTO-1 本地暂存包 — `qm9s_pouter_trace_20260927`

本地暂存目录，**尚未上传到 GitHub**。内容为 `qm9s_pouter_trace_20260927` 实验的设置、
代码、日志、训练历史、报告，以及本次会话新增的独立分析脚本与结果。

- 暂存时间：2026-09-28 01:52 (+08:00)
- 来源：`inspur@USTC-A800:/home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927`
- 规模：130 个文件，约 12 MB
- 校验：`SHA256SUMS`（129 条，不含自身与 `STAGING_README.md`），已全部核验通过

## ⚠️ 实验仍在运行中

暂存时间点各组进度（四组均 `state=RUNNING`，监督器 PID 365730）：

| 组 | 已跑轮次 | best_val（各组自身目标） | best_epoch |
|---|---:|---:|---:|
| G1 | 93 | 0.539437 | 25 |
| G2 | 91 | 0.158231 | 26 |
| G3 | 95 | 0.572582 | 92 |
| G4 | 95 | 0.152948 | 91 |

设计下限为 200 轮，早停需 bad_epochs≥150、降率≥2 次、距末次降率≥50 轮。
**日志、`history.jsonl`、`status.json` 是运行中的瞬时快照**，不是最终结果。
测试集尚未打开（`evaluate.py` 要求四组全部 `FIT_COMPLETE`）。

## 目录内容

```
experiments/qm9s_pouter_trace_20260927/
  README.md                 实验设计全文（架构、目标、预检、资源、评估口径）
  campaign.json             分组、分类、选择规则、已知限制
  configs/G1..G4.json       四组完整配置
  *.py                      训练/评估/监督代码：trainer objective model_factory
                            dataset evaluate supervisor control preflight prepare
                            seal_submit check_running preflight_system
  frozen_reference/         固定参考提交 71e575f 的 MTO/DetaNet 实现（按 Git blob 校验）
  pinned_reference/         前一轮 qm9s_eta_Ef_20260926 的固定实现
  runs/G1..G4/              history.jsonl status.json run_manifest.json
                            launch_receipt.json current_order.json   ← 无 checkpoint
  logs/                     G1..G4.log supervisor.log preflight.log system_preflight.log
  reports/                  预检、数据审计、初始化、GPU 健康、来源校验、执行状态
  prepare.log
analysis/                   本次会话新增的独立复核脚本与结果（见下）
```

## `analysis/` — 本次会话新增

这些**不是**实验原生产物，是为回答"振子强度 R² 是多少 / P-only 结果预示什么"
而另写的只读脚本，**只用验证集，未打开测试集**。

| 文件 | 作用 |
|---|---|
| `r2_val.py` | 按 `evaluate.py` 的 R² 定义，在验证集上算四组的振子强度 R²（三种 E 口径） |
| `r2_val.json` / `reports_r2_val.json` | 上述结果（两份内容相同） |
| `rank_diag.py` | 检验真值标签 A 是否秩一，并对比 G1/G3 预测的秩结构 |
| `rank_diag.json` | 上述结果 |
| `mu_diag.py` | 把 `tr(A)` 拆成幅度 `\|μ\|` 与方向 `cos`，并测 `log\|μ\|` 的可学习性 |
| `mu_diag.json` | 上述结果 |

结论摘要见同目录 `ANALYSIS_POUTER_ZH.md`。

## 明确排除的内容

| 排除项 | 原因 |
|---|---|
| `runs/*/best.pt`、`runs/*/last.pt` | 用户指定不上传 checkpoint |
| `data/`（637 MB） | 数据集，仓库既有 `data_audit.json`/哈希可追溯 |
| `initial/original.pt`、`initial/p_outer.pt` | 冻结初值权重，属于 checkpoint |
| `env/` | 本地 Python 虚拟环境；`reports/environment.json` 已记录 pip freeze |
| `__pycache__/` | 编译缓存 |
| `*.lock`、`*.tar.gz` | 运行锁；空/中间归档 |

**注意**：没有 checkpoint 意味着本包**不能直接复现训练**——需要 `initial/*.pt`
（由 `prepare.py` 生成）和 `data/`。`reports/initialization.json` 记录了初值的
全部哈希与标定过程，可用于核验。

## 复现方式

原始实验目录仍在服务器上，恢复与查看：

```bash
ssh USTC-A800
cd /home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927
PY=/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python
$PY control.py status
```

查看本包校验：

```bash
sha256sum -c SHA256SUMS
```
