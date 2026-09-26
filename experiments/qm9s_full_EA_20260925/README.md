# QM9S 全量 E+A：完整 DetaNet / 完整 MTO 实验归档（2026-09-25～26）

**6 个训练正常早停完成；1 个 MTO 重复种子完成194轮后因CUDA错误中断。checkpoint按用户要求全部排除。** 本目录包含实际模型源码、实验设计、超参数、配置、全部准备后的训练输入/标签、冻结划分/统计、七组训练记录、六组全测试集预测、分析图表和环境/来源审计。

参考MTO的测试E+A总损失比DetaNet低2.87%，但配对区间跨零；光谱MSE高约11.0%。不能声称稳定全面优于基线。正式选型仍是验证集最优的mto_reference；不是测试上数值最低的32通道模型。

| 实验 | 最佳/完成epoch | 验证L | 测试L | E MAE/eV | A Frobenius RMSE/au² |
|---|---:|---:|---:|---:|---:|
| DetaNet | 107/259 | 0.576527 | 0.572771 | 0.103804 | 0.266340 |
| MTO参考 | 45/248 | **0.523346** | 0.556327 | 0.102533 | 0.262624 |
| MTO lr=3e-4 | 23/226 | 0.537891 | 0.691439 | 0.109126 | 0.293592 |
| MTO lr=3e-3 | 75/227 | 0.539912 | 0.551158 | 0.101359 | **0.261395** |
| MTO通道32 | 92/244 | 0.531822 | **0.550137** | **0.092886** | 0.262714 |
| MTO batch128 | 33/236 | 0.530437 | 0.610282 | 0.108001 | 0.275042 |
| MTO seed23 | 最优64；完成194轮后失败 | 0.546894 | 未测试 | — | — |

## 导航

- [结果复核与详细分析](ANALYSIS_REVIEWED_ZH.md)，[诊断图](reports/results_diagnostics.png)，[原始指标](reports/results.json)
- [实际执行设计](README_execution.md)，[超参数设计](HYPERPARAMETER_DESIGN.md)，[七组配置](configs/)，[实验矩阵](campaign.json)
- [模型](models_ea.py)、[原始完整上游骨干/MTO](upstream/)、[训练](train_ea.py)、[数据加载](data_ea.py)、[数据准备](prepare_data.py)、[评估](analyze.py)
- [训练日志](logs/)、[逐run训练历史/状态/全测试预测](runs/)、[检查点元数据复核](reports/checkpoint_verification.json)
- [数据划分](data/splits.json)、[归一化](data/normalization.json)、[身份分组](data/identity_audit_v2.json)
- [原始环境快照](reports/environment_freeze.txt)、[上游固定版本清单](upstream_manifest.json)、[来源提取及验证](publication/source_extraction/)
- [归档完整性清单](publication/archive_manifest.json)、[原始服务器文件与checkpoint排除清单](publication/export_inventory.json)、[数据分块映射](publication/storage_map.json)
- [复现说明](publication/REPRODUCE.md)、[初始草案历史](publication/planning_history/README.md)

## 数据和方法

实际TD日志提供133727条有效分子记录，每条10个单重激发态。训练/验证/测试为120355/6686/6686（90/5/5）；通过几何成键推断保守身份分组，278条未解决身份记录仅放训练集。输入原子序数和同一TD日志的坐标，监督E及A=μμᵀ。

训练集固定全局sE²=0.5378066634062587 eV²、sA²=0.1324285377060015 au⁴。L=mean((Ehat−E)²)/sE² + mean(sum_ij(Ahat−A)²)/sA²。无额外光谱损失。完整DetaNet骨干128维、3块、lmax3；MTO保留路由、轨道聚合、参考态CG耦合与PSD解码。DetaNet对照适配相同E/A监督，不能混称为旧的直接240点光谱头。

本轮使用7张A800 80GB，排除已有ECC故障的GPU3。主要环境Python3.10、Torch2.5.1+cu121、e3nn0.4.4、PyG2.7.0。历史原文件保持逐字节，包含旧服务器绝对路径；便携复现方法见下文。

## 完整性和恢复

```bash
python publication/verify_archive.py
python publication/verify_archive.py --restore-data --metrics
```

第一条仅需Python标准库；`--metrics`需要NumPy。16个gzip块会恢复为原始524746908字节的data/dataset.npz，SHA-256为`be8fadca203429575d70642b617730592693be40858dca7189c098a671596330`。每个分块均小于GitHub普通文件限制。恢复包含全部133727条输入、E/A标签、缓存邻接边及分区索引，无需外部数据下载即可重新训练。原始TD文本日志不重复发布，来源和提取验证保留。

checkpoint权重和优化器状态（含preflight_checkpoint.pt，共15个文件）未上传；不能直接复现原模型推理，但可复算已保存的完整测试预测指标，也可从头重新训练。

## 历史状态说明

原先要求7个fit全部完成后测试；实际分析排除了失败seed23，只测试6组，属于收尾协议偏离，已保留在报告中。supervisor_status.json是退出前的历史快照，可能比FIT_COMPLETE.json旧；以各run完成/失败记录和最终publication/final_status.json为准。不删除错误日志，不伪造第七组完成。

初始80/10/10、逐态归一化、三种子计划已被用户修改；它们仅作为planning_history保存，最终设置以本README、configs和冻结data为准。
