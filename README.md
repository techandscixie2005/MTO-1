# MTO-1

## 最新归档：原版 DetaNet UV–Vis / 规划 MTO / global-gate（2026-09-20～21）

本次使用作者 **原版 DetaNet UV–Vis 模型**作为 A：128 通道、maxl=3、3 个 block、32 个 trainable Bessel 径向基、8 个 attention head；保留原版逐原子 `128→128→240` MLP、初始化与原子求和读出。B/C 共用同一完整骨干，分别接规划 MTO 和 global-gate 诊断消融。作者源码固定于 `4f92e643ab64651b91c4a1392cf389ddfd0d89f0`，许可证、独立 reference、数值等价日志和严格 checkpoint 加载记录均已归档。

**当前是工程验收与训练子集冒烟/诊断结果，不是已完成的 1k 泛化对照。** 同一 32 个训练分子、seed=11、最多 2000 步主冒烟：A 最佳归一化 MSE **0.2596253（未过门槛）**，B **0.0757229（通过）**，C **0.0749934（通过）**。正式五种子 × 三分支 **0 次提交、0 次完成**；无本轮正式测试结果，无 `STAGE_COMPLETE`。不能据此声称 MTO 的泛化性能更好。

- [本次完整说明、模型/协议/结果与文件导航](experiments/detanet_original_mto_1k_20260920/README.md)
- [诊断结论与真实作业状态](experiments/detanet_original_mto_1k_20260920/DIAGNOSTIC_CONCLUSIONS.md)
- [原版来源、兼容适配和逐层等价审计](experiments/detanet_original_mto_1k_20260920/SOURCE_AND_MODEL_AUDIT.md)
- [主冒烟与优化器诊断曲线](experiments/detanet_original_mto_1k_20260920/reports/instrumented_v2_1299173/DIAGNOSTIC_REPORT.md)

源目标经过共同 601→240 网格适配，源展宽尚未证实，MTO 的 sigma=0.2 eV 是披露的假设；作者 GitHub 版本与论文归档尚未验证逐字一致。因此本次称“原版模型代码/架构等价、目标网格适配”，不称论文全过程复现。下列旧实验的模型、参数量和完成标记仅适用于历史版本，与本次不同口径。

## 历史归档：旧模型及 1k / 10k / full 结果

QM9S 光谱预测：DetaNet 与状态条件化 MTO 的完整模型和实验代码。

## 模型

元素与几何 → DetaNet → 状态条件化 MTO → 参考态／目标态 CG 耦合 → 逐态激发能与跃迁强度张量 → 振子强度 → 可微高斯展宽 → 光谱。

仅使用训练集 RMS 归一化后的光谱 MSE 监督，不使用激发能、张量或振子强度标签损失。逐态物理因子是光谱监督下的潜变量。

| 模块 | 对照模型 | MTO 模型 |
|---|---:|---:|
| DetaNet 骨干 | 77,056 | 77,056 |
| 状态条件化 MTO | 0 | 54,160 |
| 参考态／目标态 CG 耦合 | 0 | 28,672 |
| 物理解码头 | 102,001 | 19,084 |
| 总参数量 | 179,057 | 178,972 |

对照头隐藏宽度为 229，由总参数匹配规则决定；MTO 头宽度为 128，各目标态共享解码权重。对照模型一次输出全部 10 个态。因此本实验比较相近参数预算下的两套模型，不是严格保持解码器不变、只插入 MTO 的消融。

## 文件导航

- `src/models.py`：完整 MTO、CG 耦合、物理解码器和对照模型。
- `src/detanet_backbone/`：随仓库提交的完整 DetaNet 骨干代码及原许可证，不需要外部源码目录。
- `src/dataset.py`、`src/scale_dataset.py`：数据读取与指标。
- `scripts/`：训练、评估、逐态导出、报告、前置检查和 Slurm 顺序调度。
- `tests/`：模型结构、工作流和前置检查回归测试。
- `configs/`：冻结的科学配置、执行配置及源文件哈希。
- `data/`：划分 ID、数据清单和审计记录。
- `audit/`：结构和参数核算脚本。
- `docs/`：参数配置、环境包版本和源码完整性清单。
- `results/`：三个规模的汇总结果及完成标记。
- `EXPERIMENT_PLAN.md`：实验设计；`backbone_provenance.json`：骨干来源和修改说明。

## 环境与检查

`requirements.txt` 记录服务器实际安装的主要依赖版本，完整环境快照见 `docs/environment-packages.json`。PyTorch、torch-scatter 和 torch-cluster 的二进制包需与本机 CUDA/PyTorch 版本匹配。

在 Linux 环境从仓库根目录执行：

```bash
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
python audit/parameter_breakdown.py
```

## 复现实验

原始实验按 1k → 10k → 全量顺序完成；1k 和 10k 各 5 对种子，全量 3 对种子。实际每规模种子以 `configs/execution.json` 为准。

本仓库保留已运行的代码和冻结配置。Slurm 脚本、数据准备与部分审计脚本保留原集群路径；不是开箱即用的跨机器调度配置。迁移运行前，应在独立工作目录配置数据路径、Slurm 分区和 Python 环境，并生成新的冻结协议，保留本次原始协议以便追溯。`protocol.py` 校验源码、数据路径、大小及修改时间，直接使用另一台机器的数据文件不会通过原协议校验。

模型实现的所有源代码、三个规模的完整实验结果、最佳及末次模型检查点、逐分子预测和训练日志均随 Git 提交。原始 QM9S 数据集不随 Git 提交。运行训练仍需准备 `data/qm9s_full.npz`，数据校验值见冻结配置及数据清单。

## 第三方代码

DetaNet 第三方代码的许可证保留在 `src/detanet_backbone/LICENSE`，来源及适配记录见 `backbone_provenance.json`。


归档完整性可独立验证，无需 PyTorch：`python audit/verify_archive.py`。它核对冻结源码和服务器结果文件的 SHA-256。

