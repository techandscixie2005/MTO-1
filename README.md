# MTO-1

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

大体积 QM9S 数据、模型检查点和逐分子预测文件不随 Git 提交；模型实现的所有源代码均提交。运行训练仍需准备 `data/qm9s_full.npz`，数据校验值见冻结配置及数据清单。

## 第三方代码

DetaNet 第三方代码的许可证保留在 `src/detanet_backbone/LICENSE`，来源及适配记录见 `backbone_provenance.json`。
