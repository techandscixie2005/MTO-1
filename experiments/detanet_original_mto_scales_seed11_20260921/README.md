# 原版 DetaNet 与 planned MTO：三个规模、seed=11 完整实验归档

**最终状态：6/6 次正式训练、3/3 个规模的测试与报告全部完成。** 本目录仅包含 2026-09-21～22 这轮独立实验，不混入历史缩小版骨干的实验结果，也不把之前的冒烟/诊断成绩当作正式测试结果。

| 规模 | 模型 | 测试 MSE | 归一化 MSE | MAE | 余弦相似度 | 实际 / 最佳 epoch | 训练小时 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1k | A：DetaNet | 0.000288303271 | 0.4777819 | 0.00926062 | 0.766021 | 1000 / 958 | 0.328 |
| 1k | B：MTO | 0.000258655432 | 0.4286489 | 0.00508280 | 0.836268 | 231 / 79 | 0.098 |
| 10k | A：DetaNet | 0.000294280873 | 0.4897009 | 0.00730541 | 0.841885 | 495 / 343 | 1.121 |
| 10k | B：MTO | 0.000201356699 | 0.3350696 | 0.00378979 | 0.905210 | 238 / 86 | 0.644 |
| full | A：DetaNet | 0.000111856695 | 0.1775715 | 0.00389380 | 0.930602 | 551 / 399 | 15.935 |
| full | B：MTO | 7.76070595e-05 | 0.1232005 | 0.00242126 | 0.947602 | 236 / 84 | 8.244 |

MTO 相对测试 MSE 改善 `100 × (A − B) / A`：1k：10.28%；10k：31.58%；full：30.62%。正值表示 MTO 更好。耗时为训练记录的累计训练/验证/保存墙钟时间，每模型占用一张 GPU，不包含排队时间；它不是受控的单步吞吐基准。

## 文件导航

- [三规模总报告](reports/summary.md) / [机器可读汇总](reports/summary.json)
- 各规模正式报告：[1k](reports/1k/comparison.md)、[10k](reports/10k/comparison.md)、[全量](reports/full/comparison.md)
- [综合分析与适用范围](ANALYSIS_REPORT.md)
- [详细 1k 过程分析](audit/analysis_1k/ANALYSIS.md)（分析写于正式报告补跑前；当时的排队状态为历史记录）
- [同四个分子的跨规模光谱对照](analysis/spectrum_comparison/README.md)
- [冻结实验配置](protocol.json)、[模型实现](models.py)、[训练入口](train.py)、[数据处理](data_protocol.py)、[评估与报告入口](report.py)
- [Slurm 提交与修复记录](jobs/submission.json)、[实际作业记账记录](publication/slurm_accounting.psv)、[最终核验状态](publication/final_status.json)
- [原始运行环境](publication/environment_at_preparation.json)、[原始集群运行说明](publication/RUNNING_ON_CLUSTER.md)、[归档来源与格式说明](publication/EXPORT_NOTES.md)
- [文件完整性清单](publication/archive_manifest.json)、[远端原文件清单](publication/remote_snapshot_manifest.json)、[独立校验工具](publication/verify_archive.py)

## 实验设置与公平性

| 规模定义 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| 1k | 800 | 100 | 100 |
| 10k | 8000 | 1000 | 1000 |
| 全量有效数据 | 103785 | 12973 | 12974 |

只使用 seed=11。A=`detanet_original_uv`，完整原版 DetaNet 直接输出240点谱；B=`detanet_mto_planned`，同一完整 DetaNet 骨干加现有 planned MTO 读出。骨干沿用128通道、maxl=3、3个block、32个trainable Bessel基、8个attention head；原版模型来源见[来源审计](audit/SOURCE_AND_MODEL_AUDIT.md)与保留的[vendor许可证](vendor/LICENSE)。没有训练global-gate，也没有新增消融或种子。

同一规模内使用相同划分、同一目标、相同训练集RMS与训练集原子数统计、相同数据打乱种子；每个规模独立从头训练，沿用共同骨干初始化。所有正式训练均不加载旧实验、作者预训练、冒烟或诊断权重。既有601点目标统一线性插值到作者240点网格；归一化统计仅由各规模训练集计算，详细哈希和数值在 `data/frozen_<scale>.json`。

Adam+AMSGrad，lr=1e-3，weight_decay=0，有效batch=64，FP32，无AMP/TF32。ReduceLROnPlateau沿用factor=.5、patience=50、relative threshold=1e-4、min_lr=1e-6；最多1000 epochs，最少200 epochs后、150轮无改善、至少两次实际降学习率且距最后降学习率至少50轮才允许早停。每个模型单独按验证集选择best checkpoint。本轮没有OOM导致的micro-batch改动。

启动检查只有数据有效性和两个训练分子的短前向/反向有限值检查，随后丢弃检查模型，重新初始化正式模型。没有32分子过拟合成绩门槛，没有伪造PREFLIGHT_PASSED或STAGE_COMPLETE。阶段顺序为1k训练→测试与报告→10k训练→测试与报告→全量训练→测试与总报告；阶段完成不依赖MTO是否胜出。

## 归档内容与校验

`runs/<scale>/seed_11/<variant>/` 包含真实 `short_check.json`、`best.pt`、`last.pt`、`history.json`、`FIT_COMPLETE.json`。共12个checkpoint，保留模型、优化器、调度器、随机数状态和恢复进度。`reports/<scale>/` 包含测试预测NPZ、真实谱、分子ID、240点网格、测试指标、CSV对照表、训练/验证/LR曲线、固定分子叠图和阶段完成记录。原始日志在 `logs/`。

全部准备后的240点数据、划分和冻结统计均随本目录归档。唯一超过普通Git单文件体积限制的 `data/spectra.npy` 被按原始字节流分成无损gzip块；清单在[storage_map.json](publication/storage_map.json)。无需Git LFS或外部权重下载。原始外部601点源数据库没有重复复制，来源与哈希保存在冻结元数据中。

在仓库根目录运行（只依赖Python标准库，不需要GPU/PyTorch）：

```bash
python experiments/detanet_original_mto_scales_seed11_20260921/publication/verify_archive.py
# 若需要恢复训练读取的原始 spectra.npy：
python experiments/detanet_original_mto_scales_seed11_20260921/publication/verify_archive.py --restore-data
```

工具检查发布文件、145个远端原文件、代码/配置fingerprint、冻结数据哈希、6次训练的历史及best/last文件、3个阶段产物和最终报告，并在恢复数据时验证完整原始字节哈希。现有不同内容的缓存不会被覆盖。

原集群Slurm脚本保留绝对路径及Python环境，运行方法见[集群说明](publication/RUNNING_ON_CLUSTER.md)。这是已运行实验的冻结归档；迁移到其他环境时，应在独立目录修改路径与配置，而非篡改本轮fingerprint或完成记录。

## 运行问题与记录

首次1k报告作业1299666在Markdown写入时遇到ASCII编码错误；训练和测试预测已完成。随后仅为报告作业设置 `PYTHONUTF8=1`，补跑作业1299743成功，并修复后续依赖。10k报告1299744、全量报告1299745均成功。没有因此重新训练或调整超参数。完整修复证据在 `audit/utf8_recovery_20260921/`。

`audit/SUBMISSION_STATUS.md`、1k早期分析以及失败报告的部分产物均为当时的真实状态快照，保留供追溯；本页和 `reports/summary.md` 描述最终完成状态。

## 结论边界

这是单种子初步比较，不报告跨种子标准差或显著性。各规模测试集规模及组成不同，不能只比较跨规模均值就推断每个分子的改善；同分子图用于补充观察。MTO包含读出与光谱构造约束，结果不能单独归因于某一个张量模块。源谱展宽及绝对物理强度单位未经确认，MTO保留sigma=0.2 eV假设；不声称论文全过程复现或潜在激发态的物理正确性。
