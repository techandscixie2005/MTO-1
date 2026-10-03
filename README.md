# MTO-1

## 当前结果：单模型振子强度研究（2026-10-03）

**QM9S v2 当前验证参考为原始 MTO（seed 11、epoch 45），pooled raw-f R² = 0.447169；目标 0.60 尚未达到。** 部署使用一个几何输入模型、一个自包含 checkpoint，不平均多个模型或 checkpoint 的预测。

### Round08 完成：原始 / 局部 / 邻居张量传递

三组按冻结设置完成 **60 epoch / 112,860 次更新**，使用共同全新初始化、TRAIN 统计和原 LE+Ls；保留全部 **66,860 个有效验证标签**，包括零值。

| 单模型 | 验证选中 epoch | 选中 checkpoint 的 pooled raw-f R² | 固定 epoch 60 R² |
|---|---:|---:|---:|
| 原始 MTO | 36 | 0.417636170 | 0.40070541 |
| 局部张量传递 | 30 | 0.433923614 | 0.39583651 |
| 邻居张量传递 | 23 | 0.443968357 | 0.40861597 |

邻居组比同期原始 / 局部对照高 **0.026332 / 0.010045**，但比保留参考低 **0.003201**，未通过同时超过两个同期对照及保留参考各 **+0.003 R²** 的冻结门槛。**关闭本轮，保留 Round05 原始 MTO epoch45 的 0.447169401；不延长、不追加确认种子、不调整系数/尺度/学习率，也不进行预测平均。**

邻居组降低误报亮态误差，但对原始模型的 MAE、能量和亮尾误差变差；相对局部对照的权衡有好有坏。全部六个选中 / 固定 epoch60 的配对分子组件 bootstrap 区间均跨零，且条件于复用验证集和 checkpoint 选择，不包含训练/种子不确定性。传递分支在全量训练中已经活跃，不能用准备阶段的小作用量断言分支未工作；这仍不证明物理响应机制或因果诊断。完整逐态、亮尾、轨迹与活跃性证据见[Round08 报告](research/single_model_20260929/round08_transport_complete/round08_transport_preparation/completion/ROUND08_REPORT.md)、[精确复现命令](research/single_model_20260929/round08_transport_complete/round08_transport_preparation/completion/ROUND08_REPRODUCE.md)、[独立科学审查](research/single_model_20260929/round08_transport_complete/round08_transport_preparation/completion/INDEPENDENT_SCIENTIFIC_REVIEW.md)和[最终决定](research/single_model_20260929/round08_transport_complete/current_state/ROUND08_DECISION.md)。**TEST 仍封存；保留参考没有独立种子或 TEST 确认，0.60 目标未达到。**

258 项源码、依赖及审计记录、九次一次性 TRAIN 技术更新和操作修正保留在[准备协议](research/single_model_20260929/round08_transport_preparation/round08_transport_preparation/PROTOCOL.md)、[准备报告](research/single_model_20260929/round08_transport_preparation/round08_transport_preparation/PREPARATION_REPORT.md)、[准备独立审查](research/single_model_20260929/round08_transport_preparation/round08_transport_preparation/INDEPENDENT_PREPARATION_REVIEW.md)和[准备接受决定](research/single_model_20260929/round08_transport_preparation/current_state/ROUND08_PREPARATION_ACCEPTANCE.md)。[信息流审计](research/single_model_20260929/round08_transport_preparation/bottleneck_audit_20261002/INFORMATION_FLOW_AUDIT.md)与[既往容量和训练配方审计](research/single_model_20260929/round08_transport_preparation/post_round07_bottleneck_audit/HISTORICAL_CAPACITY_AUDIT.md)保留其证据边界。

发布后仅获准在不改动原始 MTO 的前提下，提出一个预先指定的正则化对照方案：明确衰减语义、参数组/排除项、单一固定强度及同期零衰减对照。验证后期恶化本身不证明过拟合；方案须独立审查，尚未授权新的实现、数值实验、拟合或 TEST 评分。

### Round07 完成：共享 PSD 合同变换对照

原始 / 标量 / 张量三组按冻结设置完成 **60 epoch / 112,860 次更新**，使用共同全新初始化、TRAIN 统计和原 LE+Ls；全部 66,860 个有效验证标签均保留，包括零值。

| 单模型 | 验证选中 epoch | 选中 checkpoint 的 pooled raw-f R² | 固定 epoch 60 R² |
|---|---:|---:|---:|
| 原始 MTO | 52 | 0.409104081 | 0.3845332 |
| 共同标量缩放 | 49 | 0.429339013 | 0.3937752 |
| 共享张量合同变换 | 34 | 0.418755177 | 0.3277767 |

张量组比同期原始对照高 **0.009651**，但比标量对照低 **0.010584**，比保留的 Round05 epoch45 参考低 **0.028414**，未通过同时超过三个参考各 **+0.003 R²** 的冻结门槛。**关闭本轮，保留 v2 参考 0.447169401；不延长、不调整系数/边界、不追加确认种子。** 标量是本轮最强模型，但未替换保留参考；不进行预测平均。

张量组降低误报亮态 SSE，但对两个同期对照的 MAE、能量误差及 q99 误差变差。门控活跃且出现各向异性上下文，不代表已识别物理跃迁方向或因果机制。选中 checkpoint 的配对分子组件 bootstrap 区间跨零；固定 epoch60 的张量比较区间低于两个对照。它们仅描述复用验证集上的条件不确定性，不包含训练/种子不确定性或独立确认。完整逐态、亮尾、门控和轨迹证据见[Round07 报告](research/single_model_20260929/round07_congruence_complete/round07_congruence_preparation/completion/ROUND07_REPORT.md)、[精确复现命令](research/single_model_20260929/round07_congruence_complete/round07_congruence_preparation/completion/ROUND07_REPRODUCE.md)、[独立科学审查](research/single_model_20260929/round07_congruence_complete/round07_congruence_preparation/completion/INDEPENDENT_SCIENTIFIC_REVIEW.md)和[最终决定](research/single_model_20260929/round07_congruence_complete/current_state/ROUND07_DECISION.md)。**TEST 仍封存，0.60 目标未达到。**

185 项源码、依赖及审计记录与九次一次性 TRAIN 技术更新保留在[准备协议](research/single_model_20260929/round07_congruence_preparation/round07_congruence_preparation/PROTOCOL.md)、[准备报告](research/single_model_20260929/round07_congruence_preparation/round07_congruence_preparation/PREPARATION_REPORT.md)、[准备独立审查](research/single_model_20260929/round07_congruence_preparation/round07_congruence_preparation/INDEPENDENT_PREPARATION_REVIEW.md)和[准备接受决定](research/single_model_20260929/round07_congruence_preparation/current_state/ROUND07_PREPARATION_ACCEPTANCE.md)。后续 Round08 已按单独授权完成训练并关闭，结果见上节；没有新的训练或 TEST 评分授权。

### Round06 完成：原始 PSD 的损失函数对照

两组使用相同全新初始化、TRAIN 统计及固定设置，均完成 **60 epoch / 112,860 次更新**；只改变原始 PSD 模型的强度损失。验证保留全部 66,860 个有效标签，包括零值。

| 目标函数 | 验证选中 epoch | 选中 checkpoint 的 pooled raw-f R² | 固定 epoch 60 R² |
|---|---:|---:|---:|
| 原始 LE+Ls | 49 | 0.404797567 | 0.389052216 |
| LE+归一化 raw-f MSE | 18 | 0.422028167 | 0.393659486 |

raw-f 候选比同期对照高 **0.017231**，但比保留的 Round05 原始 MTO epoch45 参考低 **0.025141**，未通过“同时超过两个参考各 +0.003 R²”的冻结门槛。**关闭本轮，保留 v2 参考 0.447169401，不延长、不调整系数/学习率、不追加确认种子。** 候选的 MAE、能量误差及真实亮态 q90/q99 RMSE 变差，误报亮态 SSE 降低；完整权衡见[Round06 报告](research/single_model_20260929/round06_objective_complete/round06_objective_preparation/completion/ROUND06_REPORT.md)、[精确复现命令](research/single_model_20260929/round06_objective_complete/round06_objective_preparation/completion/ROUND06_REPRODUCE.md)、[独立科学审查](research/single_model_20260929/round06_objective_complete/round06_objective_preparation/completion/INDEPENDENT_SCIENTIFIC_REVIEW.md)和[最终决定](research/single_model_20260929/round06_objective_complete/current_state/ROUND06_DECISION.md)。

配对分子组件 bootstrap 区间跨零，且仅针对复用验证集和已选 checkpoint，不包含训练/种子不确定性。同 seed11 原始对照与 Round05 存在明显差异；执行图及并发工作量不同，但原因未确定，不能把控制组波动当作独立确认或物理机制。[重复对照的比较边界](research/single_model_20260929/round06_objective_complete/round06_objective_preparation/completion/CONTROL_REPEAT_CONTEXT.md)。**TEST 仍封存，没有本轮 TEST 评分或独立种子确认，0.60 目标未达到。**

准备期的 134 文件源闭包、6 次一次性 TRAIN 技术更新及失败记录保留在[冻结协议](research/single_model_20260929/round06_objective_preparation/round06_objective_preparation/PROTOCOL.md)、[准备报告](research/single_model_20260929/round06_objective_preparation/round06_objective_preparation/PREPARATION_REPORT.md)、[准备独立审查](research/single_model_20260929/round06_objective_preparation/round06_objective_preparation/INDEPENDENT_PREPARATION_REVIEW.md)和[准备接受决定](research/single_model_20260929/round06_objective_preparation/current_state/ROUND06_PREPARATION_ACCEPTANCE.md)。后续 Round07 共享 PSD 合同变换已按单独授权完成训练并关闭，结果见上节。

### Round05 完成：新划分上的四组从头训练

四组均使用全新初始化和仅新 TRAIN 统计，按冻结协议完成 60 epoch、112,860 次更新。QM9S v2 的 TRAIN / VALID / TEST 为 120,355 / 6,686 / 6,686；这是按已审计保守分组规则得到的历史暴露数据新分区，**不是外部新留出集**。验证选择保留全部 66,860 个有效标签，包括零值。

| 单模型 | 验证选中 epoch | 选中 checkpoint 的 pooled raw-f R² | 固定 epoch 60 R² |
|---|---:|---:|---:|
| **原始 MTO（v2 参考）** | **45** | **0.447169401** | **0.402691932** |
| 右侧共享 F | 41 | 0.415020369 | 0.401177943 |
| 原始 M 弱去相关 | 23 | 0.412399193 | 0.392013476 |
| 两者联合 | 20 | 0.424966054 | 0.402391091 |

所有修改组均未通过预先规定的相对原模型 **+0.003 R²** 分配门槛；本轮关闭，不延长训练或追加确认种子。F 的部分亮尾改善不能抵消 pooled 误差变差。完整逐态、亮尾、固定 epoch 60 对照及不确定性见[Round05 报告](research/single_model_20260929/round05_scratch_complete/round05_scratch_preparation/completion/ROUND05_REPORT.md)；[Round05 精确复现命令](research/single_model_20260929/round05_scratch_complete/round05_scratch_preparation/completion/ROUND05_REPRODUCE.md)；[独立科学审查](research/single_model_20260929/round05_scratch_complete/round05_scratch_preparation/completion/INDEPENDENT_SCIENTIFIC_REVIEW.md)；[最终决定](research/single_model_20260929/round05_scratch_complete/current_state/ROUND05_DECISION.md)。

新 TEST 目标保持封存，**没有本轮 TEST 评分或独立种子确认**。按分子组件 bootstrap 的区间仅描述这些已选定 checkpoint 在复用验证集上的不确定性，不包含选模或跨种子不确定性。不能把本轮数字与下面旧划分的校准结果直接比较并宣称提升。

v2 参考的服务器 checkpoint：`/home/inspur/MTO-1/research/single_model_20260929/round05_scratch_preparation/runs/control/geometry_best.pt`；SHA256：`e71c63da8bb3b8214e014ca64946fecab97fbc210cb068c0b1a3eefa3bbf8f1e`。权重及预测数组保留在服务器，不随本次轻量归档发布。

### 旧划分的单模型历史结果

旧划分仍保留 eta0（seed 11、epoch 33）及其固定仿射校准，作为单独的历史证据。

主指标把所有有效“分子 × 激发态”的原始振子强度 f 展平，计算 `R² = 1 − SSE/SST`，不是逐态 R² 的平均值，也不是展宽光谱分数。旧验证集含 6,686 个分子、66,860 个有效标签，包括零值。

| 单模型配方 | 旧验证集 pooled raw-f R² | 已暴露的历史测试 R² | 证据范围 |
|---|---:|---:|---|
| 原生 eta0 | 0.405294 | 0.455815 | 单模型基线 |
| **eta0 + 固定仿射校准（保留）** | **0.418119** | **0.459667** | 校准在整个旧验证集拟合后的分数 |
| Round03：内部 TRAIN 留出来源的仿射校准 | 0.416375 | 未评估 | 系数先冻结，再评估同一旧验证集 |

历史校准 OOF R² 约 **0.416658**；它与全验证集重拟合的 0.418119 是不同证据，且仍继承此前模型/检查点选择的影响。上述测试分数为已有历史结果的引用；本次 Round01–04 没有重新评分测试集。旧验证集已反复用于研究，**尚无独立的新测试确认**。保留配方的 MAE 和部分亮态尾部误差并非最优，逐态和尾部权衡见报告。

### 已完成的受控实验

| 轮次 | 对照与结论 |
|---|---|
| Round01 | 全模型继续训练 20 epoch；对照、右侧共享等变 F、原始 M 弱去相关、两者联合，四组均选择 epoch 0，没有原生 f-R² 提升。 |
| Round02 | 冻结原模型，仅训练 F；原损失 / 原生 f 损失的最佳 R² 为 0.405408 / 0.405388，未达到预定 +0.003 门槛。 |
| Round03 | 从干净随机初始化训练内部 TRAIN 的 80% 来源模型；留出来源仿射映射达到 0.416375，优于匹配的样本内映射 0.402024，但未超过保留配方。部署仍只使用一个完整 eta0。 |
| Round04 | 两个来源使用相同固定四系数分段线性映射，各拟合一次，统一验证一次。样本内 / 留出来源 R² 为 0.398331 / 0.411775，分别低于其仿射对照 0.003693 / 0.004599；关闭此固定配方，不扩展或追加种子。 |

Round04 的拟合误差下降，但验证表现未改善；不能据此认定所有非线性模型都会失败。预定的逐态、亮尾及误报亮态分析完整保留，没有通过删除标签提高分数。

提交 `69bf39f` 的 F5 验证 / 测试 R² 为 0.487700 / 0.508625，使用五模型预测平均，**不符合当前单模型目标**。以下旧实验的光谱 MSE、损失与 ensemble 结果均保留其原始口径，不能与本节 raw-f R² 混用。

### 复现、checkpoint 与后续数据边界

- [单模型完整报告、基线推理代码及配置](research/single_model_20260929/round03_transfer_complete/round03_transfer/SINGLE_MODEL_RESEARCH_REPORT.md)（记录到 Round03）；[Round03 精确复现命令](research/single_model_20260929/round03_transfer_complete/round03_transfer/ROUND03_REPRODUCE.md)。
- [Round04 完整结果、逐态/亮尾指标和命令](research/single_model_20260929/round04_fonly_complete/round04_fonly_preparation/ROUND04_REPORT.md)；[独立终审](research/single_model_20260929/round04_fonly_complete/round04_fonly_preparation/INDEPENDENT_TERMINAL_REVIEW.md)；[最终决定](research/single_model_20260929/round04_fonly_complete/current_state/ROUND04_DECISION.md)。
- 旧划分保留的服务器 checkpoint：`/home/inspur/MTO-1/research/single_model_20260929/baselines/calibrated_eta0.pt`；SHA256：`bcb0e51d8d877983abd02ab768892a8f8f8d3dfe0f73f4ec43d30c21c16b3db9`。
- 推理仅需元素与几何及其派生图；配置、训练归一化与固定校准均包含在一个 checkpoint 中，不需量子化学标签。校准只改 f，输出 E/A 一般不再严格重构该 f；这不是已识别的物理响应算符。
- 新版 QM9S v2 分子组划分已按固定 seed 20260930 生成并通过独立重建核验：TRAIN / VALID / TEST 为 **120,355 / 6,686 / 6,686**，在已审计的保守分组规则下无跨分区重叠；278 个身份含糊样本仅进入 TRAIN。[划分协议与限制](research/single_model_20260929/round05_scratch_preparation/dataset_audit_20260930/BENCHMARK_V2_REPORT.md)。这是历史暴露数据的新分区，不是独立外部新数据；必须全新初始化并仅使用新 TRAIN 统计，新 TEST 目标保持封存，不得用其调参。外部 UV–Vis 数据的方法兼容性与分子重叠仍待审计。
- Round05 已按独立绑定授权完成并关闭；共同设置为 seed 11、60 epoch、原 LE+Ls 和学习率 0.001。原始准备失败、工程修订及独立审查保留在[冻结协议](research/single_model_20260929/round05_scratch_preparation/round05_scratch_preparation/PROTOCOL.md)、[准备报告](research/single_model_20260929/round05_scratch_preparation/round05_scratch_preparation/PREPARATION_REPORT.md)和[准备独立审查](research/single_model_20260929/round05_scratch_preparation/round05_scratch_preparation/INDEPENDENT_PREPARATION_REVIEW.md)中。Round06 损失对照也已完成并关闭，未通过双参考门槛；后续 Round07 共享 PSD 合同变换也已完成并关闭；后续 Round08 也已完成并关闭；下一阶段仅限一个正则化对照方案，没有新的训练或 TEST 评分授权。尚不能将这些结果解释为已确立的因果机制。
- 当前研究发布只包含轻量源码、设置、日志、汇总结果及审计记录，先下载到 `D:\MTO\archives\` 校验，再提交。此次不新增模型权重、优化器、原始数据、成员/预测数组、缓存或凭据；下面旧归档中已有的历史文件保持原样。

## 历史项目与运行文档

以下为此前实验和安装/运行说明；其中“当前”、参数量、训练目标及提交文件的描述仅适用于各自历史版本。


## 历史归档：QM9S 全量 E+A 监督（2026-09-25～26，无 checkpoint）

**完整 DetaNet 与完整 MTO：6组训练正常完成，1个MTO重复种子因CUDA错误中断。** 使用实际TD-B3LYP/TZVP逐态标签，133727条有效记录，90/5/5分组划分，按训练集固定全局尺度监督E和A。

参考MTO的测试E+A损失比DetaNet低2.87%，但配对置信区间跨零；辅助光谱MSE高约11.0%，不能认定稳定全面领先。验证集选出的主配置是mto_reference。失败种子和六组测试的收尾协议偏离均保留。

- [完整归档与数据恢复](experiments/qm9s_full_EA_20260925/README.md)
- [实验设计与超参数](experiments/qm9s_full_EA_20260925/HYPERPARAMETER_DESIGN.md) · [七组实际配置](experiments/qm9s_full_EA_20260925/configs/)
- [详细结果分析](experiments/qm9s_full_EA_20260925/ANALYSIS_REVIEWED_ZH.md) · [诊断图](experiments/qm9s_full_EA_20260925/reports/results_diagnostics.png)

本次包含代码、全部准备后的输入/标签（可校验无损分块）、冻结划分/归一化、训练日志、全测试集预测、结果和环境审计。**本轮所有15个checkpoint文件均排除**，仅留审计哈希；仓库以前实验的权重保持原状。

## 历史完成实验：原版 DetaNet / planned MTO，1k → 10k → 全量（seed=11，2026-09-21～22）

**6次正式训练、三个规模的测试与报告均已完成。** 使用完整原版DetaNet骨干；每个规模A/B各从头训练一次；没有global-gate或额外种子。三个规模MTO相对测试MSE改善分别为 **10.28%、31.58%、30.62%**。这是单种子初步比较，不是显著性或跨种子稳定性结论。

| 规模 | A 测试MSE | B 测试MSE | MTO改善 |
|---|---:|---:|---:|
| 1k | 0.000288303 | 0.000258655 | 10.28% |
| 10k | 0.000294281 | 0.000201357 | 31.58% |
| 全量 | 0.000111857 | 0.0000776071 | 30.62% |

- [完整实验归档、配置与文件导航](experiments/detanet_original_mto_scales_seed11_20260921/README.md)
- [三规模总报告](experiments/detanet_original_mto_scales_seed11_20260921/reports/summary.md) · [综合分析](experiments/detanet_original_mto_scales_seed11_20260921/ANALYSIS_REPORT.md)
- [同分子跨规模预测谱图](experiments/detanet_original_mto_scales_seed11_20260921/analysis/spectrum_comparison/README.md)
- [独立完整性校验工具](experiments/detanet_original_mto_scales_seed11_20260921/publication/verify_archive.py)

归档包含12个best/last checkpoint、全测试集预测、训练日志/曲线、全部准备后的240点数据、冻结划分/统计及真实完成记录。大目标数组以可校验的无损gzip分块保存，其他权重/结果直接随Git提交。以下各节为此前独立实验的历史记录，其状态与结论保留原意。


## 历史归档：原版 DetaNet UV–Vis / 规划 MTO / global-gate（2026-09-20～21）

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

