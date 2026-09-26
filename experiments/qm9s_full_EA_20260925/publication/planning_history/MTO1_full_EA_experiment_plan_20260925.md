# MTO-1：完整 DetaNet+MTO 的全量 E+A 监督实验方案

状态：拟执行方案。本轮完成了源码、数据和服务器资源核查，尚未改动模型、安装环境或启动训练。

本方案以实际 TD 日志和当前仓库代码为依据。以前的课题总纲及仓库历史实验说明仅作为背景；本次不执行 1k→10k 阶梯实验，直接进入全量数据规模。

## 1. 实验对象与代码版本

- 仓库：<https://github.com/techandscixie2005/MTO-1>
- 固定提交：`8e1e0a871c65ff982d6d991855b983082727820b`。
- 使用完整原版骨干的实现：`experiments/detanet_original_mto_scales_seed11_20260921/models.py`，分支 `detanet_mto_planned`。
- 原版 DetaNet 来源提交：`4f92e643ab64651b91c4a1392cf389ddfd0d89f0`，沿用仓库内 vendor 与许可证。
- 仓库记录的完整模型参数量：1,552,092；其中公共骨干 1,371,840，MTO/耦合/解码读出合计 180,252。迁移后重新核算，不能引用根目录历史小模型的约 18 万参数数字。
- 正式运行从头初始化；三个种子共用数据划分与归一化统计，不加载旧光谱模型权重。

模型路径保持：

`Z, R → 完整 DetaNet → 状态条件化 MTO → 参考/目标态 CG 耦合 → E 与 A`。

源码已经实现正激发能 `E = softplus(...)`，以及

`C = β I/√3 + Q`，`A = C Cᵀ`。

因此 A 为对称半正定张量，不需要另造普通向量头。本次保留该参数化；不额外施加秩一惩罚。标签 A=μμᵀ 为秩至多一，预测的偏离作为诊断指标报告。

源码依据：[模型实现](https://github.com/techandscixie2005/MTO-1/blob/8e1e0a871c65ff982d6d991855b983082727820b/experiments/detanet_original_mto_scales_seed11_20260921/models.py)、[完整骨干来源与参数审计](https://github.com/techandscixie2005/MTO-1/blob/8e1e0a871c65ff982d6d991855b983082727820b/experiments/detanet_original_mto_scales_seed11_20260921/audit/SOURCE_AND_MODEL_AUDIT.md)。

## 2. 实际数据与划分

服务器源目录：

```text
/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925/
```

| 项目 | 实际核查结果 |
|---|---|
| 原始日志 | 133,885 份，ID 1–133885，无编号缺口或重复 ID |
| 可训练候选记录 | 133,727 条正常结束、几何和逐态标签完整的记录 |
| 无最终光谱记录 | 158 条，仅保留审计，不进入训练/验证/测试 |
| 激发态 | 每条有效记录 10 个 Singlet 根，共 1,337,270 条跃迁 |
| 计算设置 | 所有日志 route 为 `#p td=(nstates=10) b3lyp/TZVP nosymm geom=check guess=read` |
| 元素、原子数 | H/C/N/O/F；有效结构为 3–29 个原子 |
| 电荷、自旋 | 有效记录均为净电荷 0、基态多重度 1 |
| 几何 | 同一 TD 日志中的 Input orientation，Å；不替换为其他 QM9 几何 |
| 标签 | E：eV；跃迁电偶极：原子单位；A：偶极原子单位的平方 |
| 能量范围 | 实际打印 E 为 0.3111–18.9494 eV |
| 完整性 | 上传时已通过文件 SHA-256、全量数组读取及 JSONL/NPZ 对照 |

本次的“全量”指对全部有效数据进行一次完整数据集实验，保留独立验证集和测试集。目标比例为 80/10/10；不另做 1k、10k 子集学习曲线。

拟固定划分种子 `20260925`。若每条记录均属于独立分组，目标数量为 train 106,981 / validation 13,373 / test 13,373。正式划分前进行分子身份与重复结构审计：相同分子身份、重复结构及其构象整体进入同一分区，所有有效记录均保留。分组约束可能使数量略有变化，最终以保存的 `splits.json` 为准。

当前提取数据没有 SMILES/键连接，不能把文件 ID 唯一宣称为化学身份唯一。准备时优先获得与当前几何/ID核验一致的身份映射；缺失部分用 RDKit 几何成键产生待审计身份，同时检查置换/旋转不变的重复几何。未解决的身份冲突不进入验证/测试，记录其处理原因与实际分区。不得直接用旧数据集行号拼接。

旧归档的“全量”为 129,732 条，且包含基于 canonical identity 的过滤。本次不沿用旧的 240 点光谱、旧归一化常数或旧行号划分。旧分数只作历史背景，不能与本次 E/A 指标直接计算提升百分比。

## 3. 监督目标与损失的精确定义

每个分子输入仅为原子序数与坐标；状态索引是模型内部条件。E、μ、A 和 f 均不作为模型输入。

对训练集的第 k 个根计算：

\[
s_{E,k}=\max\bigl(\operatorname{Std}_{\mathrm{train}}(E_k),10^{-3}\ \mathrm{eV}\bigr),
\]

\[
s_{A,k}=\max\left(\sqrt{\operatorname{Mean}_{\mathrm{train}}\|A_k\|_F^2},10^{-6}\ \mathrm{au}^2\right).
\]

这两个下限是避免除零的工程设置。统计量、有效样本数量与哈希在训练前冻结。N_ref 同样取训练集原子数中位数，不硬编码历史值 18。

\[
\mathcal L_E=\operatorname{Mean}_{n,k\ \mathrm{valid}}
\frac{(\hat E_{nk}-E_{nk})^2}{s_{E,k}^2},
\]

\[
\mathcal L_A=\operatorname{Mean}_{n,k\ \mathrm{valid}}
\frac{\|\hat A_{nk}-A_{nk}\|_F^2}{s_{A,k}^2},
\qquad \mathcal L=\mathcal L_E+\mathcal L_A.
\]

- 初始并固定 `λ_E=1, λ_A=1`。分别记录两项损失和梯度范数，第一轮不做测试集驱动的权重搜索。
- A 监督完整 3×3 张量；Frobenius 平方等于三个对角误差平方加两倍三个独立非对角误差平方。若实现只存 6 分量，必须保留这个权重。
- A 使用每态一个旋转不变量尺度，不逐 xyz 分量独立标准化，不逐样本除以自身强度，不把暗态设成缺失。
- 不增加 f、μ 或展宽光谱损失；f 和光谱只作派生评估。
- 用源 `state_index` 对齐 E 与 A；不单独排序预测 E。若为了画图重排，E 与 A 必须使用同一个置换。
- 保留原始根顺序进行主训练。近简并/打印相同能量的根单独标记；逐态 A 可受简并子空间基底选择影响，另报告簇内 ΣA 指标。它不替代主损失，也不通过测试集选择匹配规则。

标签旋转必须一致：`R′=R Uᵀ`（坐标按行存储），`A′=U A Uᵀ`。本轮默认不额外加入随机旋转训练增强，先验证模型本身的 O(3) 协变性。

## 4. 模型和优化超参数

| 项目 | 拟采用值 | 来源/说明 |
|---|---|---|
| DetaNet 通道 | 128 | 保留完整实现 |
| 最高角动量 | lmax=3 | 所有骨干块保留 1o/2e/3o；MTO 接口取 0e/1o/2e |
| 交互块 | 3 | 保留原版算子与残差 |
| 径向基 | 32 个 trainable Bessel | 保留原版 |
| attention heads | 8 | 保留原版 |
| 邻接半径 | 5 Å | 无自环，max_num_neighbors=32；当前最多29原子，理论度上限28 |
| 额外 cosine cutoff | 关闭 | 保留仓库 `use_cutoff=False` |
| 激活、dropout | 原版 learnable Swish；0 | 保留骨干 |
| MTO 通道 | 每种类型16 | 保留完整 planned MTO |
| 状态 | 1个参考+10个目标 | 独立状态 query |
| query 维度 | 32 | 保留 |
| router | 隐层128，SiLU，tanh gate | 保留状态/类型/通道条件化系数 |
| CG 耦合 | 参考态×目标态，输出0e与2e | 保留 FullyConnectedTensorProduct |
| 解码器隐层 | 128 | 保留 invariant LayerNorm 与物理头 |
| E 初始偏置 | 2–13 eV 等间距对应的 inverse-softplus 偏置 | 保留源码初始化；不限制最终输出范围 |
| 优化器 | Adam + AMSGrad | 沿用最新完整实验 |
| 学习率 | 1e-3 | 初始建议，尚未用本次 E/A 训练验证 |
| Adam 参数 | betas=(0.9,0.999)，eps=1e-8 | 明确默认值 |
| weight decay | 0 | 沿用 |
| 有效 batch | 64 分子/次更新 | 每个种子独立单卡 |
| micro batch | 初始64；OOM时32/16并累积 | 保持有效batch与样本权重不变 |
| 梯度裁剪 | global norm 5.0 | 本次新增的优化设置；记录裁剪前范数及触发率 |
| 精度 | FP32；AMP关闭；TF32关闭 | 初次迁移先保持数值口径 |
| LR 调度 | ReduceLROnPlateau | 监控验证 L_E+L_A |
| 降LR参数 | factor=0.5，patience=50，relative threshold=1e-4，min_lr=1e-6 | 沿用最新实验 |
| epoch预算 | 最多1000，最少200后允许早停 | 全量运行 |
| 早停 | 验证loss连续150轮无改善；至少2次实际降LR；最后降LR后至少50轮 | 沿用控制规则 |
| 正式种子 | 11、23、37 | 固定同一数据划分，报告均值与标准差 |
| DataLoader | 初始8 workers/任务，pin_memory，persistent_workers | 以正式首轮吞吐调整，不改变科学配置 |
| CPU线程 | PyTorch/OMP各2 | 避免线程过量竞争 |
| checkpoint | 每个epoch保存last；最佳验证保存best；长epoch中每10分钟保存可恢复状态 | 记录优化器、调度器、RNG、shuffle顺序/游标、数据与代码哈希 |

原模型初始 router/head 分布与参数化保持源码设置。除明确写出的 E/A 损失、轻量输出接口、数据适配和梯度裁剪外，不额外改写 backbone 或扩展 MTO。

## 5. 服务器资源实测与任务分配

查询时间：2026-09-25 21:58–21:59（Asia/Shanghai），主机 `NF5468A7`，SSH 别名 `USTC-A800`。以下是当时快照，不是资源预留。

| 资源 | 实测 |
|---|---|
| GPU | 8×NVIDIA A800 80GB PCIe |
| 当前占用 | 各卡约10 MiB、利用率0%，仅Xorg，未见训练进程 |
| GPU 3 | volatile DRAM uncorrectable=187，aggregate=352；2行不可纠正重映射，Pending=Yes |
| 可优先使用 | 0、1、2、4、5、6、7；GPU3暂不用于本次训练 |
| GPU拓扑 | 0–3同NUMA节点，4–7另一节点；未显示NVLink |
| CPU | 2×AMD EPYC 9754，256物理核/512逻辑CPU |
| 内存 | 总503 GiB，available约478 GiB |
| 系统盘/NVMe | 可用约2.6 TiB |
| /data | 可用约11 TiB |
| 驱动 | 590.48.01；nvidia-smi显示CUDA兼容上限13.1 |
| 可用Python栈 | Python3.10.19、PyTorch2.5.1+cu121；实际Torch CUDA为12.1 |
| 扩展 | PyG2.7.0、scatter2.1.2+pt25cu121、cluster1.6.3+pt25cu121、NumPy2.2.6 |
| 待补依赖 | 当前环境没有e3nn；最新完整归档实际使用0.4.4 |
| 作业入口 | 检查未发现squeue，发现tmux；准备独立直接运行入口 |

正式任务拟分配 GPU0→seed11、GPU1→seed23、GPU2→seed37，每张卡一个完整训练进程，三次全量训练并行。4–7保留为资源变化时的候选卡。本模型约155万参数，优先并行独立种子，不预设跨卡DDP能提高效率。启动前再次检查占用；不重置GPU3，不终止已有任务。

在项目独立虚拟环境中复用已匹配的Torch/CUDA扩展，并优先锁定 `e3nn==0.4.4`。先做import、梯度及等变检查；若旧e3nn与当前Torch确实不兼容，再建立归档版本的独立环境（Torch2.3.0+cu121/PyG2.6.1/pt23扩展）。不把根目录 `e3nn==0.5.6` 当作最新完整实验实测版本，也不覆盖现有共享环境。

版本依据：[最新完整实验环境快照](https://github.com/techandscixie2005/MTO-1/blob/8e1e0a871c65ff982d6d991855b983082727820b/experiments/detanet_original_mto_scales_seed11_20260921/publication/environment_at_preparation.json)。

## 6. 实施安排

1. **建立新实验目录并冻结来源。** 拟用 `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/`。仅拉取所需源码、vendor和许可证，避免下载约2GB历史权重/结果。保存上游commit、修改diff、依赖版本与数据哈希。
2. **准备全部有效数据。** 一次性将NPZ分片转成训练用连续/可内存映射数组，保存原始ID和mask；核查重复身份、近重复结构、零距离与孤立原子，冻结分组划分。只从train计算 E/A scales、N_ref；验证和测试不参与统计或梯度。
3. **适配训练接口。** 新 `forward_properties` 直接返回有计算图的 E/A；常规训练不导出H/B/F/c/M，也不必计算展宽谱。保留按需导出解释量的接口。新训练器读取E/A和mask，取消旧1k/10k阶段依赖及Slurm硬编码，支持独立seed和中断恢复。
4. **必要工程验收。** 用少量真实训练记录做有限值、前向/反向、根对齐、A对称/PSD、平移/原子置换、旋转/反演协变及断点恢复检查；验证A的坐标变换约定。不做1k/10k正式训练，不以小样本成绩作为全量开跑门槛。检查模型丢弃，正式模型重新初始化。
5. **全量正式训练。** 三个种子在冻结配置下从头训练。正式第一轮记录steps/s、epoch用时、峰值显存与各损失，用于更新运行时间估计；不因第一轮测试成绩更改配置。
6. **统一测试与归档。** 三个种子分别用最低验证联合loss的checkpoint。全部训练完成后统一在冻结测试集评估，导出逐分子逐态预测、分组统计、曲线、固定样本图及汇总报告。

本轮主任务是指定的完整DetaNet+MTO模型，3个随机种子用于重复性。如果后续要量化“E+A监督相对光谱监督”的贡献，应另立相同架构、相同新划分的光谱监督对照；不能直接拿旧归档的30.62%改善套用。若需要纯DetaNet的E+A对照，则还需一个明确的等变张量头，现有240点标量输出基线不能只改Loss就监督A。

## 7. 指标与图表

- **主指标**：E的MAE/RMSE（eV，整体及S1–S10）；A的Frobenius RMSE及归一化误差；验证联合loss用于唯一的best选择。
- **张量诊断**：trace误差、traceless部分误差、对称/PSD误差、最大特征值之外的谱权重。暗态不用未经保护的逐样本相对误差。
- **近简并诊断**：预先标记相邻能隙≤1e-4 eV的簇，报告簇内ΣA误差，并明确该阈值依据打印分辨率，只是诊断分组，不宣称物理严格简并。
- **派生f**：`f_hat=(2/3)*(E_hat/27.211386245988)*trace(A_hat)`，与日志f比较；同时记录日志舍入误差基线。
- **谱图辅助指标**：评估时统一使用归一化Gaussian，拟定能量网格0–21 eV、步长0.02 eV、sigma=0.20 eV。该宽度是本次可视化/评估约定，不是原始数据的额外标签。报告有限网格丢失的积分强度；E/A训练不受该网格或展宽影响。
- 每个seed分别报告，再给均值±标准差；保存每条测试记录的E、A、f及ID。固定少量样本导出MTO组装项，避免每轮导出全数据原子级中间张量。

## 8. 时间与存储预算

尚未在A800上实测这次E+A训练吞吐，不给出确定完成时间。

历史归档的完整MTO全量运行：236 epochs、8.244小时、batch64，约2.10分钟/epoch，记录峰值allocated显存约1.37 GiB。该记录来自另一集群且目标为光谱，不是本次A800的基准；准备环境快照中的4090也不能被当成训练任务g0087的已确认GPU型号。

初始资源预算：每个seed单卡；预留每seed24小时的首轮作业窗口，必要时从完整checkpoint继续，1000-epoch硬上限保持不变。粗略规划按1–4分钟/epoch计，300–600轮为5–40小时，跑满1000轮约17–67小时。三seed并行时墙钟取最慢任务，不把三次耗时相加；这只是预算区间，正式首轮后用实测值替换。

数据/环境适配与工程检查预计1–3小时，实际取决于依赖安装和身份审计。为新实验预留约50GB磁盘（含缓存、环境、checkpoint与预测）；中间组装量只导出固定样本。模型结构、E/A监督质量或显存不足均不得通过悄悄缩小骨干来解决。

历史耗时依据：[完整MTO运行记录](https://github.com/techandscixie2005/MTO-1/blob/8e1e0a871c65ff982d6d991855b983082727820b/experiments/detanet_original_mto_scales_seed11_20260921/runs/full/seed_11/detanet_mto_planned/FIT_COMPLETE.json)。

## 9. 交付物

`experiment_config.json`、环境lock与版本快照、代码diff/source hashes、冻结划分、身份/重复审计、训练集归一化统计、三个seed的best/last checkpoint、完整恢复状态、逐轮E/A损失和学习率、正式测试预测、逐态及简并簇指标、光谱图、汇总报告。

当前随方案提供的JSON是“拟执行配置”，不是已经修改好的原仓库命令行配置。代码适配、环境安装、正式划分和训练均待进入执行阶段后完成。
