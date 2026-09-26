# QM9S 全量 E+A：实际执行记录

用户于 2026-09-25 授权直接在 USTC-A800 执行。原 80/10/10 方案已被本次 90/5/5 设置替代；此前逐态尺度也已替换为用户图片规定的两个全局尺度。

## 数据和损失

- 原始提取数据：`/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925`。
- 133727 条有效记录，10 个单重态/记录；失败原始日志不进入训练。
- 训练/验证/测试为 **120355 / 6686 / 6686**；分组划分种子 20260925。
- 分子身份由实际 XYZ 的 RDKit 成键推断得到；先检查中性价态可行性，再按忽略键级/立体化学、保留显式氢的保守连接图分组。同一组不能跨分区。不是拿日志编号当化学身份。
- 133469 个连接图组。278 条未解决身份的记录仅进入训练集，全部有效标签保留。这是结果解释中的适用性限制；见 `data/identity_audit_v2.json`。
- 用训练集逐态能量均值计算一个公共方差：`sE² = mean_na((E_na - mean_n_train(E_na))²) = 0.5378066634062587 eV²`。
- 用训练集全部张量计算一个公共矩阵均方：`sA² = mean_na(sum_ij A_naij²) = 0.1324285377060015 au⁴`。
- `LE = mean_na((Ehat-E)²)/sE²`，`LA = mean_na(sum_ij(Ahat-A)²)/sA²`，`L = LE+LA`。九个元素内部求和，再对样本和状态平均。无逐态权重、无按 batch 重算、无每样本强度归一化。
- 能量仍直接预测 E；逐态均值仅用于定标和统一初始化。训练/验证/测试均使用冻结尺度。
- 所有输入都来自同一 TD 日志。预先缓存与原模型半径图一致的 5 Å 邻接边，数据驻留各自 GPU，避免重复解压和数据加载瓶颈。

## 固定实现与对照

- 上游仓库 https://github.com/techandscixie2005/MTO-1 ，提交 `8e1e0a871c65ff982d6d991855b983082727820b`。
- 保留 `experiments/detanet_original_mto_scales_seed11_20260921/` 完整原版 DetaNet 骨干及 MTO 路由、轨道聚合、参考态耦合、PSD 解码器。
- 完整骨干：128 维、3 块、lmax=3、8 头、32 个可训练 Bessel 基、rc=5 Å。参数 1371840。
- 参考完整 MTO 总参数 1552092；32 通道版本 1783820。
- DetaNet 对照保留相同完整骨干，通过原生 MLP 与 Equivariant_Multilayer 原子读出、分子求和得到逐态 E/C，再以相同的 softplus 和 A=CCᵀ 输出。总参数 1408872。没有 MTO 查询/路由、轨道聚合或参考态 CG 耦合。此对照是适配 E/A 属性头的完整 DetaNet，不是原来的 240 维光谱输出头。
- 两个参考模型使用相同种子和逐元素一致的初始骨干参数；所有模型从头训练。能量偏置统一初始化为训练集逐态均值的 inverse-softplus。
- 直接全量；无 1k/10k 前置训练、无预训练权重。

## GPU 与实验矩阵

| GPU | 实验 | 学习率 | MTO 通道 | batch | seed |
|---|---|---:|---:|---:|---:|
| 0 | detanet_reference | 1e-3 | 不适用 | 64 | 11 |
| 1 | mto_reference | 1e-3 | 16 | 64 | 11 |
| 2 | mto_lr3e4 | 3e-4 | 16 | 64 | 11 |
| 4 | mto_lr3e3 | 3e-3 | 16 | 64 | 11 |
| 5 | mto_channels32 | 1e-3 | 32 | 64 | 11 |
| 6 | mto_batch128 | 1e-3 | 16 | 128 | 11 |
| 7 | mto_seed23 | 1e-3 | 16 | 64 | 23 |

GPU 3 存在不可纠正 ECC 错误（启动时 187），排除。其余 7 张 A800 80GB 各运行一个任务。没有重置卡或处理其他用户进程。

公共设置：Adam AMSGrad，betas=(0.9,0.999)，eps=1e-8，weight decay=0，梯度范数裁剪 5，FP32，禁用 AMP/TF32；验证总损失 plateau patience=50、factor=0.5、relative threshold=1e-4、min_lr=1e-6。最多 1000 epoch，至少 200；150 epoch 无最佳值改善、至少两次实际降率、最后降率后至少 50 epoch 才可早停。每 epoch 和每 10 分钟保存模型/优化器/调度器/RNG/批次顺序及游标。

## 环境、检查、运行与分析

- 独立 venv：`env/`，复用现有 PyTorch 2.5.1+cu121/PyG 2.7.0 的匹配二进制扩展；离线安装 e3nn 0.4.4、opt-einsum-fx 0.1.4、opt-einsum 3.4.0。未修改原有共享环境。确切环境见 `reports/environment_freeze.txt`。
- 已验证归一化基准（训练均值能量预测 LE=1；零张量预测 LA=1）、标签/图一致性、输出形状、对称 PSD、旋转/反射/平移/置换性质、梯度到达骨干/路由/耦合/解码器、检查点读写。新 MTO 属性路径与上游物理输出在 FP32 数值容差内一致。
- 根目录：`/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925`。
- tmux 会话：`qm9s_full_ea`；进程 PID 和 GPU 映射在 `launch_receipt.json`。
- 训练日志：`logs/<name>.log`，状态：`runs/<name>/status.json`，检查点：`last.pt`/`best.pt`，曲线历史：`history.jsonl`。
- Supervisor 只管理本实验进程；检测到新的不可纠正 ECC 时给本实验进程发 SIGTERM，触发保存并报告，不重置 GPU。
- 所有训练成功结束后自动执行 `analyze.py`。先冻结验证集最佳 MTO 配置，再打开测试集评价全部最佳检查点，不用测试调参。
- 报告 E 逐态/总体 MAE、RMSE；A Frobenius、迹、无迹部分、秩一偏离及 PSD；近简并态簇 ΣA；派生振子强度和 0–21 eV/0.02 eV 步长、sigma=0.2 eV 的辅助光谱误差。后者只是评估约定，不是新的训练标签。
- 匹配结构对照：detanet_reference vs mto_reference。调参最优配置由验证集选出。MTO 两种子用于有限的稳定性观察，不把不同超参数当重复种子；DetaNet 只有一个种子。测试组 bootstrap 区间只覆盖测试样本抽样不确定性，不代表全部训练随机性。
- 结果路径：`reports/analysis.md`、`reports/results.json`、`reports/learning_curves.png` 以及各 run 的测试指标和预测。
- 当前任务已安排每 30 分钟自动跟进；正常无事不重复通知，结果完成后下载并分析。

本记录表示实际已启动的设置，不表示训练已收敛或已有最终测试结果。
