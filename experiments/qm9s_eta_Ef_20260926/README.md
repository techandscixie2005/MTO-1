# QM9S：三组 MTO η + 一组原版 DetaNet E/f

## 状态

2026-09-26 在 `ssh USTC-A800` 实际执行。三组MTO及一组原版DetaNet E/f均已启动，尚未完成。用户已批准原文设置优先、缺项参考MTO，详见确认记录。

- 远端目录：`/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926`。
- tmux：`qm9s_eta_ef_20260926`。
- Python：`/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`，只读复用已有环境，没有更改共享依赖。
- MTO固定提交：`bcf67d2a80b1d5ba35706582387f528537b5283c`。
- DetaNet固定提交：`4f92e643ab64651b91c4a1392cf389ddfd0d89f0`。
- GPU 1/2/4分别运行η=0/0.1/1，GPU6运行DetaNet；避开有不可纠正ECC历史的3/7。GPU5预检时记录1次可纠正ECC，正式DetaNet改用无ECC历史的GPU6。

## 冻结协议

沿用133727个有效分子、十态、120355/6686/6686分组划分，不重划分、不删除离群点。冻结标签、坐标及ID逐项匹配实际提取NPZ，mask明确保留。原始f中的25079个零值保持为有效值。排除的158条仅因未在既有冻结ID集合中，而非本轮新筛选。

原normalization.json逐字复制：sE²=0.5378066634062587，sA²=0.1324285377060015。MTO完整骨干、MTO/CG/C/CCᵀ不改，1,552,092参数（骨干1,371,840）；仅损失中最终A的无迹部分权重η不同。

三组相同seed=11，初始化状态哈希一致；独立NumPy Generator生成相同每轮排列，逐轮保存order_sha256。CUDA scatter/index_add沿用原FP32实现，未声称GPU训练逐比特确定性。

损失数组顺序为 `[L_eta, LE, Ls, LQ]`。LE按有效E标签均值；Ls/LQ按有效A标签均值。Q=A-tr(A)I/3。三组各自使用验证Lη调度、早停与best。跨η只用统一验证光谱MSE评价各自的best，绝不以不同η的总Loss排序。

统一光谱：0–21 eV、0.02步长、单位积分Gaussian σ=0.2 eV，不对有限区间重新归一化。原始f用于共同真值；MTO由预测E/A推导f，DetaNet直接输出f；均不裁剪。

四组配置已确认并冻结。全部训练完成后，先保存各best的验证结果与`selection_before_test.json`，冻结检查点哈希和选型，再统一计算测试指标。仅追加探索性实验，不据测试结果继续调参，不声称单种子的跨种子优势。

## DetaNet实际训练规则

完整官方架构（1,391,188参数），原生20维标量读出，直接联合监督十态E/eV和原始f，官方MSE，无标签定标、无输出变换、无梯度裁剪。Adam+AMSGrad、lr=1e-3、batch=64来自正文。

正文的验证间隔50、lr减半、停止阈值1e-5和上限1,000,000优先于MTO参考规则。结合官方Trainer用批次step打印Epoch，将其解释为优化器更新：每50次更新全量验证，最多1,000,000次更新（约532个完整数据轮次），或间隔训练MSE及完整验证MSE都≤1e-5。计数单位和损失聚合是披露的执行解释。

未公开的判据参考MTO：ReduceLROnPlateau的patience=50次验证、relative threshold=1e-4、min_lr=1e-6，按完整验证MSE选择best。**DetaNet不采用MTO的200轮下限、1000轮上限或150轮无改善早停。**训练曲线的epoch字段为完整数据轮次的浮点进度，steps为更新次数。

## 文件

- `frozen_reference/`：指定提交中的只读参考源码和配置；`official_detanet/`：固定官方版本。
- `reports/source_downloads.json`：逐文件Git blob与SHA-256；`baseline_tree.json`：提交树。
- `data/dataset.npz`、`splits.json`、`normalization.json`及`hashes.json`：冻结数据副本。
- `data/raw_labels.npz`：按相同ID对齐的float64原始E/f/A和mask。
- `reports/data_audit.json`：全量标签对应、零值、打印精度差异与哈希。
- `reports/preflight.json`：η=1数值/梯度、三η旋转、等变/PSD、mask NaN验证。
- `reports/initialization.json`及`initial_model.pt`：共同初始参数和前三轮顺序哈希。
- `configs/`、`runs/<name>/run_manifest.json`：实际配置、源代码指纹和环境。
- `runs/<name>/best.pt`、`last.pt`：best与含优化器/调度器/RNG/批次游标/历史的断点；每轮及最长600秒保存。
- `logs/`、`runs/<name>/history.jsonl`、`status.json`：完整日志与曲线。
- `reports/detanet_protocol.md`、`detanet_confirmation.json`：DetaNet证据、具体缺项、已批准的补全与原文优先级。
- `trainer_detanet.py`、`configs/detanet_ef.json`：DetaNet单独训练器和实际冻结配置；`detanet_training_checks.json`记录scratch恢复测试。

## 失败及恢复

第一次三组启动因新目录漏复制data/hashes.json而在任何优化器更新前退出；保留`FAILED_attempt_1.json`与完整日志，补齐清单后重新从头启动。未更改实验超参数。

Supervisor仅管理本目录任务。检测新的不可纠正ECC时只SIGTERM对应worker，保存断点、不重置GPU、不干扰其他任务。失败不会自动反复重试。

从断点恢复时先确认目标GPU健康空闲和该组无活动worker，再用相同环境运行：

```bash
CUDA_VISIBLE_DEVICES=<healthy-id> /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python -u trainer.py mto_eta0
```

代码/配置指纹不一致将拒绝恢复。恢复不需要改config中的原始GPU记录（设备实际绑定由CUDA_VISIBLE_DEVICES决定，迁移需另存记录）。不重新运行prepare_round覆盖已经冻结的数据。

DetaNet恢复使用同一命令形式，但执行`trainer_detanet.py detanet_ef`。监督器持续运行时不要手工重复启动。每50次更新的完整验证后保存last及曲线，包含优化器、调度器、RNG、批次游标和验证间隔累计值。
