# MTO / P-only 向量外积 × 完整 E+A / 纯 trace

远端目录：`/home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927`

固定参考提交：`71e575f573cfc94c7f0a301b663b0169290ce9bd`。固定配置：`experiments/qm9s_eta_Ef_20260926/configs/mto_eta1.json`。
参考文件通过 GitHub 固定提交读取，按 Git blob 校验；证据见 `reports/source_verification.json` 和 `reports/reference_tree.json`。

| 组 | 架构 | 目标 | 初值 |
|---|---|---|---|
| G1 | 原 MTO，含 M 旁路，A=CCᵀ | LE+Ls+LQ | initial/original.pt |
| G2 | 原 MTO，含 M 旁路，A=CCᵀ | Ls | initial/original.pt |
| G3 | P-only 普通 1o 向量，A=μμᵀ | LE+Ls+LQ | initial/p_outer.pt |
| G4 | P-only 普通 1o 向量，A=μμᵀ | Ls | initial/p_outer.pt |

`s=tr(A)`、`Q=A-sI/3`，`LE=mean_valid((E-Et)²)/sE2`，`Ls=mean_valid((s-st)²)/(3*sA2)`，`LQ=mean_valid(sum_ij((Q-Qt)²))/sA2`。η=0 分支完全不读取 E 标签，也没有逐对角元 MSE。

完整 DetaNet 骨干、MTO router/query/汇聚及 128→16 投影保留。P-only 模型将 `16x0e+16x1o+16x2e` 的 M0、Mk 经原 component/element 归一化的 FullyConnectedTensorProduct 输出 `16x0e+16x1o`；其可学习 CG 权重重新初始化。两态耦合保留所有 i,j 原子对。decoder 的唯一输入为 P。

P0 经 LayerNorm(16)→Linear(16,128)→SiLU，E 经 softplus(Linear(h)+energy_offset)；g=tanh(Linear(128,16)(h))；mu=alpha_init*sum(g*P1)/sqrt(16)。vector_gate 权重 N(0,0.005²)、偏置 0.25。无 beta、C、2e 矩阵读出、方向偏置、epsilon I、向量归一化或额外损失。alpha_init 为不可训练 buffer，一次性在冻结 train 数组前512个分子上匹配 G1 初始平均 trace；具体 ID、数值及哈希见 `reports/initialization.json`。

## 数据和训练

133727 分子，120355/6686/6686 原划分；所有标签、mask、零值与冻结 sE2/sA2 原样保留。数据为独立副本，哈希见 `data/hashes.json`。旧实验目录仅被读取。环境只读复用：`/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`。

四组 seed=11，从冻结随机初值开始，无旧 checkpoint 预训练。G1/G2 全状态相同；G3/G4 全状态相同；四组骨干/MTO 状态相同。独立 NumPy Generator(11) 产生每轮排列，1000轮预期哈希及运行哈希分别保存于 `reports/epoch_order_hashes.json` 和各组 `current_order.json`、`history.jsonl`。

Adam AMSGrad，lr=1e-3，weight_decay=0，batch=64，FP32，AMP/TF32关闭，梯度裁剪5。每轮全量验证；ReduceLROnPlateau factor=0.5、patience=50、relative threshold=1e-4、min_lr=1e-6。最多1000轮，至少200轮；早停须 bad_epochs≥150、降率至少2次、距最后降率至少50轮。每组按自身验证目标调度、早停、选best。不同总Loss不能跨组排名。G2/G4 的 energy_head 和 energy_offset requires_grad=False，共享特征仍更新；其 E 输出不能当有效预测。

`last.pt` 保存模型、优化器、调度器、epoch/cursor/order、累计loss、best/早停状态和所有随机数状态。每轮及训练/验证中每540秒触发保存，为600秒要求留出余量。`best.pt` 在首次完整验证后生成。SIGTERM/SIGINT 在当前batch后保存并退出；断点恢复不会重新排列当前轮。恢复重新检查配置和代码/数据/初值哈希。

## 预检

`reports/preflight.json`：FP32/FP64 原参考前向/损失/梯度等价；trace而非对角MSE；E标签改变不影响G2/G4梯度；能量专属参数冻结；mask/NaN/零标签；P-only接口；旋转、反演、原子置换；对称/PSD/rank≤1/trace=||mu||²；非零初始化；四组batch64梯度及AMSGrad恢复。

`reports/system_preflight.json`：实际trainer入口的保存→恢复→继续更新，以及oracle/common/native、谱和R²定义的合成检查。预检权重全部丢弃。CUDA index_add 汇聚存在浮点舍入差异，预检报告记录误差与容差，不宣称逐位确定性。

中心对称合成几何的普通极向量1o必须为零；一般具有无不变极向量的高对称点群也受此限制。该头因此无法在此类几何表示非零跃迁强度。样本保留，架构不改。rank=0暗态允许。

## 资源和队列

仅使用提交时复查为空闲、当前/历史ECC均零、无重映射/待修复的GPU。健康池为1/2/4/6；GPU6旧任务退出并完成后才纳入。0/5有纠正ECC记录，3/7有不可纠正ECC及待重映射，均排除。无法获取资源时排队，不抢占或重置。监督器及worker均加锁；每组每次提交只启动一次，失败不自动重试。健康变化时只向本实验对应worker发送SIGTERM。

## 自动评估

四组全部FIT_COMPLETE、无失败标记后自动执行 `evaluate.py`；先写 `reports/selection_before_test.json` 冻结四个按各自验证目标选出的best及SHA256，然后统一验证/测试。任何一组失败均不跳过，不生成整体完成标记。

首要指标为共同trace MAE/RMSE/R²及逐态结果。R²=1−SSE/SST。

- oracle：使用真实能量的诊断，f=(2/3)(E_true/27.211386245988)tr(A_pred)，峰位置使用E_true。
- common：四组统一使用G1验证选出best提供E_ref，f=(2/3)(E_ref/27.211386245988)tr(A_pred)，峰位置使用E_ref；仅用于评估。
- native：仅G1/G3报告自身E及其f/光谱。G2/G4不报告原生E/f/光谱正式指标。

光谱0–21eV、步长0.02eV、单位积分Gaussian σ=0.2eV，真值用原始E/f。预测、真值、ID、索引、mask、E_ref、各口径f和逐分子谱误差保存于各组 `val_predictions.npz`、`test_predictions.npz`。G2/G4仅以 `E_unsupervised_diagnostic` 字段保存未监督E，明确不纳入指标。

这是单种子追加探索性实验；旧测试集曾被分析，不能凭本轮单独归因于去旁路或秩一某一项。

## 查看与恢复

```bash
ssh USTC-A800
cd /home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927
PY=/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python
$PY control.py status
tail -f logs/G1.log
cat runs/G1/status.json
```

停机/失败后检查对应 `FAILED.json`、`PROCESS_EXIT.json` 或 `GPU_STOP.json`，确认原因排除再显式恢复指定组。运行中的worker不能重复恢复：

```bash
$PY control.py resume G1
```

恢复沿用该组last.pt，失败证据移入该组resume_archive目录，资源仍需通过空闲/健康检查。仅监督器退出、worker仍存活时：

```bash
$PY control.py submit
```

新监督器会识别现有PID，避免重复worker，并接续完成后评估。全部训练完成且自动评估失败时，排除原因后可在已确认空闲健康GPU上手动运行 `CUDA_VISIBLE_DEVICES=1 $PY evaluate.py`；脚本仍核查冻结checkpoint，完成前不报告完整结果。
