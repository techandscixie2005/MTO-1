# 原版 DetaNet UV–Vis 与规划 MTO：1k 前置验收及 v2 有界诊断

**状态：诊断已完成；A 主冒烟未通过，正式 15 次 1k 训练尚未提交。** 本目录为 2026-09-20～21 在 SSH 别名 `bjhpc` 上真实执行的独立实验归档。没有本轮正式测试集预测、五种子泛化指标或 `STAGE_COMPLETE`。

## 模型与原版来源

作者 [WeiHuQLU/DetaNet](https://github.com/WeiHuQLU/DetaNet/tree/4f92e643ab64651b91c4a1392cf389ddfd0d89f0) 固定 SHA：`4f92e643ab64651b91c4a1392cf389ddfd0d89f0`。原源码、MIT 许可证、论文与补充 PDF、两个 notebook 保存在 `reference/`；`vendor/` 是独立源码副本。作者 UV checkpoint 仅用于严格加载和独立数值验收，所有冒烟模型从头初始化。

| 分支 | 实际模型 | 总参数 | 共同骨干参数 | 读出/新增模块参数 |
|---|---|---:|---:|---:|
| A `detanet_original_uv` | 完整原版 DetaNet，逐原子 MLP 后求和，直接输出 240 点谱 | 1,419,568 | 1,371,840 | 47,728 |
| B `detanet_mto_planned` | 同一完整骨干 → 规划逐原子 MTO → CG → E/Q/C/A/f → 展宽 | 1,552,092 | 1,371,840 | 180,252 |
| C `detanet_mto_global_gate` | 与 B 同类、同宽度，仅将组装系数的局部上下文换成分子均值 | 1,552,092 | 1,371,840 | 180,252 |

A 保持作者 `uv_model()`：`num_features=128, maxl=3, num_block=3, radial_type='trainable_bessel', num_radial=32, attention_head=8, rc=5.0, act='swish', dropout=0.0, use_cutoff=False, max_atomic_number=9, atom_ref=None, scale=1.0, scalar_outsize=240, irreps_out=None, summation=True, norm=False, out_type='scalar', grad_type=None`。保留原版逐原子 `128→128→240` MLP、偏置、初始化与求和顺序。没有输出截负、Softplus、逐谱归一化或新增骨干 RMS。

B/C 保留骨干 3o 消息，只在 MTO 接口选择原始 0e/1o/2e latent。每类型 16 通道、1 个参考查询和 10 个目标查询、query_dim=32、router/head_hidden=128；同 seed 的共同骨干逐张量相同，B/C 整个初始 state_dict 相同。A 的实测有梯度参数数为 1,370,032，B/C 为 1,552,092；不是靠调整原版头做参数匹配。

规划公式：`B=W H`（只混同类型通道）、`c=tanh(phi(u_i,g,q_a))`、`F=c B`、`M=gamma sum_i F`，`gamma=1/sqrt(18)`；系数可正可负，m 维统一广播，没有原子 softmax。使用 e3nn 的七类合法 CG 路径输出 0e/2e，保留规划不变量读出与张量门控。`E>0`，`C=beta I/sqrt(3)+Q`，`A=C C^T`，`f=(2/3)(E/27.211386245988)tr(A)`，最后用单位积分 Gaussian 合成谱。仅有光谱监督，不能唯一识别真实态、轨道或 A 的方向。

完整来源、白名单适配、逐层误差及上游边界行为见 [SOURCE_AND_MODEL_AUDIT.md](SOURCE_AND_MODEL_AUDIT.md)；完整规划文本与 [规划—实现—测试映射](PLAN_IMPLEMENTATION_TEST_MAP.md) 一并归档。

## 数据、网格与冻结训练协议

- 使用原身份固定的 800 train / 100 validation / 100 test；主冒烟只用固定 32 个训练 ID，保存在 `data/frozen.json`，共同目标形状 `[32,240]`。
- 保留 601 点源谱，按固定线性插值适配到作者 notebook 的 `torch.linspace(1.5,13.5,240)` 坐标。数据、坐标、清单、插值检查和 SHA256 均在 `data/` 与 `audit/`。
- 仅由原 800 个训练分子计算 `train_RMS=0.024564612974391966`、原子数中位数 `N_ref=18`。公共损失为 `mean(((prediction-target)/train_RMS)^2)`，缩放不进入 A 网络。
- 32 个训练分子的零谱归一化 MSE 为 **0.7710882439**，逐点共同均值谱为 **0.4633770329**；零谱误差不是 1。
- `source_broadening_status=unknown`；MTO `sigma=0.2 eV` 是披露的初始假设。未知的源单位/面积转换未被臆造。历史测试集不宣称“完全未接触”。
- v2 主协议唯一政策变化是三组共同 `smoke_max_steps: 600→2000`，保持最佳归一化训练 MSE `<0.1` 且低于初值 30% 的门槛和达标提前结束。600 步是工程预算，不是论文要求。诊断工具的两项工程修复另行记录。
- 正式配置仍为五种子 `[11,23,37,53,71]` × A/B/C；Adam(AMSGrad)、lr=1e-3、weight_decay=0、有效 batch=64、FP32、AMP/TF32 关闭。平台期调度 factor=0.5、patience=50、min_lr=1e-6；max_epochs=1000、min_epochs=200、early_stopping_patience=150，并要求至少两次实际降学习率、最后一次降学习率后至少 50 epochs。
- 最多两张调度分配 GPU、只做 1k。正式初始化不能使用作者、旧实验、冒烟或诊断权重。因 A 门槛失败，本轮没有执行正式训练或测试。

**本次是原版模型代码/架构等价、目标数据网格适配，不是论文全过程复现。** 论文归档 DOI 访问失败；上述 GitHub 版本与论文归档尚未验证逐字一致。本轮小数据预算和公共标签缩放也不是已证实的作者原样训练设置。

## 实际结果与科学解释

同一 seed=11、同一训练子集的新主轨迹（job 1299173）：

| 分支 | 初始归一化 MSE | 最佳归一化 MSE | 实际步数 | 主冒烟 |
|---|---:|---:|---:|---|
| A 原版 DetaNet | 33109.1953 | 0.2596252561 | 2000 | 未通过 |
| B 规划 MTO | 0.9325631 | 0.0757228807 | 170 | 通过 |
| C global-gate | 0.9221783 | 0.0749933794 | 160 | 通过 |

A 的初始分子输出 RMS 为 4.4696686，目标 RMS 为 0.0215706，约相差 207 倍；独立新建作者模型的初始张量、初始化调用顺序及 CPU RNG 均相同。没有通过加载共同权重来代替实际初始化审计。标签对齐、求和及 `normalized_MSE=raw_MSE/train_RMS²` 检查通过。

独立冻结副本的最后线性层 SVD 诊断使用 CPU float64，`X=[sum_i h_i,N]` 为 `[32,129]`，末列是原子数。初始化与 step2000 均秩 32，条件数分别 7997.497、3471.232；写回原版最后层后完整前向 MSE 分别约 1.02e-26、2.10e-28。它仅说明该训练子集可由线性读出插值，不证明泛化，也不替代主冒烟。

从本轮真实 step600 复制后各运行 300 步：D0 保留全部 AMSGrad 状态，最佳 MSE 0.4955131；D1 仅重置历史最大二阶矩，最佳 0.4719530；D2 重建同配置 AMSGrad，最佳 0.4282789。均为 **DIAGNOSTIC_ONLY**，未进入正式实验。D2 还改变了一阶矩、二阶矩和偏置修正，并出现步前损失 1221.6510 的反弹，不能仅归因于历史最大值。

第一份 D1 因优化器 CPU step 张量别名污染而作废，已保留其 JSON、权重和执行源码；另一次修正因观察器空状态检查失败，在 0 个训练步时退出。最终有效 D1 在 job 1299178 完成。只将 `reports/D1_correction_1299178/D1/` 用于有效比较。

CPU float64、GPU float64、GPU float32 各 7 项模型验收通过；诊断工具 6 项回归通过。独立原版层级输出及共同梯度的 CPU 双精度最大差为 0；实际环境中的 O(3)、平移/置换/batch、MTO/CG、物理代数及梯度、保存恢复检查与误差记录均已保留。19 份本轮有效检查点真实重载并重算损失，最大差 1.19209e-7。另有训练分子 decoder-oracle、合成可实现谱测试、初始组装导出和早期续训验收原始产物。

以上不能推出“MTO 泛化优于原版”。A↔B 对比完整读出系统；B↔C 仅诊断逐原子选择。详细分析见 [DIAGNOSTIC_CONCLUSIONS.md](DIAGNOSTIC_CONCLUSIONS.md) 和 [曲线报告](reports/instrumented_v2_1299173/DIAGNOSTIC_REPORT.md)。

## 作业与环境

| Job ID | 内容 | 状态/退出码 | 节点 | 时长 |
|---|---|---|---|---|
| 1298973 | v1 600 步前置检查 | FAILED / 1:0 | g0006 | 7:25 |
| 1299137 | 早先 v2 2000 步前置检查 | FAILED / 1:0 | g0017 | 9:20 |
| 1299173 | 本轮带完整记录的主冒烟、SVD、优化器诊断 | FAILED / 1:0，A 未过门槛 | g0056 | 10:21 |
| 1299177 | D1 修正尝试，训练 0 步 | FAILED / 1:0 | g0048 | 2:11 |
| 1299178 | 有效 D1 修正及权重复核 | COMPLETED / 0:0 | g0024 | 1:15 |

本轮主作业使用 1 GPU、8 CPU；Python 3.10.18、PyTorch 2.3.0+cu121、e3nn 0.4.4、PyG 2.6.1、scatter 2.1.2+pt23cu121、cluster 1.6.3+pt23cu121、RTX 4090。具体路径、驱动、依赖和 job 环境以 `logs/environment-*.json` 为准。GPU scatter 不保证后续训练轨迹逐位确定；保存恢复验收区分严格状态还原与数值演化误差。

远端：`/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_1k_20260920`。远端归档时 Git HEAD 为 `15af391115fb5179521397ea7f75b59153dac482`。主作业执行源码与最终修正源码分别保存在对应 `executed_source/`，不能把归档 HEAD 误当作每个作业的执行版本。版本和归档来源见 [archive_manifests/publication.json](archive_manifests/publication.json)。

## 文件与验证

| 路径 | 内容 |
|---|---|
| `models.py`, `train.py`, `preflight.py`, `smoke_diagnostics.py` | 实际模型、统一训练入口、验收与诊断实现 |
| `protocol.json`, `configs/`, `jobs/` | 冻结配置、真实 Slurm 脚本及提交记录 |
| `reference/`, `vendor/`, `planning/`, `audit/` | 作者源码/许可、完整规划、来源与修改审计 |
| `data/` | 本次固定 1k 数据副本、网格与划分；不含全量原数据 |
| `logs/`, `tests/`, `reports/` | CPU/GPU 真实日志、测试、曲线、JSON、状态恢复和张量导出 |
| `reports/instrumented_v2_1299173/` | 主冒烟、SVD、有效 D0/D2 及作废 D1 原记录 |
| `reports/D1_correction_1299178/` | 唯一有效 D1 修正结果及检查点 |
| `history/` | v1、先前 v2 的源码、失败、日志及检查点 |
| `archive_manifests/` | 508 个远端实验文件的路径/大小/SHA256、原诊断包清单及发布核验 |

所有实验文件以实际文件归档，含模型与优化器权重。两份原 ZIP 已展开；重复压缩容器不再嵌套提交。远端环境目录、`.git` 内部对象、Python 缓存及 `deliverables/` 的重复打包容器不归档；源 ZIP 的 SHA256 与文件清单保留。原始报告中 `deliverables/` 的包链接是当时交付路径，本仓库以展开的内容和清单提供同一材料，并补齐其余历史产物。

从本目录运行以下只读归档核验，无需安装 PyTorch：

```bash
python verify_archive.py
```

这验证归档传输完整性与主要状态一致性，不重新冒充 CPU/GPU 科学验收。集群脚本保留已核实的远端路径；归档并不自动启动任何作业。正式训练仍受原门槛阻断。

旧仓库根目录的 32 通道模型和 `results/1k`、`results/10k`、`results/full` 保留为历史实验。它们的头、骨干和目标协议与本次不同，旧完成标记只属于旧实验，不能拼接为本次的新结果。
