# 复核与从头复现

## 不需要checkpoint的结果复核

在本实验目录运行：

```bash
python publication/verify_archive.py --restore-data --metrics
```

该命令核验发布文件/原始服务器文件、禁止checkpoint、检查gzip块并恢复byte-identical数据；用六组完整NPZ预测复算主要测试指标。更详细的分析脚本为analyze_final_independent.py，运行时设置RESULTS_ROOT为本实验绝对路径；它需要NumPy和Matplotlib，不需要checkpoint或GPU。该脚本会重写同名诊断报告，建议先复制到独立目录。

## 环境

原始环境完整记录在reports/environment_freeze.txt。它来自已有conda环境，包含与本任务无关的软件及不可移植的file://来源路径；不要直接在另一机器pip install -r该快照。

实际主要依赖：Python3.10.19、torch2.5.1+cu121、torch-geometric2.7.0、torch_scatter2.1.2+pt25cu121、torch_cluster1.6.3+pt25cu121、e3nn0.4.4、opt-einsum-fx0.1.4、opt-einsum3.4.0、numpy2.2.6、scipy1.15.3、sympy1.13.1、matplotlib3.10.8、scikit-learn1.7.2。只有重做几何分组时才需原版RDKit2025.9.4。PyTorch扩展须与Torch/CUDA/Python ABI匹配。现有冻结数据复现不要求重新推断成键。

## 从头训练，不覆盖历史记录

恢复dataset后，准备一个不存在的新目录：

```bash
python publication/prepare_fresh_run.py --destination /path/to/new_qm9s_ea
cd /path/to/new_qm9s_ea
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python preflight.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python train_ea.py detanet_reference
```

其余配置名见configs/*.json，可以分别绑定健康GPU并行运行。历史配置的gpu字段只是原始分配元数据；train_ea.py由CUDA_VISIBLE_DEVICES决定实际设备。需要兼容CUDA环境，CPU-only不支持原训练入口。新目录不会带历史run状态或checkpoint，不会被FIT_COMPLETE跳过。

prepare_fresh_run只复制冻结代码/配置/数据并建立空输出目录，不会启动训练。原launch.sh/supervisor.py保留原服务器路径，是历史运行入口，不应直接在别的机器执行。

## 测试使用规则

原analyze.py在历史故障后被修改成允许排除未完成run。这是已披露的历史例外，不能用作新实验的默认做法。对新的七组复现，应先确认七个FIT_COMPLETE.json均存在、按验证选型后再运行分析；不要因某组测试差或未完成而静默剔除。任何新架构/损失设计都不能继续把已经观察过的本次测试集当未知调参依据。

无checkpoint意味着无法重复旧模型的前向推理或恢复旧优化器。原指标可由存储预测独立复算，训练可从头重跑；FP32 GPU原子归约及硬件/依赖差异可能导致数值轨迹不同。
