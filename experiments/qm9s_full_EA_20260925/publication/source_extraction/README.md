# QM9S 实际 TD 日志提取数据

数据来源为用户提供的 `E:\DATA\QM9S\tdlog1-20000`。目录名不用于判断样本范围。
以实际 Gaussian 日志为准；没有根据课题总纲预设协议、态数、展宽参数、样本剔除或数据划分。
统计结果见 `dataset_manifest.json` 和 `dataset_audit.md`。

## 文件

- `arrays/part-*.npz`：直接供 NumPy/PyTorch 读取的数组，float64 保留提取精度，无 pickle。
- `records/part-*.jsonl.gz`：每行一个分子的完整提取记录，包括原始打印值、route、单位约定、来源和异常。
- `source_manifest.jsonl`：每个原始日志的文件名、ID、字节数及 SHA-256。
- `anomalies.jsonl`：有警告的记录；所有记录仍保留在数据集中。
- `load_qm9s.py`：最小加载器和可选 Gaussian 展宽函数。网格和 sigma 必须由调用方指定。
- `extract_qm9s.py`：可复现提取脚本。依赖 Python 3.10+ 和 NumPy。
- `validation_report.json`：全量 JSONL/NPZ 和派生张量核对结果。
- `SHA256SUMS`：发布文件的 SHA-256 校验清单。

原始 `.log` 继续保留在本地源目录；上传的是提取数据及脚本，并未复制全部原始日志。

## 数组约定

每个分片的 N、Amax、Kmax 取自该片实际记录。原子和跃迁仅为批处理补齐；未改动原子顺序和打印态顺序。

| 字段 | 形状 | 含义 |
|---|---|---|
| molecule_id | N | 文件名中的 ID，仅用于追溯 |
| atomic_numbers / atom_mask | N,Amax | 原子序数 / 真实原子位置 |
| positions_angstrom | N,Amax,3 | 同一日志 orientation 表，Å |
| state_index / state_mask | N,Kmax | 原始态编号 / 真实打印态 |
| energy_eV / wavelength_nm | N,Kmax | 原始打印激发能 / 波长 |
| oscillator_strength | N,Kmax | 原始长度规范振子强度，dimensionless |
| transition_dipole_au | N,Kmax,3 | 长度规范跃迁电偶极，atomic units |
| transition_dipole_table_au | N,Kmax,3 | 原电偶极表的 4 位小数值 |
| A_au2 | N,Kmax,3,3 | 从选用偶极计算的 μμᵀ，派生量 |
| scalar_label_mask / vector_label_mask | N,Kmax | 字段存在、有限且根编号唯一；不代表计算成功 |
| normal_termination | N | 日志最后一次结束标记正常 |
| response_convergence_reported | N | 日志包含响应向量收敛标记 |
| unambiguous_single_calculation | N | 单 route、单电偶极表、单几何表且根编号唯一 |
| geometry_present | N | 坐标数与日志 NAtoms 一致 |
| charge / multiplicity | N | 日志打印净电荷和基态自旋多重度 |

另含速度规范偶极与振子强度、磁跃迁偶极、电偶极表强度及从 E、μ 计算的 f，用于审计。
缺失浮点值为 NaN；补齐的原子序数/态编号为 0，必须使用 mask。物理暗态打印的 0 没有改成缺失。
偶极高精度来源在 JSONL 中逐态记录，只有和带标题表逐分量吻合后才采用；原表始终保留。
日志能量与波长分别保留其打印精度，不通过互相换算覆盖。没有把迭代器中间的 Root 当作最终跃迁。

## 使用

```python
from load_qm9s import iter_molecules
for sample in iter_molecules('/home/inspur/datasets/QM9S/qm9s_td_extracted_20260925'):
    z = sample['atomic_numbers']
    pos = sample['positions_angstrom']
    energy = sample['energy_eV']
    f = sample['oscillator_strength']
    mu = sample['transition_dipole_au']
    A = sample['A_au2']
    # 使用对应 label_mask，并自行定义模型的态选择、batch 和 train/validation/test。
```

加载器默认要求正常结束、完整几何和无歧义的单次计算；`require_normal=False` 可遍历所有记录。
用于 PyTorch 时，可显式转换类型，例如 `torch.as_tensor(z, dtype=torch.long)` 和
`torch.as_tensor(pos, dtype=torch.float32)`；其余浮点标签按训练精度转换，并带上对应 mask。
未自动剔除离群光谱，也未假设所有记录都应有特定数目的态。warnings 应在具体训练前查看。
未推断 bonds/SMILES、几何优化方法或重复分子身份；未执行去重及数据划分。
若对坐标做旋转数据增强，μ 必须按同一旋转变换，A 按 RARᵀ 变换。
展宽函数返回以能量为横轴的归一化强度密度；它没有实验绝对截面单位。

复现提取：

```bash
python extract_qm9s.py --source /path/to/tdlogs --output /path/to/new_output --workers 8
python finalize_qm9s.py /path/to/new_output
python verify_source_samples.py /path/to/new_output
```

验证本次发布的文件：

```bash
sha256sum -c SHA256SUMS
python remote_verify.py .
```
