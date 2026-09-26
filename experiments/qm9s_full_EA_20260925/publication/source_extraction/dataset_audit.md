# 实际日志提取审计

- 来源：`E:\DATA\QM9S\tdlog1-20000`
- 日志：133,885；未删除任何记录。
- 正常结束：133,727；明确报告响应收敛：133,727；完整几何：133,727。
- 实际态数分布：`{10: 133727, 0: 158}`。
- 标量标签：1,337,270；向量标签：1,337,270。
- 实际 route：`{'#p td=(nstates=10) b3lyp/TZVP nosymm geom=check guess=read': 133885}`。
- 坐标框架：`{'Input': 133727, 'None': 158}`。
- 异常统计：`{'not_exactly_one_electric_dipole_table': 158, 'no_final_normal_termination': 158, 'response_convergence_marker_absent': 158, 'no_excited_state_labels': 158, 'missing_or_incomplete_geometry': 158}`；详见 anomalies.jsonl。
- 振子强度与能量/偶极的一致性绝对残差：`{'count': 1337270, 'min': 0.0, 'max': 7.498808266692691e-05, 'mean': 2.4865487175549464e-05, 'p50': 2.4840616140542815e-05, 'p99': 4.951328958654084e-05}`。

保留全部实际打印的跃迁，不把迭代求解过程中的临时 Root 当作最终光谱。原始波长、能量、f、电偶极表和可用的速度/磁偶极均保留。没有施加总纲里的态数、展宽、划分或筛选设定。此处光谱是日志计算出的离散跃迁，不声称覆盖无限高能激发态或实验谱。

坐标来自同一 TD 日志的 orientation 表，单位 Å，原子顺序不变。优化协议、连接关系和分子身份去重未从其他来源推断。长度规范电偶极采用带标题的原子单位表；经逐分量核对后使用末尾 transition elements 中的高精度分量，并保留原表。A=μμᵀ 是明确标记的派生量，f_from_E_mu 仅用于一致性审计，不替换原始 f。暗态的打印零保持为零，缺失值使用 NaN/mask。

一致性关系参考 [PySCF 官方实现](https://pyscf.org/_modules/pyscf/gw/bse.html)，规范说明参考 [PySCF TDDFT 文档](https://pyscf.org/user/tddft.html)。


全部 NumPy 数组与 JSONL 逐分子、逐态核对一致；A 张量与偶极外积逐态核对通过。所有源文件 SHA-256 保存在 source_manifest.jsonl。

正常结束、几何完整、单次计算无歧义且全部标量标签齐全的记录：133,727；全部向量标签齐全的记录：133,727。
