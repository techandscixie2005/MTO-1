# 提交状态快照

核查时间：2026-09-27 21:57:31 +08:00

远端目录：`/home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927`

监督器 PID：365730

| 组 | 状态 | PID | GPU | 轮次 | 已完成更新 | checkpoint更新数 | 日志 |
|---|---|---:|---:|---:|---:|---:|---|
| G1 | 正在训练 | 365739 | 1 | 2 | 2480 | 1881 | /home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927/logs/G1.log |
| G2 | 正在训练 | 365745 | 2 | 2 | 2140 | 1881 | /home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927/logs/G2.log |
| G3 | 正在训练 | 365754 | 4 | 2 | 2180 | 1881 | /home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927/logs/G3.log |
| G4 | 正在训练 | 365765 | 6 | 2 | 2300 | 1881 | /home/inspur/MTO-1/experiments/qm9s_pouter_trace_20260927/logs/G4.log |

四组首轮全量验证均已完成，best.pt及可恢复last.pt均已生成。训练尚未完成，测试尚未开始。各组完成后由监督器统一冻结best及哈希，再自动评估。

预检见 preflight.json、system_preflight.json；现场核查见 running_checks.json。查看和恢复命令见 ../README.md。
