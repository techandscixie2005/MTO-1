# 四个固定测试分子的跨规模光谱对照

每行同一分子；左到右为1k、10k、全量；黑色真实谱、蓝色原版DetaNet、橙色planned MTO。同一分子的三列使用相同纵轴范围。A/B表示图内两模型的原始尺度MSE。

分子ID为19441、114196、43669、122349，沿用原先固定的1k测试列表前四个分子，不按结果筛选。提取时验证了三个规模均在测试集且目标逐点相同。所有预测来自各自最佳验证checkpoint；无平滑、逐谱归一化或额外训练。

![19441与114196](same_molecules_comparison_1.png)

![43669与122349](same_molecules_comparison_2.png)

`common_spectra.json`保存精确网格、真实谱、预测谱与MSE；`plot_comparison.py`使用NumPy/Matplotlib重画。SVG文件保留矢量版本。`extract_remote.py`记录原远端提取路径。
