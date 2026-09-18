import json,re,zipfile
from pathlib import Path
base=Path('D:/MTO大纲/完整架构实验');root=base/'报告'
reports=[root/f'{s}_报告.md' for s in ('1k','10k')]+[root/'完整模型结构复核.md']
files=reports+[root/'代码核验数据.json',root/'scaling_common_test.png']
for report in reports:
    for link in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
        if link.startswith(('http:','https:')):continue
        link=re.sub(r':\d+$','',link)
        p=Path(link) if re.match(r'^[A-Za-z]:',link) else report.parent/link
        assert p.exists(),(report,link)
for scale in ('1k','10k'):
    files+=list((root/scale).glob('*.png'))
    files += [root/scale/name for name in ('comparison.json','per_seed.csv','STAGE_COMPLETE.json')]
    assert not [p for p in (root/scale).iterdir() if p.is_file() and p.stat().st_size==0]
    assert json.loads((root/scale/'STAGE_COMPLETE.json').read_text())['passed']
archive=root/'1k_10k_报告包.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(root).as_posix())
    for p in [base/'source/src/models.py',base/'source/src/dataset.py',base/'source/scripts/train_scale.py',
              base/'source/configs/grid.json',base/'source/audit/verify_trained_architecture.py']:
        z.write(p,'代码/'+p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print('Package verified, bytes:',archive.stat().st_size)
