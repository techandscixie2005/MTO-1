"""Read code cells only; never execute source notebooks."""
import json
from pathlib import Path
p=Path('/data/run01/sczc698/czl/shuju-323-527-fuwuqi/323/shuju')
for f in p.glob('*.ipynb'):
    for c in json.loads(f.read_text()).get('cells',[]):
        source=''.join(c.get('source',[]))
        if c.get('cell_type')=='code' and any(t in source for t in ('Gaussian','sigma','1.5-13.5','to_csv')):
            print(str(f), '\n', source[:5000])
