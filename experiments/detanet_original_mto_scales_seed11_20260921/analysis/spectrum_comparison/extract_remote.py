import json
from pathlib import Path
import numpy as np
root=Path('/data/run01/sczc698/xxy/MTO/experiments/detanet_original_mto_scales_seed11_20260921')
ids=[19441,114196,43669,122349]
result={'selection':'first four pre-fixed 1k test molecules; no selection by prediction quality','ids':ids,'seed':11,'scales':{}}
for scale in ('1k','10k','full'):
    result['scales'][scale]={}
    for name,variant in [('DetaNet','detanet_original_uv'),('MTO','detanet_mto_planned')]:
        path=root/f'reports/{scale}/test_{variant}_seed11.npz'
        with np.load(path,allow_pickle=False) as a:
            lookup={int(mid):i for i,mid in enumerate(a['ids'])}
            assert all(mid in lookup for mid in ids),(scale,'molecules not all shared')
            ix=[lookup[mid] for mid in ids]
            entry={'target':a['target'][ix].tolist(),'prediction':a['prediction'][ix].tolist(),
                'mse':np.mean((a['target'][ix]-a['prediction'][ix])**2,axis=1).tolist(),'source':str(path)}
            result['scales'][scale][name]=entry
            if 'grid' not in result:result['grid']=a['grid'].tolist()
            else:assert np.array_equal(result['grid'],a['grid'])
            if scale!='1k':assert np.array_equal(result['scales']['1k'][name]['target'],entry['target'])
out=Path('/tmp/mto_seed11_common_spectra_20260922.json')
out.write_text(json.dumps(result),encoding='utf-8')
print(json.dumps({'output':str(out),'ids':ids,'scales':list(result['scales']),'same_targets':True}))
