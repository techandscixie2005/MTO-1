import importlib
import inspect
import json
import os
import platform
import sys
import torch
from e3nn import o3
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
result=dict(python=sys.version,executable=sys.executable,node=platform.node(),job_id=os.getenv('SLURM_JOB_ID'),
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
            tensor_product_signature=str(inspect.signature(o3.TensorProduct)),packages={})
for name in ['torch','e3nn','torch_geometric','torch_scatter','torch_cluster','numpy','scipy','rdkit']:
    module=importlib.import_module(name)
    result['packages'][name]=dict(version=module.__version__,file=module.__file__)
assert result['packages']['e3nn']['version']=='0.4.4'
print(json.dumps(result,indent=2))
