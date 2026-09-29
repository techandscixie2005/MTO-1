"""Small dependency metadata, without environment variables or secrets."""
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
packages={}
for name in ('torch','e3nn','numpy','torch-scatter','torch-cluster','torch-geometric','scipy','sympy','opt-einsum','opt-einsum-fx'):
    try:
        packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name]=None
record={'python_executable':sys.executable,'python_version':platform.python_version(),
        'packages':packages,'torch_file':torch.__file__,'cuda_build_version':torch.version.cuda,
        'cudnn_version':torch.backends.cudnn.version(),'platform':platform.platform()}
(ROOT/'ENVIRONMENT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
