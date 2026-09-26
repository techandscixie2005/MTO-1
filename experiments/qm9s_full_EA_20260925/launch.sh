#!/usr/bin/env bash
set -euo pipefail
cd /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925
test -f reports/preflight.json
env/bin/python -m compileall -q train_ea.py models_ea.py data_ea.py analyze.py supervisor.py
env/bin/python -m pip freeze > reports/environment_freeze.txt
nvidia-smi -q > reports/nvidia_before_launch.txt
env/bin/python - <<'PY'
import json,pathlib,sys,torch,e3nn,torch_geometric,rdkit
from train_ea import fingerprint,atomic_json
root=pathlib.Path('.')
atomic_json(dict(python=sys.version,torch=torch.__version__,e3nn=e3nn.__version__,torch_geometric=torch_geometric.__version__,rdkit=rdkit.__version__,cuda=torch.version.cuda,fingerprint=fingerprint()),root/'reports/execution_environment.json')
from analyze import metric_arrays
import numpy as np
E=np.tile(np.arange(1.,11.),(2,1));A=np.tile(np.eye(3),(2,10,1,1))
m,per=metric_arrays(E,A,E,A,dict(sE2=2.,sA2=3.))
assert m['L']==0 and m['spectrum_MSE']==0 and (per==0).all()
assert json.loads((root/'reports/preflight.json').read_text())['status']=='PASS'
print('Launch gates passed')
PY
tmux new-session -d -s qm9s_full_ea 'env/bin/python supervisor.py > logs/supervisor.log 2>&1'
