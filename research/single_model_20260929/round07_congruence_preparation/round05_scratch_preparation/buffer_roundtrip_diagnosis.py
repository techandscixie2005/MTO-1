"""Synthetic initialization-only diagnosis of the preserved CPU fixture failure."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import copy,json
from common import ROOT,read,sha,immutable_json
from fresh_model import build_fresh
native=build_fresh(read(ROOT/'model_config.json'),read(ROOT/'TRAIN_STATISTICS.json'),True)
roundtrip=copy.deepcopy(native).double().float()
records=[]
for name,a in native.named_buffers():
    b=dict(roundtrip.named_buffers())[name]
    if a.dtype!=b.dtype or not __import__('torch').equal(a,b):
        records.append({'buffer':name,'native_dtype':str(a.dtype),'roundtrip_dtype':str(b.dtype),
            'max_abs_difference':float((a.double()-b.double()).abs().max()) if a.numel() else 0.})
assert records
out={'passed':True,'purpose':'Diagnose engineering fixture dtype roundtrip only','model_updates':0,'model_inference':False,
     'real_geometry_or_target_arrays_read':False,'changed_buffers':records,
     'initial_attempt_log_sha256':sha(ROOT/'CPU_PREFLIGHT_01.log'),
     'initial_attempt_source_sha256':sha(ROOT/'preflight_history/cpu_attempt01/cpu_preflight.py'),
     'current_source_sha256':sha(ROOT/'cpu_preflight.py'),'diagnostic_source_sha256':sha(__file__),
     'repair':'Fresh native-dtype model receives only hand-set adapter tensors; strict checkpoint checks/tolerances unchanged.'}
immutable_json(out,ROOT/'BUFFER_ROUNDTRIP_DIAGNOSIS.json');print(json.dumps(out,indent=2))
