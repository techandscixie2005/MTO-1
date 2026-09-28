"""Single-use CPU supervisor for reviewed 803-feature descriptor baseline."""
import json,os,secrets,sys,time
from pathlib import Path
from resource_guard import WALL_SECONDS,LIMIT_BYTES,CPU_LIMIT_SECONDS,supervised_fit,admit
from execution_gate import ROOT,RUN,h,reviewed

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def write_once(path,obj):
 p=Path(path)
 if p.exists():raise RuntimeError(f"single-use output already exists: {p}")
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def resource_stop(phase,why,remaining):
 write_once(RUN/"INCOMPLETE_RESOURCE.json",{"event":"INCOMPLETE_RESOURCE","phase":phase,"reason":why,"remaining_wall_seconds":remaining,"time":time.time()})
def main():
 auth,review,seal=reviewed()
 need(not RUN.exists(),"one-shot run already exists")
 start=time.monotonic();RUN.mkdir(parents=True)
 try:first_admission=admit()
 except Exception as exc:
  resource_stop("train","fresh_CPU_or_memory_admission_failed: "+str(exc),WALL_SECONDS)
  raise
 write_once(RUN/"SUPERVISOR_STARTED.json",{"event":"SUPERVISOR_STARTED","time":time.time(),"pid":os.getpid(),"start_ticks":int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),"auth_sha256":h(ROOT/"FIT_AUTHORIZATION.json"),"review_sha256":h(ROOT/"IMPLEMENTATION_REVIEW.json"),"source_seal_sha256":h(ROOT/"SOURCE_SEAL.json"),"admission":first_admission,"command":[sys.executable,str(ROOT/"fit_worker.py")],"test_access":False})
 env_base=os.environ.copy();env_base.update({"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","VECLIB_MAXIMUM_THREADS":"1","PYTHONUTF8":"1"})
 command=[sys.executable,str(ROOT/"fit_worker.py")]
 phase_status={}
 for phase in ("train","validation"):
  remaining=WALL_SECONDS-(time.monotonic()-start)
  if remaining<=0:
   resource_stop(phase,"total_four_hour_budget_exhausted_between_phases",remaining)
   raise RuntimeError("descriptor resource envelope expired")
  if phase=="train":admission=first_admission
  else:
   try:admission=admit()
   except Exception as exc:
    resource_stop(phase,"fresh_CPU_or_memory_admission_failed: "+str(exc),remaining)
    raise
  token=secrets.token_hex(32)
  receipt={"phase":phase,"time":time.time(),"parent_pid":os.getpid(),"parent_start_ticks":int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),"token":token,"auth_sha256":h(ROOT/"FIT_AUTHORIZATION.json"),"review_sha256":h(ROOT/"IMPLEMENTATION_REVIEW.json"),"source_seal_sha256":h(ROOT/"SOURCE_SEAL.json"),"limit_bytes":LIMIT_BYTES,"remaining_wall_seconds":remaining,"selected_cpus":admission["selected_cpus"],"admission":admission}
  write_once(RUN/f"SUPERVISOR_PHASE_{phase}.json",receipt)
  env=dict(env_base,MTO_DESCRIPTOR_PHASE_TOKEN=token)
  status=supervised_fit(command+["--phase",phase],ROOT,RUN/"INCOMPLETE_RESOURCE.json",env,RUN/f"{phase}.log",remaining,limit_bytes=LIMIT_BYTES,cpu_limit_seconds=CPU_LIMIT_SECONDS if phase=="train" else None,admission=admission)
  phase_status[phase]=status
  write_once(RUN/f"PHASE_{phase.upper()}_RESOURCE.json",status)
  if status["resource_failure"] or status["returncode"]!=0:
   if not (RUN/"INCOMPLETE_RESOURCE.json").exists():
    write_once(RUN/"FAILED.json",{"event":"FAILED","phase":phase,"status":status,"classification":"nonresource execution failure; no model result"})
   raise RuntimeError(f"descriptor {phase} failed; retained receipts/log")
  need((RUN/("FIT_FREEZE.json" if phase=="train" else "VALIDATION_RESULTS.json")).is_file(),"missing phase result")
 write_once(RUN/"PROBE_COMPLETE.json",{"event":"PROBE_COMPLETE","elapsed_seconds":time.monotonic()-start,"fit_freeze_sha256":h(RUN/"FIT_FREEZE.json"),"validation_results_sha256":h(RUN/"VALIDATION_RESULTS.json"),"phase_resource_receipts":{k:h(RUN/f"PHASE_{k.upper()}_RESOURCE.json") for k in phase_status},"source_seal_sha256":h(ROOT/"SOURCE_SEAL.json"),"review_sha256":h(ROOT/"IMPLEMENTATION_REVIEW.json"),"test_access":False})
 print(json.dumps({"event":"PROBE_COMPLETE","elapsed_seconds":time.monotonic()-start}))
if __name__=="__main__":main()
