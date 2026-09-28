"""Review, authorization and supervised child identity gates before data access."""
import hashlib,json,os,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/"runs/one_probe"
LIMIT_BYTES=16*1024**3

def h(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def reviewed():
 auth_path=ROOT/"FIT_AUTHORIZATION.json";review_path=ROOT/"IMPLEMENTATION_REVIEW.json";seal_path=ROOT/"SOURCE_SEAL.json";pre_path=ROOT/"IMPLEMENTATION_PREFLIGHT.json"
 need(auth_path.is_file() and review_path.is_file() and seal_path.is_file() and pre_path.is_file(),"PENDING: authorization/review/seal")
 auth=read(auth_path);review=read(review_path);seal=read(seal_path)
 need(auth.get("approved") is True and auth.get("one_run") is True and review.get("passed") is True,"authorization or review not passed")
 need(auth.get("review_sha256")==h(review_path) and auth.get("protocol_sha256")==h(ROOT/"CONDITIONAL_PROTOCOL.md") and auth.get("source_seal_sha256")==h(seal_path),"authorization hashes")
 need(review.get("source_seal_sha256")==h(seal_path) and review.get("preflight_sha256")==h(pre_path) and review.get("protocol_sha256")==h(ROOT/"CONDITIONAL_PROTOCOL.md"),"review hashes")
 for name,field in (("fit_worker.py","fit_worker_sha256"),("resource_guard.py","resource_guard_sha256"),("run.py","supervisor_sha256"),("execution_gate.py","execution_gate_sha256")):
  need(review.get(field)==h(ROOT/name),"reviewed code drift "+name)
 for item in seal["files_pre_fit"]:need(h(item["path"])==item["sha256"],"source drift "+item["path"])
 return auth,review,seal

def child(phase):
 need(phase in ("train","validation"),"phase")
 reviewed()
 token=os.environ.get("MTO_DESCRIPTOR_PHASE_TOKEN")
 need(token is not None and len(token)>=32,"missing supervised phase token")
 path=RUN/f"SUPERVISOR_PHASE_{phase}.json"
 need(path.is_file(),"missing supervisor phase receipt")
 d=read(path);ppid=os.getppid()
 need(d.get("phase")==phase and d.get("parent_pid")==ppid and d.get("token")==token,"supervisor identity/token")
 stat=Path(f"/proc/{ppid}/stat").read_text().split()
 need(int(stat[21])==d.get("parent_start_ticks") and stat[2]!="Z","supervisor process start")
 need(d.get("auth_sha256")==h(ROOT/"FIT_AUTHORIZATION.json") and d.get("review_sha256")==h(ROOT/"IMPLEMENTATION_REVIEW.json") and d.get("source_seal_sha256")==h(ROOT/"SOURCE_SEAL.json"),"supervisor provenance")
 need(d.get("limit_bytes")==LIMIT_BYTES and 0<d.get("remaining_wall_seconds",0)<=4*3600,"supervisor budget")
 selected=d.get("selected_cpus")
 need(isinstance(selected,list) and len(selected)==8 and set(os.sched_getaffinity(0))==set(selected),"child CPU affinity")
 lim=resource.getrlimit(resource.RLIMIT_AS)
 need(lim==(LIMIT_BYTES,LIMIT_BYTES),"child address-space limit")
 return d
