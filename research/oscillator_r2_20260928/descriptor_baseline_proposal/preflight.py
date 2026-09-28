"""Synthetic and 16-row TRAIN-only descriptor preflight. No labels or fit."""
import hashlib,json,os,sys,tempfile,time,subprocess
from pathlib import Path
import numpy as np
import sklearn,rdkit,psutil
from features import extract,SCHEMA_NAMES,FEATURES,PAIR_KEYS,CENTERS
from array_io import member
from resource_guard import admit,supervised_fit

ROOT=Path(__file__).resolve().parent
DATA=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/data")
METHANE_SHA="23db7694c5a0c75d3d636850673e35e9308958e3f45f10fcac20a40e6ba4f24e"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise RuntimeError(msg)
def same(a,b,msg):
 need(np.allclose(a,b,rtol=1e-5,atol=1e-6),msg)
 return float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
def main():
 cfg=json.loads((ROOT/"IMPLEMENTATION_CONFIG.json").read_text())
 need(sha(ROOT/"CONDITIONAL_PROTOCOL.md")==cfg["protocol_sha256"],"frozen protocol changed")
 pinned=json.loads((DATA/"hashes.json").read_text())
 for name in ("dataset.npz","splits.json","normalization.json","identity_audit_v2.json"):
  need(sha(DATA/name)==pinned[name],"frozen input hash "+name)
 need(sha(DATA/"raw_labels.npz")==cfg["raw_labels_sha256"],"raw label source hash")
 need(sklearn.__version__=="1.7.2" and rdkit.__version__=="2025.09.4","library version")
 need(FEATURES==803 and len(SCHEMA_NAMES)==803 and len(PAIR_KEYS)==15 and len(CENTERS)==16,"feature schema")
 synthetic={}
 z=np.array([6,1,1,1,1,0],dtype=np.int64)
 xyz=np.array([[0,0,0],[.6,.6,.6],[-.6,-.6,.6],[-.6,.6,-.6],[.6,-.6,-.6],[0,0,0]],dtype=np.float64)
 base,info=extract(z,xyz)
 need(hashlib.sha256(base.tobytes()).hexdigest()==METHANE_SHA,"literal methane fingerprint fixture")
 need(info["edges"]==4 and info["components"]==1,"methane graph")
 M=np.array([[0,1,0],[-1,0,0],[0,0,1]],dtype=np.float64)
 transforms=(xyz+np.array([2.5,-9.25,3]),xyz@M,xyz*np.array([-1,1,1]),xyz.copy())
 maxima=[]
 for x in transforms[:3]:maxima.append(same(base,extract(z,x)[0],"synthetic rigid invariance"))
 perm=np.array([5,4,2,1,3,0])
 maxima.append(same(base,extract(z[perm],xyz[perm])[0],"synthetic padding permutation"))
 synthetic["max_abs_feature_difference"]=max(maxima)
 coincident,coincident_diag=extract(np.array([1,1,0]),np.zeros((3,3),dtype=np.float64))
 need(coincident_diag["edges"]==1 and np.isfinite(coincident).all(),"coincident synthetic geometry")
 for x,expect,near in ((.806-2e-9,1,0),(.806,1,1),(.806+2e-9,0,0)):
  _,g=extract(np.array([1,1,0]),np.array([[0,0,0],[x,0,0],[0,0,0]],dtype=np.float64))
  need(g["edges"]==expect and g["near_graph_threshold_pairs"]==near,"boundary behavior")
 with np.load(DATA/"dataset.npz",allow_pickle=False) as d:
  train=np.asarray(member(DATA/"dataset.npz","train")).copy()
  need(train.ndim==1 and len(train)==120355,"train split count")
  zall=member(DATA/"dataset.npz","z");pall=member(DATA/"dataset.npz","pos")
  # Only these frozen train indices are transformed into descriptors.
  locations=np.rint(np.linspace(0,len(train)-1,16)).astype(np.int64)
  indices=train[locations]
  train_details=[];maxdiff=0.0;near=0
  for ix in indices:
   zrow=zall[ix];prow=pall[ix]
   f,g=extract(zrow,prow)
   perm=np.arange(len(zrow))[::-1]
   variants=(extract(zrow,prow+np.array([1.125,-2.5,3.875]))[0],
             extract(zrow,np.asarray(prow,dtype=np.float64)@M)[0],
             extract(zrow,np.asarray(prow,dtype=np.float64)*np.array([-1,1,1]))[0],
             extract(zrow[perm],prow[perm])[0])
   for v in variants:maxdiff=max(maxdiff,same(f,v,"TRAIN row invariance "+str(ix)))
   near+=g["near_graph_threshold_pairs"]
   train_details.append({"dataset_index":int(ix),"atoms":g["atoms"],"edges":g["edges"],
       "components":g["components"],"near_graph_threshold_pairs":g["near_graph_threshold_pairs"],
       "minimum_graph_threshold_gap_angstrom":g["minimum_graph_threshold_gap_angstrom"],
       "descriptor_sha256":hashlib.sha256(f.tobytes()).hexdigest()})
 need(near==0,"TRAIN sample graph threshold ambiguity requires review")
 admission=admit()
 with tempfile.TemporaryDirectory(prefix="descriptor_guard_fixture_") as td:
  log=Path(td)/"synthetic_child.log"
  status=supervised_fit([sys.executable,"-c","print(\"SYNTHETIC_GUARD_OK\")"],ROOT,
                        Path(td)/"resource_failure.json",os.environ.copy(),log,10)
  need(status["returncode"]==0 and status["resource_failure"] is None and
       b"SYNTHETIC_GUARD_OK" in log.read_bytes(),"synthetic resource guard")
  timeout=supervised_fit([sys.executable,"-c","import time;time.sleep(2)"],ROOT,
      Path(td)/"timeout_resource.json",os.environ.copy(),Path(td)/"timeout.log",.12)
  need(timeout["resource_failure"]=="wall_clock_over_total_budget" and
       json.loads((Path(td)/"timeout_resource.json").read_text())["event"]=="INCOMPLETE_RESOURCE","short wall resource classification")
  memory=supervised_fit([sys.executable,"-c","raise MemoryError(\"synthetic low-memory fixture\")"],ROOT,
      Path(td)/"memory_resource.json",os.environ.copy(),Path(td)/"memory.log",10,limit_bytes=128*1024**2)
  need(memory["resource_failure"]=="address_space_or_allocation_failure" and
       json.loads((Path(td)/"memory_resource.json").read_text())["event"]=="INCOMPLETE_RESOURCE","low memory resource classification")
  direct=subprocess.run([sys.executable,str(ROOT/"fit_worker.py"),"--phase","train"],cwd=ROOT,
       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=20)
  need(direct.returncode!=0 and "PENDING: authorization/review/seal" in direct.stdout and
       not (ROOT/"runs/one_probe").exists(),"direct child blocked before arrays")
  import execution_gate as eg
  old_root,old_run=eg.ROOT,eg.RUN
  try:
   eg.ROOT=Path(td);eg.RUN=Path(td)/"runs/one_probe"
   for name,content in (("FIT_AUTHORIZATION.json",{"approved":False,"one_run":True}),("IMPLEMENTATION_REVIEW.json",{"passed":True}),("SOURCE_SEAL.json",{"files_pre_fit":[]}),("IMPLEMENTATION_PREFLIGHT.json",{"passed":True})):
    (Path(td)/name).write_text(json.dumps(content))
   try:eg.reviewed()
   except RuntimeError as exc:need("authorization or review not passed" in str(exc),"invalid authorization classification")
   else:raise RuntimeError("invalid authorization accepted")
  finally:eg.ROOT,eg.RUN=old_root,old_run
 provenance_path=ROOT.parent/"postrun_geometry/FIXED_SEVEN_SECONDARY_PROVENANCE.json"
 need(sha(provenance_path)==cfg["fixed_five_provenance_sha256"],"fixed five provenance changed")
 provenance=json.loads(provenance_path.read_text())
 pre_files=[ROOT/name for name in ("CONDITIONAL_PROTOCOL.md","IMPLEMENTATION_AUTHORIZATION.md",
   "IMPLEMENTATION_CONFIG.json","features.py","array_io.py","resource_guard.py","execution_gate.py","preflight.py","fit_worker.py","run.py")]
 pre_files += [DATA/name for name in ("dataset.npz","raw_labels.npz","splits.json",
   "normalization.json","identity_audit_v2.json","hashes.json")]
 post_files=[ROOT.parent/"fixed_seven_sidecar/fixed_seven.py",
   ROOT.parent/"fixed_seven_sidecar/IMPLEMENTATION_REVIEW.json",
   ROOT.parent/"postrun_geometry/postrun_geometry.py",provenance_path]
 post=[{"path":str(p),"sha256":sha(p)} for p in post_files]
 for name,item in provenance["frozen_five_components"].items():
  post.append({"path":item["prediction_path"],"sha256":item["prediction_sha256"],
               "validation_array_verification_deferred_until_frozen_fit":True})
 seal={"protocol_sha256":cfg["protocol_sha256"],
       "files_pre_fit":[{"path":str(p),"sha256":sha(p)} for p in pre_files],
       "files_post_fit":post,
       "postfit_five_array_contents_not_opened_in_preflight":True}
 sealpath=ROOT/"SOURCE_SEAL.json"
 sealpath.write_text(json.dumps(seal,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 report={"passed":True,"classification":"synthetic and fixed 16-row TRAIN-only; no full table/fit/val/test",
         "time":time.time(),"protocol_sha256":cfg["protocol_sha256"],
         "authorization_sha256":sha(ROOT/"IMPLEMENTATION_AUTHORIZATION.md"),
         "config_sha256":sha(ROOT/"IMPLEMENTATION_CONFIG.json"),
         "features_sha256":sha(ROOT/"features.py"),
         "resource_guard_sha256":sha(ROOT/"resource_guard.py"),
         "preflight_script_sha256":sha(__file__),
         "source_seal_sha256":sha(sealpath),
         "data_hashes_verified":{k:pinned[k] for k in ("dataset.npz","splits.json","normalization.json","identity_audit_v2.json")},
         "versions":{"python":sys.version.split()[0],"sklearn":sklearn.__version__,"rdkit":rdkit.__version__,
                     "numpy":np.__version__,"psutil":psutil.__version__},
         "schema_columns":len(SCHEMA_NAMES),
         "schema_names_sha256":hashlib.sha256(json.dumps(SCHEMA_NAMES,separators=(",",":")).encode()).hexdigest(),
         "literal_methane_descriptor_sha256":METHANE_SHA,
         "synthetic":synthetic,"train_sample_rows":train_details,
         "max_train_invariance_abs":maxdiff,"near_threshold_train_pairs":near,
         "resource_admission_snapshot":admission,"synthetic_resource_guard_passed":True,"negative_resource_and_direct_child_checks":{"short_wall":timeout["resource_failure"],"low_memory":memory["resource_failure"],"direct_child_returncode":direct.returncode,"invalid_authorization_rejected":True},
         "full_feature_table":False,"fit":False,"validation_access":False,"test_access":False}
 path=ROOT/"IMPLEMENTATION_PREFLIGHT.json"
 path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 print(json.dumps({"passed":True,"train_rows":len(train_details),"max_abs_diff":maxdiff,"preflight_sha256":sha(path)}))
if __name__=="__main__":main()

