"""Future one-shot descriptor fit/validation worker. Never run during implementation preflight."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import joblib,numpy as np,sklearn
from sklearn.ensemble import ExtraTreesRegressor
from features import extract,SCHEMA_NAMES,FEATURES
from array_io import member
from execution_gate import child as child_gate

ROOT=Path(__file__).resolve().parent
SOURCE=Path("/home/inspur/MTO-1/experiments/qm9s_eta_Ef_20260926/data")
RESEARCH=ROOT.parent
RUN=ROOT/"runs/one_probe"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
def save_json(path,obj):
 path=Path(path)
 if path.exists():raise RuntimeError(f"immutable output already exists: {path}")
 tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
 os.replace(tmp,path)
def need(ok,why):
 if not ok:raise RuntimeError(why)
def metric(y,p):
 y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
 need(y.shape==p.shape and y.ndim==2 and y.shape[1]==10 and np.isfinite(y).all() and np.isfinite(p).all(),"metric shape/finiteness")
 sse=float(np.square(p-y).sum(dtype=np.float64));sst=float(np.square(y-y.mean()).sum(dtype=np.float64))
 need(sst>0,"zero SST")
 return {"molecules":len(y),"states":10,"sse":sse,"sst":sst,"r2":1-sse/sst,
         "mae":float(np.abs(p-y).mean()),"rmse":float(np.sqrt(sse/y.size))}
def load_seal(phase):
 seal=json.loads((ROOT/"SOURCE_SEAL.json").read_text())
 for item in seal["files_pre_fit"]+(seal["files_post_fit"] if phase=="validation" else []):
  need(sha(item["path"])==item["sha256"],"source changed "+item["path"])
 need(seal["protocol_sha256"]==sha(ROOT/"CONDITIONAL_PROTOCOL.md"),"protocol changed")
 return seal
def main_train():
 child_gate("train")
 need(RUN.is_dir() and not (RUN/"TRAIN_STARTED.json").exists(),"single-use train phase already entered")
 start=time.monotonic()
 cfg=json.loads((ROOT/"IMPLEMENTATION_CONFIG.json").read_text())
 seal=load_seal("train")
 save_json(RUN/"TRAIN_STARTED.json",{"event":"TRAIN_STARTED","source_seal_sha256":sha(ROOT/"SOURCE_SEAL.json"),"time":time.time()})
 train=np.asarray(member(SOURCE/"dataset.npz","train")).copy()
 ids=member(SOURCE/"dataset.npz","ids");z=member(SOURCE/"dataset.npz","z");pos=member(SOURCE/"dataset.npz","pos")
 rawids=member(SOURCE/"raw_labels.npz","ids")
 need(np.array_equal(rawids[train],ids[train]),"raw train label IDs")
 y=np.asarray(member(SOURCE/"raw_labels.npz","f")[train],dtype=np.float64)
 mask=np.asarray(member(SOURCE/"raw_labels.npz","mask_f")[train],dtype=bool)
 need(len(train)==120355 and y.shape==(120355,10) and mask.all(),"train rows/masks")
 need(np.isfinite(y).all() and (y>=0).all(),"train raw-f targets")
 X=np.lib.format.open_memmap(RUN/"TRAIN_FEATURES.npy",mode="w+",dtype=np.float32,shape=(len(train),FEATURES))
 near=0;min_gap=float("inf")
 feature_start=time.monotonic()
 for j,ix in enumerate(train):
  features,diag=extract(z[ix],pos[ix]);X[j]=features
  near+=diag["near_graph_threshold_pairs"]
  if diag["minimum_graph_threshold_gap_angstrom"] is not None:min_gap=min(min_gap,diag["minimum_graph_threshold_gap_angstrom"])
  if (j+1)%10000==0:print(json.dumps({"event":"TRAIN_FEATURE_PROGRESS","done":j+1,"total":len(train)}),flush=True)
 X.flush();train_feature_seconds=time.monotonic()-feature_start
 del z,pos,ids
 need(np.isfinite(X).all(),"nonfinite train features")
 mean=float(y.mean());scale=float(np.sqrt(np.mean(np.square(y-mean))))
 need(scale>0 and np.isfinite(scale),"target pooled scale")
 params=cfg["learner"]
 model=ExtraTreesRegressor(**params)
 need(sklearn.__version__=="1.7.2","sklearn version")
 fit_start=time.monotonic()
 model.fit(X,y/scale)
 fit_seconds=time.monotonic()-fit_start
 need(len(model.estimators_)==256,"forest incomplete")
 pred_start=time.monotonic()
 pred=np.asarray(model.predict(X),dtype=np.float64)*scale
 train_predict_seconds=time.monotonic()-pred_start
 train_metrics=metric(y,pred)
 joblib.dump(model,RUN/"FOREST.joblib",compress=0)
 np.savez_compressed(RUN/"TRAIN_PREDICTIONS.npz",indices=train,truth=y,pred=pred)
 nodes=[int(t.tree_.node_count) for t in model.estimators_]
 freeze={"event":"FIT_FREEZE","time":time.time(),"source_seal_sha256":sha(ROOT/"SOURCE_SEAL.json"),
         "preflight_sha256":sha(ROOT/"IMPLEMENTATION_PREFLIGHT.json"),
         "schema_names_sha256":hashlib.sha256(json.dumps(SCHEMA_NAMES,separators=(",",":")).encode()).hexdigest(),
         "feature_hash":sha(RUN/"TRAIN_FEATURES.npy"),"forest_hash":sha(RUN/"FOREST.joblib"),
         "train_predictions_hash":sha(RUN/"TRAIN_PREDICTIONS.npz"),
         "train_indices_sha256":hashlib.sha256(train.tobytes()).hexdigest(),
         "train_ids_sha256":hashlib.sha256(np.asarray(rawids[train]).tobytes()).hexdigest(),
         "effective_estimator_params":model.get_params(deep=True),
         "train_feature_seconds":train_feature_seconds,
         "fit_seconds":fit_seconds,"train_prediction_seconds":train_predict_seconds,
         "train_target_scale":scale,"train_target_mean":mean,
         "train_metrics":train_metrics,"near_threshold_train_pairs":near,
         "minimum_graph_threshold_gap_angstrom":min_gap,
         "n_estimators":len(nodes),"total_tree_nodes":sum(nodes),"max_tree_nodes":max(nodes),
         "model_bytes":(RUN/"FOREST.joblib").stat().st_size,
         "train_features_bytes":(RUN/"TRAIN_FEATURES.npy").stat().st_size,
         "elapsed_seconds":time.monotonic()-start,"test_access":False,"validation_access":False}
 save_json(RUN/"FIT_FREEZE.json",freeze)
 print(json.dumps({"event":"FIT_FREEZE","train_r2":train_metrics["r2"],"forest_sha256":freeze["forest_hash"]}),flush=True)
def main_validation():
 child_gate("validation")
 freeze_path=RUN/"FIT_FREEZE.json"
 need(freeze_path.is_file(),"fit is not frozen")
 need(not (RUN/"VALIDATION_STARTED.json").exists(),"validation is single-use")
 freeze=json.loads(freeze_path.read_text())
 need(freeze["event"]=="FIT_FREEZE" and freeze["source_seal_sha256"]==sha(ROOT/"SOURCE_SEAL.json"),"fit seal changed")
 seal=load_seal("validation")
 need(sha(RUN/"FOREST.joblib")==freeze["forest_hash"] and sha(RUN/"TRAIN_FEATURES.npy")==freeze["feature_hash"],"frozen fit artifact changed")
 save_json(RUN/"VALIDATION_STARTED.json",{"event":"VALIDATION_STARTED","fit_freeze_sha256":sha(freeze_path),"time":time.time()})
 model=joblib.load(RUN/"FOREST.joblib")
 sys.path.insert(0,str(RESEARCH))
 from fixed_seven_sidecar import fixed_seven as f7
 provenance,old,pins,rows=f7.static_pins()
 five,F5=f7.load_five(provenance,rows)
 val=np.asarray(member(SOURCE/"dataset.npz","val")).copy()
 ids=member(SOURCE/"dataset.npz","ids");z=member(SOURCE/"dataset.npz","z");pos=member(SOURCE/"dataset.npz","pos")
 need(np.array_equal(val,rows["indices"]) and np.array_equal(ids[val],rows["ids"]),"validation row alignment")
 X=np.lib.format.open_memmap(RUN/"VAL_FEATURES.npy",mode="w+",dtype=np.float32,shape=(len(val),FEATURES))
 feature_start=time.monotonic()
 for j,ix in enumerate(val):
  X[j]=extract(z[ix],pos[ix])[0]
 X.flush();validation_feature_seconds=time.monotonic()-feature_start
 need(np.isfinite(X).all(),"nonfinite val features")
 predict_start=time.monotonic()
 D=np.asarray(model.predict(X),dtype=np.float64)*freeze["train_target_scale"]
 validation_prediction_seconds=time.monotonic()-predict_start
 blend=(D+F5)/2.0
 truth=rows["truth"];need(D.shape==(6686,10) and np.isfinite(D).all() and (D>=0).all(),"prediction shape/finite/nonnegative")
 np.savez_compressed(RUN/"VAL_PREDICTIONS.npz",ids=rows["ids"],indices=val,
                     f_true=truth,mask_f_true=rows["mask"],D=D,blend=blend)
 groups,keys=f7.identity_groups(rows)
 comparisons={"D_vs_F5":f7.compare(rows,F5,D,groups,keys),
              "equal_blend_vs_F5":f7.compare(rows,F5,blend,groups,keys)}
 result={"event":"VALIDATION_COMPLETE","classification":"one fixed descriptor forest and equal F5 blend; exploratory validation only",
         "time":time.time(),"fit_freeze_sha256":sha(freeze_path),
         "source_seal_sha256":sha(ROOT/"SOURCE_SEAL.json"),
         "val_feature_hash":sha(RUN/"VAL_FEATURES.npy"),
         "val_predictions_hash":sha(RUN/"VAL_PREDICTIONS.npz"),
         "metrics":{"D":metric(truth,D),"F5":metric(truth,F5),"equal_blend":metric(truth,blend)},
         "signed_error_pred_minus_truth":{label:{"overall_mean":float((pred-truth).mean()),"per_state_mean":[float(v) for v in (pred-truth).mean(axis=0)]} for label,pred in (("D",D),("F5",F5),("equal_blend",blend))},
         "train_metrics":freeze["train_metrics"],"comparisons":comparisons,
         "validation_feature_seconds":validation_feature_seconds,
         "validation_prediction_seconds":validation_prediction_seconds,
         "inference_cost":"D: 803 descriptors plus 256-tree CPU estimator; blend adds five neural forwards, not equal-latency passes",
         "test_access":False}
 save_json(RUN/"VALIDATION_RESULTS.json",result)
 print(json.dumps({"event":"VALIDATION_COMPLETE","D_r2":result["metrics"]["D"]["r2"],"blend_r2":result["metrics"]["equal_blend"]["r2"]}),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("--phase",choices=("train","validation"),required=True)
 args=parser.parse_args()
 try:
  if args.phase=="train":main_train()
  else:main_validation()
 except MemoryError:
  print("MemoryError: address space resource envelope",flush=True)
  raise SystemExit(86)

