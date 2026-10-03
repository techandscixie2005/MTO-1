"""Hash-bound proposal/source metadata review, without scientific imports or payload decoding."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
mp=ROOT/"REFERENCE_MANIFEST.json"
assert sha(mp)=="8585ef71f7f45f3a7f33efbbc4a99b13d2d24f2e30fe72d6f30488c866c77886"
m=read(mp);assert m["scope"]=="round09_proposal_only" and m["reference_count"]==len(m["references"])==34
assert len({v["path"] for v in m["references"]})==34
for e in m["references"]:
 p=Path(e["path"]);assert p.suffix in {".py",".md",".json",".cfg"}
 assert p.is_file() and not p.is_symlink()
 b=p.read_bytes();b.decode("utf-8");assert b"\x00" not in b
 assert len(b)==e["bytes"] and sha(p)==e["sha256"]
for path,digest in m["documents"].items():assert sha(Path(path))==digest
s=read(ROOT/"PROPOSAL_SETTINGS.json")
assert s["arm_order"]==["zero_decay","coupled_l2"]
assert s["arms"]=={"zero_decay":{"model_mode":"original","weight_decay":0.0},"coupled_l2":{"model_mode":"original","weight_decay":0.0001}}
assert s["groups"]==1 and s["expected_trainable_parameter_tensors"]==135 and s["other_parameter_exclusions"]==[]
assert s["frozen_exclusions"]==["right_adapter.*","core.blocks.1.transport.theta","core.blocks.2.transport.theta"]
assert s["foreach"] is None and s["fused"] is None and s["amsgrad"] and not s["decoupled_weight_decay"]
assert s["lr"]==.001 and s["betas"]==[.9,.999] and s["eps"]==1e-8 and s["task_gradient_clip_norm"]==5
assert s["epochs"]==60 and s["batch_size"]==64 and s["optimizer_updates_per_arm"]==112860
assert s["seed"]==s["order_seed"]==11 and s["promotion_delta_r2"]==.003 and s["retained_reference_r2"]==.44716940136585204
assert len(s["train_statistics_sha256"])==64 and s["train_statistics_sha256"]==sha(ROOT.parent/"round08_transport_preparation/TRAIN_STATISTICS.json")
for k in ("test_access","coefficient_search","implementation_authorized","production_authorized","proposal_stage_numerical_execution","proposal_stage_model_or_array_access"):
 assert s[k] is False
r={"passed":True,"scope":"proposal_and_exact_source_metadata_only","reference_manifest_sha256":sha(mp),"document_hashes":m["documents"],"text_reference_count_rehashed":34,"all_reference_bytes_match":True,"settings_and_full_train_statistics_pin_checked":True,"no_model_construction_scientific_imports_arrays_or_numerical_experiments":True,"reference_paths_not_recursive_archive_allowlist":True,"source_sha256":sha(Path(__file__))}
out=ROOT/"INDEPENDENT_REFERENCE_CHECKS.json";assert not out.exists();out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps({"passed":True,"receipt_sha256":sha(out),"references":34}))
