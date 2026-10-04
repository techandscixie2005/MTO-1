"""Review exact proposal text/metadata; no scientific imports or payload decoding."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

mp = ROOT / "REFERENCE_MANIFEST.json"
assert sha(mp) == "fa76f9849f5b3c4257b305d6f42f3611daee1799936239242c9b555e4fbd70c2"
m = read(mp)
assert m["scope"] == "round10_schedule_proposal_only"
assert m["reference_count"] == len(m["references"]) == 38
assert len({e["path"] for e in m["references"]}) == 38
for entry in m["references"]:
    p = Path(entry["path"])
    assert p.suffix in {".py", ".md", ".json"}
    assert p.is_file() and not p.is_symlink()
    b = p.read_bytes()
    b.decode("utf-8")
    assert b"\x00" not in b
    assert len(b) == entry["bytes"] and sha(p) == entry["sha256"]
for path, digest in {**m["documents"], **m["preserved_initial_draft"]}.items():
    p = Path(path)
    assert p.suffix in {".md", ".json"} and sha(p) == digest
s = read(ROOT / "PROPOSAL_SETTINGS.json")
assert s["scope"] == "round10_schedule_proposal_only"
assert list(s["arms"]) == ["fixed_lr", "step_lr"]
assert s["arms"]["fixed_lr"] == {"mode": "original", "weight_decay": 0., "epoch_learning_rates": [.001] * 60}
assert s["arms"]["step_lr"] == {"mode": "original", "weight_decay": 0., "epoch_learning_rates": [.001] * 30 + [.0003] * 30}
assert s["epochs"] == 60 and s["epoch0_eligible"] is True
assert s["seed"] == s["order_seed"] == 11
assert s["batch_size"] == 64 and s["batches_per_epoch"] == 1881
assert s["optimizer_updates"] == 112860 == 60 * 1881
t = s["candidate_transition"]
assert t["last_high_rate_epoch"] == 30 and t["first_low_rate_epoch"] == 31
assert t["last_high_rate_update_one_based"] == 56430 == 30 * 1881
assert t["first_low_rate_update_one_based"] == 56431
assert t["high_lr"] == .001 and t["low_lr"] == .0003
for k in ("validation_feedback", "reset_moments", "warmup", "within_epoch_change"):
    assert t[k] is False
o = s["optimizer"]
assert o == {"name": "torch.optim.Adam", "amsgrad": True, "betas": [.9, .999], "eps": 1e-8,
             "weight_decay": 0., "grad_clip": 5., "foreach": None, "fused": None,
             "capturable": False, "differentiable": False, "maximize": False,
             "group_count": 1, "trainable_tensor_count": 135, "trainable_numel": 1552092,
             "excluded_frozen_tensors": 11}
assert s["parameter_roster_sha256"] == sha(ROOT.parent / "round09_regularization_preparation/PARAMETER_ROSTER.json")
assert s["train_statistics_sha256"] == sha(ROOT.parent / "round09_regularization_preparation/TRAIN_STATISTICS.json")
assert s["split_manifest_sha256"] == sha(ROOT.parent / "dataset_audit_20260930/SPLIT_MANIFEST.json")
assert s["independent_split_verification_sha256"] == sha(ROOT.parent / "dataset_audit_20260930/INDEPENDENT_SPLIT_VERIFICATION.json")
assert (s["train_molecules"], s["validation_molecules"], s["test_molecules"], s["validation_raw_f_labels"]) == (120355, 6686, 6686, 66860)
assert s["promotion_delta_r2"] == .003 and s["retained_reference_r2"] == .44716940136585204
assert s["promotion_comparators"] == ["contemporaneous selected fixed_lr", "retained Round05 control45"]
assert s["selection"] == "Earliest minimum pooled native validation raw-f SSE among epochs0-60"
assert s["bootstrap"] == {"draws": 2000, "seed": 20260930, "unit": "validation identity component, all molecules/states", "sst": "recomputed each draw", "interval_percentiles": [2.5, 97.5], "conditional_only": True}
for k in ("implementation_authorized", "numerical_execution_authorized", "test_access", "historical_weights", "technical_weights_reused", "automatic_seed_allocation", "automatic_extension", "prediction_averaging"):
    assert s[k] is False
p = s["proposed_later_technical_cap"]
assert p["requires_separate_root_preparation_decision"] is True
assert (p["train_rows"], p["unique_batches_per_arm"], p["executed_updates_per_arm"], p["total_full_model_updates"]) == (128, 2, 3, 6)
assert p["schedule_fixture_labels"] == [30, 31, 31] and p["actual_adam_counters_after_calls"] == [1, 2, 2]
assert p["validation_rows"] == p["test_rows"] == 0
assert p["model_adam_atol"] == 2e-6 and p["model_adam_rtol"] == 1e-5
assert p["loss_atol"] == 1e-6 and p["loss_rtol"] == 1e-5 and p["rng_order_exact"] is True
assert "cannot support a schedule benefit" in s["preintervention_selection_caveat"]
r = {"passed": True, "scope": "round10_proposal_text_and_metadata_only",
     "reference_manifest_sha256": sha(mp), "document_hashes": m["documents"],
     "text_reference_count_rehashed": 38, "all_reference_bytes_match": True,
     "preserved_draft_files_rehashed": 4, "explicit_60_rate_tables_and_transition_metadata_pass": True,
     "roster_split_statistics_pins_pass": True, "dual_gate_and_future_scope_metadata_pass": True,
     "no_model_scientific_import_arrays_or_numerical_experiment": True,
     "reference_paths_not_recursive_archive_allowlist": True, "source_sha256": sha(Path(__file__))}
out = ROOT / "INDEPENDENT_REFERENCE_CHECKS.json"
assert not out.exists()
out.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"passed": True, "references": 38, "receipt_sha256": sha(out)}))
