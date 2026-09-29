# Independent-seed confirmation preparation

This is an isolated, CPU-tested contract and configuration package. **It contains no training entry point, assigns no GPU, selects no candidate, and cannot launch a fit.** The original factorial remains frozen. Preparation is not promotion or launch approval.

## What is pinned

- Source seed23: legacy-selected epoch53, checkpoint SHA256 `7c28a2fe204f8804d05d3dc58bb9acf853f2d0505ce83a9f66819f02fd485a18`.
- Source seed37: legacy-selected epoch44, checkpoint SHA256 `9570b5b3743be89cd8ca3f7f75a95bc43cd8445336a3d73694858c79ae05dd82`.
- `SOURCE_CONTRACTS.json` contains exact checkpoint paths/configurations, canonical config hashes, tensor-schema hashes, historical replay expectations, and the full existing frozen manifest, including code/data/weights hashes and split-index hashes. Source model weights remain server-only.
- Historical selected-checkpoint replays are .36870884832620643 and .39445533914883546. These differ slightly from older rounded fit-time scores; the unchanged epoch-zero R² tolerance is 1e-5. Full validation replay under the future runner is still required before fitting.
- Source configurations differ from seed11 only in seed/name/GPU/100-epoch cap. Their selected epochs differ and all used the same historical validation set. This is conditional continuation evidence, not a fresh holdout or a precisely matched full-training-budget comparison.

## Preserved experiment rules

The generator starts from the pinned pilot configuration. It changes the source checkpoint/contract, seed-specific RNGs and epoch-zero replay expectation, round name, and the candidate/control arm roster. It leaves 20 epochs, Adam AMSGrad reset, lr1e-5, batch64, weight decay0, clip5, FP32, two threads, no AMP/TF32/scheduler, original LE+Ls, and λ=.001 when applicable unchanged. Epoch zero remains eligible; checkpoint selection is native validation pooled raw-f SSE with earliest exact ties. The exact same historical α/β map remains a secondary metric without refit or checkpoint reselection.

Each seed uses its own seed value for runtime RNG, train-order RNG and adapter initialization. Control and candidate within that seed use the same source weights and order sequence; reset runtime RNG after model construction exactly as in the frozen loop. Keep original train/validation masks and training-derived tail thresholds .0549/.2412. Do not compare scores across seeds as a causal adapter effect. Report each within-seed contrast separately, with state/tail/energy diagnostics. Never average predictions or checkpoints.

`check_preparation.py` exercises all four candidate schemas for both sources **in memory**, rejects missing/draft/invalid promotion and wrong source epoch/hash/order seed, and tests 20 paired synthetic order sequences plus order-generator restoration. It opens only checkpoints and lightweight JSON/source files; no Data instance, dataset, predictions or test labels are loaded. It does not repeat optimizer-resume testing; the future runner must inherit the already reviewed loop and be reviewed again after integration.

## Promotion and generation

The root orchestrator must first author a promotion record after the complete pilot analysis. Required fields are documented by `validate_promotion` in `contracts.py`:

```json
{
  "status": "approved",
  "approved_by": "research_orchestrator",
  "candidate": "ROOT_MUST_SELECT_ONE_OF_control_adapter_decorrelation_both",
  "pilot_manifest_sha256": "eaf2c75e239f7c04717adb97220cc35731df8d1a598b14b7b014c6754bf5ebd3",
  "selection_metric": "minimum_validation_pooled_raw_f_SSE_earliest_tie",
  "confirmation_seeds": [23, 37],
  "allow_test_evaluation": false,
  "prediction_averaging": false,
  "decision_rationale": "ROOT_MUST_SUPPLY",
  "pilot_final_analysis": {"path": "ROOT_MUST_SUPPLY", "sha256": "ROOT_MUST_SUPPLY"},
  "confirmation_claim_criterion": {
    "both_new_seeds_positive": true,
    "minimum_mean_within_seed_delta_r2": null,
    "contrast": "ROOT_MUST_SUPPLY"
  }
}
```

The template is deliberately invalid. Root must predeclare the practical mean-delta threshold; no default effect threshold is silently selected. Contrast is `control_vs_own_epoch_zero` for a control-only schedule confirmation, or `candidate_vs_matched_control_and_own_epoch_zero` for an architecture confirmation. `both_new_seeds_positive` prevents a positive mean from hiding a reversal. The generator checks the analysis file hash but does not independently judge the scientific promotion decision.

After that record exists:

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/confirmation_preparation
CUDA_VISIBLE_DEVICES= /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python generate.py --promotion PROMOTION.json --output approved_specs
```

This writes `seed23.CONFIRMATION_SPEC.json` and `seed37.CONFIRMATION_SPEC.json`, both `launch_enabled:false`, with GPUs unset. No `train.py`, `round_config.json`, `FROZEN_MANIFEST.json` or launcher is generated. Output is confined to this new preparation directory and existing differing records are never overwritten.

## Minimal future integration, after promotion

Avoid maintaining a second scientific training loop during preparation. In a separate approved confirmation runtime namespace, reuse the pinned loop and replace only its hardcoded source-loading block:

```python
from contracts import load_source
contracts = json.loads((ROOT / 'SOURCE_CONTRACTS.json').read_text())
source_cfg, original = load_source(cfg, contracts)
```

The following existing `build_model(source_cfg, data.stats, ac['adapter'], cfg['adapter_seed'])` and `load_baseline` calls remain unchanged. The helper verifies checkpoint bytes, expected epoch53/44, legacy-selection provenance, exact config and tensor schema. It returns CPU-loaded weights; the runtime moves the model to its assigned GPU later. It must run only after that runtime's own current code/data freeze and promotion checks.

A future independent review must also cover namespace-relative imports/ROOT, freeze roster, source-contract pins, promotion/analysis pins, healthy GPU assignment and lock, monitor registration, archive/publication receipt, full validation anchor replay, and final selected-checkpoint replay. Preserve atomic last/best/selected-epoch transactions, Adam state and all Python/NumPy/Torch/CUDA/order RNG states. Resume stays at the last completed epoch; interrupted epochs are replayed. Do not silently claim mid-epoch cursor resume. The legacy Data class allocates shared arrays including test rows during actual training, but no test indexing, inference, scoring, fitting or selection is permitted. This preparation itself loads no dataset.

## Interpretation and deferred decisions

The original MTO head already contains coherent cross-channel interference. The adapter is O(3)-equivariant; its signed scalar0e gate does **not** establish electronic state-phase covariance `F(-M)=-F(M)`. The squared-cosine representation penalty is sign-invariant. Raw learned M is not a calibrated wavefunction, so these facts do not by themselves identify a code defect or a physical response operator.

Do not attribute early pilot degradation to the loss, learning rate or architecture before completion. Conditional later hypotheses include frozen-base training of the right-CG adapter (different readout location from the previously tried frozen h-head), gentle native-f optimization with a matched low-LR control, and the deferred empirical factor mixer. None is selected or implemented here. Direct shared-H eigendecomposition remains rejected by the separate degeneracy/gauge feasibility audit.

## Reproduce CPU preparation

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/confirmation_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python capture_contracts.py
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python check_preparation.py
```

Lightweight files may be archived in the next round. Do not add them to an already frozen round00 inventory. All new experiment records must follow server → `D:\MTO\archives\` → inspected commit → verified push before launch. No checkpoint, raw data, cache or prediction array belongs in publication.
