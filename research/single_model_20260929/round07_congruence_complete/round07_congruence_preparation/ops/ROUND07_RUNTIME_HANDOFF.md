# COMPLETE — historical launch/first-epoch record below

Round07 has completed and root closed the failed triple-gate study. Do not relaunch or repeat checks. Read CURRENT_RUNTIME_HANDOFF.md and completion/ROUND07_REPORT.md for final verified status; the active-run language below is preserved historical evidence.

# Round07 production ACTIVE — original launch, no rerun

Science is sole executor. Root's separate execution authorization dfb6a5fe356e30da37260830f8032fe7892f2749bb0afc30b6daf277652afaaa is mirrored and passes the exact gate. It binds source68c052e909d0fcbe1671084edf7de1968f81f8f5e333bbfc2431fe3181c8c203, reviewe3ae0804141dc37fdbcb3797411a301e75dabe223eea4005d0c043197f423826 and publicationce3f342916bb6ecb6a76516bc8da8b64fc1c9839f406dcd9e852c6c12185d6c0 at commit271d61f6dc9e21009c31d10467fe704e44b974ad. Earlier blocked-preparation handoffs are historical.

## Actual launch

On2026-10-02 at00:01:51–52Z, the unchanged reviewed launcher admitted physical GPUs1/2/4 sequentially, inherited separate exclusive locks, prebound each child's UUID before imports and registered all three workers successfully. Launcher exit0, each registration_returncode0. No existing production run directory was present before launch. Full receipts, admission XML and registration output are under runs/<arm>/attempts/<attempt>/; aggregate launcher output is ops/production_launch_01.log.

| Arm | Physical GPU | PID | Start ticks | Registry/attempt ID |
|---|---:|---:|---:|---|
| original |1|1591761|1274697650|round07_original_1790899311738424814|
| scalar |2|1591766|1274697673|round07_scalar_1790899311967426770|
| tensor |4|1591771|1274697696|round07_tensor_1790899312200172164|

All share boot793423dc-86b6-4fb9-9a9d-e74cfa8135a3, UID1000 and cwd /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation. Exact argv hashes and UUIDs are in each LAUNCH_RECEIPT.json. Environment: own CUDA UUID, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2, PYTHONUNBUFFERED=1. No other GPU/process was changed.

Exact parent command:

```bash
cd /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  /usr/bin/python3 launch.py \
  --authorization /home/inspur/MTO-1/research/single_model_20260929/round07_congruence_preparation/PRODUCTION_EXECUTION_AUTHORIZATION.json \
  --arms original scalar tensor
```

## First committed-epoch readiness PASS

Science completed exactly one independently source-reviewed CPU checkpoint metadata inspection. ops/FIRST_EPOCH_METADATA.json SHA4c43a3b3dd2825088f1650ab1f4335e4f6578e478ee783c92748f02f2fbdcb0c passed: all three observed committed epoch1/1881 steps, complete AMSGrad state, saved RNG, exact order/initial/source/authority bindings, matching best/last provenance and TRAIN/validation access ledgers with zero TEST numeric rows. Source-review SHA0631fad28df56043947e29831ab3478e38bc9f6c92dabbe9cec110dd0a02bf6f binds inspector f09cfd8f. No model construction/inference or raw-target decoding; private checkpoint tensors were loaded on CPU only and not copied. Do not repeat this completed inspection.

First-epoch measured seconds: original170.55, scalar183.01, tensor174.52. Rough fixed60 training pace is2.84–3.05h, indicative only; no budget/settings adjustment. At inspection workers were naturally progressing in epoch2, with no failure marker. QC owns one independent combined startup/due check of full identities, actual GPU placement and locks, and updates the existing four-hour heartbeat. Science does not duplicate that check or poll through completion. At the next scheduled wakeup inspect actual terminal/identity state before any continuation; normal completion is expected before that wakeup at the measured pace, not guaranteed by it.

## Frozen scope and recovery

Each arm runs exactly60 epochs/112860 optimizer updates from fresh seed11, common data order, original LE+Ls and fixed reviewed settings. Checkpoint selection is earliest minimum native pooled raw-f validation SSE across0–60, all66860 labels; TEST remains sealed. The tensor allocation gate is +.003 vs both contemporary controls and retained .44716940136585204. No early outcome can change recipe, budget, controls, normalization, bound or loss.

Preserve resumable last.pt after each committed epoch; best.pt is selected snapshot and geometry_best.pt is one-file geometry export. Preflight and historical trained states are forbidden initialization/recovery inputs. A mid-epoch interruption may only recover from the preceding committed last.pt after explicit diagnosis/recovery decision. Never blindly launch all arms again, rerun a completed arm, silently change GPU, or signal unknown processes. Observe actual registered identity/terminal status before deciding a process failed. Preserve failure/attempt/log/checkpoint evidence.

At natural terminal completion verify all source/authorization/order/epoch/checkpoint/access receipts and use saved validation metrics/predictions for reviewed final analysis. No model/test replay, extension or new seed is authorized. History owns independent terminal review; QC owns D-first lightweight result archive/publication after root decision. Weights, optimizer tensors, raw data, caches and all private arrays stay server-side.

## Independent startup confirmation complete

QC's single combined due/startup check at2026-10-02T00:07:21Z passed all full registered identities, actual physicalGPU1/2/4 UUID placement, environment and inherited GPU/worker lock descriptors. All three were in epoch2; no new failure or intervention. Receipt monitoring/ROUND07_STARTUP_CONFIRMATION_20261002T0001.json SHAb7d0cb125e06f17f18edf2e1a05f7d1744fae96eeaa3b9e49ffd8f059eae7e6c; canonical CHECK_20261002T000721Z.json. Existing four-hour cadence preserved. First-epoch and startup checks are complete; no active polling, additional checkpoint inspection or numerical work remains in this launch turn. Continue the exact fixed60 protocol to natural completion, then the reviewed terminal workflow at the scheduled wakeup.
