# Round08 future production and recovery handoff

**Production is currently blocked.** Preparation review, acceptance and publication alone are not execution authority. Root must separately authorize the exact original/local/neighbor sixty-epoch study after the final source/review and D-first publication receipts are available. No validation or TEST scoring is authorized by this handoff.

Server namespace: `/home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation`.
Pinned interpreter: `/home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python`.
The complete entry and launcher are implemented; the missing production authority is the intentional gate.

## Exact authorization schema

Root writes `PRODUCTION_EXECUTION_AUTHORIZATION.json` in the namespace, using the keys in the preserved `PRODUCTION_AUTHORIZATION_TEMPLATE.json`. It must set authorized=true, scope=`round08_three_arm_60epoch_fit`, epochs=60 and ordered arms=[original,local,neighbor], while test_access=false and historical_weights=false. Bind exact SHA256 values for FROZEN_MANIFEST.json, INDEPENDENT_PREPARATION_REVIEW.json and the future campaign `ops/ROUND08_PREPARATION_PUBLICATION_RECEIPT.json`, plus the frozen split/verifier pins. The publication receipt must verify the remote push, archive-before-stage/commit/push chronology, and the identical source/review hashes. The independent review's complete source dictionary must equal the frozen manifest. Every source is rehashed before launch; no wildcard bypass or fallback exists.

## Launch command, only after that authority exists

```sh
cd /home/inspur/MTO-1/research/single_model_20260929/round08_transport_preparation
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  /home/inspur/MTO-1/experiments/qm9s_full_EA_20260925/env/bin/python \
  launch.py --authorization PRODUCTION_EXECUTION_AUTHORIZATION.json --arms original local neighbor
```

The parent performs admissions sequentially; the three workers then train concurrently on separate GPUs. Original uses physicalGPU1 UUID GPU-b22353bd-fc06-efd3-5819-cfd3e85b3800; local uses GPU2 UUID GPU-cb4dc1ea-9ea3-e426-bdeb-df0bef8a23fa; neighbor uses GPU4 UUID GPU-e212aefc-f1d6-cc7a-5594-e87abeaf1184. Each requires current idle/healthy admission, an exclusive `/tmp/mto_pouter_gpu_<index>.lock`, and an exact owned registry entry. GPU UUID visibility is set before the child interpreter imports scientific modules. Logical cuda:0 then denotes that one admitted physical GPU. GPU0 and all unrelated jobs remain untouched. No substitution is automatic if an assigned GPU fails admission.

The child first executes `registered_entry.py`. Its pipe barrier requires registrar success bound to the owned PID and authority SHA before entering unchanged `train.py` in the same PID. Registration failure leaves the scientific stage blocked and stops only the owned child, with a preserved failure record. There is no automatic retry. A later-arm admission failure does not cancel an already registered healthy arm; report exact partial launch and let root decide the remaining admission.

## Runtime records and explicit recovery

Each `runs/<arm>/attempts/<unique-id>` contains launch/identity/admission/registration/log evidence. `status.json`, `history.jsonl`, `BEST.json`, `last.pt`, `best.pt` and final `FIT_COMPLETE.json` belong to that arm. All checkpoints, optimizer/RNG states and prediction arrays remain server-only. The complete checkpoint schema includes dormant transport tensors for original; local/neighbor have exactly768 active added parameters. No technical preflight state may seed a production run.

`last.pt` is the committed completed-epoch resume point: optimizer/AMSGrad, RNGs/order generator, history, selected-record references, source and split bindings. A killed incomplete epoch restarts from the preceding committed epoch with its prescribed order. CUDA model/update equality is only the fixed numerical preflight contract, not a claim of bitwise identical entire trajectories. `best.pt` is a selected model/optimizer snapshot; resume this implementation from `last.pt`, not `best.pt`. Selected snapshot/export is the earliest strict minimum validation raw-f SSE among epochs0–60.

For a genuine failure, first preserve the exact FAILED/status/log and process identity; determine whether the registered process still exists. Never equate an app interruption with a model failure, kill an unknown process, delete private artifacts or blindly relaunch. Only an explicit reviewed recovery may invoke the same launcher for the authorized affected arm with the same source/config/authority; it reacquires resource/worker locks, makes a distinct attempt/registry ID and resumes committed last.pt. A completed FIT_COMPLETE arm refuses relaunch. The runner archives an old FAILED marker after a successful checkpoint resume. No extra epochs or changed settings are authorized by recovery.

At normal completion require sixty epochs,112860 updates, all sixty prescribed order hashes, unchanged source/split, exact checkpoint/export hashes and zero TEST numeric access. Analyze saved selected/final validation arrays once under an independently reviewed analysis scope. Report all labels, states, tails, energy and aligned-final comparisons; preserve the triple+.003 screen against both contemporary controls and retained .44716940136585204. Conditional component bootstraps do not replace independent training seeds. No automatic extension, confirmation allocation or TEST release.
