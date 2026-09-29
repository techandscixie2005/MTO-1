# Confirmation preparation handoff

Prepared 2026-09-30 while the original frozen 20-epoch factorial was running. Root requested a separate namespace and no launch. Candidate remains unset.

## Status

- CPU-only capture and checks PASS on USTC-A800. Both source checkpoint bytes match the predeclared hashes. Payload epochs53/44, source seed23/37, legacy selection, exact configurations and 178-tensor schemas verified.
- All eight seed/arm configuration combinations pass in-memory tests without modifying inherited scientific settings. Eleven explicit rejection checks pass. Synthetic paired train-order sequences match across all20epochs; restoring the generator reproduces the next order. Synthetic order hashes are not actual-data order hashes.
- No Data instance, raw dataset, prediction array or test labels loaded. No new validation inference, GPU use, training, optimizer update, candidate selection, commit or push.
- Server package: `/home/inspur/MTO-1/research/single_model_20260929/confirmation_preparation/`.
- Local mirror: `C:/Users/master/Documents/ChatGPT/MTO/research_state/single_model_20260929/confirmation_preparation/`.
- Science agent has been asked for independent source review. Its review is separate from this self-check receipt.

## Design decision

Do not duplicate the full runner before a promotion exists. `contracts.py` supplies the generalized source loader and strict promotion schema; `generate.py` writes only named nonlaunchable specifications after a root-authored decision with pinned final-analysis evidence. Actual source checkpoints are loaded only to CPU. No `train.py`, `round_config.json`, freeze or launcher is created here. `README.md` documents the minimal source-loader substitution and all subsequent review/archive/resource gates.

Full historical validation replay expectations are seed23 .36870884832620643 and seed37 .39445533914883546. Original rounded fit-time scores differ by <5e-8; use the pinned replay receipts with the inherited 1e-5 epoch-zero tolerance. Before any continuation, replay each full source validation anchor with the integrated runner.

## Required later decision

Read root `research_state/DECISION_RULES.md`. Control-schedule improvement is eligible for confirmation independently of architecture improvement. For an architecture, retain its paired control for each seed. Native pooled raw-f SSE selects checkpoints, including epoch0; the historical fixed calibration is secondary with no refit. Predeclare an effect/consistency criterion before confirmation outcomes. Both new seeds must improve on the relevant within-seed contrast; the generator additionally requires root to specify the practical mean-delta threshold, without supplying a default.

No fresh holdout is available. Independent seeds remain training replication on the shared historical validation set. Retain all states, valid labels, fixed train-derived bright tails and energy errors. Compare best-selected recipes separately from same-epoch factorial curves; do not claim synergy from different selected epochs.

## Scientific caveat

The signed0e nonlinear adapter gate is rotation invariant, but does not imply electronic sign-gauge covariance `F(-M)=-F(M)`. Penalty sign invariance is a separate verified property. Raw M is an unidentified learned representation. Deferred frozen-base right-CG tuning, matched gentle native-f optimization and empirical factor mixing remain conditional ideas, with no new fit authorized.

## Archive boundary

Archive only the explicit lightweight files in `PREPARATION_MANIFEST.json`, plus a later independent-review receipt if produced. Python `__pycache__` is execution cache and excluded. Do not alter frozen round00 inventories. All files must first be downloaded from server to `D:/MTO/archives/` in the next appropriate round, before inspected commit and verified push. Model weights/checkpoints remain server-only.
