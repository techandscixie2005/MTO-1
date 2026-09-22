# full: original DetaNet vs planned MTO, seed=11

Split counts: {'train': 103785, 'validation': 12973, 'test': 12974}. Train-only RMS: 0.0250983051138.
Both complete DetaNet backbones start from the same fresh backbone initialization; each model uses its own best validation checkpoint.
Single-seed preliminary comparison; no cross-seed standard deviation or significance claims. Existing historical test identities are reused.
Source spectra are linearly interpolated to the same author 240-point grid. Source broadening is unknown; MTO sigma=0.2 eV is the retained assumption.
Training hours measure cumulative training/validation/checkpoint wall time on one GPU per model; queue and short-check time are excluded.
Cosine is averaged per molecule; zero-norm pairs are assigned zero and their count is recorded.

| Model | MSE (source) | MSE / train RMS² | MAE | Cosine | Epochs | Best epoch | Training hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| detanet_original_uv | 0.00011185669 | 0.17757147 | 0.0038937968 | 0.930602 | 551 | 399 | 15.935 |
| detanet_mto_planned | 7.7607059e-05 | 0.12320049 | 0.0024212579 | 0.947602 | 236 | 84 | 8.244 |

Lower test MSE: detanet_mto_planned.
MTO relative MSE improvement = 100 × (A − B) / A: 30.61920908522305% (negative means MTO is worse).
