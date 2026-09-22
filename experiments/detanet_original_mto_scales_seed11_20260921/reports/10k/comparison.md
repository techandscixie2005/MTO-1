# 10k: original DetaNet vs planned MTO, seed=11

Split counts: {'train': 8000, 'validation': 1000, 'test': 1000}. Train-only RMS: 0.024514077575.
Both complete DetaNet backbones start from the same fresh backbone initialization; each model uses its own best validation checkpoint.
Single-seed preliminary comparison; no cross-seed standard deviation or significance claims. Existing historical test identities are reused.
Source spectra are linearly interpolated to the same author 240-point grid. Source broadening is unknown; MTO sigma=0.2 eV is the retained assumption.
Training hours measure cumulative training/validation/checkpoint wall time on one GPU per model; queue and short-check time are excluded.
Cosine is averaged per molecule; zero-norm pairs are assigned zero and their count is recorded.

| Model | MSE (source) | MSE / train RMS² | MAE | Cosine | Epochs | Best epoch | Training hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| detanet_original_uv | 0.00029428087 | 0.48970092 | 0.0073054075 | 0.841885 | 495 | 343 | 1.121 |
| detanet_mto_planned | 0.0002013567 | 0.33506956 | 0.0037897903 | 0.90521 | 238 | 86 | 0.644 |

Lower test MSE: detanet_mto_planned.
MTO relative MSE improvement = 100 × (A − B) / A: 31.576695094480847% (negative means MTO is worse).
