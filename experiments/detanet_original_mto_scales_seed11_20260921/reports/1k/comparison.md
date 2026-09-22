# 1k: original DetaNet vs planned MTO, seed=11

Split counts: {'train': 800, 'validation': 100, 'test': 100}. Train-only RMS: 0.0245646129744.
Both complete DetaNet backbones start from the same fresh backbone initialization; each model uses its own best validation checkpoint.
Single-seed preliminary comparison; no cross-seed standard deviation or significance claims. Existing historical test identities are reused.
Source spectra are linearly interpolated to the same author 240-point grid. Source broadening is unknown; MTO sigma=0.2 eV is the retained assumption.
Training hours measure cumulative training/validation/checkpoint wall time on one GPU per model; queue and short-check time are excluded.
Cosine is averaged per molecule; zero-norm pairs are assigned zero and their count is recorded.

| Model | MSE (source) | MSE / train RMS² | MAE | Cosine | Epochs | Best epoch | Training hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| detanet_original_uv | 0.00028830327 | 0.47778193 | 0.0092606173 | 0.766021 | 1000 | 958 | 0.328 |
| detanet_mto_planned | 0.00025865543 | 0.42864894 | 0.0050827966 | 0.836268 | 231 | 79 | 0.098 |

Lower test MSE: detanet_mto_planned.
MTO relative MSE improvement = 100 × (A − B) / A: 10.283559653299415% (negative means MTO is worse).
