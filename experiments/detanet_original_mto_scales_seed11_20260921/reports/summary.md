# DetaNet / MTO: single-seed scale comparison

Only this run, seed=11. Six fresh fits across existing 1k, 10k, full splits. No significance claims or cross-seed statistics.

## 1k
Counts: {'train': 800, 'validation': 100, 'test': 100}

| Model | MSE (source) | MSE / train RMS² | MAE | Cosine | Epochs | Best epoch | Training hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| detanet_original_uv | 0.00028830327 | 0.47778193 | 0.0092606173 | 0.766021 | 1000 | 958 | 0.328 |
| detanet_mto_planned | 0.00025865543 | 0.42864894 | 0.0050827966 | 0.836268 | 231 | 79 | 0.098 |

Lower MSE: detanet_mto_planned; MTO relative MSE improvement: 10.283559653299415%; cumulative training GPU hours: 0.427.

## 10k
Counts: {'train': 8000, 'validation': 1000, 'test': 1000}

| Model | MSE (source) | MSE / train RMS² | MAE | Cosine | Epochs | Best epoch | Training hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| detanet_original_uv | 0.00029428087 | 0.48970092 | 0.0073054075 | 0.841885 | 495 | 343 | 1.121 |
| detanet_mto_planned | 0.0002013567 | 0.33506956 | 0.0037897903 | 0.90521 | 238 | 86 | 0.644 |

Lower MSE: detanet_mto_planned; MTO relative MSE improvement: 31.576695094480847%; cumulative training GPU hours: 1.765.

## full
Counts: {'train': 103785, 'validation': 12973, 'test': 12974}

| Model | MSE (source) | MSE / train RMS² | MAE | Cosine | Epochs | Best epoch | Training hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| detanet_original_uv | 0.00011185669 | 0.17757147 | 0.0038937968 | 0.930602 | 551 | 399 | 15.935 |
| detanet_mto_planned | 7.7607059e-05 | 0.12320049 | 0.0024212579 | 0.947602 | 236 | 84 | 8.244 |

Lower MSE: detanet_mto_planned; MTO relative MSE improvement: 30.61920908522305%; cumulative training GPU hours: 24.179.

