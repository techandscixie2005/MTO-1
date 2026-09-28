# Numerical preflight note

The first preflight compared two-molecule model forwards with features cached from 64-molecule GPU batches. It stopped at a maximum absolute difference of 1.9744e-6, which is consistent with FP32 rounding that can depend on batch shape and exceeded the declared absolute tolerance of 1e-6 for near-zero values. No optimization run had started.

The corrected preflight compares the first **64** train molecules and first **64** validation molecules with the identical batch size, device, precision settings, input order, and model evaluation mode used during cache generation. It passed the declared elementwise combined tolerance (absolute 1e-6, relative 1e-5) for h, E, base32 and base64; the largest absolute h difference was still about 2.1e-6 on a nonzero tensor. Cached versus uncached head loss, gradients and copied one-step Adam updates passed the same tolerance. Initial validation native-f R² was 0.405294118949 versus frozen eta0 0.405294118341 (difference 6.08e-10). Two-step optimizer/order/cursor/RNG resume produced maximum parameter difference 0. All 20 training-order hashes match the archived residual run.

This is a tolerance-based numerical equivalence claim. It is not bitwise identity, even at the matched batch size. The initial two-molecule failed check is preserved here for review.

