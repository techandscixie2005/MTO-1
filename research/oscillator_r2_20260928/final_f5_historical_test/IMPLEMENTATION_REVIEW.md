# Independent final F5 evaluator review

PASS for the exact evaluator and preflight hashes in IMPLEMENTATION_REVIEW.json. This is preparation approval, not test execution authorization.

The reviewer read the complete evaluator, matched the native FP64 E/trace formula to the audited G1/G3 validation replay, independently verified all 125 frozen source/artifact hashes and five checkpoint identities, and reproduced F5 validation R2 = 0.48769960489832564. Fixed-row input assembly exactly matches original Data.batch on 64 and 30 saved-validation rows without constructing the eager loader or forwarding a model.

Synthetic checks reject changed authorization, source/array hashes, alignment, nonfinite predictions, split overlap/incomplete coverage and preexisting partial outputs. Independent 2,000-draw molecule and group calculations exactly match the evaluator; integer molecule units preserve dataset order. Group gross contributions, extrema and absolute-SSE rows are verified.

The runtime has exactly two new selected checkpoint passes, G1 then G3; old eta arrays are byte-checked and aligned only after authorization. No targets enter model inputs, and native predictions are not clipped. All population, TRAIN tail, frozen q2, cost and concentration reporting follows the unchanged protocol. Source/checkpoint checks, publication-bound authorization, physical GPU4 health/shared lock and durable single-use STARTED gate precede test consumption. Failures preserve evidence and do not permit silent retries.

Required next steps: D/SSH publication, root authorization bound to published hashes, then fresh runtime admission. Test remains unopened in this review. Bootstrap remains descriptive because test and validation have been reused.
