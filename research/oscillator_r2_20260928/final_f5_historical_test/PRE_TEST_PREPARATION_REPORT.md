# Final F5 historical-test preparation: reviewed, unpublished

The candidate is the fixed equal native-f mean of eta0, eta01, eta1, G1 and G3. Its validation R² is 0.48769960489832564. The primary historical-test comparison is F5 against the original equal-three predictor. The historical test has been used before and cannot serve as an untouched holdout.

The isolated evaluator passed a saved-validation, synthetic and CPU-checkpoint-metadata preflight. It reproduced the frozen F5 validation score; the fixed-row adapter reproduced the original input batch tensors exactly on 64 and 30 validation rows. Independent code review passed. Neither preflight nor review read a test array or ran a model forward.

- Candidate freeze SHA-256: \`46a8153e490e16f03bae0927bee4954f2cba95977050fb9f788aa9dbacda4855\`.
- Protocol SHA-256: \`05afd33a04567cfab711a95d10734d0bcf8841dcf4f08c7ebcc036efafaf4e94\`.
- Evaluator SHA-256: \`aa4fdb125a099d0858e0a24430dd6676b0adecee1d33879469244390b39998ab\`.
- Preflight SHA-256: \`da209894bc4a78bb50241742a2cb5bcfe1651bc1882408247ab6f3f6dc967c91\`.
- Independent review SHA-256: \`07dbcf0efb98e4ab9a3e631b450fa40382b5e73265a7285e18eba6505ac9812e\`.

Execution is **not authorized by this preparation**. The frozen code requires D-first hash-verified archive and GitHub SSH publication, then a separate root authorization file binding the published commit, D manifest and these exact hashes. It requires fresh healthy idle physical GPU4 with an exclusive shared lock and creates an exclusive STARTED receipt before any test member or prediction array is opened. No test score exists for F5 yet. Historical eta test-array hashes are inherited from the prior receipt but their current bytes have not been opened or rehashed in this preparation.
