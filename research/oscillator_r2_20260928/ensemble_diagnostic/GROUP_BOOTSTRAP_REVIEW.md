# Additive review: connectivity bootstrap sensitivity

PASS for the code and summarized sensitivity report. Reviewer did not open test prediction arrays. The bootstrap resamples connectivity groups, retaining each group's molecules and states, pairs candidate/control on identical draws, and recomputes SST with variable group size. Its full-sample delta matches the preceding frozen comparison.

Reported6671groups give95% delta-R2 interval[+.003586,+.056691], positive fraction.9875. This supports the same directional sensitivity conclusion. It introduces no new model selection, does not establish scaffold generalization, and does not make the previously exposed test a fresh holdout. Original ensemble-review claims remain unchanged.
