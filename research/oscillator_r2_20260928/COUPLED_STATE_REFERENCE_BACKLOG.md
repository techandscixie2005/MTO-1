# Deferred coupled-state reference

Recorded 2026-09-29 from root's research note; this is a bookmark, not a reviewed implementation proposal.

Juergens et al., *Latent unified smooth Hamiltonians for excited state chemistry* (reported 2026-09-01): https://arxiv.org/abs/2609.01871 ; PDF https://arxiv.org/pdf/2609.01871 . Root reports a latent Hamiltonian and additional operators transformed by common eigenvectors, demonstrated for thymine/azobenzene. This reviewer has not independently read the paper in this turn.

Potential later relevance: a consistent coupled-state representation for energies and other operators. Reported examples are not the cross-molecule QM9S benchmark; they do not establish transfer to MTO or explain the current all-zero-f amplitude outlier. Retain for a future coupled-state design review only. No change to the approved Gram probe or seed replication, no architecture implementation or launch authorized by this note.

## Independent primary-paper review (2026-09-29T04:00:46.528540+08:00)

The original bookmark above is retained as historical context. The paper has now been independently read through its accessible main text and supplemental text/captions. Actual scope: excited-state training covers all nine QeMFi species with a random 90/10 geometry split, with detailed thymine and azobenzene studies; QM9 evidence is ground-state energy prediction only. This is not a QM9S connectivity-held-out excited-state benchmark. [Primary paper, pp. 7–9 and SI S2](https://arxiv.org/pdf/2609.01871v1).

See `lush_feasibility/FEASIBILITY_REVIEW.md` for the conceptual assessment. Distinguish per-state spatial PSD tensors from a shared state-space strength Gram under the Hamiltonian eigenvectors. Phase/parity, incomplete label identifiability, finite-state truncation and degeneracy remain design constraints. A small synthetic gauge/degeneracy stress test is only a possible later prerequisite; it is NOT authorized or run. No coupled-state QM9S launch, active-study change or test access follows from this reference.
