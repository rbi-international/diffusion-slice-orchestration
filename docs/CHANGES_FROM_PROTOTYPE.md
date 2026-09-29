# Corrections relative to the earlier prototype (Rohit16)

The earlier prototype (`simulator.py`, `models.py`, `gan_predictor.py`,
`diffusion_predictor.py`, and the document `GAN_Diffusion.docx`) was rebuilt in
this repository. The table records every behavioural difference, why it was
changed, and where the corrected version lives. None of the numbers in
`GAN_Diffusion.docx` should be reused: they came from a single seed, 50 test
slots, and the defects below.

| # | Item | Prototype | This repository | Reason |
|---|---|---|---|---|
| 1 | DDPM noise schedule | linear beta 1e-4 to 0.02 with K = 20, giving alpha_bar_K = 0.817 | cosine schedule, alpha_bar_K about 6e-6 (`ddpm.py`) | With the old schedule the forward process never reached noise, yet sampling started from N(0, I): a train/sample mismatch. The old setting is kept only as an ablation (`linear_legacy`). |
| 2 | DDPM reverse variance | sqrt(beta_k), while the document stated the posterior variance | posterior variance beta tilde by default, beta as an ablation | Text and code now agree. |
| 3 | Reservation rule, Eq. (20) | (alpha_s + kappa Q) additive | alpha_s * max(u_bar, u) * (1 + kappa Q) | Matches the equation in the paper. |
| 4 | Risk margins alpha_s | 1.25 / 1.10 / 1.15 | 1.35 / 1.30 / 1.18 | Values stated in the paper. |
| 5 | Queue pressure Q | L2 norm of backlog | L1 norm, capped at 1 | Paper defines Q = min(1, norm-1(q) / 45). |
| 6 | Unassigned backlog, Eq. (8) | 0.55 q + 0.15 u | q + 0.15 u | Paper's second branch has no retention factor. |
| 7 | Transmission delay, Eq. (5) | chi_tx applied to the ratio terms only | chi_tx applied to propagation plus ratio terms | Paper places the propagation term inside the bracket. |
| 8 | Rejection latency penalty | 3.0 d_s | 2.2 d_s | Value stated in the paper. |
| 9 | Fragmentation, Eq. (12) | mean absolute difference of utilizations | 1 minus mean of min(bw util, cpu util) | Paper definition. |
| 10 | LWMA weights, Eq. (17) | 1, 2, ..., k | 0.35 + 0.65 (i-1)/(k-1) | Paper definition. |
| 11 | Forecast history | window ended at slot t (included the demand being forecast) | window t-k .. t-1 predicts slot t | Removes look-ahead. |
| 12 | Generator output head | GAN sigmoid (cannot exceed the fitting-segment maximum); DDPM clamped to [0, 1.2] | linear head for both, identical scaling | Neither generator is artificially capped, so the comparison is fair. |
| 13 | Training length | fixed 40 epochs on 50 slots in the benchmark script | selected per model on the validation window (slots 80-99, separate seeds) by CRPS | Principled and identical procedure for every learned model. |
| 14 | Random draws | global RNG state, order dependent | per (seed, model, slot, slice) generators; `torch.use_deterministic_algorithms` | Same seed gives the same numbers; batching does not matter. |
| 15 | Evaluation protocol | 150 slots, test slots 100-149, one seed | 260 slots, fit 0-99, test 100-259, ten seeds, paired tests with Holm correction | Protocol of the base paper, with a larger seed set. |
| 16 | Diffusion sample count | document said M = 100, code used 12 | M = 12 (paper value) by default; 64 and 256 as ablations | Text and code agree. |
| 17 | Reported numbers | hand-copied into the document | produced by `scripts/analyze.py` from saved CSVs with a manifest (git commit, versions, config) | Every figure and table is traceable. |

## Controls added

* **QuantileMLP-JCSO**: direct quantile regression with the same condition and
  width as the generators. Tests whether a generative sampler is needed.
* **Matched-budget frontiers** (`configs/frontier.yaml`): each controller is
  compared at equal over-reservation, by scaling the reservation envelope and,
  separately, by varying the reservation quantile.
* **Proactive information setting**: reservations made before the current
  request is observed, where the demand estimate carries the full burden.

## Behaviour inherited from the GAN-JCSO model (kept unchanged)

**Isolation admission cliff.** The candidate screen admits a co-location when
the summed coupling is at most theta = 0.52, so eMBB and URLLC (rho = 0.42) may
share a node. The admission rule then requires the slot isolation index
Gamma_t = 1 - mean(phi) / theta to be at least 0.25, but that single pair gives
Gamma_t = 1 - 0.42 / 0.52 = 0.19. Whenever the placement co-locates only eMBB
and URLLC, every request of the slot fails admission. With more capacity the
largest node can hold both slices, which is why several controllers miss more
at capacity multiplier 1.0 than at 0.82 under normal bursts. The rule is kept
exactly as published so that results remain comparable with GAN-JCSO; the
effect is reported rather than tuned away.
