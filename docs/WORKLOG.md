# Work log

Shared hand-off file for everyone working on this repository (Rohit, the
cloud Claude session, and Claude Code on Rohit's laptop). Newest entry at the
bottom. Claim open tasks with `[local]` or `[cloud]` before starting.

## Open tasks

- [ ] Rohit: approve `docs/PREREGISTRATION_B.md` version 2 (blocks all study-B work).
- [ ] [local] Independent review of PREREGISTRATION_B v2, then of the RB code; local rerun of the RB validation step (check only; the cloud selection is official).
- [ ] [cloud] Implement rule RB (`src/dsorch/policies.py` or a new `reservation.py`):
      joint risk target from samples, Bonferroni variant for QuantileMLP,
      split-conformal calibration, capacity-aware relaxation. Tests first.
- [ ] Validation run of RB on seeds 1, 2, 5, 7, 9 to select eps_s; record
      the selection in `results/validation_b/selected.yaml`.
- [ ] Study-B experiment config and frontier runs on the test seeds (once).
- [ ] Find why 5-8 non-learned baseline rows differ between Linux and
      Windows (suspected float ties at the 0.85 admission threshold).
- [ ] Per-method cross-platform stability table (needs Rohit's
      `results/verify/orchestration/orchestration.csv`).
- [ ] Manuscript (after study B): Scientific Reports format, Word,
      Introduction / Results / Discussion / Methods, at most 8 display items.

## Log

### 2026-09-29 (cloud)
Built the repository: simulator per base-paper Eqs. (4)-(12), (20)-(22);
GAN per Table 5; DDPM with cosine schedule (K = 20); QuantileMLP control;
six experiments; 30 tests. Training lengths selected on validation seeds
(DDPM 1600, GAN 400, QuantileMLP 160 epochs). Full suite run from commit
e3d48be; rerun reproduced every CSV exactly on Linux.

### 2026-09-30 (cloud + Rohit)
History rewritten to remove AI co-author trailers; pushed (e696c08).
Thread oversubscription fixed (Windows orchestration run: 3 h to 4 min).
Windows verification rerun: 1168/1920 rows identical; every learned-model
row differs; scenario means within 1.9 pp miss rate; Diffusion-vs-GAN sign
agreement 0.79 (miss) and 0.75 (over-reservation), all other comparisons
0.96 to 1.00. Study B pre-registered (a474dd1). CLAUDE.md added.

### 2026-09-30 (cloud)
Pre-registration revised to version 2 after the local review. Checked the
review's claims against the data: QuantileMLP-JCSO below Diffusion-JCSO in
16/16 proactive scale-lever cells (26/28 including the quantile lever); 0/24
proactive Diffusion-vs-GAN matched-budget comparisons Holm-significant.
Power: median paired SD 8.8 pp; at alpha/12 and 80 percent power, 10 seeds
detect about 13 pp, 80 seeds about 3.7 pp. Adopted 80 fresh test seeds
(1001-1080), per-hypothesis Holm families, "no cell significant in the
opposite direction" rule, H4 as per-seed difference-in-differences,
per-method tuning with a stated objective, explicit missing-budget rule.
