# Work log

Shared hand-off file for everyone working on this repository (Rohit, the
cloud Claude session, and Claude Code on Rohit's laptop). Newest entry at the
bottom. Claim open tasks with `[local]` or `[cloud]` before starting.

## Open tasks

- [x] Rohit: approve `docs/PREREGISTRATION_B.md` (approved v3.1, 2026-09-30).
- [x] [local] Independent review of PREREGISTRATION_B v2 (verdict in the log, 2026-09-30).
- [x] [cloud] Resolve review items B1-B6 in PREREGISTRATION_B (version 3, a500906).
- [x] [local] Re-check B1-B6 against v3 (verdict in the log, 2026-09-30).
- [x] [cloud] One-line fix R1 (v3.1, 93184d3); confirmed by local review.
- [x] [local] Review of Amendment 1 (verdict in the log, 2026-09-30).
- [x] [cloud] Amendment 1 conditions A1-A3 (968d6b0).
- [x] [local] Re-check A1-A3 against 968d6b0 (verdict in the log, 2026-09-30): all PASS.
- [x] Rohit: approve Amendment 1 (approved 2026-09-30 as of 968d6b0).
- [x] [local] Review of the RB code against PREREGISTRATION_B sections 2-5 (log, 2026-10-01). Local validation rerun dropped by Rohit: the cloud selection is official and GAN results are platform-sensitive.
- [x] [cloud] RB code review follow-ups F1-F5: F1 fixed in code (b3784fe); F2-F5 stated in the manuscript (log, 2026-10-01).
- [x] [cloud] Implement rule RB (`src/dsorch/reservation.py`) per
      PREREGISTRATION_B v3.1 section 2 and Amendment 1 (4b28b3c).
- [x] [cloud] Pre-specified analysis of sections 5-6 (`src/dsorch/analysis_b.py`,
      `scripts/analyze_b.py`), tested on synthetic frontiers before any test run.
- [x] [cloud] Validation run of RB on seeds 1, 2, 5, 7, 9 (07f690c, 58 min).
- [x] [cloud] Study-B test run on seeds 1001-1080 (once, 54 min), analysis per sections 5-6.
- [x] [local] Windows rerun of the study-B test run for the cross-platform check (section 8); no verdict changes (log, 2026-10-01).
- [ ] [local] Exploratory public-trace replay of the six study-B controllers:
      `python scripts/run_experiment.py configs/b5g_b.yaml --jobs 8`, then
      `python scripts/analyze_b5g_b.py`; commit `results/b5g_b` (needs
      `data/external/public_b5g_trace.csv`, which stays uncommitted).
- [ ] Find why 5-8 non-learned baseline rows differ between Linux and
      Windows (suspected float ties at the 0.85 admission threshold).
- [ ] Per-method cross-platform stability table (needs Rohit's
      `results/verify/orchestration/orchestration.csv`).
- [ ] [cloud] Manuscript (after study B): Scientific Reports format, Word and
      LaTeX (sn-jnl), at most 8 display items. Draft complete; awaiting the
      authors' final edits. Kept out of the public repository.

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

### 2026-09-30 (local)
Independent review of PREREGISTRATION_B version 2 (commit af9e38e).

**Verdict: not ready for approval; minor revision required.** The design is
sound and every point of the earlier review is addressed. Six specification
gaps remain (B1-B6). Each leaves a choice to the implementer that can move
the primary outcome, so each must be fixed before approval, not by
amendment after the code exists.

Checked against the study-A CSVs (`results/frontier/matched_budget.csv`,
`results/tables/matched_budget_tests.csv`), all confirmed exactly:
QuantileMLP-JCSO lower mean miss than Diffusion-JCSO in 16/16 proactive
scale-lever cells (the earlier local review said 24; 16 is correct);
0/24 proactive Diffusion-vs-GAN rows Holm-significant; paired SD median
8.80 pp, range 1.94-17.92, lower quartile 5.52; minimum detectable
difference at alpha/12, 80 percent power (noncentral t): 13.4 pp with 10
seeds, 3.75 pp (median SD) and 2.35 pp (lower quartile) with 80 seeds.
`scripts/check_style.py` passes.

Blocking items:

- **B1. Over-reservation under relaxation.** `metrics.py:28` measures
  over-reservation on the target `tgt`. State whether RB's target is taken
  before or after capacity-aware relaxation (section 2, step 5).
  Post-relaxation lowers RB's measured budget whenever capacity binds,
  which moves every matched-budget cell.
- **B2. Sample count confounds H3 and H4.** RB uses M = 200; the JCSO
  comparators keep M = 12. In study A, M = 256 alone lowered Diffusion-JCSO
  proactive miss from 10.46 to 9.46 percent (heavy, capacity 1.0). H3 then
  credits RB with a sampling effect. Fix: run Diffusion-JCSO and GAN-JCSO
  with M = 200 for H3 and H4 (keep M = 12 as a study-A reference only), or
  restate H3 as "RB with M = 200 vs Eq. (20) with M = 12".
- **B3. Calibration multiplier ties.** Coverage pooled over 60 slice-slots
  moves in steps of 1/60, so many of the 41 values of g give the same
  coverage. Give a tie rule (for example the smallest g, the most
  conservative), and say whether the [0.002, 0.60] clip is reapplied to
  g * eps_s.
- **B4. Levels outside the resolvable range.** QuantileMLP: 1 - eps/2 >
  0.995 whenever eps < 0.01, and `quantile_at` (`np.interp`) then returns
  the 0.995 quantile silently. Generators: eps below 1/M = 0.005 cannot be
  resolved with 200 samples, and lambda = 6 may not reach 1 - eps. State
  the rule for both (clamp, and report how often it happens).
- **B5. Missing seeds in paired tests and in tuning.** The 72-of-80 rule
  should count seeds present for both methods (H1-H3) or all four
  (H4). The validation objective (mean miss over budgets 30/40/50 on 5
  seeds) needs the same rule for budgets a validation seed does not reach.
- **B6. Relaxation ladder.** Specify the 12 ladder levels, the order of
  relaxation and queue correction (step 6), and a deterministic tie-break.
  "Reduces the penalty least" should read "increases the expected
  shortfall penalty least".

Non-blocking notes:

- Calibrating g on the slots 0-79 model and applying it to a model refit on
  0-99 is not split conformal (no coverage guarantee). Either keep the 0-79
  model for the test slots or call the step "calibration" rather than
  "split-conformal".
- QuantileMLP-RB uses a retrained model (extended level grid), so it is not
  the study-A QuantileMLP. State this next to H2.
- Give a run-time estimate before the test run: 80 seeds x 2 workloads x 3
  learned models x 2 fits (0-79 and 0-99), with DDPM at 1600 epochs. The
  Windows rerun of the test runs (section 8) is about 4 times slower.

Next: cloud revises to version 3; local re-checks B1-B6 only, then Rohit
approves.

### 2026-09-30 (local)
Re-check of B1-B6 only, against PREREGISTRATION_B version 3 (a500906).

**Verdict: B1, B2, B3, B5 and B6 resolved. B4 resolved except one residual
(R1), a one-line fix. Approve once R1 is fixed; no further review round is
needed beyond confirming that line.**

- **B1 resolved** (section 5). Budget is measured on the requested level-0
  target, before relaxation; Eq. (20) on its target before capping. This
  matches `metrics.py:28`. Realized ratio reported as secondary.
- **B2 resolved** (section 3). Diffusion-JCSO and GAN-JCSO use M = 200;
  QuantileMLP-JCSO uses the same retrained model as QuantileMLP-RB.
- **B3 resolved** (section 2.4). Tie rule: smallest |log g|, then smaller
  g; the 41-point grid contains g = 1 exactly (checked). Clip is reapplied
  (eps_s' = clip(g * eps_s)). Implementation note: compare coverage as
  integer counts out of 60, not floats, so ties are exact.
- **B4 resolved, with residual R1** (section 2.3). Common range [0.01, 0.60]
  is resolvable by both model types (0.99 = 198/200 samples; Bonferroni
  0.995 is the top QuantileMLP level); lambda-capped and clipped fractions
  are reported.
  **R1:** v2 clipped the base level eps_s = c * b_s; v3 dropped that clip
  and clips only effective levels. The recalibration target
  mean_s(1 - eps_s) then uses unclipped eps_s, which exceeds 1 (a negative
  target coverage) for 12 of the 25 lever values when b_s = 1, 10 of 25
  when b_s = 0.5, and 7 of 25 when b_s = 0.25 (checked on the stated grid).
  The g selected at those lever values is then arbitrary, and it affects the
  other slices through the pooled target. Fix: in sections 2.4 and 5, define
  eps_s = clip(c * b_s) to [0.01, 0.60] before recalibration and use the
  clipped values in the target.
- **B5 resolved** (sections 4, 5, 6). Seeds must reach the budget for every
  controller in the comparison (2 for H1-H3, 4 for H4); tuning requires all
  60 points, with a stated fallback; excluded cells count toward neither
  side of the decision rule.
- **B6 resolved** (section 2.5). Ladder eps_s' * 1.5^k, k = 0..11, clipped
  and merged; queue correction at every level; trigger, choice (smallest
  increase in unit-free expected shortfall), deadline-based tie-break,
  12-step stop and fallback are all specified. The tie-break order (eMBB,
  mMTC, URLLC) is the reverse of the deadline order in `configs/base.yaml`,
  as intended.

Outside B1-B6 (not reviewed in depth): the open task for the RB code still
says "split-conformal calibration"; v3 renamed the step "validation-window
recalibration". `scripts/check_style.py` passes on v3.

Next: cloud applies R1 in the approval commit or a v3.1; local confirms the
line; Rohit sets the status to approved.

### 2026-09-30 (cloud)
Rohit approved PREREGISTRATION_B version 3.1 (text as of 93184d3, main at
8514090). R1 confirmed by local review before approval. Status line set to
approved; no study-B code existed at approval. RB task renamed to
"validation-window recalibration" to match the pre-registration. CLAUDE.md
section 8 updated to the new workflow: Rohit pushes to GitHub (public), the
cloud reads GitHub directly and returns bundles that fast-forward `main`.
Next: cloud implements RB (tests first).

### 2026-09-30 (cloud)
RB code written on branch b/rb-implementation (not delivered; 45 tests pass).
A smoke run and a reachability check on the validation seeds showed that the
RB risk lever spans too narrow an over-reservation band: no budget is
reached by all six controllers in more than 15 of 20 validation
seed-conditions, and 30/40/50 percent in at most 6. Study B as approved
would be "not supported" by construction. Proposed Amendment 1 in
PREREGISTRATION_B (scale lever for RB, risk level tuned). Miss rates were
not inspected for this decision. No test seed has been run. Needs local
review and Rohit's approval before the tuning run.

### 2026-09-30 (local)
Review of Amendment 1 to PREREGISTRATION_B (ca7aa52).

**Verdict: accept in principle; not ready for approval until A1-A3 are
met.** The reason is legitimate and the timing is correct: the problem was
found on validation seeds, from over-reservation only, before tuning and
before any test seed. As approved, section 5 would exclude every cell, so
every hypothesis would be "not supported" whatever the controllers do. The
fix is also an improvement in its own right: both rules are now traced by
the same scale lever, which removes a lever asymmetry present in v3.1.

Context checked locally (study A, `results/frontier/frontier.csv`,
proactive, scale lever 0.70-1.60, M = 12): the Eq. (20) controllers reached
the 30/40/50 percent budgets in 95-100 percent of seed-conditions (GAN-JCSO
lowest, 0.95 at 30 and 40; its highest per-seed minimum over-reservation
was 47 percent). So the shortfall comes from the RB side, as the amendment
states, and lowering the Eq. (20) grid to 0.6 is useful margin. I could not
check the RB reachability figures: they come from code on
b/rb-implementation, which has not been delivered.

Conditions before approval:

- **A1. Commit the evidence, for both levers.** A script and a CSV with
  over-reservation only (min and max per controller and validation
  seed-condition; no miss-rate columns), covering (a) the c lever as
  approved, which reproduces the figures in the Reason, and (b) the
  proposed s lever for all six controllers at M = 200, for at least the
  extreme candidates of the tuning grid (c = 0.02 and c = 0.4, all six
  shapes). The amendment must show that 30/40/50 percent is reached by all
  six controllers in at least 18 of 20 validation seed-conditions (the
  90 percent of section 5). Otherwise the same failure could return and
  need an Amendment 2. The tuning eligibility rule protects the RB side only
  for the selected (c, b), not the Eq. (20) side.
- **A2. Recalibration is independent of s.** State that the recalibration
  of section 2.4 is computed on the unscaled reservation (s = 1), so g
  depends on the seed and (c, b) only, and s then multiplies the
  recalibrated ladder. A coverage target of 1 - eps is meaningless on a
  scaled reservation.
- **A3. "Miss rates were not inspected".** This cannot be verified if the
  smoke run printed miss rates. Record in the amendment exactly what was
  run (commit of b/rb-implementation, command, output columns), so the
  claim is checkable.

Non-blocking:

- With a common scale lever, eps no longer sets a risk budget; RB becomes a
  distribution-shaped reservation with a global scale, and recalibration
  only matters for how the reservation is shared across slices and slots.
  The amendment says this. The manuscript should describe RB the same way
  and not claim calibrated per-slice risk control.
- In the tuning grid, c * b_s < 0.01 clips to 0.01 (for example c = 0.02,
  b_s = 0.25 or 0.5), so some of the 30 candidates are identical. This is
  harmless (identical candidates give identical results), but report the
  number of distinct candidates.

Next: cloud adds A1-A3 to Amendment 1; local re-checks A1-A3 only; Rohit
approves Amendment 1 before the tuning run.

### 2026-09-30 (local)
Re-check of Amendment 1 conditions A1-A3 against the committed files at
968d6b0 (GitHub main 52a1168; the only change after 968d6b0 is the previous
WORKLOG entry). No study-B tuning or test run was started.

**Verdict: A1 PASS, A2 PASS, A3 PASS. No new blocking issue. Amendment 1
is ready for Rohit's approval.**

- **A1 PASS.** All four files exist in `results/validation_b/`. The manifest
  (`reachability_manifest/manifest.json`) records commit 4b28b3c,
  `dirty: false`, command `scripts/reachability_b.py --jobs 2`. Recomputed
  from the CSVs: `reachability_fullgrid.csv` holds 186 frontiers x 21 values
  of s (3906 rows), all 186 non-decreasing in s. `reachability_endpoints.csv`
  (3720 rows = 93 frontiers x 20 seed-conditions x 2 endpoints): the three
  Eq. (20) controllers and all 30 candidates of Diffusion-RB and of GAN-RB
  reach 30/40/50 percent in 20 of 20 seed-conditions; QuantileMLP-RB 23 of
  30 candidates at 20/20, 5 reach 30 percent in 19/20, and 2 (c = 0.02,
  b = (1,0.25,0.5) and (0.75,0.25,0.5)) in 17/20; the failures are only at
  30 percent, where even s = 0.6 reserves 32-36 percent. The committed
  `reachability_summary.csv` agrees with the recomputation on all 93 rows.
  Only seeds 1, 2, 5, 7, 9 appear.
- **A2 PASS.** `recalibrate` (`src/dsorch/reservation.py:133-151`) takes no
  scale argument and clips the base levels (R1). `rb_setup`
  (`src/dsorch/experiments_b.py:84-90`) runs recalibration and
  `build_ladders` once per seed and candidate; `rb_frontier`
  (`experiments_b.py:100-103`) calls it before the loop over s. The scale is
  applied to every ladder level in `RBPolicy.plan`
  (`reservation.py:230-231`), which returns the requested level-0 target
  (`reservation.py:233, 256`), so the budget is measured before relaxation.
  g does not depend on s.
- **A3 PASS.** `scripts/reachability_b.py:59` asserts each row's keys equal
  `REACH_COLUMNS` (`experiments_b.py:108`) before `workload` and `seed` are
  added, so the stored columns are exactly method, rule, model, candidate,
  c, shape, lever, capacity, over_reservation, workload, seed (the CSV
  headers and the manifest's `stored_columns` agree). The script prints
  only reach counts, monotonicity and the number of distinct candidates.
  Committed `results/` files added since approval are the four reachability
  files only; none has a miss-rate column (the manifest's only "miss"
  matches are in "admission"), and no committed CSV contains a seed in
  1001-1080. `tests/test_reservation.py` uses synthetic data only.
- **Section 4 eligibility.** `select_shapes` (`experiments_b.py:177-201`)
  counts interpolated points per candidate over workload x seed x capacity
  x budget (60) and selects among candidates with all 60 reached; the
  fallback follows section 4. Checked with a synthetic frame: a candidate
  with a better objective but 59 of 60 points was rejected in favour of a
  worse one with 60 of 60.
- `python -m pytest`: 46 passed (14.5 s, Windows, dsorch env).
  `python scripts/check_style.py`: passes.

Non-blocking notes:

- Monotonicity in s was verified on one of the 20 seed-conditions only.
  This does not weaken A1: `frontier_points` uses the minimum and maximum
  over the whole grid, which contain the s = 0.6 and s = 1.8 values, so the
  endpoint reach counts are lower bounds.
- `QuantileBank._interp` (`reservation.py:90-98`) is flat beyond the grid
  ends (0.05 and 0.995), so the QuantileMLP expected shortfall used to order
  relaxation steps ignores the extreme tails. It affects only which slice is
  relaxed first, for QuantileMLP-RB; worth a sentence in Methods.
- The column guard is an `assert`, which `python -O` would skip. Harmless for
  the committed run.

Next: Rohit approves Amendment 1; cloud runs the validation (tuning) step
and commits `results/validation_b/selected.yaml` with the measured run time.

## 2026-09-30 (cloud): Amendment 1 approved; analysis code committed

Rohit approved Amendment 1 as of 968d6b0 after the local re-check
(1091399). The pre-specified analysis (`analysis_b.py`, per-cell paired
tests with Holm per hypothesis, the section 6 decision rule, seed-level
bootstrap with seed 20260930) is committed with its synthetic-data tests
before any tuning or test-seed run. Tests: 49 passed; style check passes.
Local note for Methods: QuantileMLP expected shortfall is flat beyond the
0.05 and 0.995 grid ends. Next: tuning on validation seeds.

## 2026-09-30 (cloud): study B tuning and test run

First tuning launch was lost when the cloud container restarted while idle
(no job had finished; nothing written). Added resumable per-job checkpoints
(51f1009, gitignored under results/scratch/checkpoints) and reran.

Tuning (validation seeds only, 07f690c): Diffusion-RB c = 0.4, b = (1, 0.25, 1);
GAN-RB c = 0.1, b = (1, 0.25, 0.5); QuantileMLP-RB c = 0.4, b = (1, 0.25, 1).
Two of three selections sit at the top of the c grid (report as a threat to
validity; grid kept as registered).

Test run (seeds 1001-1080, run once, config test_b.yaml): 160 jobs, 3230 s,
2 workers. Analysis by `scripts/analyze_b.py` unchanged since fb6fb99.
Verdicts (section 6): H1 not supported (Diffusion-RB minus GAN-RB +1.9 pp,
95% CI -1.5 to 5.8; no cell significant). H2 contradicted (QuantileMLP-RB
better than Diffusion-RB in 12 of 12 cells, +6.5 pp). H3 not supported
(RB lowers Diffusion miss rate by 2.1 pp, CI -3.0 to -1.2; 4 cells
significant in the predicted direction at capacity 1.0 and budgets 30/40,
1 cell opposite). H4 not supported by the cell rule (-2.5 pp, CI -4.2 to
-0.9; no cell significant). All 12 cells analysed for every hypothesis.

### 2026-10-01 (local)
Cross-platform check of study B (PREREGISTRATION_B section 8), from 88e1166
(clean tree). Check only: `results/study_b` and `results/validation_b` were
not modified (index hashes identical before and after).

Run: `python scripts/run_experiment.py configs/test_b.yaml --jobs 4 --out
results/verify/study_b`, Windows 11, torch 2.14.0+cpu, Python 3.11.16:
160 jobs, 5770 s (1 h 36 min); official cloud run (Linux, 2 workers):
3230 s. Then `python scripts/analyze_b.py results/verify/study_b`.
`compare_runs.py` keys do not fit `frontier_b.csv` (no `information`
column, no lever); compared directly on workload, seed, method, lever,
capacity (40320 of 40320 rows matched).

**Verdicts: 4 of 4 agree; no hypothesis changes verdict.**

| | official (Linux) | Windows rerun |
|---|---|---|
| H1 | not supported, 1.90 pp [-1.52, 5.81] | not supported, 1.18 pp [-2.52, 5.27] |
| H2 | contradicted (12/12 opposite), 6.45 [3.38, 10.05] | contradicted (12/12 opposite), 6.50 [3.42, 10.11] |
| H3 | not supported (4 predicted, 1 opposite), -2.09 [-2.96, -1.18] | identical to two decimals |
| H4 | not supported (0 predicted), -2.52 [-4.15, -0.94] | not supported (1 predicted), -3.23 [-5.13, -1.35] |

Per-cell mean_diff sign agreement: H1 8/12, H2 12/12, H3 12/12, H4 12/12.
The four H1 sign flips are all non-significant cells with |mean_diff| below
1.4 pp in both runs. Holm significance status agrees in 47/48 cells; the
exception is H4 extreme, capacity 1.0, budget 30 (p_holm 0.062 official,
0.004 rerun), which leaves H4 at 1 of the 6 cells needed. Excluded seeds
differ by one in two cells (76 vs 75, 78 vs 79); no cell is excluded in
either run.

Per method (cell means of miss rate, 12 cells each; rerun minus official):

- **Diffusion-RB, Diffusion-JCSO:** miss rate identical in every frontier
  row; over-reservation differs by at most 0.13 and 0.49 pp per row.
  Cell means unchanged.
- **QuantileMLP-RB, QuantileMLP-JCSO:** max cell-mean difference 0.23 and
  0.18 pp; no cell over 1 pp.
- **GAN-JCSO:** max 1.49 pp (heavy, 1.0, budget 30); 1 cell over 1 pp.
- **GAN-RB:** max 2.08 pp (extreme, 1.0, budget 30); 6 of 12 cells over
  1 pp, mean shift +0.63 pp. Row level: 78 percent of GAN-RB rows and
  55 percent of GAN-JCSO rows differ by more than 0.5 pp.

Flagged (more than 1 pp): GAN-RB (6 cells) and GAN-JCSO (1 cell). As in
the study-A rerun, the GAN rows are the platform-sensitive ones; this moves
H1 and H4 estimates by 0.7 pp but no verdict.

Note: `checkpoint_dir` (`src/dsorch/experiments_b.py:204-207`) names the
folder by the last part of `--out`, so this verify run and the official
output share `results/scratch/checkpoints/study_b`. A later official run on
this machine would silently resume from the verify checkpoints. Clear that
folder before any official rerun here, or key it on the full output path.

### 2026-10-01 (cloud)
- Fixed the checkpoint collision reported in the cross-platform entry:
  `checkpoint_dir` now keys on the full output path
  (`results__study_b` vs `results__verify__study_b`); test added. The old
  `results/scratch/checkpoints/study_b` folder is no longer read by any
  run and can be deleted.
- Added figure and analysis scripts used by the manuscript:
  `scripts/figure_style.py`, `make_figure1.py` (overview figure),
  `make_figures_b.py`, `exploratory_b.py` (five non-pre-registered
  contrasts, labelled exploratory), `sensitivity_b.py` (Wilcoxon
  signed-rank sensitivity: same four verdicts). Figures are written as
  PDF, EPS, 600 dpi PNG and TIFF.
- Manuscript (Scientific Reports, task 4) is drafted in the cloud session;
  it is kept out of this public repository until the authors decide.

### 2026-10-01 (local)
Code review of rule RB: `src/dsorch/reservation.py` and
`src/dsorch/experiments_b.py` read in full at 608ffd6 against
PREREGISTRATION_B sections 2-5 and Amendment 1. Also read the code they
call (`engine.py`, `metrics.orchestration_metrics`, forecaster sampling,
`QuantileMLP`) and the section 5 seed rule in `analysis_b.py`. No rerun.

**Verdict: the implementation matches sections 2-5 with Amendment 1. No
finding changes a reported study-B number or verdict.** Five reporting
and documentation items (F1-F5); F1 should be fixed before any further run.

Checked and consistent (reference -> section):

- 2.1: RB target is `scale * r * (1 + kappa * Q)` with no alpha_s, no LWMA
  and no max(u_bar, u) (`reservation.py:230-231`).
- 2.2 generators: median, 0.95-quantile spread floored at 1e-6
  (`reservation.py:47-48`); smallest lambda on the 0.01 grid with joint
  coverage >= 1 - eps, computed exactly from per-sample minimal lambdas
  plus a float guard; lambda capped at 6 and flagged (`:55-71`).
  QuantileMLP: Bonferroni `Q(1 - eps/2)` (`:103-108`); quantiles sorted, so
  they cannot cross (`baselines.py:81`); extended grid applied by
  `study_b_config` (`experiments_b.py:32-38`).
- 2.3: base levels, recalibrated levels and every ladder level are clipped
  to [0.01, 0.60] (`experiments_b.py:87`, `reservation.py:139, 144, 176,
  180`).
- 2.4: f_val fitted on slots 0-79, reservations on 80-99 (60 slice-slots),
  41-point g grid, target from clipped base levels, ties on integer counts
  then |log g| then smaller g (`reservation.py:133-151`); refit on 0-99 for
  the test slots (`experiments_b.py:73-77`).
- 2.5: ladder clip(eps' * 1.5^k), k = 0..11, equal levels merged
  (`reservation.py:176-182`); queue correction at every level (`:231`);
  trigger on unassigned or under-allocated in either resource (`:238`);
  smallest weighted increase in unit-free shortfall, tie by longest
  deadline (`:242-249`); at most 12 steps, last placement kept (`:237-256`).
- Amendment 1: recalibration once per seed and candidate, before the loop
  over s (`experiments_b.py:100-103`); scale on every ladder level.
- 3: Eq. (20) comparators with M = 200 (`experiments_b.py:144`);
  QuantileMLP-JCSO uses the same retrained model as QuantileMLP-RB (`:142`);
  study-A epochs (1600/400/160) in the run config.
- 4: 30 candidates; objective and P95 tie-break; eligibility requires all
  60 points, with the stated fallback (`experiments_b.py:177-201`).
- 5: 80 test seeds; common 21-point scale grid (`:41-44`); budget measured
  on the requested level-0 target (`reservation.py:256`, `metrics.py:28`);
  interpolation without extrapolation (`experiments_b.py:285-301`);
  missing-seed rule on seeds that have every controller of the comparison,
  at least ceil(0.9 * 80) = 72 (`analysis_b.py:32-50`).
- No leakage: RB plans use only the predictive banks built from history
  t-5..t-1 (`common.py:99-107`); `run_episode`'s current request is never
  passed to RB (`engine.py:80-84`).

Findings:

- **F1 (fix before any rerun). Manifests record the wrong config.**
  `run_experiment.py:52` writes the config before `exp_test_b` /
  `exp_validation_b` apply `study_b_config` (`experiments_b.py:221, 270`).
  Both `results/study_b/manifest.json` and
  `results/validation_b/manifest.json` therefore show
  `information: request_observed` and the 12-level QuantileMLP grid. The
  runs themselves used proactive information and the 15-level grid:
  QuantileMLP-RB reaches eps = 0.01 (level 0.995) with no capped decisions
  in `frontier_b.csv`, which the 12-level grid could not do. Fix: write the
  effective config into the manifest (for example return it from the
  experiment function), or add a note to both manifests' README entries.
- **F2 (document). Relaxation order ignores s and Q.** The shortfall used to
  choose which slice to relax is precomputed on the unscaled ladder r(eps_k)
  (`reservation.py:189, 246`), not on the reservation actually requested,
  s * r * (1 + kappa * Q). This matches "precomputed" in section 2.5, but
  after Amendment 1 the order does not depend on s. Say so in Methods.
- **F3 (label). `realized_joint_coverage`** (`experiments_b.py:124, 130`) is
  the coverage of the unscaled level-0 reservation r(eps'), not of the
  scaled, queue-corrected request or of the allocation. Describe it as
  "calibrated coverage at s = 1" wherever it is reported.
- **F4 (missing secondary outcome). Inference time for RB.** Section 6 lists
  inference time; RB rows carry none (`experiments_b.py:125-133`), only the
  Eq. (20) rows do (`:153`), and `fit_seconds` (`:79`) includes training.
  Time `build_bank` separately, or report RB inference time as the
  Eq. (20) sampling time, which uses the same M = 200 draws.
- **F5 (definition). Clip fractions.** `decision_flags`
  (`reservation.py:259-270`) counts a decision as clipped if the level
  after recalibration (level 0) or on the ladder would leave [0.01, 0.60];
  clipping of the base level c * b_s before recalibration is not counted.
  GAN-RB reaches frac_clip_low = 0.67 in some rows. State the definition
  next to the table.

Still open from the Amendment 1 review: QuantileMLP shortfall is flat
beyond the grid ends 0.05 and 0.995 (`reservation.py:90-98`), so it affects
only the relaxation order for QuantileMLP-RB.

### 2026-10-01 (cloud): RB code review follow-ups F1-F5
- F1: `run_experiment.py` now records `study_b_config(cfg)` (proactive,
  15-level QuantileMLP grid) in study-B manifests; test runs smoke_b and
  checks the manifest. The committed manifests are left as the original
  record; `scripts/effective_config_b.py` writes `effective_config.json`
  next to each (both show proactive and 15 levels). No rerun needed.
- F2: Methods states that the relaxation order uses the unscaled
  reservations, independent of the frontier scale and the queue correction.
- F3: realized joint coverage is reported as coverage of the unscaled
  level-0 reservation (s = 1, before queue correction).
- F4: inference times are reported from the published-rule controllers
  (same 200 samples); the text says they were not recorded for RB rows.
- F5: clipping is defined next to the numbers (slice-slot decisions whose
  effective level fell outside [0.01, 0.60]); GAN-RB maximum 67% per
  frontier setting is reported.

### 2026-10-01 (cloud): public-trace replay prepared (exploratory)
Reviewers will ask whether the study-B ranking holds on real traffic; all
study-B evidence is synthetic. Added experiment `b5g_b`: the six controllers
on the three public-trace segments x 10 model seeds (2001-2010), RB settings
frozen from `results/validation_b/selected.yaml` (nothing tuned on the
trace), scale lever widened to 41 values in [0.4, 2.4] so the budgets are
reachable, fixed before any run. Analysis `scripts/analyze_b5g_b.py`:
budgets 30/40/50, 2 capacities, H1-H4 and E1, E3, E4 contrasts over pairs
(segment, model seed), Holm over 6 cells, a cell analysed only if 90% of
pairs reach the budget, per-segment means and sign agreement reported. Not
pre-registered; reported as exploratory. Tested end to end on a stand-in
file with the trace's columns (`DSORCH_B5G_TRACE` overrides the path).

### 2026-10-01 (local)
Exploratory public-trace replay `b5g_b` (not pre-registered), from fb3945d
with the design as committed there; no config, budget or setting changed.
`data/external/public_b5g_trace.csv` present and git-ignored; not committed
(output has no demand columns; manifest does not reference the file).
`python -m pytest`: 53 passed. `python scripts/run_experiment.py
configs/b5g_b.yaml --jobs 8` (Windows, 8 workers): 30 jobs (3 segments x
10 model seeds), 999 s (16.6 min). Then `python scripts/analyze_b5g_b.py`.

Mean miss rate (%) at matched over-reservation:

| capacity | budget | Diffusion-RB | GAN-RB | QuantileMLP-RB | Diffusion-JCSO | GAN-JCSO | QuantileMLP-JCSO |
|---|---|---|---|---|---|---|---|
| 0.82 | 30 | 80.9 | 82.9 | 82.4 | 89.0 | 89.6 | 88.2 |
| 0.82 | 40 | 73.6 | 74.5 | 75.7 | 77.5 | 78.7 | 76.6 |
| 0.82 | 50 | 66.8 | 67.7 | 68.3 | 68.7 | 70.0 | 69.2 |
| 1.00 | 30 | 85.2 | 90.6 | 89.7 | 93.0 | 94.4 | 93.0 |
| 1.00 | 40 | 73.6 | 77.5 | 76.5 | 86.4 | 87.0 | 84.7 |
| 1.00 | 50 | 63.8 | 67.8 | 67.6 | 72.9 | 73.2 | 70.6 |

Budgets not reached, reported as they are: GAN-RB reaches 30 percent in
26/30 (capacity 0.82) and 25/30 (1.0) pairs, 40 percent in 27/30, 50
percent in 29/30; GAN-JCSO reaches 30 percent in 29/30. With the 90 percent
rule (27 pairs), the 30 percent cells of H1, H4 and E3 are not analysed.
All other controllers reach every budget in 30/30.

Contrasts (Holm over 6 cells each; first minus second, negative favours the
first): H3 (Diffusion-RB vs Diffusion-JCSO) significant in 6/6 cells,
-1.9 to -12.8 pp; H2 (Diffusion-RB vs QuantileMLP-RB) significant in 5/6,
-1.5 to -4.5 pp, the opposite direction to the synthetic test; H1 significant
in 2/4 analysed cells (capacity 1.0, budgets 40 and 50, about -4 pp);
E1 (QuantileMLP-RB vs QuantileMLP-JCSO) significant in 4/6; H4, E3, E4
significant in none.

Points for the write-up (facts, not a change to the analysis):

- Absolute miss rates are 64 to 94 percent at these budgets, so every
  contrast compares controllers that miss most deadlines on this trace.
- The 30 pairs come from 3 trace segments; pairs within a segment share the
  trace, so the pooled paired p-values treat model seeds as independent
  replicates. The per-segment column `segments_same_sign` (of 3) is the
  trace-level evidence: H3 3/3 in 5 cells, H2 3/3 in 5 cells.

### 2026-10-01 (cloud): public-trace replay written up
- Read d205a03. At the pre-specified budgets every controller misses 64-94%
  of deadlines on the public trace. Added `scripts/describe_b5g_b.py`
  (post hoc, descriptive, no tests): miss rates at 60-200% over-reservation
  (`results/b5g_b/tables/descriptive_budgets.csv`) and Figure 5
  (`results/figures/fig_b5g_b_frontiers.*`). At 150-200% the published-rule
  controllers reach 0.2-3.8% (capacity 1.0) and 8.4-12.2% (0.82), within
  about 3 pp of each other, Diffusion-JCSO lowest by 0.4-0.9 pp; the RB
  controllers stay above them and Diffusion-RB levels off at 22-25%
  (capacity 1.0). The RB settings tuned on synthetic traffic do not
  transfer to this trace at service-relevant budgets.
- Manuscript reframed accordingly (title: diffusion forecasts do not
  consistently improve slice reservation at matched budgets). Kept out of
  this repository.
