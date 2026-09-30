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
- [ ] Rohit: approve Amendment 1 (blocks the tuning run).
- [ ] [local] Review of the RB code; local rerun of the RB validation step (check only; the cloud selection is official).
- [ ] [cloud] Implement rule RB (new `src/dsorch/reservation.py`) exactly as
      PREREGISTRATION_B v3.1 section 2: joint risk target from samples,
      Bonferroni variant for QuantileMLP, validation-window recalibration,
      capacity-aware relaxation ladder. Tests first.
- [ ] [cloud] Validation run of RB on seeds 1, 2, 5, 7, 9 (section 4); commit
      `results/validation_b/selected.yaml` and the measured run time before
      the test run.
- [ ] [cloud] Study-B test run on seeds 1001-1080 (once), analysis per sections 5-6.
- [ ] [local] Windows rerun of the study-B test run for the cross-platform check (section 8).
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
