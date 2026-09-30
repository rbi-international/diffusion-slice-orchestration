# Pre-registration: uncertainty-driven reservation (study B)

Status: **approved** (version 3.1, text as of commit 93184d3, main at
8514090). Approved by Rohit Bharti (corresponding author) on 2026-09-30,
after two independent local reviews and the local confirmation of fix R1.
No study-B code had been written or run at approval. Every later change
goes in the "Amendments" section with its reason and date, committed before
the run it affects.

Revision history:

* v1 (a474dd1): initial design.
* v2 (af9e38e): after the first local review. Missing-budget rule, frontier
  lever, H4 as difference-in-differences, per-hypothesis Holm families,
  opposite-direction rule, per-method tuning objective, fresh test seeds,
  power analysis.
* v3 (this version): after the second local review (e41b661 on branch
  b/prereg-review). Settles six open choices: budget measured on the
  requested (pre-relaxation) reservation (section 5); equal sample counts
  for the Eq. (20) comparators (section 3); tie-breaking in recalibration
  (section 2.4); a common resolvable risk range with reported clipping
  (section 2.3); joint seed availability for missing budgets, including
  tuning (sections 4 and 5); a fully specified relaxation ladder (section
  2.5). Also renames the recalibration step (it is not split-conformal),
  states that QuantileMLP is retrained for both rules in study B, and adds
  a run-time estimate with a pre-run coarsening rule (section 9).
* v3.1: after the third local review (f8810fb on branch
  b/prereg-review-v3). The base risk levels c * b_s are clipped to
  [0.01, 0.60] **before** recalibration, and the recalibration target uses
  the clipped values (sections 2.3, 2.4 and 5). Without this, the target
  1 - c * b_s is negative for 12 of the 25 lever values when b_s = 1 and
  for 7 of 25 when b_s = 0.25.

## 1. Motivation

In study A every forecaster feeds the GAN-JCSO reservation rule (Eq. 20):

    a_tar = alpha_s * max(u_bar, u) * (1 + kappa * Q)
    u_bar = 0.62 * Q_0.86(samples) + 0.38 * LWMA

The fixed margins alpha_s = (1.35, 1.30, 1.18) and the LWMA anchor inflate
every reservation by the same factor whatever the forecast uncertainty. In
study A the DDPM was better calibrated than the GAN, but under Eq. (20) this
did not yield fewer misses at matched over-reservation (0 of 24 proactive
Diffusion-vs-GAN comparisons survived Holm correction), and QuantileMLP-JCSO
had the lower mean miss rate in 16 of 16 proactive scale-lever cells.

Study B asks whether a reservation rule that uses the predictive
distribution directly converts calibrated generative uncertainty into a
better miss-rate versus capacity trade-off.

## 2. Reservation rule RB (risk-budgeted joint reservation)

Notation: slice s, slot t, predictive samples {x_m}, m = 1..M, of the
bandwidth-CPU demand vector, predicted from history t-5..t-1 (proactive
information: the current request is not used). M = 200 for every
generator-based controller in study B (section 3).

### 2.1 No fixed margins

alpha_s = 1 and the LWMA anchor weight is 0.

### 2.2 Reservation for a risk level eps

* **Generators (DDPM, GAN).** Per resource d, med_d is the sample median and
  h_d the sample 0.95-quantile minus med_d (floored at 1e-6). For
  lambda >= 0, r(lambda) = med + lambda * h componentwise. The reservation
  is r(lambda*), where lambda* is the smallest value on the grid
  {0, 0.01, ..., 6.00} whose empirical joint coverage
  (1/M) sum_m 1[x_m,bw <= r_bw and x_m,cpu <= r_cpu] is at least 1 - eps.
  If no grid value reaches 1 - eps, lambda* = 6 and the decision is counted
  as **lambda-capped**.
* **QuantileMLP (marginal quantiles only).** r_d = Q_d(1 - eps / 2) (the
  Bonferroni bound for joint coverage), by linear interpolation on its
  quantile grid. For study B the pinball-loss grid is extended with the
  levels 0.975, 0.99 and 0.995, and the model is retrained with the study-A
  epoch count (160). This retrained model is used for **both**
  QuantileMLP-RB and QuantileMLP-JCSO in study B, so the two rules are
  compared on the same model.

### 2.3 Resolvable risk range

With M = 200 samples, joint coverage is resolved in steps of 0.005; the
highest QuantileMLP level, 0.995, resolves eps = 0.01 under Bonferroni. To
give every method the same resolvable range, **every effective risk level
(after recalibration and relaxation) is clipped to [0.01, 0.60]** for all
methods. The base risk levels are clipped to the same range before
recalibration (section 2.4). For every method and cell, the fraction of decisions clipped at
0.01, clipped at 0.60, and lambda-capped is reported.

### 2.4 Validation-window recalibration

This step recalibrates coverage; it is **not** split-conformal prediction
and no finite-sample coverage guarantee is claimed, because the correction
is estimated on one fitted model and applied to a refitted one.

For each seed and each lever value c, the base risk levels are
eps_s = clip(c * b_s) with clip to [0.01, 0.60]. Fit the forecaster on slots 0-79;
compute RB reservations on slots 80-99 (60 slice-slots) with risk levels
g * eps_s for g on a grid of 41 log-spaced values in [0.1, 10]; choose the g
whose pooled empirical joint coverage is closest to the target
mean_s(1 - eps_s), computed with the clipped eps_s. **Ties** (equal absolute coverage error) are broken by
the smallest |log g| (least correction), then by the smaller g (more
conservative). Then refit on slots 0-99 and use eps_s' = clip(g * eps_s) on
the test slots.

### 2.5 Capacity-aware relaxation

* **Ladder.** For each slice the ladder has 12 risk levels
  eps_s,k = clip(eps_s' * 1.5^k), k = 0..11. Consecutive levels that are
  equal after clipping are merged. Reservations and expected shortfall are
  precomputed for every level.
* **Queue correction is applied at every level:** the requested target at
  level k is r(eps_s,k) * (1 + kappa * Q_s,t). Relaxation therefore acts on
  the queue-corrected target.
* **Trigger.** After placement (study-A placement rule, deadline order
  URLLC, mMTC, eMBB), a slice triggers relaxation if it is unassigned or its
  allocation is below its target in either resource.
* **Choice.** Among all slices of the slot that are not at their last
  level (a non-triggering slice may be relaxed to free capacity for a
  triggering one), relax the one with the smallest increase in the penalty
  w_s * sum_d E[(x_d - r_d)+] / med_d (expected shortfall, unit-free;
  estimated from samples, or from the quantile grid for QuantileMLP) when
  moving from level k to k + 1. **Ties** are broken by relaxing the slice
  with the longest deadline first (eMBB, then mMTC, then URLLC).
* **Stop.** Placement is repeated after each step; at most 12 steps per
  slot in total. If placement is still infeasible, the last placement is
  used. Admission and all simulator equations are exactly as in study A.

## 3. Controllers

| Name | Demand model | Rule | Samples M |
|---|---|---|---|
| Diffusion-RB | conditional DDPM | RB | 200 |
| GAN-RB | conditional GAN | RB | 200 |
| QuantileMLP-RB | quantile regression (retrained, extended grid) | RB, Bonferroni | n/a |
| Diffusion-JCSO | conditional DDPM | Eq. (20) | 200 |
| GAN-JCSO | conditional GAN | Eq. (20) | 200 |
| QuantileMLP-JCSO | same retrained model as QuantileMLP-RB | Eq. (20) | n/a |

The Eq. (20) comparators use M = 200 in study B (study A used the published
M = 12), so that H3 and H4 compare rules and not sample counts. In study A,
raising M from 12 to 256 alone lowered Diffusion-JCSO's proactive heavy-burst
miss rate from 10.46 to 9.46 percent. Forecaster architectures and
validation-selected training lengths are those of study A (DDPM 1600,
GAN 400, QuantileMLP 160 epochs).

## 4. Tuning (validation only)

* Validation seeds 1, 2, 5, 7, 9; heavy and extreme bursts; capacity 1.0 and
  0.82; proactive. No test seed is used.
* Each RB method is tuned **separately and identically**. The base shape b
  is chosen from {(1,1,1), (1,0.5,1), (1,0.25,1), (1,0.5,0.75),
  (1,0.25,0.5), (0.75,0.25,0.5)} (eMBB, URLLC, mMTC).
* **Objective:** mean miss rate over budgets 30, 40 and 50 percent, read off
  each validation seed's frontier, averaged over the 5 seeds x 4 conditions
  x 3 budgets = 60 points. Ties are broken by the smaller mean P95 latency.
* **Missing points in tuning.** A candidate shape is eligible only if its
  frontier reaches all 60 points. If no candidate is eligible for a method,
  the candidate reaching the most points is chosen, ties broken by the
  objective over the points it reaches; this is reported.
* The selection is written to `results/validation_b/selected.yaml` and
  committed before any test run. The cloud run's selection is the official
  frozen value; a rerun on another machine is a check only.

## 5. Test protocol

* **Test seeds 1001 to 1080 (80 seeds)**, never used in study A or in any
  tuning.
* Conditions (4): heavy and extreme bursts, capacity 1.0 and 0.82, proactive.
* **Frontier lever.** RB: c on 25 log-spaced values in [0.05, 20]; base
  risk levels eps_s = clip(c * b_s) to [0.01, 0.60] before recalibration.
  Lever values that give identical clipped levels for all slices give
  identical reservations and are kept as duplicate frontier points. Eq. (20) controllers: the target
  multiplied by a scalar on 21 values in [0.6, 1.8].
* **Budget measurement.** Over-reservation is computed on the **requested**
  reservation: for RB the level-0 target of section 2.5 (after
  recalibration and queue correction, **before** any capacity-aware
  relaxation); for Eq. (20) the Eq. (20) target (before capacity capping).
  RB therefore receives no budget credit for relaxation. The realized
  (post-relaxation, post-capping) reservation ratio is reported as a
  secondary outcome.
* **Budgets:** over-reservation 30, 40 and 50 percent; metrics are linearly
  interpolated along each seed's frontier, never extrapolated.
  4 conditions x 3 budgets = **12 cells per hypothesis**.
* **Seeds that do not reach a budget.** For a given comparison and cell,
  the analysed seeds are those whose frontiers reach the budget **for every
  controller in that comparison** (two controllers for H1 to H3, four for
  H4). The cell is analysed if at least 72 of the 80 seeds qualify;
  otherwise it counts as **not supporting** the hypothesis. The number of
  excluded seeds is reported for every cell.

## 6. Hypotheses

Primary outcome: submitted-request miss rate at matched over-reservation.

* **H1.** Diffusion-RB < GAN-RB.
* **H2.** Diffusion-RB < QuantileMLP-RB. Study A makes this unlikely; it is
  tested and reported whatever the outcome.
* **H3.** Diffusion-RB < Diffusion-JCSO.
* **H4.** RB helps the DDPM more than the GAN. Per seed and cell,
  D = (Diffusion-RB - Diffusion-JCSO) - (GAN-RB - GAN-JCSO); H4 predicts
  mean D < 0, tested with a one-sample two-sided t-test on D.

H1 to H3 use two-sided paired t-tests over seeds.

**Multiplicity.** The Holm family is the 12 cells of one hypothesis.

**Decision rule.** A hypothesis is **supported** if (i) the Holm-adjusted
p < 0.05 with the predicted sign in at least 6 of its 12 cells and (ii) no
cell is Holm-significant in the opposite direction. It is **contradicted**
if at least 6 cells are Holm-significant in the opposite direction, and
**not supported** otherwise. Cells excluded under section 5 count toward
neither the 6 supporting nor the 6 contradicting cells. For every
hypothesis the mean difference across analysed cells with a 95 percent
seed-level bootstrap confidence interval (10,000 resamples) is reported.

**Secondary outcomes** (reported, not used for decisions): acceptance, P95
and P99 latency, utilization, per-slice miss rate, realized reservation
ratio, realized joint coverage, clipping and lambda-cap fractions, number
of relaxation steps, and inference time.

## 7. Power

The standard deviation of paired per-seed miss-rate differences
(Diffusion-JCSO minus GAN-JCSO) in the study-A proactive matched-budget
cells had a median of 8.8 percentage points (range 1.9 to 17.9). For a
two-sided paired t-test at the most stringent Holm level (alpha / 12) and
80 percent power, 10 seeds detect only differences of about 13 pp; 80 seeds
detect about 3.7 pp at the median spread and about 2.4 pp at the
lower-quartile spread (5.5 pp).

## 8. Reporting commitments

All four hypotheses and all 12 cells of each are reported whatever the
outcome. Study A stays in the manuscript unchanged. The cross-platform check
(Windows rerun of the test runs, `scripts/compare_runs.py`) is reported for
every supported hypothesis.

## 9. Run-time estimate and pre-run coarsening rule

Estimate for the test run on the cloud machine (2 cores): 80 seeds x 2
burst levels = 160 jobs. Each job fits every forecaster twice (slots 0-79
and 0-99), draws 200 samples per slice-slot, and simulates about 25 x 3 x 2
RB and 21 x 3 x 2 Eq. (20) frontier points. From the study-A timings
(about 10 s per job for 5 fits and 48 episodes), about 60 to 120 s per job,
that is about 1.5 to 3 hours in total; roughly 4 times longer on Rohit's
laptop per core.

The validation run (section 4) is timed first. If its timing implies more
than 8 hours for the test run, the frontier grids are coarsened by an
amendment committed **before** the test run, never after it.

## Amendments

### Amendment 1 (proposed 2026-09-30; not yet approved; no test seed run)

**Reason.** Before the tuning run, the reachable over-reservation range of
every controller was measured on the validation seeds only (1, 2, 5, 7, 9;
heavy and extreme; capacity 1.0 and 0.82; 20 seed-conditions), with the
study-A training lengths. Miss rates were not inspected for this decision.
With the risk lever alone, RB spans only a narrow band of over-reservation,
because eps is confined to [0.01, 0.60]: at the most conservative setting
(eps = 0.01) the maximum over-reservation was 24 to 44 percent for
Diffusion-RB, 17 to 80 percent for GAN-RB and 37 to 108 percent for
QuantileMLP-RB across seed-conditions. The budgets 30, 40 and 50 percent
were reached by all six controllers in 6, 4 and fewer than 4 of the 20
seed-conditions; no budget from 10 to 40 percent was reached in more than
15 of 20 (75 percent). Under section 5 (72 of 80 seeds required), every
cell would be excluded and every hypothesis would be "not supported" by
construction, independently of the controllers' performance.

**Change.**

1. *Frontier lever for RB* (replaces the c lever of section 5). The RB
   reservation at every ladder level is multiplied by a scalar s on the same
   grid as the Eq. (20) controllers: 21 values in [0.6, 1.8]. The requested
   target is s * r(eps_s,0) * (1 + kappa * Q) and the budget is measured on
   it (section 5, "Budget measurement", otherwise unchanged). Relaxation
   (section 2.5) moves along the scaled ladder s * r(eps_s,k).
2. *Risk level fixed per method by tuning* (extends section 4). Base levels
   eps_s = clip(c * b_s) with (c, b) chosen jointly on the validation seeds
   from c in {0.02, 0.05, 0.1, 0.2, 0.4} and the six shapes of section 4
   (30 candidates per method). Recalibration (section 2.4) is applied at the
   selected (c, b) exactly as before. The objective, tie-break and
   eligibility rule of section 4 are unchanged, evaluated on the s frontier.
3. *Unchanged:* budgets 30, 40 and 50 percent; hypotheses H1 to H4; decision
   rule; Holm families; test seeds 1001 to 1080; M = 200; eps range
   [0.01, 0.60]; missing-seed rule; recalibration; relaxation rules.

**Consequence for interpretation.** Both rules are now traced by the
identical scale lever, so a matched-budget comparison contrasts how each
rule and forecaster distributes the same total reservation across slots
and slices. The risk level becomes a tuned shape parameter of RB rather
than its frontier lever.

**Cost.** Tuning: 30 candidates x 21 scale values x 2 capacities x 3
methods per job, about 35 minutes on 2 cores. Test run unchanged at about
2 hours.
