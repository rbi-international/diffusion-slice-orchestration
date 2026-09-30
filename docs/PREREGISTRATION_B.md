# Pre-registration: uncertainty-driven reservation (study B)

Status: **draft for approval** (version 2). No study-B code has been written
or run. After Rohit approves, the status line changes to "approved" in a
commit of its own, and every later change goes in the "Amendments" section
with its reason and date, committed before the run it affects.

Version 2 incorporates the independent review by Claude Code on Rohit's
machine (2026-09-30): explicit handling of budgets a seed does not reach,
the frontier lever, a difference-in-differences test for H4, a per-hypothesis
Holm family, a sign rule robust to noise, per-method tuning with a stated
objective, fresh test seeds, and a power analysis.

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
bandwidth-CPU demand vector (M = 200 for RB), predicted from history
t-5..t-1 (proactive information: the current request is not used).

1. **No fixed margins, no LWMA anchor** (alpha_s = 1, anchor weight 0).
2. **Reservation for a risk level eps.**
   * Generators (DDPM, GAN). Per resource d, let med_d be the sample median
     and h_d the sample 0.95-quantile minus med_d. For lambda >= 0 let
     r(lambda) = med + lambda * h (componentwise). The reservation is
     r(lambda*) with lambda* the smallest value on a grid of step 0.01 in
     [0, 6] whose empirical joint coverage
     (1/M) sum_m 1[x_m,bw <= r_bw and x_m,cpu <= r_cpu] is at least 1 - eps.
   * QuantileMLP (marginal quantiles only). r_d = Q_d(1 - eps / 2), the
     Bonferroni bound for joint coverage, by linear interpolation on its
     quantile grid, extended to the levels {0.975, 0.99, 0.995} by adding
     them to the pinball-loss grid (retrained with the study-A epoch count).
3. **Per-slice risk budgets.** eps_s = c * b_s, where b = (b_eMBB, b_URLLC,
   b_mMTC) is the method's base shape (section 4) and c > 0 is the frontier
   lever. eps_s is clipped to [0.002, 0.60].
4. **Split-conformal calibration** (identical procedure for every method):
   fit the forecaster on slots 0-79, compute the RB reservations on slots
   80-99 with risk levels g * eps_s for a pooled multiplier g on a grid of
   41 log-spaced values in [0.1, 10], and choose the g whose empirical joint
   coverage, pooled over the 60 slice-slots, is closest to the target
   mean_s(1 - eps_s). Then refit on slots 0-99 and use eps_s' = g * eps_s on
   the test slots. g is recomputed for every seed and every c.
5. **Capacity-aware relaxation.** Reservations are precomputed on a ladder
   of 12 risk levels per slice. Placement, admission and every simulator
   equation are exactly as in study A. If after placement a slice is
   unassigned or its allocation is below its target, the slice whose next
   ladder step reduces the weighted expected shortfall penalty least,
   w_s * E[(x - r)+] summed over resources (from samples, or from the
   quantile grid for QuantileMLP), is relaxed by one step and placement is
   repeated, at most 12 times per slot.
6. **Queue correction.** The reservation is multiplied by (1 + kappa * Q)
   as in Eq. (20).

## 3. Controllers

| Name | Demand model | Rule |
|---|---|---|
| Diffusion-RB | conditional DDPM | RB |
| GAN-RB | conditional GAN | RB |
| QuantileMLP-RB | quantile regression | RB, Bonferroni joint |
| Diffusion-JCSO, GAN-JCSO, QuantileMLP-JCSO | as study A | Eq. (20) |

All forecasters keep the architectures and validation-selected training
lengths of study A (DDPM 1600, GAN 400, QuantileMLP 160 epochs).

## 4. Tuning (validation only)

* Validation seeds 1, 2, 5, 7, 9 (heavy and extreme bursts, capacity 1.0
  and 0.82, proactive). No test seed is used.
* Each RB method is tuned **separately and identically**. The base shape b
  is chosen from the grid {(1,1,1), (1,0.5,1), (1,0.25,1), (1,0.5,0.75),
  (1,0.25,0.5), (0.75,0.25,0.5)} (entries for eMBB, URLLC, mMTC).
* **Objective:** the mean miss rate over the three budgets 30, 40 and 50
  percent, read off each validation seed's frontier and averaged over seeds
  and conditions. Ties are broken by the smaller mean P95 latency.
* The selected shapes are written to `results/validation_b/selected.yaml`
  and committed before any test run.
* The cloud run's selection is the official frozen value. A rerun on
  another machine is a check only and does not replace it; disagreements are
  reported.

## 5. Test protocol

* **Test seeds: 1001 to 1080 (80 seeds), never used in study A or in any
  tuning.** The study-A seeds are not reused.
* Conditions (4): heavy and extreme bursts, capacity multipliers 1.0 and
  0.82, proactive information.
* **Frontier lever.** For RB methods, the lever c on a grid of 25 log-spaced
  values in [0.05, 20]. For JCSO methods, the study-A scale lever (target
  multiplied by a scalar on a grid of 21 values in [0.6, 1.8]).
* **Budgets:** over-reservation 30, 40 and 50 percent of demand; miss rate
  and the other metrics are linearly interpolated along each seed's
  frontier. 4 conditions x 3 budgets = **12 cells per hypothesis**.
* **Seeds that do not reach a budget.** No extrapolation. A cell is analysed
  with the seeds that reach it if they are at least 72 of the 80 (90
  percent); otherwise the cell is counted as **not supporting** the
  hypothesis, and the number of missing seeds is reported for every cell.

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

**Multiplicity.** The Holm family is the 12 cells of one hypothesis; each
hypothesis is corrected within its own family.

**Decision rule.** A hypothesis is **supported** if (i) the Holm-adjusted
p < 0.05 with the predicted sign in at least 6 of its 12 cells, and (ii) no
cell is Holm-significant in the opposite direction. It is **contradicted**
if at least 6 cells are Holm-significant in the opposite direction, and
**not supported** otherwise. For every hypothesis the mean difference
across cells with a 95 percent confidence interval (seed-level bootstrap,
10,000 resamples) is also reported.

**Secondary outcomes** (reported, not used for decisions): acceptance, P95
and P99 latency, utilization, per-slice miss rate, realized joint coverage,
and inference time.

## 7. Power

The standard deviation of paired per-seed miss-rate differences
(Diffusion-JCSO minus GAN-JCSO) in the study-A proactive matched-budget
cells had a median of 8.8 percentage points (range 1.9 to 17.9). For a
two-sided paired t-test at the most stringent Holm level (alpha / 12) and
80 percent power, 10 seeds would detect only differences of about 13 pp;
80 seeds detect about 3.7 pp at the median spread and about 2.4 pp at the
lower-quartile spread (5.5 pp). This is the reason for 80 test seeds.

## 8. Reporting commitments

All four hypotheses and all 12 cells of each are reported whatever the
outcome. Study A stays in the manuscript unchanged. The cross-platform
check (Windows rerun of the test runs, `scripts/compare_runs.py`) is
reported for every hypothesis that is supported.

## Amendments

(none)
