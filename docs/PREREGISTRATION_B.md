# Pre-registration: uncertainty-driven reservation (study B)

Committed before any study-B code is written or run. Changes after this
commit must be recorded in the "Amendments" section with a reason.

## Motivation

In study A, every forecaster feeds the GAN-JCSO reservation rule (Eq. 20):

    a_tar = alpha_s * max(u_bar, u) * (1 + kappa * Q)
    u_bar = 0.62 * Q_0.86(samples) + 0.38 * LWMA

The fixed margins alpha_s (1.35, 1.30, 1.18) and the LWMA anchor were
tuned for a GAN whose samples are too narrow. They inflate every reservation
by the same factor whatever the forecast uncertainty, so a forecaster that
knows *when* demand is uncertain gains nothing from that knowledge. Study A
found that the DDPM is better calibrated than the GAN, yet the rule converts
this into extra reservation rather than fewer misses at equal cost.

Study B replaces the rule with one that uses the predictive distribution
directly, and asks whether calibrated generative uncertainty then yields a
better miss-rate versus capacity trade-off.

## Reservation rule RB (risk-budgeted joint reservation)

For slice s at slot t, with predictive samples {x_m} of the bandwidth-CPU
demand vector:

1. **No fixed margins, no LWMA anchor.** alpha_s = 1 and the anchor weight is 0
   for every distributional forecaster.
2. **Joint risk target.** Choose the smallest reservation vector r on a
   one-parameter family r(lambda) = median + lambda * (upper spread), computed
   per resource from the samples, such that the empirical joint coverage
   P(x_bw <= r_bw and x_cpu <= r_cpu) >= 1 - eps_s. eps_s is a per-slice risk
   budget ordered by deadline strictness (URLLC strictest).
   *Generators* estimate joint coverage from their samples. *QuantileMLP*
   has only marginal quantiles, so it uses the Bonferroni bound (each
   resource at 1 - eps_s / 2). This is the one place where a joint generative
   model can legitimately differ from marginal quantile regression.
3. **Split-conformal calibration** (identical for all forecasters): on the
   validation window (slots 80-99, fitting on 0-79) compute the lambda
   correction that makes empirical joint coverage equal to 1 - eps_s, then
   refit on slots 0-99 and apply that correction on the test slots.
4. **Capacity-aware degradation.** If the placement cannot host all targets,
   relax the slice whose relaxation increases the weighted expected shortfall
   E[(x - r)+] least, as estimated from the samples (quantile grid for
   QuantileMLP), and retry. Placement, admission and all simulator equations
   are unchanged from study A.
5. **Queue correction** (1 + kappa * Q) is kept, as in Eq. (20).

## Controllers compared

| Name | Demand model | Rule |
|---|---|---|
| Diffusion-RB | conditional DDPM | RB |
| GAN-RB | conditional GAN | RB |
| QuantileMLP-RB | quantile regression | RB (Bonferroni joint) |
| Diffusion-JCSO, GAN-JCSO, QuantileMLP-JCSO | as study A | Eq. (20) |
| MovingAverage-JCSO, JointHeuristic | as study A | as study A |

All forecasters keep the architectures and validation-selected training
lengths of study A. The risk budgets eps_s are selected once on the five
validation seeds (1, 2, 5, 7, 9) and then frozen.

## Hypotheses (test seeds 3, 11, 27, 41, 59, 73, 89, 97, 113, 131)

Primary outcome: submitted-request miss rate at matched over-reservation,
interpolated on frontiers traced by scaling eps (budgets 30, 40, 50 percent),
heavy and extreme bursts, capacity 1.0 and 0.82, proactive information.

* **H1.** Diffusion-RB < GAN-RB.
* **H2.** Diffusion-RB < QuantileMLP-RB.
* **H3.** Diffusion-RB < Diffusion-JCSO (the new rule helps the DDPM).
* **H4.** The gain of RB over Eq. (20) is larger for the DDPM than for the GAN
  (calibration is what the rule exploits).

Two-sided paired t-tests over the ten seeds, Holm correction across all
hypothesis x scenario x budget comparisons. A hypothesis is **supported**
if the Holm-adjusted p < 0.05 in at least half of its scenario-budget cells
and the mean difference has the predicted sign in all cells; **not
supported** otherwise. Cross-platform agreement (Linux and Windows reruns)
is reported for every supported hypothesis.

## Reporting commitment

All four hypotheses are reported whatever their outcome. Study A remains in
the manuscript unchanged. No hyperparameter is tuned on test seeds.

## Amendments

(none)
