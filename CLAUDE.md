# CLAUDE.md

Guidance for Claude Code working in this repository. Read this file fully,
then `docs/WORKLOG.md` (current state and open tasks), before doing anything.

## 1. What this project is

Research code and results for a manuscript by Gurpreet Singh and Rohit Bharti
(Lovely Professional University), target journal **Scientific Reports**.
Corresponding author: Rohit Bharti (rohit.33572@lpu.co.in).

The study compares generative demand models for joint bandwidth-CPU slice
orchestration in 5G/6G cloud-edge networks:

* **Study A (done):** conditional DDPM (Diffusion-JCSO) vs conditional GAN
  (GAN-JCSO) vs direct quantile regression (QuantileMLP-JCSO) and reactive
  baselines, all inside the published GAN-JCSO orchestration model.
* **Study B (in progress):** a new uncertainty-driven reservation rule (RB)
  that uses the predictive distribution directly. Pre-registered in
  `docs/PREREGISTRATION_B.md`. Read it before touching study-B code.

The orchestration model reimplements, from its published equations, Qiu and
Zhang, "GAN-Assisted joint computation-communication slice orchestration for
5G/6G cloud-edge networks", J. Cloud Comput. 2026,
doi:10.1186/s13677-026-00985-4 ("the base paper"). Equation numbers in code
comments refer to that paper. The PDF is not in the repo (copyright); ask
Rohit for it if you need to check an equation.

## 2. Non-negotiable rules

1. **Scientific integrity.** Never tune anything on test seeds
   (3, 11, 27, 41, 59, 73, 89, 97, 113, 131). Selection uses validation seeds
   (1, 2, 5, 7, 9) and the validation window (fit slots 0-79, score 80-99).
   Never edit result CSVs by hand. Never report a number that
   `scripts/analyze.py` did not produce. Report negative and null results.
2. **Pre-registration.** Study-B hypotheses, metrics and pass criteria are
   fixed in `docs/PREREGISTRATION_B.md`. Any deviation goes in its
   "Amendments" section with the reason and date, committed before the run
   it affects.
3. **Held-out protocol.** Forecasters see only slots `< fit_slots` when
   fitting; predictions for slot t use history t-k..t-1 only. The tests in
   `tests/test_forecasters.py::test_no_test_leakage` must keep passing.
4. **Do not change the base-paper model** (`src/dsorch/system.py`, the
   admission rule, placement, parameter values in `configs/base.yaml`)
   without an explicit instruction from Rohit. Known quirks are documented,
   not fixed (see section 7).
5. **Commits.** Author is Rohit Bharti <rohit.33572@lpu.co.in>. Do **not** add
   `Co-Authored-By`, `Claude-Session`, "Generated with" or any other AI
   attribution to commit messages, PR descriptions, code comments or docs.
6. **Writing style** (docs, comments, and especially manuscript text):
   * Never use the em dash character (U+2014). Use commas, parentheses,
     colons or separate sentences. Check before every commit:
     `grep -rn $'—' --include='*.md' --include='*.py' --include='*.yaml' .`
   * Formal research prose for anything that may enter the paper. No
     marketing tone, no filler, no "delve", "crucial", "seamless",
     "comprehensive", "robust" as praise, "In conclusion".
   * With Rohit: terse. Explain in plain terms first, then the technical
     version. Give facts and evidence, disagree when the data disagree.
7. **Files over snippets.** Deliver working files, run them, and show output.

## 3. Environment and commands

Conda environment `dsorch` (Python 3.11, pinned in `environment.yml`, CPU
only, no GPU needed). Rohit works on Windows in Git Bash (MINGW64).

```bash
conda activate dsorch                     # if it fails in Git Bash:
source /c/Users/rbhar/anaconda3/etc/profile.d/conda.sh && conda activate dsorch
pip install -e . --no-deps                # once, after cloning
python -m pytest                          # 30+ tests, about 15 s on Windows
```

Experiments (always pass `--jobs`; 8 is right for Rohit's 16-thread laptop):

```bash
python scripts/run_experiment.py configs/<name>.yaml --jobs 8
python scripts/run_experiment.py configs/<name>.yaml --jobs 8 --out results/verify/<name>   # verification rerun
python scripts/compare_runs.py results/orchestration results/verify/orchestration
python scripts/analyze.py                 # tables + results/SUMMARY.md
python scripts/make_figures.py            # results/figures/*.pdf, *.png
python scripts/run_all.py --jobs 8        # everything, about 30-40 min on Windows
```

`results/verify/` is git-ignored. Never run a canonical experiment without
`--out` unless the intent is to replace the committed results, and then
commit the new results together with the code that produced them.

## 4. Repository map

```
configs/base.yaml          every model parameter, with its source
configs/<exp>.yaml         experiment grids (extends base.yaml)
src/dsorch/system.py       Eqs. (4)-(12): latency, backlog, isolation, fragmentation
src/dsorch/policies.py     Eq. (20) reservation, Eqs. (21)-(22) placement, baselines
src/dsorch/engine.py       slot loop (run_episode), demand estimates (jcso_estimates)
src/dsorch/forecasters/    gan.py (Table 5), ddpm.py, baselines.py (QuantileMLP etc.)
src/dsorch/experiments.py  validation, orchestration, forecast, frontier, ablation, b5g
src/dsorch/metrics.py      miss rate etc., CRPS, pinball, energy score
src/dsorch/stats.py        paired t-test, Wilcoxon, Holm
src/dsorch/repro.py        seeding (derive_seed), manifests
scripts/                   runners, analysis, figures, run comparison
tests/                     equations vs hand calculations, determinism, leakage, smoke
results/                   canonical CSVs, tables/, figures/, SUMMARY.md, manifests
docs/                      CHANGES_FROM_PROTOTYPE.md, PREREGISTRATION_B.md, WORKLOG.md
data/external/             third-party B5G trace, git-ignored (see data/README.md)
```

Key design points:

* Every random draw is seeded by `derive_seed(seed, model_id, slot, slice)`,
  so results do not depend on batching or execution order.
* `jcso_estimates` precomputes demand estimates for all slots once per
  forecaster; `run_episode` then simulates any policy against them.
* A new controller = a new `Policy` subclass in `policies.py` plus a line in
  the relevant experiment; a new forecaster = a `Forecaster` subclass with a
  unique `model_id` registered in `forecasters/__init__.py::build`.

## 5. Current results (study A, 10 seeds, from results/SUMMARY.md)

* Diffusion is better calibrated than the GAN: lower CRPS at every burst
  level; 0.86-quantile coverage about 0.79 vs 0.61 (heavy). Significant after
  Holm on normal and medium bursts, not on heavy and extreme.
* The GAN discriminator loss stays at 2 ln 2 (chance) throughout training.
* Diffusion-JCSO has a lower mean miss rate than GAN-JCSO, but reserves more;
  no difference is significant at matched over-reservation.
* QuantileMLP-JCSO matches or beats Diffusion-JCSO.
* Cross-platform (Linux vs Windows) rerun: scenario means move at most
  1.9 pp; Diffusion-vs-GAN sign agrees in 19/24 scenarios (GAN rows are the
  unstable ones); all other comparisons agree in 23/24 or 24/24.

Do not describe study A as "Diffusion outperforms GAN" without these
qualifications.

## 6. How to verify work (do this before claiming anything is done)

1. `python -m pytest` passes.
2. New behaviour has a test that would fail without it (equation checks by
   hand calculation, determinism, leakage).
3. Experiments run end to end; results regenerate through `analyze.py`.
4. For any headline claim, a verification rerun with `--out` and
   `compare_runs.py` shows the claim survives.
5. No em dashes; no AI attribution; `git status` clean after commit.

## 7. Known issues (documented, do not silently fix)

* **Isolation admission cliff** (base-paper model): the candidate screen
  allows eMBB with URLLC (rho 0.42 <= theta 0.52) but the admission rule
  needs Gamma_t >= 0.25, and that pair alone gives 1 - 0.42/0.52 = 0.19, so
  the whole slot fails. Explains non-monotone miss rate vs capacity.
* **Cross-platform floats**: PyTorch is not bit-identical across Windows and
  Linux; trained networks differ, so per-seed rows differ. Also 5-8 rows of
  non-learned baselines differed in the Windows rerun (open task: find the
  cause, suspected ties at the 0.85 admission threshold).
* `data/external/public_b5g_trace.csv` is third-party (base-paper
  supplementary) and must never be committed.

## 8. Collaboration protocol (two Claude instances)

Rohit works with two Claude instances: a cloud session (no GitHub push
access) and Claude Code on his laptop (this repository).

* `docs/WORKLOG.md` is the shared hand-off file. Before starting, read it.
  When finishing a piece of work, append a dated entry: what changed, what
  was run, the exact result, what is next. Keep entries short and factual.
* Claim a task in the WORKLOG "Open tasks" list (`[local]` or `[cloud]`)
  before starting it, so both do not work on the same thing.
* Work on a branch per task (`b/<short-name>`), merge to `main` only when
  tests pass. Rohit pushes to GitHub.
* Moving work between instances: the sender runs
  `git bundle create ../handoff.bundle main` and Rohit passes the file on;
  the receiver runs `git fetch <path>/handoff.bundle main` and merges
  (`git merge --ff-only FETCH_HEAD` when possible).
* Independent checking is welcome: if one instance produced a result, the
  other may re-derive it from the code and CSVs and record agreement or
  disagreement in the WORKLOG.
