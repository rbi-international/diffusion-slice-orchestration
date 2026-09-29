# Diffusion-JCSO

Conditional denoising diffusion for joint computation-communication slice
orchestration in 5G/6G cloud-edge networks.

This repository accompanies the manuscript by **Gurpreet Singh** and
**Rohit Bharti** (School of Computer Science and Engineering, Lovely
Professional University). The study asks two questions about generative demand
forecasting for joint bandwidth-CPU slice orchestration in 5G/6G cloud-edge
networks:

1. Does a conditional denoising diffusion probabilistic model (DDPM) represent
   the uncertainty of bursty slice demand more faithfully than a conditional
   GAN?
2. Does better-calibrated demand uncertainty translate into better slice
   reservation once resource consumption is held equal?

### Contributions of this work

* **Diffusion-JCSO**, a conditional DDPM sampler for two-dimensional
  bandwidth-CPU demand, designed for short sampling chains (K = 20 with a
  cosine noise schedule whose forward process reaches noise), with training
  length selected on a validation window.
* **A distributional evaluation of demand generators** (CRPS, energy score,
  pinball loss, quantile coverage), which exposes the under-dispersion of the
  adversarially trained generator.
* **Matched-resource evaluation**: miss-rate versus over-reservation frontiers,
  traced by two independent levers, so that forecast quality is separated from
  the amount of capacity a controller consumes.
* **Additional controls and conditions**: a direct quantile-regression
  estimator, a proactive reservation setting, four burst regimes, three
  capacity levels, public B5G trace replay, and ten seeds with paired,
  Holm-corrected tests.
* **An audit of the published orchestration model**, including an
  inconsistency between its isolation screen and its admission threshold
  (see `docs/CHANGES_FROM_PROTOTYPE.md`).
* **An open, tested implementation**, in which every reported number is
  regenerated from saved CSV files with recorded provenance.

### Relation to prior work

The orchestration environment (slice classes, edge nodes, latency and queue
model, reservation rule and isolation-aware placement) reimplements, from its
published equations, the GAN-JCSO framework of Qiu and Zhang
([J. Cloud Comput. 2026, doi:10.1186/s13677-026-00985-4](https://doi.org/10.1186/s13677-026-00985-4)).
It is deliberately held fixed and used as a published, citable testbed: when
every controller runs in the same environment, differences in outcome can be
attributed to the demand model rather than to changes in the simulator. All
equations and parameter values taken from that work are cited in
`configs/base.yaml` and in the source code.

## Repository layout

```
configs/            YAML configs; base.yaml holds every model parameter
src/dsorch/         the package
  system.py         system model, Eqs. (4)-(12) of GAN-JCSO
  policies.py       reservation rules and placement, Eqs. (20)-(22)
  engine.py         slot-by-slot simulation and demand-estimate precomputation
  forecasters/      GAN (Table 5), DDPM, QuantileMLP, Persistence, MovingAverage
  experiments.py    the six experiments
  metrics.py        orchestration and forecast metrics (CRPS, pinball, coverage)
  stats.py          paired t-test, Wilcoxon, Holm correction
  repro.py          seeding and run manifests
scripts/            run_experiment.py, run_all.py, analyze.py, make_figures.py
tests/              pytest suite (equations, determinism, leakage, end-to-end)
results/            saved CSVs, tables, figures and SUMMARY.md
docs/               CHANGES_FROM_PROTOTYPE.md
data/               README for the optional public B5G trace
```

## Setup (Windows, macOS or Linux)

```bash
git clone https://github.com/rbi-international/diffusion-slice-orchestration.git
cd diffusion-slice-orchestration
conda env create -f environment.yml
conda activate dsorch
pip install -e . --no-deps
python -m pytest            # 30 tests, about 10 seconds
```

Without conda: `python -m venv .venv`, activate it, then
`pip install -r requirements.txt` and `pip install -e . --no-deps`.
On Linux, the default PyPI PyTorch wheel includes CUDA libraries; for a smaller
CPU-only install use
`pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`.

In VS Code, open the folder, choose the `dsorch` interpreter
(Python: Select Interpreter), and the Testing panel will discover the tests.

## Reproducing the results

```bash
python scripts/run_all.py --jobs 4        # all experiments, then tables and figures
```

or one step at a time:

```bash
python scripts/run_experiment.py configs/validation.yaml --jobs 4   # selects training lengths
python scripts/run_experiment.py configs/orchestration.yaml --jobs 4
python scripts/run_experiment.py configs/forecast.yaml --jobs 4
python scripts/run_experiment.py configs/frontier.yaml --jobs 4
python scripts/run_experiment.py configs/ablation.yaml --jobs 4
python scripts/run_experiment.py configs/b5g.yaml --jobs 4           # needs data/external, see data/README.md
python scripts/analyze.py
python scripts/make_figures.py
```

Each experiment folder in `results/` contains a `manifest.json` with the git
commit, library versions, platform and the full resolved configuration.
`results/SUMMARY.md` collects the headline tables.

### Protocol

* Traces of 260 slots. Forecasters are fitted on slots 0-99; every reported
  metric uses the held-out slots 100-259 only.
* Ten seeds: 3, 11, 27, 41, 59, 73, 89, 97, 113, 131.
* Training length of each learned forecaster is selected once, before testing,
  on the validation window (fit on slots 0-79, score CRPS on 80-99) with five
  separate validation seeds (1, 2, 5, 7, 9).
* All other parameters are the validation-selected values published for
  GAN-JCSO and are not re-tuned.
* Comparisons are two-sided paired tests over seeds with Holm correction
  within each metric family.

### Verifying the results on another machine

Write the rerun to a separate folder so the saved results are not overwritten,
then compare:

```bash
python scripts/run_experiment.py configs/orchestration.yaml --jobs 8 --out results/verify/orchestration
python scripts/compare_runs.py results/orchestration results/verify/orchestration
```

The whole suite needs no GPU. On two CPU cores it takes about 25 minutes; the
runner fixes every numerical library to one thread per worker process.

### Numerical reproducibility

Every random draw is seeded by its logical position (seed, model, slot, slice),
PyTorch runs single-threaded with deterministic algorithms, and all versions
are pinned. A rerun on the same platform reproduces the saved CSV files
exactly. PyTorch does not guarantee bit-identical floating point across
operating systems or CPU instruction sets, and neural-network training
amplifies last-digit differences over many epochs, so a run on a different
platform yields different individual trained models and therefore different
per-seed rows. The claims in the manuscript are stated over ten seeds and are
checked for agreement across platforms with `scripts/compare_runs.py`.

## Citation

See `CITATION.cff`. Please also cite the GAN-JCSO paper on which the system
model is based.

## Licence

Code: MIT (see `LICENSE`). The optional public B5G trace is third-party data
and is not redistributed here; see `data/README.md`.
