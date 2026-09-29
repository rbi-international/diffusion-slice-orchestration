# Diffusion-JCSO

Conditional denoising diffusion for joint computation-communication slice
orchestration in 5G/6G cloud-edge networks.

This repository contains the complete, tested code and the saved results for
the manuscript by **Gurpreet Singh** and **Rohit Bharti** (School of Computer
Science and Engineering, Lovely Professional University). It extends the
GAN-JCSO framework of Qiu and Zhang
([J. Cloud Comput. 2026, doi:10.1186/s13677-026-00985-4](https://doi.org/10.1186/s13677-026-00985-4))
by replacing its conditional GAN demand sampler with a conditional DDPM, while
keeping the system model, reservation rule and isolation-aware placement
unchanged. The only difference between GAN-JCSO and Diffusion-JCSO is the
generator.

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

### Numerical reproducibility

Every random draw is seeded by its logical position (seed, model, slot, slice),
PyTorch runs single-threaded with deterministic algorithms, and all versions
are pinned. Re-running on the same platform reproduces the saved CSVs exactly.
PyTorch does not guarantee bit-identical floating point across operating
systems or CPU instruction sets, so a run on a different machine may differ in
the last digits; conclusions should not change.

## Citation

See `CITATION.cff`. Please also cite the GAN-JCSO paper on which the system
model is based.

## Licence

Code: MIT (see `LICENSE`). The optional public B5G trace is third-party data
and is not redistributed here; see `data/README.md`.
