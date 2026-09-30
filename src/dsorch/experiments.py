"""Experiment definitions. Each ``exp_*`` function writes tidy CSV files to ``out``.

All experiments share one unit of work, the *job*: one workload (burst level
or public-trace segment) and one seed. A job builds the trace, fits every
forecaster once on slots 0..fit_slots-1, precomputes demand estimates, and then
simulates every controller configuration it needs. Jobs are independent and
run in parallel processes.
"""
from __future__ import annotations

import copy
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .engine import jcso_estimates, run_episode
from .forecasters import build
from .metrics import crps_quantiles, crps_samples, energy_score, orchestration_metrics, pinball
from .policies import JCSO, Independent, JointHeuristic, StaticReserve
from .repro import set_global_determinism
from .system import SystemModel
from .workload import b5g_trace, controlled_trace

FORECASTERS = ["Persistence", "MovingAverage", "QuantileMLP", "GAN", "DDPM"]
METHOD_NAME = {
    "Persistence": "Persistence-JCSO",
    "MovingAverage": "MovingAverage-JCSO",
    "QuantileMLP": "QuantileMLP-JCSO",
    "GAN": "GAN-JCSO",
    "DDPM": "Diffusion-JCSO",
}
REFERENCE = ["Independent", "StaticReserve", "JointHeuristic"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def run_parallel(fn: Callable, jobs: List[tuple], n_proc: int, checkpoint: Optional[Path] = None,
                 labels: Optional[List[str]] = None) -> List:
    """Run jobs, print one progress line per finished job, return results in job order.

    With ``checkpoint``, each finished job's result is pickled to
    ``checkpoint/<label>.pkl`` and reused on a rerun, so an interrupted run
    resumes without recomputing finished jobs. Jobs are deterministic, so a
    resumed run gives the same results as an uninterrupted one.
    """
    t0 = time.perf_counter()
    n = len(jobs)
    labels = labels or [str(i) for i in range(n)]
    results: List = [None] * n
    todo = list(range(n))
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
        todo = []
        for i, lab in enumerate(labels):
            f = checkpoint / f"{lab}.pkl"
            if f.exists():
                with f.open("rb") as fh:
                    results[i] = pickle.load(fh)
            else:
                todo.append(i)
        if len(todo) < n:
            print(f"  {fn.__name__.strip('_')}: resumed {n - len(todo)}/{n} jobs from {checkpoint}", flush=True)

    def finish(i: int, res, done: int) -> None:
        results[i] = res
        if checkpoint is not None:
            tmp = checkpoint / f"{labels[i]}.pkl.tmp"
            with tmp.open("wb") as fh:
                pickle.dump(res, fh)
            tmp.replace(checkpoint / f"{labels[i]}.pkl")
        el = time.perf_counter() - t0
        print(f"  {fn.__name__.strip('_')}: {done}/{n} jobs done, {el:.0f} s elapsed", flush=True)

    done = n - len(todo)
    if n_proc <= 1:
        _init_worker()
        for i in todo:
            done += 1
            finish(i, fn(*jobs[i]), done)
        return results
    with ProcessPoolExecutor(max_workers=n_proc, initializer=_init_worker) as ex:
        futures = {ex.submit(fn, *jobs[i]): i for i in todo}
        for fut in as_completed(futures):
            done += 1
            finish(futures[fut], fut.result(), done)
    return results


def apply_selected(cfg: Dict, root: Path) -> Dict:
    """Overlay validation-selected training lengths if the config asks for them."""
    sel_path = cfg.get("use_selected")
    if not sel_path:
        return cfg
    path = root / sel_path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the validation experiment first")
    with path.open() as fh:
        sel = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    for model, epochs in sel["epochs"].items():
        cfg["forecasters"][model]["epochs"] = int(epochs)
    return cfg


def with_override(cfg: Dict, model_key: str, override: Dict) -> Dict:
    c = copy.deepcopy(cfg)
    c["forecasters"][model_key].update(override)
    return c


def make_trace(cfg: Dict, workload: str, seed: int) -> np.ndarray:
    sm = SystemModel.from_config(cfg)
    if workload.startswith("b5g"):
        return b5g_trace(sm.slices, seed, cfg["protocol"]["slots"])
    bf = cfg["workload"]["burst_factors"][workload]
    return controlled_trace(sm.slices, cfg["protocol"]["slots"], bf, seed)


def fit_all(cfg: Dict, traces: np.ndarray, seed: int, names: Iterable[str] = FORECASTERS) -> Dict:
    fit_slots = cfg["protocol"]["fit_slots"]
    out = {}
    for name in names:
        set_global_determinism(seed)
        t0 = time.perf_counter()
        f = build(name, cfg).fit(traces[:fit_slots], seed)
        f.fit_seconds = time.perf_counter() - t0
        out[name] = f
    return out


def reference_policies(cfg: Dict, traces: np.ndarray) -> list:
    b = cfg["baselines"]
    fit_mean = traces[: cfg["protocol"]["fit_slots"]].mean(axis=0)
    return [Independent(b["independent_margin"]), StaticReserve(fit_mean, b["static_safety"]), JointHeuristic()]


def simulate(cfg, traces, capacity, policy, estimate, info) -> Dict:
    c = copy.deepcopy(cfg)
    c["protocol"]["information"] = info
    sm = SystemModel.from_config(c, capacity)
    df = run_episode(sm, traces, policy, c, estimate)
    return orchestration_metrics(df, c["protocol"]["test_start"])


# ---------------------------------------------------------------------------
# 1. Validation-based selection of training length
# ---------------------------------------------------------------------------

def _validation_job(cfg: Dict, seed: int) -> List[Dict]:
    vcfg = copy.deepcopy(cfg)
    val = cfg["validation"]
    traces = make_trace(vcfg, val["workload"], seed)
    fit_end, val_end = val["fit_slots"], val["val_end"]
    t_idx = np.arange(fit_end, val_end)
    y = traces[fit_end:val_end]
    rows = []
    for model_key, name in [("gan", "GAN"), ("ddpm", "DDPM"), ("quantile_mlp", "QuantileMLP")]:
        for epochs in val["epoch_grid"]:
            c = with_override(vcfg, model_key, {"epochs": epochs})
            set_global_determinism(seed)
            f = build(name, c).fit(traces[:fit_end], seed)
            if name == "QuantileMLP":
                lv = np.array(c["forecasters"]["quantile_mlp"]["levels"])
                crps = crps_quantiles(f.quantiles(traces, t_idx), lv, y)
                q86 = f.quantile_at(traces, t_idx, cfg["control"]["quantile"])
            else:
                s = f.sample(traces, t_idx, cfg["evaluation"]["forecast_eval_samples"], seed)
                crps = crps_samples(s, y)
                q86 = np.quantile(s, cfg["control"]["quantile"], axis=2)
            rows.append({
                "model": model_key, "epochs": epochs, "seed": seed,
                "val_crps": float(crps.mean()),
                "val_pinball": float(pinball(y, q86, cfg["control"]["quantile"]).mean()),
                "val_coverage": float((y <= q86).mean()),
            })
    return rows


def exp_validation(cfg: Dict, out: Path, n_proc: int) -> None:
    seeds = cfg["validation"]["seeds"]
    rows = sum(run_parallel(_validation_job, [(cfg, s) for s in seeds], n_proc), [])
    df = pd.DataFrame(rows)
    df.to_csv(out / "validation_grid.csv", index=False)
    mean = df.groupby(["model", "epochs"])["val_crps"].mean().reset_index()
    selected = {m: int(g.loc[g["val_crps"].idxmin(), "epochs"]) for m, g in mean.groupby("model")}
    with (out / "selected.yaml").open("w") as fh:
        yaml.safe_dump({"criterion": "mean validation CRPS", "epochs": selected}, fh)


# ---------------------------------------------------------------------------
# 2. Main orchestration grid
# ---------------------------------------------------------------------------

def _orchestration_job(cfg: Dict, workload: str, seed: int) -> Tuple[List[Dict], List[Dict]]:
    traces = make_trace(cfg, workload, seed)
    fcs = fit_all(cfg, traces, seed)
    ests = {n: jcso_estimates(f, traces, cfg, seed) for n, f in fcs.items()}
    rows = []
    for info in cfg["grid"]["information"]:
        for cap in cfg["grid"]["capacity"]:
            for pol in reference_policies(cfg, traces):
                m = simulate(cfg, traces, cap, pol, None, info)
                rows.append({"workload": workload, "seed": seed, "information": info, "capacity": cap,
                             "method": pol.name, **m})
            for n in FORECASTERS:
                m = simulate(cfg, traces, cap, JCSO(METHOD_NAME[n]), ests[n]["estimate"], info)
                rows.append({"workload": workload, "seed": seed, "information": info, "capacity": cap,
                             "method": METHOD_NAME[n], **m})
    losses = []
    for n in ("GAN", "DDPM", "QuantileMLP"):
        for rec in fcs[n].loss_history:
            losses.append({"workload": workload, "seed": seed, "model": n, **rec})
    return rows, losses


def exp_orchestration(cfg: Dict, out: Path, n_proc: int) -> None:
    jobs = [(cfg, w, s) for w in cfg["grid"]["workloads"] for s in cfg["protocol"]["seeds"]]
    res = run_parallel(_orchestration_job, jobs, n_proc)
    pd.DataFrame(sum((r[0] for r in res), [])).to_csv(out / "orchestration.csv", index=False)
    pd.DataFrame(sum((r[1] for r in res), [])).to_csv(out / "training_loss.csv", index=False)


# ---------------------------------------------------------------------------
# 3. Forecast quality
# ---------------------------------------------------------------------------

def _forecast_job(cfg: Dict, workload: str, seed: int) -> List[Dict]:
    traces = make_trace(cfg, workload, seed)
    fcs = fit_all(cfg, traces, seed)
    start = cfg["protocol"]["test_start"]
    t_idx = np.arange(start, traces.shape[0])
    y = traces[start:]
    tau = cfg["control"]["quantile"]
    n_eval = cfg["evaluation"]["forecast_eval_samples"]
    rows = []
    for n, f in fcs.items():
        est = jcso_estimates(f, traces, cfg, seed)
        u_bar = est["estimate"][start:]
        row = {"workload": workload, "seed": seed, "forecaster": n,
               "params_online": f.num_parameters().get("online", 0),
               "params_training_only": f.num_parameters().get("training_only", 0),
               "fit_seconds": getattr(f, "fit_seconds", 0.0),
               "ms_per_prediction": est["ms_per_prediction"],
               "estimate_pinball": float(pinball(y, u_bar, tau).mean()),
               "estimate_coverage": float((y <= u_bar).mean()),
               "estimate_joint_coverage": float((y <= u_bar).all(axis=-1).mean())}
        if n == "QuantileMLP":
            lv = np.array(cfg["forecasters"]["quantile_mlp"]["levels"])
            q = f.quantile_at(traces, t_idx, tau)
            med = f.quantile_at(traces, t_idx, 0.5)
            row.update(crps=float(crps_quantiles(f.quantiles(traces, t_idx), lv, y).mean()),
                       energy_score=np.nan, sample_spread=np.nan)
        elif f.distributional:
            s = f.sample(traces, t_idx, n_eval, seed)
            q = np.quantile(s, tau, axis=2)
            med = np.median(s, axis=2)
            row.update(crps=float(crps_samples(s, y).mean()),
                       energy_score=float(energy_score(s.reshape(-1, n_eval, 2), y.reshape(-1, 2)).mean()),
                       sample_spread=float(s.std(axis=2).mean()))
        else:
            q = med = f.point(traces, t_idx)
            row.update(crps=float(np.abs(med - y).mean()), energy_score=float(np.linalg.norm(med - y, axis=-1).mean()),
                       sample_spread=0.0)
        # identical quantile-grid CRPS for every forecaster, so the column is comparable
        lv = np.array(cfg["forecasters"]["quantile_mlp"]["levels"])
        if n == "QuantileMLP":
            grid = f.quantiles(traces, t_idx)
        elif f.distributional:
            grid = np.moveaxis(np.quantile(s, lv, axis=2), 0, 2)
        else:
            grid = np.repeat(med[:, :, None, :], len(lv), axis=2)
        row["crps_grid"] = float(crps_quantiles(grid, lv, y).mean())
        row.update(nmae=float(np.abs(med - y).sum() / y.sum()),
                   rmse=float(np.sqrt(((med - y) ** 2).mean())),
                   quantile_pinball=float(pinball(y, q, tau).mean()),
                   quantile_coverage=float((y <= q).mean()),
                   quantile_joint_coverage=float((y <= q).all(axis=-1).mean()))
        rows.append(row)
    return rows


def exp_forecast(cfg: Dict, out: Path, n_proc: int) -> None:
    jobs = [(cfg, w, s) for w in cfg["grid"]["workloads"] for s in cfg["protocol"]["seeds"]]
    pd.DataFrame(sum(run_parallel(_forecast_job, jobs, n_proc), [])).to_csv(out / "forecast.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Matched-budget frontiers
# ---------------------------------------------------------------------------

def _frontier_job(cfg: Dict, workload: str, seed: int) -> List[Dict]:
    traces = make_trace(cfg, workload, seed)
    fr = cfg["frontier"]
    fcs = fit_all(cfg, traces, seed, fr["forecasters"])
    rows = []
    # (a) scale frontier: multiply the whole Eq. (20) target by c
    ests = {n: jcso_estimates(f, traces, cfg, seed)["estimate"] for n, f in fcs.items()}
    for info in cfg["grid"]["information"]:
        for cap in cfg["grid"]["capacity"]:
            for n in fr["forecasters"]:
                for c in fr["scales"]:
                    m = simulate(cfg, traces, cap, JCSO(METHOD_NAME[n], c), ests[n], info)
                    rows.append({"workload": workload, "seed": seed, "information": info, "capacity": cap,
                                 "method": METHOD_NAME[n], "lever": "scale", "level": c, **m})
                if fcs[n].distributional:
                    for q in fr["quantiles"]:
                        e = jcso_estimates(fcs[n], traces, cfg, seed, quantile=q)["estimate"]
                        m = simulate(cfg, traces, cap, JCSO(METHOD_NAME[n]), e, info)
                        rows.append({"workload": workload, "seed": seed, "information": info, "capacity": cap,
                                     "method": METHOD_NAME[n], "lever": "quantile", "level": q, **m})
    return rows


def exp_frontier(cfg: Dict, out: Path, n_proc: int) -> None:
    jobs = [(cfg, w, s) for w in cfg["grid"]["workloads"] for s in cfg["protocol"]["seeds"]]
    df = pd.DataFrame(sum(run_parallel(_frontier_job, jobs, n_proc), []))
    df.to_csv(out / "frontier.csv", index=False)
    matched = matched_budget(df, cfg["frontier"]["budgets"])
    matched.to_csv(out / "matched_budget.csv", index=False)


def matched_budget(df: pd.DataFrame, budgets: List[float]) -> pd.DataFrame:
    """Interpolate every metric at fixed over-reservation budgets along each frontier.

    Budgets outside a method's observed over-reservation range are reported as NaN
    rather than extrapolated.
    """
    keys = ["workload", "seed", "information", "capacity", "method", "lever"]
    metrics = ["miss_rate", "acceptance", "p95_latency_ms", "p99_latency_ms", "utilization"]
    rows = []
    for k, g in df.groupby(keys):
        g = g.sort_values("over_reservation")
        x = g["over_reservation"].to_numpy()
        for b in budgets:
            row = dict(zip(keys, k))
            row["budget"] = b
            inside = x.min() <= b <= x.max()
            for m in metrics:
                row[m] = float(np.interp(b, x, g[m].to_numpy())) if inside else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Diffusion ablations
# ---------------------------------------------------------------------------

def _ablation_job(cfg: Dict, workload: str, seed: int) -> List[Dict]:
    traces = make_trace(cfg, workload, seed)
    start = cfg["protocol"]["test_start"]
    t_idx = np.arange(start, traces.shape[0])
    y = traces[start:]
    tau = cfg["control"]["quantile"]
    rows = []
    for var in cfg["ablation"]["variants"]:
        c = with_override(cfg, "ddpm", var.get("ddpm", {}))
        set_global_determinism(seed)
        t0 = time.perf_counter()
        f = build("DDPM", c).fit(traces[: cfg["protocol"]["fit_slots"]], seed)
        fit_s = time.perf_counter() - t0
        M = var.get("samples", cfg["control"]["samples"])
        est = jcso_estimates(f, traces, c, seed, samples=M)
        s = f.sample(traces, t_idx, cfg["evaluation"]["forecast_eval_samples"], seed)
        q = np.quantile(s, tau, axis=2)
        base = {"workload": workload, "seed": seed, "variant": var["name"], "group": var["group"],
                "terminal_alpha_bar": f.terminal_alpha_bar, "fit_seconds": fit_s,
                "ms_per_prediction": est["ms_per_prediction"], "crps": float(crps_samples(s, y).mean()),
                "quantile_coverage": float((y <= q).mean()), "quantile_pinball": float(pinball(y, q, tau).mean())}
        for info in cfg["grid"]["information"]:
            for cap in cfg["grid"]["capacity"]:
                m = simulate(c, traces, cap, JCSO("Diffusion-JCSO"), est["estimate"], info)
                rows.append({**base, "information": info, "capacity": cap, **m})
    return rows


def exp_ablation(cfg: Dict, out: Path, n_proc: int) -> None:
    jobs = [(cfg, w, s) for w in cfg["grid"]["workloads"] for s in cfg["protocol"]["seeds"]]
    pd.DataFrame(sum(run_parallel(_ablation_job, jobs, n_proc), [])).to_csv(out / "ablation.csv", index=False)


# ---------------------------------------------------------------------------
# 6. Public B5G trace replay
# ---------------------------------------------------------------------------

def _b5g_job(cfg: Dict, segment: int, model_seed: int) -> List[Dict]:
    traces = make_trace(cfg, "b5g", segment)
    fcs = fit_all(cfg, traces, model_seed)
    ests = {n: jcso_estimates(f, traces, cfg, model_seed)["estimate"] for n, f in fcs.items()}
    rows = []
    for info in cfg["grid"]["information"]:
        for cap in cfg["grid"]["capacity"]:
            for pol in reference_policies(cfg, traces):
                m = simulate(cfg, traces, cap, pol, None, info)
                rows.append({"segment": segment, "model_seed": model_seed, "information": info,
                             "capacity": cap, "method": pol.name, **m})
            for n in FORECASTERS:
                m = simulate(cfg, traces, cap, JCSO(METHOD_NAME[n]), ests[n], info)
                rows.append({"segment": segment, "model_seed": model_seed, "information": info,
                             "capacity": cap, "method": METHOD_NAME[n], **m})
    return rows


def exp_b5g(cfg: Dict, out: Path, n_proc: int) -> None:
    from .workload import b5g_available, b5g_segments

    if not b5g_available():
        raise FileNotFoundError("data/external/public_b5g_trace.csv is missing; see data/README.md")
    jobs = [(cfg, seg, ms) for seg in b5g_segments() for ms in cfg["b5g"]["model_seeds"]]
    pd.DataFrame(sum(run_parallel(_b5g_job, jobs, n_proc), [])).to_csv(out / "b5g.csv", index=False)


EXPERIMENTS = {
    "validation": exp_validation,
    "orchestration": exp_orchestration,
    "forecast": exp_forecast,
    "frontier": exp_frontier,
    "ablation": exp_ablation,
    "b5g": exp_b5g,
}
