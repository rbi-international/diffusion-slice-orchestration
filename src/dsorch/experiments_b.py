"""Study B experiments (docs/PREREGISTRATION_B.md v3.1, sections 3 to 5).

Job = one workload (burst level) and one seed. A job fits every forecaster
twice (slots 0-79 for recalibration, slots 0-99 for the test slots), builds
the predictive banks once, and traces RB and Eq. (20) frontiers.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from .engine import jcso_estimates, run_episode
from .experiments import make_trace, run_parallel
from .forecasters import build
from .metrics import orchestration_metrics
from .policies import JCSO
from .repro import set_global_determinism
from .reservation import (
    RBPolicy, build_bank, build_ladders, clip_eps, decision_flags, recalibrate,
)
from .system import SystemModel

MODELS = {"DDPM": "Diffusion", "GAN": "GAN", "QuantileMLP": "QuantileMLP"}


def study_b_config(cfg: Dict) -> Dict:
    """QuantileMLP with the extended level grid (section 2.2); proactive information."""
    c = copy.deepcopy(cfg)
    lv = sorted(set(c["forecasters"]["quantile_mlp"]["levels"]) | set(cfg["study_b"]["extra_levels"]))
    c["forecasters"]["quantile_mlp"]["levels"] = lv
    c["protocol"]["information"] = "proactive"
    return c


def scale_grid(sb: Dict) -> np.ndarray:
    """Frontier lever shared by both rules (Amendment 1): 21 values in [0.6, 1.8]."""
    lo, hi, n = sb["scale_lever"]
    return np.linspace(lo, hi, int(n))


def rb_candidates(sb: Dict):
    """Tuning candidates (c, b) of Amendment 1, item 2."""
    return [(float(c), tuple(float(v) for v in b)) for c in sb["rb_c_grid"] for b in sb["shapes"]]


def candidate_key(c: float, shape) -> str:
    return f"c={c:g};b=" + ",".join(f"{v:g}" for v in shape)


def _fit(cfg, name, traces, n_fit, seed):
    set_global_determinism(seed)
    return build(name, cfg).fit(traces[:n_fit], seed)


def prepare(cfg: Dict, workload: str, seed: int) -> Dict:
    """Traces, fitted models and predictive banks for one job."""
    traces = make_trace(cfg, workload, seed)
    sb = cfg["study_b"]
    M = int(sb["samples"])
    k = int(cfg["control"]["history"])
    val_lo, val_hi = int(sb["recal_fit_slots"]), int(cfg["protocol"]["fit_slots"])
    t_val = np.arange(val_lo, val_hi)
    t_all = np.arange(k, traces.shape[0])
    out = {"traces": traces, "t_all": t_all, "y_val": traces[val_lo:val_hi], "models": {}}
    for name in MODELS:
        t0 = time.perf_counter()
        f_val = _fit(cfg, name, traces, val_lo, seed)
        f_all = _fit(cfg, name, traces, val_hi, seed)
        out["models"][name] = {
            "bank_val": build_bank(f_val, traces, t_val, M, seed),
            "bank_all": build_bank(f_all, traces, t_all, M, seed),
            "forecaster": f_all,
            "fit_seconds": time.perf_counter() - t0,
        }
    return out


def rb_setup(prep: Dict, name: str, shape, c: float):
    """Recalibrated ladders for one (c, b); independent of the scale lever (Amendment 1, A2)."""
    m = prep["models"][name]
    eps_base = clip_eps(c * np.asarray(shape, dtype=float))
    rec = recalibrate(m["bank_val"], prep["y_val"], eps_base)
    eps_raw = rec.g * eps_base
    return rec, eps_raw, build_ladders(m["bank_all"], eps_raw)


def rb_frontier(cfg: Dict, prep: Dict, name: str, shape, c: float, scales, capacities,
                outcome: bool = True) -> List[Dict]:
    """RB frontier for one (c, b): one row per scale and capacity.

    outcome=False drops every outcome column and keeps only the requested
    over-reservation (used for the reachability check of Amendment 1, A1/A3).
    """
    rec, eps_raw, ladders = rb_setup(prep, name, shape, c)
    rows = []
    for scale in scales:
        for r in _rb_run(cfg, prep, name, shape, c, float(scale), capacities, rec, eps_raw, ladders):
            rows.append(r if outcome else {k: r[k] for k in REACH_COLUMNS})
    return rows


REACH_COLUMNS = ["method", "rule", "model", "candidate", "c", "shape", "lever", "capacity", "over_reservation"]


def _rb_run(cfg, prep, name, shape, c, scale, capacities, rec, eps_raw, ladders) -> List[Dict]:
    rows = []
    for cap in capacities:
        sm = SystemModel.from_config(cfg, cap)
        pol = RBPolicy(f"{MODELS[name]}-RB", ladders, int(prep["t_all"][0]), prep["traces"], scale=scale)
        df = run_episode(sm, prep["traces"], pol, cfg)
        met = orchestration_metrics(df, cfg["protocol"]["test_start"])
        log = df.attrs["plan_log"]
        start = cfg["protocol"]["test_start"]
        levels = np.array([e["levels"] for e in log])[start:]
        t_test = np.arange(start, prep["traces"].shape[0])
        flags = decision_flags(ladders, levels, int(prep["t_all"][0]), t_test)
        y = prep["traces"][start:]
        r0 = np.stack([ladders[s].r[0, start - prep["t_all"][0]:] for s in range(y.shape[1])], axis=1)
        rows.append({
            "method": f"{MODELS[name]}-RB", "rule": "RB", "model": name, "candidate": candidate_key(c, shape),
            "c": float(c), "shape": ",".join(f"{v:g}" for v in shape), "lever": float(scale), "capacity": cap,
            "g": rec.g, "recal_coverage": rec.coverage, "recal_target": rec.target,
            **{f"eps_{i}": float(clip_eps(e)) for i, e in enumerate(eps_raw)},
            "realized_joint_coverage": float((y <= r0).all(axis=-1).mean()),
            "mean_relax_steps": float(np.mean([e["relax_steps"] for e in log[start:]])),
            **flags, **met,
        })
    return rows


def jcso_frontier(cfg: Dict, prep: Dict, capacities, outcome: bool = True) -> List[Dict]:
    """Eq. (20) frontiers with M = 200 samples (section 3)."""
    sb = cfg["study_b"]
    rows = []
    for name in MODELS:
        f = prep["models"][name]["forecaster"]
        t0 = time.perf_counter()
        est = jcso_estimates(f, prep["traces"], cfg, int(prep["seed"]), samples=int(sb["samples"]))
        for scale in scale_grid(sb):
            for cap in capacities:
                sm = SystemModel.from_config(cfg, cap)
                df = run_episode(sm, prep["traces"], JCSO(f"{MODELS[name]}-JCSO", float(scale)), cfg,
                                 est["estimate"])
                met = orchestration_metrics(df, cfg["protocol"]["test_start"])
                row = {"method": f"{MODELS[name]}-JCSO", "rule": "JCSO", "model": name, "candidate": "",
                       "c": np.nan, "shape": "", "lever": float(scale), "capacity": cap,
                       "ms_per_prediction": est["ms_per_prediction"], **met}
                rows.append(row if outcome else {k: row[k] for k in REACH_COLUMNS})
    return rows


# ---------------------------------------------------------------------------
# Section 4: tuning on validation seeds
# ---------------------------------------------------------------------------

def _validation_job(cfg: Dict, workload: str, seed: int) -> List[Dict]:
    t0 = time.perf_counter()
    prep = prepare(cfg, workload, seed)
    prep["seed"] = seed
    sb = cfg["study_b"]
    rows = []
    for name in MODELS:
        for c, shape in rb_candidates(sb):
            for r in rb_frontier(cfg, prep, name, shape, c, scale_grid(sb), sb["capacity"]):
                rows.append({"workload": workload, "seed": seed, **r})
    for r in rows:
        r["job_seconds"] = time.perf_counter() - t0
    return rows


def select_shapes(df: pd.DataFrame, budgets: List[float]) -> Dict:
    """Section 4 objective and eligibility rule."""
    pts = frontier_points(df, budgets, ["workload", "seed", "capacity", "method", "candidate"])
    out, table = {}, []
    for method, g in pts.groupby("method"):
        cand = []
        for key, h in g.groupby("candidate"):
            reached = int(h["miss_rate"].notna().sum())
            cand.append({"method": method, "candidate": key, "points_reached": reached,
                         "points_total": len(h), "objective": float(h["miss_rate"].mean()),
                         "p95": float(h["p95_latency_ms"].mean())})
        cand = pd.DataFrame(cand)
        full = cand[cand["points_reached"] == cand["points_total"]]
        if len(full):
            pick = full.sort_values(["objective", "p95"]).iloc[0]
            rule = "eligible (all points reached)"
        else:
            pick = cand.sort_values(["points_reached", "objective"], ascending=[False, True]).iloc[0]
            rule = "fallback (no candidate reached all points)"
        c_txt, b_txt = pick["candidate"].split(";")
        out[method] = {"c": float(c_txt.split("=")[1]),
                       "shape": [float(v) for v in b_txt.split("=")[1].split(",")],
                       "objective": float(pick["objective"]), "selection_rule": rule}
        table.append(cand)
    return {"selected": out, "table": pd.concat(table)}


def checkpoint_dir(out: Path) -> Path:
    """Per-job checkpoints (gitignored) so an interrupted run resumes; see run_parallel."""
    root = Path(__file__).resolve().parents[2]
    return root / "results" / "scratch" / "checkpoints" / Path(out).name


def exp_validation_b(cfg: Dict, out: Path, n_proc: int) -> None:
    cfg = study_b_config(cfg)
    sb = cfg["study_b"]
    jobs = [(cfg, w, s) for w in sb["workloads"] for s in sb["validation_seeds"]]
    t0 = time.perf_counter()
    labels = [f"{w}_seed{s}" for _, w, s in jobs]
    df = pd.DataFrame(sum(run_parallel(_validation_job, jobs, n_proc, checkpoint_dir(out), labels), []))
    wall = time.perf_counter() - t0
    df.to_csv(out / "validation_b.csv", index=False)
    sel = select_shapes(df, sb["budgets"])
    sel["table"].to_csv(out / "shape_selection.csv", index=False)
    per_job = float(df.groupby(["workload", "seed"])["job_seconds"].first().mean())
    n_cand = len(rb_candidates(sb))
    # test job: 1 RB candidate instead of all, plus the Eq. (20) frontier (about one candidate's cost)
    est_job = per_job * 2.0 / n_cand + per_job * 0.05
    n_test_jobs = len(sb["workloads"]) * len(sb["test_seeds"])
    selected = {
        "preregistration": "docs/PREREGISTRATION_B.md v3.1 (93184d3)",
        "criterion": "mean miss rate over budgets 30/40/50 percent, 5 seeds x 4 conditions (section 4, Amendment 1)",
        "shapes": sel["selected"],
        "timing": {"validation_wall_seconds": round(wall, 1), "mean_job_seconds": round(per_job, 1),
                   "workers": n_proc,
                   "note": "wall seconds cover the final invocation only if the run resumed from checkpoints; mean_job_seconds is per job",
                   "test_run_estimate_hours": round(est_job * n_test_jobs / max(n_proc, 1) / 3600, 2)},
    }
    with (out / "selected.yaml").open("w") as fh:
        yaml.safe_dump(selected, fh, sort_keys=False)


# ---------------------------------------------------------------------------
# Section 5: test run
# ---------------------------------------------------------------------------

def _test_job(cfg: Dict, workload: str, seed: int, shapes: Dict) -> List[Dict]:
    t0 = time.perf_counter()
    prep = prepare(cfg, workload, seed)
    prep["seed"] = seed
    sb = cfg["study_b"]
    rows = []
    for name in MODELS:
        sel = shapes[f"{MODELS[name]}-RB"]
        rows += [{"workload": workload, "seed": seed, **r}
                 for r in rb_frontier(cfg, prep, name, sel["shape"], sel["c"], scale_grid(sb), sb["capacity"])]
    rows += [{"workload": workload, "seed": seed, **r} for r in jcso_frontier(cfg, prep, sb["capacity"])]
    for r in rows:
        r["job_seconds"] = time.perf_counter() - t0
    return rows


def exp_test_b(cfg: Dict, out: Path, n_proc: int) -> None:
    cfg = study_b_config(cfg)
    sb = cfg["study_b"]
    root = Path(__file__).resolve().parents[2]
    with (root / sb["selected_file"]).open() as fh:
        shapes = yaml.safe_load(fh)["shapes"]
    jobs = [(cfg, w, s, shapes) for w in sb["workloads"] for s in sb["test_seeds"]]
    labels = [f"{w}_seed{s}" for _, w, s, _ in jobs]
    df = pd.DataFrame(sum(run_parallel(_test_job, jobs, n_proc, checkpoint_dir(out), labels), []))
    df.to_csv(out / "frontier_b.csv", index=False)


# ---------------------------------------------------------------------------
# Matched budgets (section 5)
# ---------------------------------------------------------------------------

def frontier_points(df: pd.DataFrame, budgets: List[float], keys: List[str]) -> pd.DataFrame:
    """Linear interpolation along each frontier at fixed over-reservation; NaN if not reached."""
    metrics = ["miss_rate", "acceptance", "p95_latency_ms", "p99_latency_ms", "utilization",
               "realized_reservation_ratio"]
    rows = []
    for k, g in df.groupby(keys):
        g = g.sort_values(["over_reservation", "lever"])
        x = g["over_reservation"].to_numpy()
        for b in budgets:
            row = dict(zip(keys, k))
            row["budget"] = b
            inside = x.min() <= b <= x.max()
            for m in metrics:
                if m in g:
                    row[m] = float(np.interp(b, x, g[m].to_numpy())) if inside else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


EXPERIMENTS_B = {"validation_b": exp_validation_b, "test_b": exp_test_b}
