"""Multi-objective Bayesian optimisation: Sobol design + batch ParEGO (Eqs. 12-18).

Every candidate configuration is trained with several random seeds (default 5); the objective vector
used by the optimiser is the mean over the seeds.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# (name, kind, argument); kind: choice | uniform | log
SEARCH_SPACES = {
    "timexer": [
        ("d_model", "choice", [32, 48, 64, 96]),
        ("num_heads", "choice", [2, 4]),
        ("ff_dim", "choice", [64, 128, 256]),
        ("num_layers", "choice", [1, 2, 3]),
        ("dropout", "uniform", (0.0, 0.20)),
        ("patch_len", "choice", [1, 2, 4]),
        ("head_units", "choice", [16, 32, 64]),
        ("learning_rate", "log", (3e-4, 3e-3)),
    ],
    "xgboost": [
        ("n_estimators", "choice", [200, 350, 500, 750, 1000]),
        ("max_depth", "choice", [4, 6, 8, 10, 12]),
        ("learning_rate", "log", (0.01, 0.15)),
        ("subsample", "uniform", (0.65, 1.0)),
        ("colsample_bytree", "uniform", (0.65, 1.0)),
        ("min_child_weight", "log", (0.5, 20.0)),
        ("reg_lambda", "log", (1e-3, 30.0)),
    ],
    "elm": [
        ("hidden_units", "choice", [128, 256, 512, 768, 1024]),
        ("alpha", "log", (1e-5, 10.0)),
        ("activation", "choice", ["tanh", "relu", "sigmoid"]),
    ],
    "gru": [
        ("units", "choice", [32, 64, 96, 128]),
        ("num_layers", "choice", [1, 2, 3]),
        ("dropout", "uniform", (0.0, 0.30)),
        ("head_units", "choice", [16, 32, 64]),
        ("learning_rate", "log", (3e-4, 3e-3)),
    ],
    "cnn_lstm": [
        ("filters", "choice", [32, 64, 96, 128]),
        ("kernel_size", "choice", [2, 3, 5]),
        ("lstm_units", "choice", [32, 64, 96, 128]),
        ("dropout", "uniform", (0.0, 0.30)),
        ("head_units", "choice", [16, 32, 64]),
        ("learning_rate", "log", (3e-4, 3e-3)),
    ],
    "nonstationary_transformer": [
        ("d_model", "choice", [32, 48, 64, 96]),
        ("num_heads", "choice", [2, 4]),
        ("ff_dim", "choice", [64, 128, 256]),
        ("num_layers", "choice", [1, 2, 3]),
        ("dropout", "uniform", (0.0, 0.25)),
        ("projector_units", "choice", [32, 64, 128]),
        ("head_units", "choice", [16, 32, 64]),
        ("learning_rate", "log", (3e-4, 3e-3)),
    ],
}
TASK_WEIGHT_MODELS = {"timexer"}      # task weights are searched only for the proposed model


def decode(space: list, z: np.ndarray) -> dict:
    cfg = {}
    for i, (name, kind, arg) in enumerate(space):
        u = float(z[i])
        if kind == "choice":
            cfg[name] = list(arg)[min(int(u * len(arg)), len(arg) - 1)]
        elif kind == "uniform":
            cfg[name] = float(arg[0] + u * (arg[1] - arg[0]))
        elif kind == "log":
            cfg[name] = float(math.exp(math.log(arg[0]) + u * (math.log(arg[1]) - math.log(arg[0]))))
        else:
            raise ValueError(kind)
    return cfg


def task_weights_from_unit(a, temperature: float = 3.0) -> list:
    """lambda = K * softmax(3a), a in [0,1]^K (Eq. 12)."""
    a = np.asarray(a, dtype=float)
    e = np.exp(temperature * a)
    return list(len(a) * e / e.sum())


def make_decoder(space: list, n_task_weights: int = 0):
    dim = len(space) + n_task_weights

    def decode_fn(z):
        cfg = decode(space, z[: len(space)])
        if n_task_weights:
            cfg["task_weights"] = task_weights_from_unit(z[len(space): dim])
        return cfg

    return dim, decode_fn


def pareto_mask(F: np.ndarray) -> np.ndarray:
    mask = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if (np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)).any():
            mask[i] = False
    return mask


def hypervolume(F: np.ndarray, reference: np.ndarray, mc_samples: int = 400_000) -> float:
    P = F[pareto_mask(F)]
    P = P[np.all(P < reference, axis=1)]
    if len(P) == 0:
        return 0.0
    m = P.shape[1]
    if (len(P) + 1) ** m > 3_000_000:                 # Monte-Carlo estimate for many objectives
        lo = P.min(0)
        u = np.random.default_rng(0).uniform(lo, reference, size=(mc_samples, m))
        dom = np.zeros(mc_samples, dtype=bool)
        for p in P:
            dom |= np.all(u >= p, axis=1)
        return float(dom.mean() * np.prod(reference - lo))
    grids = [np.unique(np.r_[P[:, k], reference[k]]) for k in range(m)]
    lows = np.meshgrid(*[g[:-1] for g in grids], indexing="ij")
    widths = np.meshgrid(*[np.diff(g) for g in grids], indexing="ij")
    cell = np.stack([l.ravel() for l in lows], axis=1)
    vol = np.prod(np.stack([w.ravel() for w in widths], axis=1), axis=1)
    cov = np.zeros(len(cell), dtype=bool)
    for p in P:
        cov |= np.all(cell >= p, axis=1)
    return float(vol[cov].sum())


def knee_index(F: np.ndarray, norm_ord: float = 2) -> int:
    """Eq. 18: Pareto point closest to the ideal point after range normalisation."""
    idx = np.where(pareto_mask(F))[0]
    P = F[idx]
    span = np.where(P.max(0) - P.min(0) > 0, P.max(0) - P.min(0), 1.0)
    return int(idx[np.argmin(np.linalg.norm((P - P.min(0)) / span, ord=norm_ord, axis=1))])


def _fit_gp(Z, g, seed):
    d = Z.shape[1]
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.5), length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(Z, g)
    return gp


def expected_improvement(mu, sigma, best, xi):
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def chebyshev(Fn, w, rho):
    return np.max(w * Fn, axis=1) + rho * np.sum(w * Fn, axis=1)


def run_parego(*, dim: int, decode_fn: Callable, train_eval_fn: Callable, out_dir, objective_names: list,
               seeds=DEFAULT_SEEDS, n_init: int = 10, n_iterations: int = 10, batch_candidates: int = 2,
               candidate_pool: int = 4096, min_distance: float = 0.12, rho: float = 0.05, xi: float = 1e-3,
               sobol_seed: int = 2026, log: Callable = print, **_) -> dict:
    """train_eval_fn(config, seed) -> {"objectives": [K], "seconds": s, "params": n}.
    Each candidate is evaluated with every seed in ``seeds``; the surrogate sees the seed mean."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names, seeds = list(objective_names), [int(s) for s in seeds]
    t_csv, s_csv = out / "mobo_trials.csv", out / "mobo_trials_per_seed.csv"
    rows = pd.read_csv(t_csv, encoding="utf-8-sig").to_dict("records") if t_csv.exists() else []
    per_seed = pd.read_csv(s_csv, encoding="utf-8-sig").to_dict("records") if s_csv.exists() else []
    if rows:
        log(f"[MOBO] resuming after {len(rows)} evaluations")
    rng = np.random.default_rng(sobol_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sobol = qmc.Sobol(dim, scramble=True, seed=sobol_seed).random(n_init)

    def evaluate(z, phase, iteration):
        trial = len(rows)
        cfg = decode_fn(z)
        log(f"[MOBO] trial {trial:02d} ({phase}, iter {iteration}) {json.dumps(cfg)}")
        done = {int(r["seed"]): r for r in per_seed if int(r["trial"]) == trial}
        F, secs, params = [], 0.0, 0
        for s in seeds:
            if s in done:
                r = done[s]
            else:
                res = train_eval_fn(cfg, s)
                r = {"trial": trial, "seed": s, "seconds": float(res["seconds"]), "params": int(res["params"]),
                     **{f"NMSE_{n}": float(v) for n, v in zip(names, res["objectives"])}}
                per_seed.append(r)
                pd.DataFrame(per_seed).to_csv(s_csv, index=False, encoding="utf-8-sig")
            F.append([r[f"NMSE_{n}"] for n in names])
            secs += r["seconds"]
            params = r["params"]
        F = np.asarray(F, dtype=float)
        f, sd = F.mean(0), F.std(0, ddof=1) if len(F) > 1 else np.zeros(F.shape[1])
        rows.append({"trial": trial, "phase": phase, "iteration": iteration, "n_seeds": len(seeds),
                     "duration_seconds": secs, "parameter_or_tree_count": params, "objective_mean": float(f.mean()),
                     "objective_max": float(f.max()), "objective_mean_sd_over_seeds": float(F.mean(1).std(ddof=1)) if len(F) > 1 else 0.0,
                     "config_json": json.dumps(cfg, sort_keys=True),
                     **{f"NMSE_{n}": float(v) for n, v in zip(names, f)},
                     **{f"SD_{n}": float(v) for n, v in zip(names, sd)},
                     **{f"z{i}": float(v) for i, v in enumerate(z)}})
        pd.DataFrame(rows).to_csv(t_csv, index=False, encoding="utf-8-sig")
        log(f"[MOBO]   -> mean of {len(seeds)} seeds {np.round(f, 5).tolist()}  avg={f.mean():.5f}  {secs:.1f}s")

    for i in range(len(rows), n_init):
        evaluate(sobol[i], "sobol", -1)

    for it in range(n_iterations):                         # random draws are replayed on resume
        done_it = sum(1 for r in rows if r["phase"] == "bayesian" and int(r["iteration"]) == it)
        seen = rows[: n_init + it * batch_candidates]
        Z = np.array([[r[f"z{i}"] for i in range(dim)] for r in seen])
        F = np.array([[r[f"NMSE_{n}"] for n in names] for r in seen])
        Fn = (F - F.min(0)) / np.where(np.ptp(F, 0) > 0, np.ptp(F, 0), 1.0)
        W = rng.dirichlet(np.ones(F.shape[1]), size=batch_candidates)
        pool = rng.random((candidate_pool, dim))
        chosen = []
        for j in range(batch_candidates):
            g = chebyshev(Fn, W[j], rho)
            mu, sd = _fit_gp(Z, g, sobol_seed + 100 * it + j).predict(pool, return_std=True)
            for c in np.argsort(-expected_improvement(mu, sd, float(g.min()), xi)):
                cand = pool[c]
                if all(np.linalg.norm(cand - o) > min_distance for o in chosen) and np.min(np.linalg.norm(Z - cand, axis=1)) > 1e-6:
                    chosen.append(cand)
                    break
        for k, z in enumerate(chosen):
            if k >= done_it:
                evaluate(z, "bayesian", it)

    trials = pd.DataFrame(rows)
    F = trials[[f"NMSE_{n}" for n in names]].to_numpy(float)
    ref = 1.10 * F.max(0)
    trials["pareto"] = pareto_mask(F)
    trials.to_csv(t_csv, index=False, encoding="utf-8-sig")
    trials[trials.pareto].to_csv(out / "pareto_front.csv", index=False, encoding="utf-8-sig")
    hist = []
    for stop, lab in [(n_init, -1)] + [(n_init + (it + 1) * batch_candidates, it) for it in range(n_iterations)]:
        Fs = F[:stop]
        hist.append({"iteration": lab, "evaluations": len(Fs), "hypervolume_fixed_reference": hypervolume(Fs, ref),
                     "pareto_size": int(pareto_mask(Fs).sum()), "best_mean_NMSE": float(Fs.mean(1).min()),
                     "best_mean_trial": int(np.argmin(Fs.mean(1))), "best_worst_target_NMSE": float(Fs.max(1).min())})
    pd.DataFrame(hist).to_csv(out / "mobo_iteration_summary.csv", index=False, encoding="utf-8-sig")
    k = knee_index(F)
    best = {"selection_rule": "Pareto knee, Eq. 18 (Euclidean distance to the ideal point after range normalisation)",
            "seeds_per_evaluation": seeds, "trial": int(trials.trial[k]), "best_config": json.loads(trials.config_json[k]),
            "validation_objectives": dict(zip(names, F[k].tolist())),
            "knee_with_other_distances": {"L1": int(trials.trial[knee_index(F, 1)]),
                                          "Chebyshev": int(trials.trial[knee_index(F, np.inf)])},
            "reference_point": ref.tolist(), "evaluations": len(trials)}
    (out / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    log(f"[MOBO] done: {len(trials)} evaluations x {len(seeds)} seeds, {int(trials.pareto.sum())} Pareto points, knee = trial {best['trial']}")
    return best
