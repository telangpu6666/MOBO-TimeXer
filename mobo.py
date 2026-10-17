"""Multi-objective Bayesian optimisation (MOBO): Sobol initial design + batch ParEGO (Section 2.2, Eqs. 12-18).

Algorithm (Algorithm 1, stage 2 of the paper)
--------------------------------------------
1. Draw ``n_init`` scrambled Sobol points in the unit hypercube [0,1]^d and evaluate them.
2. In each of ``n_iterations`` iterations draw ``q = batch_candidates`` Dirichlet weight vectors w_j.
   For each w_j the K objectives (range-normalised over the evaluated samples) are scalarised by the
   augmented Chebyshev function  g = max_k w_k f_k + rho * sum_k w_k f_k            (Eq. 15)
   and an anisotropic Matern-5/2 + white-noise Gaussian process is fitted to g      (Eq. 16).
   Expected Improvement (Eq. 17) is maximised over ``candidate_pool`` random points, subject to a
   minimum distance ``min_distance`` between the candidates of the same batch.
3. After the budget is spent, the non-dominated set is extracted (Eq. 14) and the knee point is the
   Pareto solution with minimum Euclidean distance to the ideal point after range normalisation
   over the Pareto set (Eq. 18).
The hypervolume (HV) of the non-dominated set is tracked with a fixed reference point
= 1.10 x the maximum of every objective over all evaluations.

The optimiser is model-agnostic: it only needs ``decode_fn(z) -> config`` and
``evaluate_fn(config, trial) -> {"objectives": [K], "seconds": float, "params": int}``.
Every evaluation is appended to ``mobo_trials.csv`` immediately, and an interrupted run resumes
from that file with exactly the same random sequence.

This file also holds the decision spaces used in the paper for the six models (Eq. 12, Table 6).
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

# ====================================================================== decision spaces of the paper
# Each entry: (name, kind, argument). kind = "choice" (list of values), "uniform" (low, high) or
# "log" (low, high; sampled on a logarithmic scale). The order defines the dimensions of z.
SEARCH_SPACES = {
    # MOBO-TimeXer: 8 structural/training hyper-parameters + 4 task-weight coefficients = 12 dims (Eq. 12)
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
# Models whose task weights lambda are part of the decision vector (only the proposed model).
TASK_WEIGHT_MODELS = {"timexer"}


# ====================================================================== decoding z in [0,1]^d -> configuration
def choice(z: float, options: list):
    return options[min(int(z * len(options)), len(options) - 1)]


def uniform(z: float, low: float, high: float) -> float:
    return float(low + z * (high - low))


def log_uniform(z: float, low: float, high: float) -> float:
    return float(math.exp(math.log(low) + z * (math.log(high) - math.log(low))))


def decode(space: list, z: np.ndarray) -> dict:
    config = {}
    for i, (name, kind, arg) in enumerate(space):
        if kind == "choice":
            config[name] = choice(float(z[i]), list(arg))
        elif kind == "uniform":
            config[name] = uniform(float(z[i]), *arg)
        elif kind == "log":
            config[name] = log_uniform(float(z[i]), *arg)
        else:
            raise ValueError(kind)
    return config


def task_weights_from_unit(a: np.ndarray, temperature: float = 3.0, total: float | None = None) -> list:
    """Softmax parameterisation of the task weights (Eq. 12): lambda = K * softmax(3 a), a in [0,1]^K.
    Keeps every weight strictly positive and their sum fixed at K (= 4 in the paper)."""
    a = np.asarray(a, dtype=float)
    total = float(len(a)) if total is None else total
    e = np.exp(temperature * a)
    return list(total * e / e.sum())


def make_decoder(space: list, n_task_weights: int = 0) -> tuple[int, Callable[[np.ndarray], dict]]:
    """Returns (dimension, decode_fn). With ``n_task_weights > 0`` the last dimensions are task-weight coefficients."""
    dim = len(space) + n_task_weights

    def decode_fn(z: np.ndarray) -> dict:
        cfg = decode(space, z[: len(space)])
        if n_task_weights:
            cfg["task_weights"] = task_weights_from_unit(z[len(space): dim])
        return cfg

    return dim, decode_fn


# ====================================================================== Pareto set, hypervolume, knee point
def pareto_mask(F: np.ndarray) -> np.ndarray:
    """Non-dominated rows of F (minimisation, Eq. 14)."""
    n = len(F)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def hypervolume(F: np.ndarray, reference: np.ndarray, mc_samples: int = 400_000, seed: int = 0) -> float:
    """Hypervolume (minimisation). Exact by coordinate compression for small problems (tens of points,
    up to ~5 objectives); Monte-Carlo estimate otherwise."""
    P = F[pareto_mask(F)]
    P = P[np.all(P < reference, axis=1)]
    if len(P) == 0:
        return 0.0
    m = P.shape[1]
    if (len(P) + 1) ** m > 3_000_000:                  # Monte-Carlo fallback for many objectives
        lo = P.min(0)
        u = np.random.default_rng(seed).uniform(lo, reference, size=(mc_samples, m))
        dom = np.zeros(mc_samples, dtype=bool)
        for p in P:
            dom |= np.all(u >= p, axis=1)
        return float(dom.mean() * np.prod(reference - lo))
    grids = [np.unique(np.r_[P[:, k], reference[k]]) for k in range(m)]
    lows = np.meshgrid(*[g[:-1] for g in grids], indexing="ij")
    widths = np.meshgrid(*[np.diff(g) for g in grids], indexing="ij")
    cell_low = np.stack([l.ravel() for l in lows], axis=1)
    volume = np.prod(np.stack([w.ravel() for w in widths], axis=1), axis=1)
    covered = np.zeros(len(cell_low), dtype=bool)
    for p in P:
        covered |= np.all(cell_low >= p, axis=1)
    return float(volume[covered].sum())


def knee_index(F: np.ndarray, norm_ord: float = 2) -> int:
    """Eq. 18: Pareto solution closest to the ideal point after range normalisation over the Pareto set.
    ``norm_ord`` = 2 (Euclidean, paper), 1 (L1) or np.inf (Chebyshev) for the sensitivity check."""
    idx = np.where(pareto_mask(F))[0]
    P = F[idx]
    span = np.where(P.max(0) - P.min(0) > 0, P.max(0) - P.min(0), 1.0)
    dist = np.linalg.norm((P - P.min(0)) / span, ord=norm_ord, axis=1)
    return int(idx[np.argmin(dist)])


# ====================================================================== surrogate and acquisition
def _fit_gp(Z: np.ndarray, g: np.ndarray, seed: int) -> GaussianProcessRegressor:
    d = Z.shape[1]
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.5), length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e-1)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(Z, g)
    return gp


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi
    zz = imp / sigma
    return imp * norm.cdf(zz) + sigma * norm.pdf(zz)


def chebyshev(Fn: np.ndarray, w: np.ndarray, rho: float) -> np.ndarray:
    return np.max(w * Fn, axis=1) + rho * np.sum(w * Fn, axis=1)


# ====================================================================== main loop
def run_parego(
    *,
    dim: int,
    decode_fn: Callable[[np.ndarray], dict],
    evaluate_fn: Callable[[dict, int], dict],
    out_dir: str | Path,
    objective_names: list,
    n_init: int = 10,
    n_iterations: int = 10,
    batch_candidates: int = 2,
    candidate_pool: int = 4096,
    min_distance: float = 0.12,
    rho: float = 0.05,
    xi: float = 1e-3,
    seed: int = 2026,
    log: Callable[[str], None] = print,
    **_,
) -> dict:
    """Run batch ParEGO. ``evaluate_fn(config, trial)`` must return
    ``{"objectives": [K values to minimise], "seconds": float, "params": int}``.

    Writes to ``out_dir``: mobo_trials.csv, pareto_front.csv, mobo_iteration_summary.csv, best_config.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(objective_names)
    trials_csv = out_dir / "mobo_trials.csv"
    rows: list[dict] = []
    if trials_csv.exists():
        rows = pd.read_csv(trials_csv, encoding="utf-8-sig").to_dict("records")
        log(f"[MOBO] found {len(rows)} finished evaluations - resuming")
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():                 # n_init need not be a power of 2
        warnings.simplefilter("ignore", UserWarning)
        sobol = qmc.Sobol(dim, scramble=True, seed=seed).random(n_init)

    def do_eval(z: np.ndarray, phase: str, iteration: int) -> None:
        trial = len(rows)
        config = decode_fn(z)
        log(f"[MOBO] trial {trial:02d} ({phase}, iter {iteration}) {json.dumps(config, ensure_ascii=False)}")
        res = evaluate_fn(config, trial)
        f = np.asarray(res["objectives"], dtype=float)
        rows.append({"trial": trial, "phase": phase, "iteration": iteration, "duration_seconds": res["seconds"],
                     "parameter_or_tree_count": res["params"], "objective_sum": float(f.sum()),
                     "objective_max": float(f.max()), "objective_mean": float(f.mean()),
                     "config_json": json.dumps(config, sort_keys=True),
                     **{f"NMSE_{n}": float(v) for n, v in zip(names, f)},
                     **{f"z{i}": float(v) for i, v in enumerate(z)}})
        pd.DataFrame(rows).to_csv(trials_csv, index=False, encoding="utf-8-sig")
        log(f"[MOBO]   -> validation NMSE {np.round(f, 5).tolist()}  mean={f.mean():.5f}  {res['seconds']:.1f}s")

    # 1) Sobol initial design
    for i in range(len(rows), n_init):
        do_eval(sobol[i], "sobol", -1)

    # 2) batch ParEGO iterations (random numbers are replayed on resume -> identical to an uninterrupted run)
    for it in range(n_iterations):
        done_in_iter = sum(1 for r in rows if r["phase"] == "bayesian" and int(r["iteration"]) == it)
        seen = rows[: n_init + it * batch_candidates]
        Z = np.array([[r[f"z{i}"] for i in range(dim)] for r in seen])
        F = np.array([[r[f"NMSE_{n}"] for n in names] for r in seen])
        fmin, fmax = F.min(0), F.max(0)
        Fn = (F - fmin) / np.where(fmax - fmin > 0, fmax - fmin, 1.0)
        chosen = []
        weights_drawn = rng.dirichlet(np.ones(F.shape[1]), size=batch_candidates)
        pool = rng.random((candidate_pool, dim))
        for j in range(batch_candidates):
            g = chebyshev(Fn, weights_drawn[j], rho)
            gp = _fit_gp(Z, g, seed + 100 * it + j)
            mu, sd = gp.predict(pool, return_std=True)
            ei = expected_improvement(mu, sd, float(g.min()), xi)
            for c in np.argsort(-ei):
                cand = pool[c]
                far_batch = all(np.linalg.norm(cand - other) > min_distance for other in chosen)
                far_seen = np.min(np.linalg.norm(Z - cand, axis=1)) > 1e-6
                if far_batch and far_seen:
                    chosen.append(cand)
                    break
        for k, z in enumerate(chosen):
            if k < done_in_iter:
                continue
            do_eval(z, "bayesian", it)

    # 3) Pareto front, hypervolume history and knee point
    trials = pd.DataFrame(rows)
    F = trials[[f"NMSE_{n}" for n in names]].to_numpy(float)
    reference = 1.10 * F.max(0)
    trials["pareto"] = pareto_mask(F)
    trials.to_csv(trials_csv, index=False, encoding="utf-8-sig")
    trials[trials.pareto].to_csv(out_dir / "pareto_front.csv", index=False, encoding="utf-8-sig")
    summary = []
    for stop, label in [(n_init, -1)] + [(n_init + (it + 1) * batch_candidates, it) for it in range(n_iterations)]:
        Fs = F[:stop]
        summary.append({"iteration": label, "evaluations": len(Fs),
                        "hypervolume_fixed_reference": hypervolume(Fs, reference),
                        "pareto_size": int(pareto_mask(Fs).sum()), "best_mean_NMSE": float(Fs.mean(1).min()),
                        "best_mean_trial": int(np.argmin(Fs.mean(1))), "best_worst_target_NMSE": float(Fs.max(1).min())})
    pd.DataFrame(summary).to_csv(out_dir / "mobo_iteration_summary.csv", index=False, encoding="utf-8-sig")
    k = knee_index(F)
    alt = {"L1": int(trials.trial[knee_index(F, 1)]), "Chebyshev": int(trials.trial[knee_index(F, np.inf)])}
    best = {"selection_rule": "Pareto knee: minimum Euclidean distance to the ideal point after range normalisation (Eq. 18)",
            "trial": int(trials.trial[k]), "best_config": json.loads(trials.config_json[k]),
            "validation_objectives": dict(zip(names, F[k].tolist())),
            "knee_with_other_distances": alt, "reference_point": reference.tolist(), "evaluations": len(trials)}
    (out_dir / "best_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[MOBO] done: {len(trials)} evaluations, {int(trials.pareto.sum())} Pareto solutions, knee = trial {best['trial']}")
    return best
