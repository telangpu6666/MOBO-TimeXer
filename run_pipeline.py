"""End-to-end reproduction of the MOBO-TimeXer study on any data set (Algorithm 1 of the paper).

Stages
------
  audit     data split, Table-3 statistics, standardisation parameters, persistence baseline
  analysis  band-separated coupling analysis of the targets (Section 4.1: Welch PSD, 0.03-Hz Butterworth split,
            Pearson correlation per band, low-/wave-frequency energy shares and peak periods)
  search    batch-ParEGO MOBO of every model (Section 2.2; Table 5, Fig. 8, Fig. 9) -> knee configuration
  final     knee configuration retrained on the full training segment with several seeds; metrics and
            window-wise predictions on every test segment (Tables 7, 8)
  ablation  A default structure + equal weights / B knee structure + equal weights /
            C four single-task TimeXers / D MOBO-TimeXer (Table 9)
  report    seed statistics (mean, SD, 95% CI), Welch t-tests with Holm correction, seed ensembles,
            improvement over the best benchmark and persistence, cost table, figures, one Excel workbook
  all       audit + analysis + search + final + (ablation if enabled) + report

Usage
-----
  python make_demo_data.py                                   # optional: synthetic data set for a test run
  python run_pipeline.py --config configs/demo_config.json --quick       # smoke test (minutes)
  python run_pipeline.py --config configs/demo_config.json               # full protocol
  python run_pipeline.py --config my_config.json --stage search --models timexer
  python run_pipeline.py --config my_config.json --stage final --knee paper --seeds 0-9
Every stage resumes from finished results (delete a result folder to recompute it).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_pipeline as DP  # noqa: E402
import mobo as MB  # noqa: E402

M = None          # models module (imports torch) - loaded lazily so that 'audit'/'analysis' run without torch


def _models():
    global M
    if M is None:
        import models as _m
        M = _m
    return M


LABELS = {"timexer": "MOBO-TimeXer", "xgboost": "XGBoost", "elm": "ELM", "gru": "GRU", "cnn_lstm": "CNN-LSTM",
          "nonstationary_transformer": "Non-stationary Transformer"}
ALL_MODELS = ["timexer", "xgboost", "elm", "gru", "cnn_lstm", "nonstationary_transformer"]

# ====================================================================== default settings (= the paper)
DEFAULTS = {
    "output_dir": "results/run",
    "models": ALL_MODELS,
    "reference_model": "timexer",
    # fixed (not searched) options of the proposed network, Section 2.1.2
    "model_options": {"timexer": {"ar_bypass": True, "window_norm": True, "window_norm_eps": 0.02}},
    "search_spaces": {},              # optional overrides of mobo.SEARCH_SPACES, same format
    "mobo": {
        "default": {                  # benchmarks in the paper: 10 Sobol + 8 x 2 ParEGO, one seed, 30k/15k windows, 6 epochs
            "n_init": 10, "n_iterations": 8, "batch_candidates": 2, "candidate_pool": 4096, "min_distance": 0.12,
            "rho": 0.05, "xi": 1e-3, "sobol_seed": 42, "seeds": [42],
            "epochs": 6, "patience": 3, "batch_size": 1024, "max_train_samples": 30000, "max_val_samples": 15000},
        "timexer": {                  # proposed model: 10 Sobol + 10 x 2 ParEGO, three seeds, 60k/20k windows, 8 epochs
            "n_iterations": 10, "sobol_seed": 2026, "seeds": [0, 1, 2],
            "epochs": 8, "max_train_samples": 60000, "max_val_samples": 20000},
    },
    "final": {"seeds": list(range(10)), "epochs": 60, "patience": 15, "batch_size": 1024,
              "max_train_samples": 0, "max_val_samples": 0, "save_models": True, "save_prediction_csv": True},
    "ablation": {"enabled": False, "network": "revised", "seeds": list(range(10)),
                 "variants": ["A_default_equal", "B_knee_equal", "C_independent", "D_mobo"]},
    "analysis": {"enabled": True, "conditions": {"normal": "train", "extreme": "extreme_test"},
                 "cutoff_hz": 0.03, "nperseg": 8192, "low_peak_range_hz": [0.002, 0.03]},
    "report": {"plot_window_samples": 300, "plot_window_position": 0.5, "figure_dpi": 300},
}

QUICK = {
    "mobo": {"n_init": 3, "n_iterations": 1, "seeds": [0], "epochs": 1, "patience": 1,
             "max_train_samples": 3000, "max_val_samples": 1000},
    "final": {"seeds": [0, 1], "epochs": 2, "patience": 1, "max_train_samples": 5000, "max_val_samples": 2000},
    "ablation_seeds": [0, 1],
    "xgb_max_trees": 30,
}


def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# ====================================================================== context
class Context:
    def __init__(self, cfg: dict, config_dir: Path, quick: bool):
        self.cfg, self.quick = cfg, quick
        out = Path(cfg["output_dir"])
        if not out.is_absolute():
            out = (config_dir / out).resolve()
        self.out = out.with_name(out.name + "_quick") if quick else out
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = Logger(self.out / "run_log.txt")
        self.data_cfg = DP.DataConfig.from_dict(cfg["data"], base_dir=config_dir)
        self._data, self._wd = None, None

    @property
    def data(self) -> DP.MotionData:
        if self._data is None:
            t0 = time.perf_counter()
            self._data = DP.load_dataset(self.data_cfg)
            self.log(f"data loaded in {time.perf_counter() - t0:.1f} s; windows per split {self._data.profile['split_windows']}")
        return self._data

    @property
    def wd(self):
        if self._wd is None:
            self._wd = _models().WindowData(self.data)
        return self._wd

    def model_options(self, key: str) -> dict:
        return dict(self.cfg["model_options"].get(key, {}))

    def mobo_budget(self, key: str) -> dict:
        b = deep_update(self.cfg["mobo"]["default"], self.cfg["mobo"].get(key, {}))
        if self.quick:
            b.update(QUICK["mobo"])
        return b

    def final_protocol(self) -> dict:
        p = dict(self.cfg["final"])
        if self.quick:
            p.update(QUICK["final"])
        return p

    def search_space(self, key: str) -> list:
        space = self.cfg["search_spaces"].get(key, MB.SEARCH_SPACES[key])
        space = [(n, k, a) for n, k, a in space]
        L = self.data.lookback
        fixed = []
        for name, kind, arg in space:                                   # keep only patch lengths that divide L
            if name == "patch_len" and kind == "choice":
                arg = [p for p in arg if L % int(p) == 0] or [1]
            fixed.append((name, kind, arg))
        return fixed

    def quick_cfg(self, key: str, cfg: dict) -> dict:
        if self.quick and key == "xgboost":
            cfg = {**cfg, "n_estimators": min(int(cfg["n_estimators"]), QUICK["xgb_max_trees"])}
        return cfg


# ====================================================================== stage: audit
def persistence_metrics(ctx: Context) -> pd.DataFrame:
    d = ctx.data
    rows = []
    for split in ["val"] + d.test_splits:
        idx = d.split_indices[split]
        rows += _models_free_metrics(d.raw_features[idx, : d.n_targets], d.raw_features[idx - d.horizon, : d.n_targets],
                                     d, split, "Persistence")
    return pd.DataFrame(rows)


def _models_free_metrics(truth, pred, d, scenario, name):
    """Same as models.regression_metrics, duplicated here so that audit runs without torch."""
    rows = []
    for k, t_name in enumerate(d.target_names):
        t, p = truth[:, k].astype(np.float64), pred[:, k].astype(np.float64)
        e = p - t
        mse = float(np.mean(e ** 2))
        rows.append({"model": name, "scenario": scenario, "target": t_name, "unit": d.target_units[k], "n": len(t),
                     "MSE": mse, "RMSE": math.sqrt(mse), "MAE": float(np.mean(np.abs(e))),
                     "NMSE": mse / (float(d.target_scale[k]) ** 2 + 1e-12),
                     "R2": float(1.0 - np.sum(e ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-30)),
                     "sMAPE_pct": float(200.0 * np.mean(np.abs(e) / (np.abs(t) + np.abs(p) + 1e-6))),
                     "bias": float(np.mean(e))})
    f = pd.DataFrame(rows)
    rows.append({"model": name, "scenario": scenario, "target": "average", "unit": "mixed", "n": len(truth),
                 **{c: float(f[c].mean()) for c in ["MSE", "RMSE", "MAE", "NMSE", "R2", "sMAPE_pct", "bias"]},
                 "NMSE_max": float(f["NMSE"].max()), "R2_min": float(f["R2"].min())})
    return rows


def stage_audit(ctx: Context) -> None:
    out = ctx.out / "data"
    DP.write_data_audit(ctx.data, out)
    persistence_metrics(ctx).to_csv(out / "persistence_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(ctx.out / "resolved_config.json", ctx.cfg)
    ctx.log(f"[audit] split time ranges {json.dumps(ctx.data.profile['split_time_range'])}")
    ctx.log(f"[audit] written to {out}")


# ====================================================================== stage: coupling analysis (Section 4.1)
def stage_analysis(ctx: Context) -> None:
    from scipy import signal
    a = ctx.cfg["analysis"]
    if not a.get("enabled", True):
        return
    d = ctx.data
    out = ctx.out / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    fs = 1.0 / d.profile["sample_step_seconds"]
    cut, nper = float(a["cutoff_hz"]), int(a["nperseg"])
    lo_a, lo_b = a["low_peak_range_hz"]
    corr_rows, band_rows, comps = [], [], {}
    for label, split in a["conditions"].items():
        if split not in d.split_indices:
            ctx.log(f"[analysis] split '{split}' not found - condition '{label}' skipped")
            continue
        v = d.raw_features[d.split_indices[split], : d.n_targets].astype(np.float64)
        det = signal.detrend(v, axis=0, type="linear")
        sos = signal.butter(4, cut / (fs / 2.0), btype="low", output="sos")
        low = signal.sosfiltfilt(sos, det, axis=0)                     # zero-phase 4th-order Butterworth
        wave = det - low
        comps[label] = {"low": low, "wave": wave, "raw": det}
        for comp, arr in (("low", low), ("wave", wave)):
            C = np.corrcoef(arr.T)
            for i in range(d.n_targets):
                for j in range(d.n_targets):
                    corr_rows.append({"condition": label, "split": split, "band": comp, "target_i": d.target_names[i],
                                      "target_j": d.target_names[j], "pearson_r": C[i, j]})
        for k, name in enumerate(d.target_names):
            f, p = signal.welch(det[:, k], fs=fs, nperseg=min(nper, len(det)))
            total = np.trapezoid(p, f)
            low_e = np.trapezoid(p[f < cut], f[f < cut])
            lm, wm = (f >= lo_a) & (f < lo_b), f >= cut
            lpk = f[lm][np.argmax(p[lm])] if lm.any() else np.nan
            wpk = f[wm][np.argmax(p[wm])] if wm.any() else np.nan
            band_rows.append({"condition": label, "split": split, "target": name, "unit": d.target_units[k],
                              "low_frequency_energy_pct": 100 * low_e / total, "wave_energy_pct": 100 * (total - low_e) / total,
                              "low_frequency_peak_period_s": 1 / lpk if lpk > 0 else np.nan,
                              "wave_peak_period_s": 1 / wpk if wpk > 0 else np.nan})
    pd.DataFrame(corr_rows).to_csv(out / "band_correlations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(band_rows).to_csv(out / "band_energy.csv", index=False, encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dpi = ctx.cfg["report"]["figure_dpi"]
        labels = list(comps)
        fig, axes = plt.subplots(len(labels), 2, figsize=(8.6, 3.9 * len(labels)), squeeze=False)
        for r, lab in enumerate(labels):
            for c, comp in enumerate(("low", "wave")):
                C = np.corrcoef(comps[lab][comp].T)
                ax = axes[r, c]
                im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
                ax.set_xticks(range(d.n_targets), d.target_names, rotation=25, ha="right")
                ax.set_yticks(range(d.n_targets), d.target_names)
                ax.set_title(f"{lab} - {'low-frequency' if comp == 'low' else 'wave-frequency'}")
                for i in range(d.n_targets):
                    for j in range(d.n_targets):
                        ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
        fig.savefig(out / "Fig_band_correlations.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        n = d.n_targets
        fig, axes = plt.subplots(int(math.ceil(n / 2)), 2, figsize=(10, 3.4 * math.ceil(n / 2)), squeeze=False)
        for k in range(n):
            ax = axes.flat[k]
            for lab in labels:
                f, p = signal.welch(comps[lab]["raw"][:, k], fs=fs, nperseg=min(nper, len(comps[lab]["raw"])))
                ax.semilogy(f[1:], p[1:], label=lab)
            ax.axvline(cut, color="0.4", ls="--", lw=0.8)
            ax.set(title=d.target_names[k], xlabel="Frequency (Hz)", ylabel=f"PSD ({d.target_units[k]}²/Hz)")
        axes.flat[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "Fig_power_spectra.png", dpi=dpi)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        ctx.log(f"[analysis] figures skipped: {e}")
    ctx.log(f"[analysis] written to {out}")


# ====================================================================== stage: MOBO search
def stage_search(ctx: Context, models: list) -> None:
    m = _models()
    d, wd = ctx.data, ctx.wd
    for key in models:
        budget = ctx.mobo_budget(key)
        space = ctx.search_space(key)
        n_tw = d.n_targets if key in MB.TASK_WEIGHT_MODELS else 0
        dim, decode_fn = MB.make_decoder(space, n_tw)
        out = ctx.out / "mobo" / key
        out.mkdir(parents=True, exist_ok=True)
        if (out / "best_config.json").exists():
            ctx.log(f"[search] {key}: best_config.json exists - skipped")
            continue
        seeds = list(budget["seeds"])
        lowfi = {k: budget[k] for k in ("epochs", "patience", "batch_size", "max_train_samples", "max_val_samples")}
        write_json(out / "optimizer_config.json", {
            "model": key, "search_space": [[n, k, a] for n, k, a in space],
            "task_weights": f"lambda = {d.n_targets}*softmax(3a), a in [0,1]^{d.n_targets}" if n_tw else "equal (not searched)",
            "decision_dimensions": dim, **budget, "low_fidelity_protocol": lowfi,
            "objective": "mean over seeds of the validation NMSE of every target (Eq. 13)",
            "fixed_model_options": ctx.model_options(key)})
        ps_file = out / "mobo_trials_per_seed.csv"
        per_seed = pd.read_csv(ps_file, encoding="utf-8-sig").to_dict("records") if ps_file.exists() else []
        opts = ctx.model_options(key)

        def evaluate(cfg, trial, key=key, seeds=seeds, lowfi=lowfi, opts=opts, per_seed=per_seed, ps_file=ps_file):
            objs, secs, params = [], 0.0, 0
            for s in seeds:
                tm = m.train_model(key, d, wd, ctx.quick_cfg(key, cfg), lowfi, s, model_options=opts)
                if key in m.NEURAL:
                    vi, p = tm.predict(d, wd, "val", lowfi["max_val_samples"], as_tensor=True)
                    f = ((p - wd.targets(vi)) ** 2).mean(0).cpu().numpy()
                else:
                    vi, p = tm.predict(d, wd, "val", lowfi["max_val_samples"])
                    f = ((p - d.features[vi, : d.n_targets]) ** 2).mean(0)
                objs.append(f)
                secs += tm.seconds
                params = tm.params
                per_seed.append({"trial": trial, "seed": s, "seconds": tm.seconds,
                                 **{f"NMSE_{n}": float(v) for n, v in zip(d.target_names, f)}})
                del tm
                _free_gpu()
            pd.DataFrame(per_seed).to_csv(ps_file, index=False, encoding="utf-8-sig")
            return {"objectives": np.mean(objs, axis=0), "seconds": secs, "params": params}

        ctx.log(f"===== [search] {LABELS[key]}: {dim}-D decision space, {budget['n_init']} Sobol + "
                f"{budget['n_iterations']} x {budget['batch_candidates']} ParEGO, seeds {seeds}")
        t0 = time.perf_counter()
        best = MB.run_parego(dim=dim, decode_fn=decode_fn, evaluate_fn=evaluate, out_dir=out,
                             objective_names=d.target_names, seed=budget["sobol_seed"], log=ctx.log,
                             **{k: budget[k] for k in ("n_init", "n_iterations", "batch_candidates", "candidate_pool",
                                                       "min_distance", "rho", "xi")})
        ctx.log(f"===== [search] {LABELS[key]} finished in {(time.perf_counter() - t0) / 60:.1f} min; knee trial "
                f"{best['trial']}: {json.dumps(best['best_config'])}")


def _free_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ====================================================================== stage: final multi-seed training
def knee_config(ctx: Context, key: str, source: str) -> tuple[dict, str]:
    m = _models()
    if source == "search":
        f = ctx.out / "mobo" / key / "best_config.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found: run '--stage search' first or use '--knee paper' / '--knee <file.json>'")
        j = json.loads(f.read_text(encoding="utf-8"))
        return j["best_config"], f"MOBO knee, trial {j['trial']}"
    if source == "paper":
        cfg = copy.deepcopy(m.PAPER_KNEE_CONFIGS[key])
        if "task_weights" in cfg and len(cfg["task_weights"]) != ctx.data.n_targets:
            cfg["task_weights"] = [1.0] * ctx.data.n_targets
        return cfg, "paper knee configuration (Table 6)"
    j = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    cfg = j[key] if key in j else j
    return cfg.get("best_config", cfg), f"file {source}"


def save_run(rd: Path, ctx: Context, name: str, tm_list, preds: dict, seed: int, cfg: dict, protocol: dict,
             source: str, save_csv: bool) -> pd.DataFrame:
    d = ctx.data
    rows, store = [], {}
    for split, (idx, p_std) in preds.items():
        truth = d.raw_features[idx, : d.n_targets]
        pred = d.standardizer.inverse_targets(p_std)
        rows += _models().regression_metrics(truth, pred, d.target_scale, d.target_names, d.target_units, split, name)
        store[f"{split}_index"] = idx.astype(np.int64)
        store[f"{split}_pred"] = pred.astype(np.float32)
        if save_csv and split != "val":
            frame = {"timestamp": pd.to_datetime(d.timestamps[idx])}
            for k, n in enumerate(d.target_names):
                frame[f"observed_{n}"] = truth[:, k]
                frame[f"predicted_{n}"] = pred[:, k]
                frame[f"error_{n}"] = pred[:, k] - truth[:, k]
            pd.DataFrame(frame).to_csv(rd / f"predictions_{split}.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(rd / "predictions.npz", **store)
    hist = [t.history for t in tm_list if t.history is not None]
    if hist:
        pd.concat(hist, ignore_index=True).to_csv(rd / "training_history.csv", index=False, encoding="utf-8-sig")
    write_json(rd / "run_info.json", {"model": name, "seed": seed, "config": cfg, "config_source": source,
                                      "protocol": protocol, "training_seconds": float(sum(t.seconds for t in tm_list)),
                                      "parameter_or_tree_count": int(sum(t.params for t in tm_list)),
                                      "epochs_run": [int(h.epoch.max()) for h in hist] if hist else None})
    m = pd.DataFrame(rows)
    m.insert(2, "seed", seed)
    m.to_csv(rd / "metrics.csv", index=False, encoding="utf-8-sig")              # written last = run complete
    return m


def stage_final(ctx: Context, models: list, knee: str, seeds: list | None) -> None:
    m = _models()
    d, wd = ctx.data, ctx.wd
    prot = ctx.final_protocol()
    seeds = seeds or prot["seeds"]
    train_prot = {k: prot[k] for k in ("epochs", "patience", "batch_size", "max_train_samples", "max_val_samples")}
    splits = ["val"] + d.test_splits
    for seed in seeds:                                   # seeds outer: every model gets some seeds early
        for key in models:
            cfg, source = knee_config(ctx, key, knee)
            cfg = ctx.quick_cfg(key, cfg)
            rd = ctx.out / "final" / key / f"seed_{seed:02d}"
            if (rd / "metrics.csv").exists():
                continue
            rd.mkdir(parents=True, exist_ok=True)
            ctx.log(f"[final] {LABELS[key]} seed {seed}: {json.dumps(cfg)} ({source})")
            tm = m.train_model(key, d, wd, cfg, train_prot, seed, model_options=ctx.model_options(key))
            preds = {s: tm.predict(d, wd, s, 0) for s in splits}
            if prot.get("save_models", True):
                tm.save(str(rd / "model"))
            met = save_run(rd, ctx, LABELS[key], [tm], preds, seed, {**ctx.model_options(key), **cfg}, train_prot,
                           source, prot.get("save_prediction_csv", True))
            avg = met[met.target == "average"].set_index("scenario").NMSE
            ctx.log(f"[final] {LABELS[key]} seed {seed} done in {tm.seconds:.0f} s | average NMSE "
                    + ", ".join(f"{s} {avg[s]:.4f}" for s in d.test_splits))
            del tm
            _free_gpu()


# ====================================================================== stage: ablation (Table 9)
ABLATION_LABELS = {"A_default_equal": "A: default structure + equal weights (no MOBO)",
                   "B_knee_equal": "B: knee structure + equal weights",
                   "C_independent": "C: independent single-task TimeXers (no sharing)",
                   "D_mobo": "D: MOBO-TimeXer (knee structure + knee weights)"}


def stage_ablation(ctx: Context, knee: str, seeds: list | None) -> None:
    m = _models()
    d, wd = ctx.data, ctx.wd
    ab = ctx.cfg["ablation"]
    prot = ctx.final_protocol()
    seeds = seeds or (QUICK["ablation_seeds"] if ctx.quick else ab["seeds"])
    train_prot = {k: prot[k] for k in ("epochs", "patience", "batch_size", "max_train_samples", "max_val_samples")}
    opts = ctx.model_options("timexer") if ab["network"] == "revised" else {"ar_bypass": False, "window_norm": False}
    knee_cfg, source = knee_config(ctx, "timexer", knee)
    equal = [1.0] * d.n_targets
    variants = {"A_default_equal": {**m.DEFAULT_TIMEXER_CONFIG, "task_weights": equal},
                "B_knee_equal": {**knee_cfg, "task_weights": equal},
                "C_independent": {k: v for k, v in knee_cfg.items() if k != "task_weights"},
                "D_mobo": knee_cfg}
    splits = ["val"] + d.test_splits
    for seed in seeds:
        for v in ab["variants"]:
            rd = ctx.out / "ablation" / v / f"seed_{seed:02d}"
            if (rd / "metrics.csv").exists():
                continue
            src = ctx.out / "final" / "timexer" / f"seed_{seed:02d}"
            if v == "D_mobo" and ab["network"] == "revised" and (src / "metrics.csv").exists() and not ctx.quick:
                shutil.copytree(src, rd, dirs_exist_ok=True)                 # identical to the final run
                met = pd.read_csv(rd / "metrics.csv", encoding="utf-8-sig")
                met["model"] = ABLATION_LABELS[v]
                met.to_csv(rd / "metrics.csv", index=False, encoding="utf-8-sig")
                continue
            rd.mkdir(parents=True, exist_ok=True)
            cfg = variants[v]
            ctx.log(f"[ablation] {v} seed {seed} ({ab['network']} network)")
            if v == "C_independent":
                tms = m.train_independent_timexer(d, wd, cfg, train_prot, seed, model_options=opts)
                preds = {}
                for s in splits:
                    cols = [t.predict(d, wd, s, 0) for t in tms]
                    preds[s] = (cols[0][0], np.column_stack([c[1][:, 0] for c in cols]))
            else:
                tms = [m.train_model("timexer", d, wd, cfg, train_prot, seed, model_options=opts)]
                preds = {s: tms[0].predict(d, wd, s, 0) for s in splits}
            save_run(rd, ctx, ABLATION_LABELS[v], tms, preds, seed, {**opts, **cfg}, train_prot, source, False)
            del tms
            _free_gpu()


# ====================================================================== stage: report
def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment of a family of p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def collect(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*/seed_*/metrics.csv"))
    return pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True) if files else pd.DataFrame()


def seed_summary(per_seed: pd.DataFrame, metrics=("NMSE", "R2", "MSE", "RMSE", "MAE", "sMAPE_pct")) -> pd.DataFrame:
    from scipy import stats
    rows = []
    for (mdl, sc, tg), g in per_seed.groupby(["model", "scenario", "target"], sort=False):
        for met in metrics:
            x = g[met].to_numpy(float)
            n = len(x)
            sd = x.std(ddof=1) if n > 1 else np.nan
            h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
            rows.append({"model": mdl, "scenario": sc, "target": tg, "metric": met, "n_seeds": n, "mean": x.mean(),
                         "sd": sd, "CV_pct": 100 * sd / abs(x.mean()) if n > 1 and x.mean() else np.nan,
                         "CI95_lo": x.mean() - h, "CI95_hi": x.mean() + h, "min": x.min(), "max": x.max()})
    return pd.DataFrame(rows)


def significance(per_seed: pd.DataFrame, ref: str, scenarios: list) -> pd.DataFrame:
    """Welch t-test (and Mann-Whitney U) of the reference model against every other model on the seed-level
    metrics; Holm correction over the comparisons within each scenario x target x metric (Section 2.4)."""
    from scipy import stats
    rows = []
    others = [mdl for mdl in per_seed.model.unique() if mdl != ref]
    for sc in scenarios:
        for tg in per_seed.target.unique():
            for met in ("NMSE", "R2"):
                fam = []
                x = per_seed[(per_seed.model == ref) & (per_seed.scenario == sc) & (per_seed.target == tg)][met].to_numpy(float)
                for o in others:
                    y = per_seed[(per_seed.model == o) & (per_seed.scenario == sc) & (per_seed.target == tg)][met].to_numpy(float)
                    if len(x) > 1 and len(y) > 1:
                        t, p = stats.ttest_ind(x, y, equal_var=False)
                        u = stats.mannwhitneyu(x, y, alternative="two-sided")
                        sp = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
                        fam.append({"scenario": sc, "target": tg, "metric": met, "model": ref, "versus": o,
                                    "mean_model": x.mean(), "mean_versus": y.mean(),
                                    "relative_difference_pct": 100 * (x.mean() - y.mean()) / abs(y.mean()) if y.mean() else np.nan,
                                    "welch_t": t, "p_welch": p, "p_mannwhitney": u.pvalue,
                                    "cohens_d": (x.mean() - y.mean()) / sp if sp > 0 else np.nan})
                if fam:
                    adj = holm([r["p_welch"] for r in fam])
                    for r, a in zip(fam, adj):
                        r["p_welch_holm"] = a
                    rows += fam
    return pd.DataFrame(rows)


def fmt(m, s, digits=4):
    return f"{m:.{digits}f} ± {s:.{digits}f}" if np.isfinite(s) else f"{m:.{digits}f}"


def main_tables(summ: pd.DataFrame, d: DP.MotionData, models: list) -> dict:
    """Paper-style Tables 7/8: mean ± SD of MSE, MAE, NMSE and R2 for every target and the average."""
    out = {}
    for sc in d.test_splits:
        rows = []
        for mdl in models:
            for met in ("MSE", "MAE", "NMSE", "R2"):
                row = {"model": mdl, "metric": met}
                for tg in d.target_names + ["average"]:
                    g = summ[(summ.model == mdl) & (summ.scenario == sc) & (summ.target == tg) & (summ.metric == met)]
                    row[tg] = fmt(g["mean"].iloc[0], g["sd"].iloc[0]) if len(g) else ""
                rows.append(row)
        out[sc] = pd.DataFrame(rows)
    return out


def ensemble_metrics(ctx: Context, root: Path, models: list) -> pd.DataFrame:
    d = ctx.data
    rows = []
    for key in models:
        runs = sorted((root / key).glob("seed_*/predictions.npz"))
        if not runs:
            continue
        zs = [np.load(r) for r in runs]
        for sc in ["val"] + d.test_splits:
            if f"{sc}_pred" not in zs[0]:
                continue
            idx = zs[0][f"{sc}_index"]
            pred = np.mean([z[f"{sc}_pred"] for z in zs], axis=0)
            rows += [{**r, "n_seeds": len(zs)} for r in _models_free_metrics(
                d.raw_features[idx, : d.n_targets], pred, d, sc, LABELS.get(key, key))]
    return pd.DataFrame(rows)


def stage_report(ctx: Context, models: list) -> None:
    d = ctx.data
    out = ctx.out / "report"
    out.mkdir(parents=True, exist_ok=True)
    ref = LABELS[ctx.cfg["reference_model"]]
    per_seed = collect(ctx.out / "final")
    tabs = {}
    if per_seed.empty:
        ctx.log("[report] no final runs found")
    else:
        order = [LABELS[k] for k in models if LABELS[k] in set(per_seed.model)]
        per_seed = per_seed[per_seed.model.isin(order)].copy()
        per_seed["_o"] = per_seed.model.map({mdl: i for i, mdl in enumerate(order)})
        per_seed = per_seed.sort_values(["_o", "seed"], kind="stable").drop(columns="_o").reset_index(drop=True)
        summ = seed_summary(per_seed)
        sig = significance(per_seed, ref, d.test_splits) if ref in order else pd.DataFrame()
        ens = ensemble_metrics(ctx, ctx.out / "final", models)
        pers = persistence_metrics(ctx)
        tabs.update({"seed_metrics": per_seed, "seed_summary": summ, "significance": sig, "ensemble_metrics": ens,
                     "persistence": pers})
        for sc, t in main_tables(summ, d, order).items():
            tabs[f"table_{sc}"] = t
        # improvement of the reference model over the best benchmark and persistence (average NMSE)
        imp = []
        for sc in d.test_splits:
            g = summ[(summ.scenario == sc) & (summ.target == "average") & (summ.metric == "NMSE")].set_index("model")
            if ref not in g.index:
                continue
            bench = g.drop(index=ref)
            p = pers[(pers.scenario == sc) & (pers.target == "average")].NMSE.iloc[0]
            row = {"scenario": sc, "reference_model": ref, "reference_NMSE_mean": g.loc[ref, "mean"],
                   "reference_NMSE_sd": g.loc[ref, "sd"], "persistence_NMSE": p,
                   "reduction_vs_persistence_pct": 100 * (1 - g.loc[ref, "mean"] / p)}
            if len(bench):
                b = bench["mean"].idxmin()
                row.update({"best_benchmark": b, "best_benchmark_NMSE": bench.loc[b, "mean"],
                            "reduction_vs_best_benchmark_pct": 100 * (1 - g.loc[ref, "mean"] / bench.loc[b, "mean"])})
            imp.append(row)
        tabs["improvement"] = pd.DataFrame(imp)
        # computational cost (Table 9, first columns)
        cost = []
        for key in models:
            infos = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((ctx.out / "final" / key).glob("seed_*/run_info.json"))]
            if infos:
                cost.append({"model": LABELS[key], "parameter_or_tree_count": infos[0]["parameter_or_tree_count"],
                             "training_seconds_mean": np.mean([i["training_seconds"] for i in infos]),
                             "training_seconds_sd": np.std([i["training_seconds"] for i in infos], ddof=1) if len(infos) > 1 else np.nan,
                             "epochs_run_mean": np.mean([np.mean(i["epochs_run"]) for i in infos]) if infos[0]["epochs_run"] else np.nan,
                             "seeds": len(infos)})
        tabs["cost"] = pd.DataFrame(cost)
    # MOBO summaries (Table 5, Table 6)
    mobo_rows, knee_rows = [], []
    for key in models:
        f = ctx.out / "mobo" / key / "mobo_trials.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f, encoding="utf-8-sig")
        b = json.loads((ctx.out / "mobo" / key / "best_config.json").read_text(encoding="utf-8"))
        it = pd.read_csv(ctx.out / "mobo" / key / "mobo_iteration_summary.csv", encoding="utf-8-sig")
        first, best = t.objective_mean.iloc[0], t.objective_mean.min()
        hv0, hv1 = it.hypervolume_fixed_reference.iloc[0], it.hypervolume_fixed_reference.iloc[-1]
        mobo_rows.append({"model": LABELS[key], "evaluations": len(t), "initial_NMSE": first, "best_NMSE": best,
                          "reduction_pct": 100 * (1 - best / first), "best_found_at_evaluation": int(t.objective_mean.idxmin()) + 1,
                          "pareto_size": int(t.pareto.sum()), "knee_trial": b["trial"],
                          "hypervolume_gain_pct": 100 * (hv1 / hv0 - 1) if hv0 > 0 else np.nan,
                          "knee_L1": b.get("knee_with_other_distances", {}).get("L1"),
                          "knee_Chebyshev": b.get("knee_with_other_distances", {}).get("Chebyshev")})
        knee_rows.append({"model": LABELS[key], "knee_trial": b["trial"],
                          "configuration": ", ".join(f"{k}={_short(v)}" for k, v in b["best_config"].items())})
    if mobo_rows:
        tabs["mobo_summary"] = pd.DataFrame(mobo_rows)
        tabs["knee_configs"] = pd.DataFrame(knee_rows)
    # ablation
    ab = collect(ctx.out / "ablation")
    if not ab.empty:
        tabs["ablation_seed_metrics"] = ab
        ab_s = seed_summary(ab, metrics=("NMSE", "R2"))
        tabs["ablation_summary"] = ab_s
        dlab = ABLATION_LABELS["D_mobo"]
        if dlab in set(ab.model):
            tabs["ablation_significance"] = significance(ab, dlab, d.test_splits)
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(out / "results_summary.xlsx", engine="openpyxl") as w:
            pd.DataFrame({"content": [
                f"data: {d.profile['source']}", f"splits: {json.dumps(d.profile['split_time_range'])}",
                f"windows: {json.dumps(d.profile['split_windows'])}",
                "NMSE = MSE / training-segment variance (Eq. 22); average = macro average over the targets",
                "seed statistics: mean, SD, CV, t-based 95% CI; significance: Welch t-test, Holm-adjusted within "
                "each scenario x target x metric", "ensemble = mean of the seed predictions"]}).to_excel(
                w, sheet_name="readme", index=False)
            for name, t in tabs.items():
                t.to_excel(w, sheet_name=name[:31], index=False)
    except Exception as e:  # noqa: BLE001
        ctx.log(f"[report] Excel workbook skipped: {e}")
    try:
        figures(ctx, models, tabs, out)
    except Exception as e:  # noqa: BLE001
        ctx.log(f"[report] figures skipped: {type(e).__name__}: {e}")
    if "improvement" in tabs:
        for _, r in tabs["improvement"].iterrows():
            ctx.log(f"[report] {r.scenario}: {ref} average NMSE {r.reference_NMSE_mean:.4f} ± {r.reference_NMSE_sd:.4f}; "
                    + (f"best benchmark {r.best_benchmark} {r.best_benchmark_NMSE:.4f} "
                       f"(reference {-r.reduction_vs_best_benchmark_pct:+.1f}% vs best benchmark); " if "best_benchmark" in r and isinstance(r.best_benchmark, str) else "")
                    + f"persistence {r.persistence_NMSE:.4f}")
    ctx.log(f"[report] written to {out}")


def _short(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "(" + ", ".join(_short(x) for x in v) + ")"
    return str(v)


def figures(ctx: Context, models: list, tabs: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = ctx.data
    rep = ctx.cfg["report"]
    dpi = rep["figure_dpi"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(zip([LABELS[k] for k in ALL_MODELS], ["#C0392B", "#2E86C1", "#7F8C8D", "#27AE60", "#8E44AD", "#D68910"]))
    # MOBO convergence (Fig. 8)
    keys = [k for k in models if (ctx.out / "mobo" / k / "mobo_trials.csv").exists()]
    if keys:
        fig, axes = plt.subplots(int(math.ceil(len(keys) / 3)), min(3, len(keys)), figsize=(10, 2.8 * math.ceil(len(keys) / 3)), squeeze=False)
        for ax, k in zip(axes.flat, keys):
            t = pd.read_csv(ctx.out / "mobo" / k / "mobo_trials.csv", encoding="utf-8-sig")
            b = json.loads((ctx.out / "mobo" / k / "best_config.json").read_text(encoding="utf-8"))
            ev = np.arange(1, len(t) + 1)
            ax.plot(ev, t.objective_mean, "o", ms=3, color="0.55", label="evaluation")
            ax.plot(ev, t.objective_mean.cummin(), "-", color=colors.get(LABELS[k], "k"), label="best so far")
            ax.scatter(ev[t.pareto], t.objective_mean[t.pareto], s=40, facecolors="none", edgecolors="k", label="Pareto")
            ax.axvline(b["trial"] + 1, ls="--", lw=0.8, color="#C0392B")
            ax.set(title=LABELS[k], xlabel="evaluation", ylabel="mean validation NMSE")
        axes.flat[0].legend(frameon=False, fontsize=7)
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "Fig_MOBO_convergence.png", dpi=dpi)
        plt.close(fig)
    # hyper-parameter landscape of the reference model (Fig. 9)
    rk = ctx.cfg["reference_model"]
    f = ctx.out / "mobo" / rk / "mobo_trials.csv"
    if f.exists():
        t = pd.read_csv(f, encoding="utf-8-sig")
        cfgs = pd.DataFrame([json.loads(c) for c in t.config_json])
        cols = [c for c in cfgs.columns if c != "task_weights"]
        fig, axes = plt.subplots(int(math.ceil(len(cols) / 4)), 4, figsize=(11, 2.6 * math.ceil(len(cols) / 4)), squeeze=False)
        for ax, c in zip(axes.flat, cols):
            ax.scatter(cfgs[c], t.objective_mean, s=14, c=np.arange(len(t)), cmap="viridis")
            if c == "learning_rate":
                ax.set_xscale("log")
            ax.set(xlabel=c, ylabel="mean val. NMSE")
        for ax in list(axes.flat)[len(cols):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "Fig_hyperparameter_landscape.png", dpi=dpi)
        plt.close(fig)
    # target-wise NMSE and R2 (Figs. 10, 13)
    if "seed_summary" in tabs:
        s = tabs["seed_summary"]
        mods = list(dict.fromkeys(s.model))
        for sc in d.test_splits:
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
            for ax, met in zip(axes, ("NMSE", "R2")):
                x = np.arange(d.n_targets + 1)
                wbar = 0.8 / len(mods)
                for i, mdl in enumerate(mods):
                    g = s[(s.model == mdl) & (s.scenario == sc) & (s.metric == met)].set_index("target").reindex(d.target_names + ["average"])
                    ax.bar(x + (i - (len(mods) - 1) / 2) * wbar, g["mean"], wbar, yerr=g["sd"], capsize=2,
                           label=mdl, color=colors.get(mdl))
                ax.set_xticks(x, d.target_names + ["average"])
                ax.set_ylabel(met)
                ax.set_title(f"{sc}: {met} (mean ± SD over seeds)")
            axes[0].legend(frameon=False, fontsize=7)
            fig.tight_layout()
            fig.savefig(out / f"Fig_targetwise_{sc}.png", dpi=dpi)
            plt.close(fig)
    # time-domain comparison of the seed-ensemble predictions (Figs. 11, 12, 14, 15)
    n_win, pos = int(rep["plot_window_samples"]), float(rep["plot_window_position"])
    for sc in d.test_splits:
        preds = {}
        for k in models:
            runs = sorted((ctx.out / "final" / k).glob("seed_*/predictions.npz"))
            if runs:
                zs = [np.load(r) for r in runs]
                preds[LABELS[k]] = (zs[0][f"{sc}_index"], np.mean([z[f"{sc}_pred"] for z in zs], axis=0))
        if not preds:
            continue
        idx0 = next(iter(preds.values()))[0]
        a = int(max(0, min(len(idx0) - n_win, pos * len(idx0) - n_win / 2)))
        sel = idx0[a: a + n_win]
        tsec = (d.timestamps[sel] - d.timestamps[sel[0]]).astype("timedelta64[ms]").astype(float) / 1000
        fig, axes = plt.subplots(d.n_targets, 1, figsize=(8, 1.9 * d.n_targets), sharex=True, squeeze=False)
        for k, ax in enumerate(axes[:, 0]):
            ax.plot(tsec, d.raw_features[sel, k], color="k", lw=1.2, label="measured")
            for mdl, (idx, p) in preds.items():
                ax.plot(tsec, p[a: a + n_win, k], lw=0.9, label=mdl, color=colors.get(mdl))
            ax.set_ylabel(f"{d.target_names[k]} ({d.target_units[k]})")
        axes[0, 0].legend(frameon=False, fontsize=7, ncol=4)
        axes[-1, 0].set_xlabel(f"time (s) from {str(d.timestamps[sel[0]])[:19]}")
        fig.tight_layout()
        fig.savefig(out / f"Fig_timeseries_{sc}.png", dpi=dpi)
        plt.close(fig)


# ====================================================================== CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="MOBO-TimeXer: full reproduction pipeline")
    ap.add_argument("--config", required=True, help="JSON configuration file (see configs/)")
    ap.add_argument("--stage", default="all", choices=["all", "audit", "analysis", "search", "final", "ablation", "report"])
    ap.add_argument("--models", default=None, help="comma-separated subset, e.g. timexer,xgboost")
    ap.add_argument("--seeds", default=None, help="seeds for final/ablation, e.g. 0-9 or 0,3,5")
    ap.add_argument("--knee", default="search", help="'search' (own MOBO), 'paper' (Table 6) or a JSON file {model: config}")
    ap.add_argument("--quick", action="store_true", help="tiny budgets for a smoke test (results in <output>_quick)")
    ap.add_argument("--output", default=None, help="override output_dir")
    a = ap.parse_args()
    cfg_path = Path(a.config).resolve()
    user = json.loads(cfg_path.read_text(encoding="utf-8-sig"))      # tolerate a BOM (Windows Notepad)
    cfg = deep_update(DEFAULTS, user)
    if a.output:
        cfg["output_dir"] = a.output
    models = a.models.split(",") if a.models else list(cfg["models"])
    bad = [m for m in models if m not in ALL_MODELS]
    if bad:
        raise SystemExit(f"unknown model(s) {bad}; choose from {ALL_MODELS}")
    ctx = Context(cfg, cfg_path.parent, a.quick)
    seeds = parse_seeds(a.seeds) if a.seeds else None
    ctx.log(f"===== stage {a.stage} | models {models} | knee {a.knee} | quick {a.quick} | output {ctx.out}")
    if any(s in (a.stage,) for s in ("all", "search", "final", "ablation")):
        dev = _models().DEVICE
        ctx.log(f"device: {dev}")
    t0 = time.perf_counter()
    if a.stage in ("all", "audit"):
        stage_audit(ctx)
    if a.stage in ("all", "analysis"):
        stage_analysis(ctx)
    if a.stage in ("all", "search"):
        stage_search(ctx, models)
    if a.stage in ("all", "final"):
        stage_final(ctx, models, a.knee, seeds)
    if a.stage == "ablation" or (a.stage == "all" and cfg["ablation"]["enabled"]):
        stage_ablation(ctx, a.knee, seeds)
    if a.stage in ("all", "report"):
        stage_report(ctx, models)
    ctx.log(f"===== finished in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
