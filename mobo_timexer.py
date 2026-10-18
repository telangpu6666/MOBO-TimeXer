"""MOBO-TimeXer in one file: data pipeline, TimeXer, 5-seed batch-ParEGO MOBO, knee point, multi-seed training, test metrics.

    python mobo_timexer.py --config configs/demo_config.json --quick     # smoke test
    python mobo_timexer.py --config configs/demo_config.json             # search + final training
    python mobo_timexer.py --config my.json --stage final --knee paper   # skip the search
Only the "data" block of the configuration is required (format: DATASET_FORMAT.md).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULTS = {
    "output_dir": "results",
    "model_options": {"ar_bypass": True, "window_norm": True, "window_norm_eps": 0.02},
    "mobo": {"seeds": [0, 1, 2, 3, 4], "n_init": 10, "n_iterations": 10, "batch_candidates": 2, "candidate_pool": 4096,
             "min_distance": 0.12, "rho": 0.05, "xi": 1e-3, "sobol_seed": 2026,
             "epochs": 8, "patience": 3, "batch_size": 1024, "max_train_samples": 60000, "max_val_samples": 20000},
    "final": {"seeds": list(range(10)), "epochs": 60, "patience": 15, "batch_size": 1024,
              "max_train_samples": 0, "max_val_samples": 0},
}
QUICK = {"mobo": {"n_init": 3, "n_iterations": 1, "epochs": 1, "patience": 1, "max_train_samples": 3000,
                  "max_val_samples": 1000},
         "final": {"seeds": [0, 1], "epochs": 2, "patience": 1, "max_train_samples": 5000, "max_val_samples": 2000}}

SEARCH_SPACE = [
    ("d_model", "choice", [32, 48, 64, 96]),
    ("num_heads", "choice", [2, 4]),
    ("ff_dim", "choice", [64, 128, 256]),
    ("num_layers", "choice", [1, 2, 3]),
    ("dropout", "uniform", (0.0, 0.20)),
    ("patch_len", "choice", [1, 2, 4]),
    ("head_units", "choice", [16, 32, 64]),
    ("learning_rate", "log", (3e-4, 3e-3)),
]
PAPER_KNEE = {"d_model": 32, "num_heads": 4, "ff_dim": 256, "num_layers": 3, "dropout": 0.16967092417631452,
              "patch_len": 4, "head_units": 32, "learning_rate": 0.0028788475595118573,
              "task_weights": [0.3121030773375151, 2.73929283782137, 0.2651447507096903, 0.683459334131425]}


# ---------------------------------------------------------------- data
class Data:
    """Clean, split by date (or ratio), standardise on the training rows, gap-aware windows X[t-h-L+1:t-h] -> y[t]."""

    def __init__(self, c: dict, base: Path):
        path = Path(c["path"]) if Path(c["path"]).is_absolute() else (base / c["path"]).resolve()
        self.L, self.h = int(c.get("lookback", 16)), int(c.get("horizon", 1))
        tcols = list(c["target_columns"])
        self.names = list(c.get("target_names") or tcols)
        self.units = list(c.get("target_units") or [""] * len(tcols))
        rules = c.get("derived_columns", [])
        d_in = [x for r in rules for x in r["inputs"]]
        frame = self._read(path, c)
        exo = c.get("exogenous_columns", "auto")
        if exo == "auto":
            skip = {c["time_column"], *tcols, *c.get("exclude_columns", [])}
            exo = [x for x in frame.columns if x not in skip and pd.api.types.is_numeric_dtype(frame[x])]
            exo += [r["name"] for r in rules if r["name"] not in exo and r["name"] not in skip]
        cols = tcols + [x for x in exo if x not in tcols]
        ts = frame[c["time_column"]]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(ts, format="mixed", errors="coerce")
        num = frame[list(dict.fromkeys([x for x in cols if x in frame.columns] + d_in))].apply(pd.to_numeric, errors="coerce")
        ok = ts.notna().to_numpy() & np.isfinite(num.to_numpy(dtype=np.float64)).all(axis=1)
        ts, num = ts[ok].reset_index(drop=True), num[ok].reset_index(drop=True)
        keep = ~ts.duplicated(keep="first")
        ts, num = ts[keep].reset_index(drop=True), num[keep].reset_index(drop=True)
        if not ts.is_monotonic_increasing:
            o = np.argsort(ts.to_numpy(), kind="stable")
            ts, num = ts.iloc[o].reset_index(drop=True), num.iloc[o].reset_index(drop=True)
        for r in rules:
            x = num[list(r["inputs"])].to_numpy(dtype=np.float64)
            v = {"hypot": lambda: np.hypot(x[:, 0], x[:, 1]) if x.shape[1] == 2 else np.sqrt((x ** 2).sum(1)),
                 "linear": lambda: float(r.get("scale", 1.0)) * x[:, 0] + float(r.get("bias", 0.0)),
                 "sum": lambda: x @ np.asarray(r.get("weights", [1.0] * x.shape[1]), dtype=np.float64)}[r["op"]]()
            m = np.ones(len(num), dtype=bool)
            if r.get("from"):
                m &= (ts >= pd.Timestamp(r["from"])).to_numpy()
            if r.get("to"):
                m &= (ts < pd.Timestamp(r["to"])).to_numpy()
            if r["name"] in num.columns and not m.all():
                num.loc[m, r["name"]] = v[m]
            else:
                num[r["name"]] = v
        self.ts = ts.to_numpy(dtype="datetime64[ns]")
        self.raw = num[cols].to_numpy(dtype=np.float32, copy=True)
        self.nt, self.exo_names = len(tcols), cols[len(tcols):]
        n = len(self.ts)
        if c.get("splits"):
            pos = lambda t, d: d if t in (None, "", "None") else int(np.searchsorted(self.ts, np.datetime64(pd.Timestamp(t)), side="left"))
            rows = {k: (pos(a, 0), pos(b, n)) for k, (a, b) in c["splits"].items()}
        else:
            r_, keys, rows, s0 = c["split_ratios"], list(c["split_ratios"]), {}, 0
            for k in keys:
                s1 = n if k == keys[-1] else s0 + int(n * r_[k] / sum(r_.values()))
                rows[k], s0 = (s0, s1), s1
        a, b = rows["train"]
        self.mean = np.nanmean(self.raw[a:b], axis=0).astype(np.float32)
        self.scale = np.nanstd(self.raw[a:b], axis=0).astype(np.float32)
        self.scale[self.scale < 1e-8] = 1.0
        self.feat = ((self.raw - self.mean) / self.scale).astype(np.float32)
        ms = self.ts.astype("datetime64[ms]").astype(np.int64)
        steps = np.diff(ms)
        step = int(round(float(c["sample_seconds"]) * 1000)) if c.get("sample_seconds") else int(np.median(steps))
        seg = np.cumsum(np.r_[True, steps != step])

        def win(a, b):
            idx = np.arange(max(a, self.L + self.h - 1), b, dtype=np.int64)
            hs = idx - self.h - self.L + 1
            return idx[(hs >= 0) & (seg[idx] == seg[hs])]

        self.idx = {k: win(*v) for k, v in rows.items()}
        self.tests = [k for k in self.idx if k not in ("train", "val")]
        self.profile = {"rows": n, "start": str(self.ts[0])[:19], "end": str(self.ts[-1])[:19],
                        "recording_gaps": int((steps != step).sum()),
                        "split_time_range": {k: [str(self.ts[a])[:19], str(self.ts[max(b - 1, a)])[:19]] for k, (a, b) in rows.items()},
                        "split_windows": {k: int(len(v)) for k, v in self.idx.items()}}
        self.gfeat = torch.from_numpy(self.feat).to(DEVICE)
        self.offsets = torch.arange(self.L, device=DEVICE)

    @staticmethod
    def _read(path, c):
        s = path.suffix.lower()
        if s in (".csv", ".txt", ".dat", ".tsv"):
            return pd.read_csv(path, encoding=c.get("encoding", "utf-8"), sep="\t" if s == ".tsv" else c.get("sep", ","),
                               low_memory=False)
        if s == ".parquet":
            return pd.read_parquet(path)
        if s in (".xlsx", ".xls"):
            return pd.read_excel(path, sheet_name=c.get("sheet_name", 0))
        return pd.read_feather(path)

    def index(self, split, maximum=0):
        i = self.idx[split]
        if maximum and 0 < maximum < len(i):
            i = i[np.linspace(0, len(i) - 1, num=maximum, dtype=np.int64)]
        return torch.from_numpy(i.copy()).to(DEVICE)

    def windows(self, idx):
        return self.gfeat[(idx - self.h - self.L + 1)[:, None] + self.offsets[None, :]]

    def targets(self, idx):
        return self.gfeat[idx][:, : self.nt]

    def to_physical(self, y):
        return y * self.scale[: self.nt] + self.mean[: self.nt]


# ---------------------------------------------------------------- model
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class MHA(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        if d % h:
            raise ValueError("d_model must be divisible by num_heads")
        self.h, self.dk = h, d // h
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def keras_init(self):
        d, h, dk = self.h * self.dk, self.h, self.dk
        a_qkv, a_out = math.sqrt(6.0 / (h * d + dk * d)), math.sqrt(6.0 / (dk * h + d * h))
        with torch.no_grad():
            for lin in (self.q, self.k, self.v):
                lin.weight.uniform_(-a_qkv, a_qkv)
            self.o.weight.uniform_(-a_out, a_out)

    def forward(self, xq, xkv):
        B, T, _ = xq.shape
        S = xkv.shape[1]
        q = self.q(xq).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(xkv).view(B, S, self.h, self.dk).transpose(1, 2)
        v = self.v(xkv).view(B, S, self.h, self.dk).transpose(1, 2)
        w = self.drop((q @ k.transpose(-1, -2) / math.sqrt(self.dk)).softmax(-1))
        return self.o((w @ v).transpose(1, 2).reshape(B, T, self.h * self.dk))


class Heads(nn.Module):
    def __init__(self, in_dim, n_targets, units):
        super().__init__()
        self.hidden = nn.ModuleList(nn.Linear(in_dim, units) for _ in range(n_targets))
        self.out = nn.ModuleList(nn.Linear(units, 1) for _ in range(n_targets))

    def forward(self, ctx, last):
        return last + torch.cat([o(F.gelu(h(ctx))) for h, o in zip(self.hidden, self.out)], dim=1)


class TimeXer(nn.Module):
    """Shared encoder (patch + global tokens, self-attn -> cross-attn to exogenous variate tokens -> FFN),
    task-specific residual heads, window-wise amplitude normalisation and linear AR bypass."""

    def __init__(self, L, n_features, nt=4, *, d_model=64, num_heads=4, ff_dim=128, num_layers=2, dropout=0.05,
                 patch_len=2, head_units=32, ar_bypass=True, window_norm=True, window_norm_eps=0.02, **_):
        super().__init__()
        if L % patch_len:
            raise ValueError("lookback must be divisible by patch_len")
        self.nt = nt
        self.patch = nn.Conv1d(nt, d_model, kernel_size=patch_len, stride=patch_len)
        self.pos = nn.Parameter(torch.empty(L // patch_len, d_model))
        self.exo = nn.Linear(L, d_model)
        self.blocks = nn.ModuleList(nn.ModuleDict(dict(
            sa=MHA(d_model, num_heads, dropout), n1=nn.LayerNorm(d_model, eps=1e-6),
            ca=MHA(d_model, num_heads, dropout), n2=nn.LayerNorm(d_model, eps=1e-6),
            ff1=nn.Linear(d_model, ff_dim), ff2=nn.Linear(ff_dim, d_model), n3=nn.LayerNorm(d_model, eps=1e-6)))
            for _ in range(num_layers))
        self.drop = nn.Dropout(dropout)
        self.heads = Heads(2 * d_model + nt, nt, head_units)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.pos)
        for b in self.blocks:
            b["sa"].keras_init()
            b["ca"].keras_init()
        self.use_ar, self.use_rev, self.eps = bool(ar_bypass), bool(window_norm), float(window_norm_eps)
        if self.use_ar:
            self.ar = nn.Linear(nt * L, nt)
            nn.init.zeros_(self.ar.weight)
            nn.init.zeros_(self.ar.bias)

    def forward(self, x):
        raw, exo = x[:, :, : self.nt], x[:, :, self.nt:]
        endo = raw - raw[:, -1:, :]
        if self.use_rev:
            s = torch.sqrt((endo - endo.mean(1, keepdim=True)).pow(2).mean(1, keepdim=True) + self.eps ** 2)
            z = endo / s
        else:
            z = endo
        p = self.patch(z.transpose(1, 2)).transpose(1, 2) + self.pos
        e = torch.cat([p.mean(1, keepdim=True), p], dim=1)
        xt = self.exo(exo.transpose(1, 2))
        for b in self.blocks:
            e = b["n1"](e + self.drop(b["sa"](e, e)))
            e = b["n2"](e + self.drop(b["ca"](e, xt)))
            e = b["n3"](e + b["ff2"](self.drop(F.gelu(b["ff1"](e)))))
        inc = self.heads(torch.cat([e[:, 0], e[:, -1], z[:, -1]], dim=1), torch.zeros_like(raw[:, -1]))
        if self.use_rev:
            inc = inc * s[:, 0, :]
        if self.use_ar:
            inc = inc + self.ar(endo.flatten(1))
        return raw[:, -1] + inc


@torch.no_grad()
def predict(model, data, idx, batch=8192):
    model.eval()
    return torch.cat([model(data.windows(idx[s: s + batch])) for s in range(0, len(idx), batch)])


def train(data, cfg, prot, seed, options, log=None):
    """Weighted multi-task MSE (Eq. 11), Adam, early stopping with best weights, LR x0.5 on plateau.
    On GPU each step is replayed as a CUDA graph (same mathematics, faster)."""
    seed_everything(seed)
    mcfg = {k: v for k, v in {**options, **cfg}.items() if k not in ("learning_rate", "task_weights")}
    model = TimeXer(data.L, data.feat.shape[1], data.nt, **mcfg).to(DEVICE)
    w = np.asarray(cfg.get("task_weights", [1.0] * data.nt), dtype=np.float32)
    w = torch.tensor(w / w.mean(), device=DEVICE)
    lr0 = float(cfg.get("learning_rate", 1e-3))
    graph_on = DEVICE.type == "cuda"
    if graph_on:
        opt = torch.optim.Adam(model.parameters(), lr=torch.tensor(lr0, device=DEVICE), eps=1e-7, capturable=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr0, eps=1e-7)
    tr, va = data.index("train", prot["max_train_samples"]), data.index("val", prot["max_val_samples"])
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    bs, n = int(prot["batch_size"]), len(tr)
    loss_of = lambda i: ((model(data.windows(i)) - data.targets(i)) ** 2 * w).mean()

    def step(i):
        loss = loss_of(i)
        opt.zero_grad(set_to_none=False)
        loss.backward()
        opt.step()
        return loss.detach()

    graph = None
    if graph_on and n >= bs:
        sidx = tr[:bs].clone()
        p0 = [p.detach().clone() for p in model.parameters()]
        model.train()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step(sidx)
        torch.cuda.current_stream().wait_stream(side)
        opt.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sloss = loss_of(sidx)
            sloss.backward()
            opt.step()
        with torch.no_grad():                                # restore the initial state after warm-up/capture
            for p, v in zip(model.parameters(), p0):
                p.copy_(v)
                if p.grad is not None:
                    p.grad.zero_()
            for st in opt.state.values():
                for t in st.values():
                    if torch.is_tensor(t):
                        t.zero_()

    def set_lr(v):
        for g in opt.param_groups:
            g["lr"].fill_(v) if torch.is_tensor(g["lr"]) else g.__setitem__("lr", v)

    get_lr = lambda: float(opt.param_groups[0]["lr"])
    best, wait, best_state, lr_best, lr_wait = math.inf, 0, None, math.inf, 0
    lr_pat = max(2, int(prot["patience"]) // 4)
    hist, t0 = [], time.perf_counter()
    for ep in range(int(prot["epochs"])):
        model.train()
        perm = tr[torch.randperm(n, generator=gen, device=DEVICE)]
        tot = torch.zeros((), device=DEVICE)
        for s in range(0, n, bs):
            i = perm[s: s + bs]
            if graph is not None and len(i) == bs:
                sidx.copy_(i)
                graph.replay()
                tot += sloss.detach() * bs
            else:
                tot += step(i) * len(i)
        vl = float(((predict(model, data, va) - data.targets(va)) ** 2 * w).mean())
        hist.append({"epoch": ep + 1, "loss": float(tot / n), "val_loss": vl, "learning_rate": get_lr()})
        if log:
            log(f"    epoch {ep + 1:02d} loss {float(tot / n):.5f} val {vl:.5f}")
        if vl < best:
            best, wait, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if vl < lr_best - 1e-4:
            lr_best, lr_wait = vl, 0
        else:
            lr_wait += 1
            if lr_wait >= lr_pat and get_lr() > 1e-6:
                set_lr(max(get_lr() * 0.5, 1e-6))
                lr_wait = 0
        if wait >= int(prot["patience"]):
            break
    sec = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    del graph
    return model, pd.DataFrame(hist), sec, int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------- MOBO
def decode(z, space, nt):
    cfg = {}
    for i, (name, kind, arg) in enumerate(space):
        u = float(z[i])
        if kind == "choice":
            cfg[name] = list(arg)[min(int(u * len(arg)), len(arg) - 1)]
        elif kind == "uniform":
            cfg[name] = float(arg[0] + u * (arg[1] - arg[0]))
        else:
            cfg[name] = float(math.exp(math.log(arg[0]) + u * (math.log(arg[1]) - math.log(arg[0]))))
    e = np.exp(3.0 * np.asarray(z[len(space): len(space) + nt], dtype=float))
    cfg["task_weights"] = list(nt * e / e.sum())                 # lambda = K softmax(3a), Eq. 12
    return cfg


def pareto_mask(F):
    return np.array([not (np.all(F <= f, axis=1) & np.any(F < f, axis=1)).any() for f in F])


def knee_index(F, ord_=2):
    idx = np.where(pareto_mask(F))[0]
    P = F[idx]
    span = np.where(np.ptp(P, 0) > 0, np.ptp(P, 0), 1.0)
    return int(idx[np.argmin(np.linalg.norm((P - P.min(0)) / span, ord=ord_, axis=1))])


def hypervolume(F, ref):
    P = F[pareto_mask(F)]
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    m = P.shape[1]
    if (len(P) + 1) ** m > 3_000_000:
        u = np.random.default_rng(0).uniform(P.min(0), ref, size=(400_000, m))
        dom = np.zeros(len(u), dtype=bool)
        for p in P:
            dom |= np.all(u >= p, axis=1)
        return float(dom.mean() * np.prod(ref - P.min(0)))
    g = [np.unique(np.r_[P[:, k], ref[k]]) for k in range(m)]
    lo = np.stack([a.ravel() for a in np.meshgrid(*[x[:-1] for x in g], indexing="ij")], 1)
    vol = np.prod(np.stack([a.ravel() for a in np.meshgrid(*[np.diff(x) for x in g], indexing="ij")], 1), 1)
    cov = np.zeros(len(lo), dtype=bool)
    for p in P:
        cov |= np.all(lo >= p, axis=1)
    return float(vol[cov].sum())


def mobo_search(data, opts, b, out, log):
    """Sobol + batch ParEGO; each candidate trained with every seed in b['seeds'], objectives = seed mean."""
    out.mkdir(parents=True, exist_ok=True)
    space = [(n, k, [p for p in a if data.L % int(p) == 0] if n == "patch_len" else a) for n, k, a in SEARCH_SPACE]
    names, nt, seeds = data.names, data.nt, [int(s) for s in b["seeds"]]
    dim = len(space) + nt
    lowfi = {k: b[k] for k in ("epochs", "patience", "batch_size", "max_train_samples", "max_val_samples")}
    (out / "optimizer_config.json").write_text(json.dumps({"search_space": [list(map(str, s)) for s in space], **b},
                                                          indent=2), encoding="utf-8")
    t_csv, s_csv = out / "mobo_trials.csv", out / "mobo_trials_per_seed.csv"
    rows = pd.read_csv(t_csv, encoding="utf-8-sig").to_dict("records") if t_csv.exists() else []
    per_seed = pd.read_csv(s_csv, encoding="utf-8-sig").to_dict("records") if s_csv.exists() else []
    rng = np.random.default_rng(b["sobol_seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sobol = qmc.Sobol(dim, scramble=True, seed=b["sobol_seed"]).random(b["n_init"])

    def evaluate(z, phase, it):
        trial, cfg = len(rows), decode(z, space, nt)
        log(f"[MOBO] trial {trial:02d} ({phase}) {json.dumps(cfg)}")
        done = {int(r["seed"]): r for r in per_seed if int(r["trial"]) == trial}
        Fs, secs = [], 0.0
        for s in seeds:
            if s not in done:
                model, _, sec, params = train(data, cfg, lowfi, s, opts)
                vi = data.index("val", lowfi["max_val_samples"])
                f = ((predict(model, data, vi) - data.targets(vi)) ** 2).mean(0).cpu().numpy()
                done[s] = {"trial": trial, "seed": s, "seconds": sec, "params": params,
                           **{f"NMSE_{n}": float(v) for n, v in zip(names, f)}}
                per_seed.append(done[s])
                pd.DataFrame(per_seed).to_csv(s_csv, index=False, encoding="utf-8-sig")
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
            Fs.append([done[s][f"NMSE_{n}"] for n in names])
            secs += done[s]["seconds"]
        Fs = np.asarray(Fs)
        f = Fs.mean(0)
        rows.append({"trial": trial, "phase": phase, "iteration": it, "n_seeds": len(seeds), "seconds": secs,
                     "objective_mean": float(f.mean()), "config_json": json.dumps(cfg, sort_keys=True),
                     **{f"NMSE_{n}": float(v) for n, v in zip(names, f)},
                     **{f"SD_{n}": float(v) for n, v in zip(names, Fs.std(0, ddof=1) if len(Fs) > 1 else 0 * f)},
                     **{f"z{i}": float(v) for i, v in enumerate(z)}})
        pd.DataFrame(rows).to_csv(t_csv, index=False, encoding="utf-8-sig")
        log(f"[MOBO]   -> mean of {len(seeds)} seeds {np.round(f, 5).tolist()}  avg {f.mean():.5f}  {secs:.0f}s")

    for i in range(len(rows), b["n_init"]):
        evaluate(sobol[i], "sobol", -1)
    q = b["batch_candidates"]
    for it in range(b["n_iterations"]):
        done_it = sum(1 for r in rows if r["phase"] == "bayesian" and int(r["iteration"]) == it)
        seen = rows[: b["n_init"] + it * q]
        Z = np.array([[r[f"z{i}"] for i in range(dim)] for r in seen])
        Fm = np.array([[r[f"NMSE_{n}"] for n in names] for r in seen])
        Fn = (Fm - Fm.min(0)) / np.where(np.ptp(Fm, 0) > 0, np.ptp(Fm, 0), 1.0)
        W = rng.dirichlet(np.ones(nt), size=q)
        pool = rng.random((b["candidate_pool"], dim))
        chosen = []
        for j in range(q):
            g = np.max(W[j] * Fn, 1) + b["rho"] * np.sum(W[j] * Fn, 1)          # augmented Chebyshev, Eq. 15
            kern = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.full(dim, 0.5), (1e-2, 1e2), nu=2.5)
                    + WhiteKernel(1e-3, (1e-8, 1e-1)))
            gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=3,
                                          random_state=b["sobol_seed"] + 100 * it + j)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(Z, g)
            mu, sd = gp.predict(pool, return_std=True)
            sd = np.maximum(sd, 1e-12)
            imp = g.min() - mu - b["xi"]
            ei = imp * norm.cdf(imp / sd) + sd * norm.pdf(imp / sd)                # Eq. 17
            for c in np.argsort(-ei):
                if all(np.linalg.norm(pool[c] - o) > b["min_distance"] for o in chosen) and np.min(np.linalg.norm(Z - pool[c], axis=1)) > 1e-6:
                    chosen.append(pool[c])
                    break
        for k, z in enumerate(chosen):
            if k >= done_it:
                evaluate(z, "bayesian", it)

    T = pd.DataFrame(rows)
    Fm = T[[f"NMSE_{n}" for n in names]].to_numpy(float)
    ref = 1.10 * Fm.max(0)
    T["pareto"] = pareto_mask(Fm)
    T.to_csv(t_csv, index=False, encoding="utf-8-sig")
    hist = [{"evaluations": s, "hypervolume": hypervolume(Fm[:s], ref), "pareto_size": int(pareto_mask(Fm[:s]).sum()),
             "best_mean_NMSE": float(Fm[:s].mean(1).min())} for s in range(b["n_init"], len(Fm) + 1, q)]
    pd.DataFrame(hist).to_csv(out / "mobo_iteration_summary.csv", index=False, encoding="utf-8-sig")
    k = knee_index(Fm)
    best = {"trial": int(T.trial[k]), "seeds_per_evaluation": seeds, "best_config": json.loads(T.config_json[k]),
            "validation_objectives": dict(zip(names, Fm[k].tolist())),
            "knee_L1": int(T.trial[knee_index(Fm, 1)]), "knee_Chebyshev": int(T.trial[knee_index(Fm, np.inf)])}
    (out / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    log(f"[MOBO] {len(T)} evaluations x {len(seeds)} seeds, {int(T.pareto.sum())} Pareto points, knee = trial {best['trial']}")
    return best


# ---------------------------------------------------------------- final training and evaluation
def metrics(truth, pred, data, scenario):
    rows = []
    for k, n in enumerate(data.names):
        t, p = truth[:, k].astype(np.float64), pred[:, k].astype(np.float64)
        e = p - t
        mse = float(np.mean(e ** 2))
        rows.append({"scenario": scenario, "target": n, "unit": data.units[k], "n": len(t), "MSE": mse, "RMSE": math.sqrt(mse),
                     "MAE": float(np.mean(np.abs(e))), "NMSE": mse / float(data.scale[k]) ** 2,
                     "R2": float(1 - np.sum(e ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-30))})
    f = pd.DataFrame(rows)
    rows.append({"scenario": scenario, "target": "average", "unit": "mixed", "n": len(truth),
                 **{c: float(f[c].mean()) for c in ["MSE", "RMSE", "MAE", "NMSE", "R2"]}})
    return rows


def final_training(data, cfg, opts, prot, out, log):
    out.mkdir(parents=True, exist_ok=True)
    tprot = {k: prot[k] for k in ("epochs", "patience", "batch_size", "max_train_samples", "max_val_samples")}
    all_rows, ens = [], {s: [] for s in data.tests}
    for seed in prot["seeds"]:
        rd = out / f"seed_{seed:02d}"
        if (rd / "metrics.csv").exists():
            all_rows.append(pd.read_csv(rd / "metrics.csv", encoding="utf-8-sig"))
            z = np.load(rd / "predictions.npz")
            for s in data.tests:
                ens[s].append(z[f"{s}_pred"])
            continue
        rd.mkdir(parents=True, exist_ok=True)
        model, hist, sec, params = train(data, cfg, tprot, seed, opts)
        rows, store = [], {}
        for s in data.tests:
            i = data.index(s)
            pred = data.to_physical(predict(model, data, i).cpu().numpy())
            ic = i.cpu().numpy()
            truth = data.raw[ic, : data.nt]
            rows += metrics(truth, pred, data, s)
            store[f"{s}_index"], store[f"{s}_pred"] = ic, pred.astype(np.float32)
            ens[s].append(pred)
            fr = {"timestamp": pd.to_datetime(data.ts[ic])}
            for k, n in enumerate(data.names):
                fr[f"observed_{n}"], fr[f"predicted_{n}"] = truth[:, k], pred[:, k]
            pd.DataFrame(fr).to_csv(rd / f"predictions_{s}.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(rd / "predictions.npz", **store)
        torch.save(model.state_dict(), rd / "model.pt")
        hist.to_csv(rd / "training_history.csv", index=False, encoding="utf-8-sig")
        (rd / "run_info.json").write_text(json.dumps({"seed": seed, "config": {**opts, **cfg}, "training_seconds": sec,
                                                      "parameters": params}, indent=2), encoding="utf-8")
        m = pd.DataFrame(rows)
        m.insert(0, "seed", seed)
        m.to_csv(rd / "metrics.csv", index=False, encoding="utf-8-sig")
        all_rows.append(m)
        avg = m[m.target == "average"].set_index("scenario").NMSE
        log(f"[final] seed {seed}: {sec:.0f} s, {params} parameters | average NMSE " + ", ".join(f"{s} {avg[s]:.4f}" for s in data.tests))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    per_seed = pd.concat(all_rows, ignore_index=True)
    per_seed.to_csv(out / "all_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summ = []
    for (sc, tg), g in per_seed.groupby(["scenario", "target"], sort=False):
        for met in ("NMSE", "R2", "MSE", "RMSE", "MAE"):
            x = g[met].to_numpy(float)
            sd = x.std(ddof=1) if len(x) > 1 else np.nan
            hw = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)) if len(x) > 1 else np.nan
            summ.append({"scenario": sc, "target": tg, "metric": met, "n_seeds": len(x), "mean": x.mean(), "sd": sd,
                         "CI95_lo": x.mean() - hw, "CI95_hi": x.mean() + hw, "min": x.min(), "max": x.max()})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "seed_summary.csv", index=False, encoding="utf-8-sig")
    erows, prows = [], []
    for s in data.tests:
        i = data.idx[s]
        truth = data.raw[i, : data.nt]
        erows += metrics(truth, np.mean(ens[s], axis=0), data, s)
        prows += metrics(truth, data.raw[i - data.h, : data.nt], data, s)
    pd.DataFrame(erows).to_csv(out / "ensemble_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prows).to_csv(out / "persistence_metrics.csv", index=False, encoding="utf-8-sig")
    for s in data.tests:
        g = summ[(summ.scenario == s) & (summ.target == "average") & (summ.metric == "NMSE")].iloc[0]
        p = [r for r in prows if r["scenario"] == s and r["target"] == "average"][0]["NMSE"]
        log(f"[final] {s}: average NMSE {g['mean']:.4f} ± {g['sd']:.4f} over {int(g.n_seeds)} seeds (persistence {p:.4f})")
    return summ


# ---------------------------------------------------------------- main
def deep_update(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def parse_seeds(t):
    out = []
    for p in str(t).split(","):
        if "-" in p:
            a, b = p.split("-")
            out += list(range(int(a), int(b) + 1))
        elif p.strip():
            out.append(int(p))
    return out


def main():
    ap = argparse.ArgumentParser(description="MOBO-TimeXer (single file)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["all", "search", "final"], default="all")
    ap.add_argument("--knee", default="search", help="search | paper | path to a JSON configuration")
    ap.add_argument("--mobo-seeds", default=None, help="seeds per MOBO evaluation, default 0-4")
    ap.add_argument("--final-seeds", default=None, help="seeds of the final training, default 0-9")
    ap.add_argument("--output", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cpath = Path(a.config).resolve()
    user = json.loads(cpath.read_text(encoding="utf-8-sig"))
    cfg = deep_update(DEFAULTS, {k: user[k] for k in ("output_dir", "model_options", "final") if k in user})
    cfg["mobo"] = deep_update(cfg["mobo"], user.get("mobo", {}).get("timexer", user.get("mobo", {})))
    if a.quick:
        cfg["mobo"].update(QUICK["mobo"])
        cfg["final"].update(QUICK["final"])
    if a.mobo_seeds:
        cfg["mobo"]["seeds"] = parse_seeds(a.mobo_seeds)
    if a.final_seeds:
        cfg["final"]["seeds"] = parse_seeds(a.final_seeds)
    out = Path(a.output) if a.output else Path(cfg["output_dir"]) / "mobo_timexer"
    out = out if out.is_absolute() else (cpath.parent / out).resolve()
    out = out.with_name(out.name + "_quick") if a.quick else out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run_log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.perf_counter()
    data = Data(user["data"], cpath.parent)
    log(f"device {DEVICE} | windows {data.profile['split_windows']} | MOBO seeds per evaluation {cfg['mobo']['seeds']}")
    (out / "data_profile.json").write_text(json.dumps(data.profile, indent=2), encoding="utf-8")
    opts = cfg["model_options"]
    if a.stage in ("all", "search") and a.knee == "search":
        f = out / "mobo" / "best_config.json"
        best = json.loads(f.read_text()) if f.exists() else mobo_search(data, opts, cfg["mobo"], out / "mobo", log)
        knee = best["best_config"]
    elif a.knee == "paper":
        knee = dict(PAPER_KNEE)
        if len(knee["task_weights"]) != data.nt:
            knee["task_weights"] = [1.0] * data.nt
    elif a.knee == "search":
        knee = json.loads((out / "mobo" / "best_config.json").read_text())["best_config"]
    else:
        j = json.loads(Path(a.knee).read_text(encoding="utf-8-sig"))
        knee = j.get("best_config", j)
    if a.stage in ("all", "final"):
        log(f"[final] configuration {json.dumps(knee)}")
        final_training(data, knee, opts, cfg["final"], out / "final", log)
    log(f"finished in {(time.perf_counter() - t0) / 60:.1f} min; results in {out.name}/")


if __name__ == "__main__":
    main()
