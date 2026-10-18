"""Prediction models (MOBO-TimeXer and five benchmarks), common training protocol and metrics (Secs. 2.1, 2.3, 2.4)."""
from __future__ import annotations

import math
import os
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_pipeline import MotionData, history_tensor, materialize_windows, select_evenly

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {"timexer": "MOBO-TimeXer", "xgboost": "XGBoost", "elm": "ELM", "gru": "GRU", "cnn_lstm": "CNN-LSTM",
          "nonstationary_transformer": "Non-stationary Transformer"}
NEURAL = ("timexer", "gru", "cnn_lstm", "nonstationary_transformer")
TABULAR = ("xgboost", "elm")

# knee configurations of Table 6
PAPER_KNEE_CONFIGS = {
    "timexer": {"d_model": 32, "num_heads": 4, "ff_dim": 256, "num_layers": 3, "dropout": 0.16967092417631452,
                "patch_len": 4, "head_units": 32, "learning_rate": 0.0028788475595118573,
                "task_weights": [0.3121030773375151, 2.73929283782137, 0.2651447507096903, 0.683459334131425]},
    "xgboost": {"n_estimators": 1000, "max_depth": 4, "learning_rate": 0.03472391831035434,
                "subsample": 0.7119709012955899, "colsample_bytree": 0.7059964653418797,
                "min_child_weight": 18.863481682220762, "reg_lambda": 0.017015462702041875},
    "elm": {"hidden_units": 1024, "alpha": 7.54939031448198, "activation": "sigmoid"},
    "gru": {"units": 128, "num_layers": 3, "dropout": 0.008510000543135264, "head_units": 16,
            "learning_rate": 0.0015734330168744731},
    "cnn_lstm": {"filters": 32, "kernel_size": 5, "lstm_units": 64, "dropout": 0.2681555184535682, "head_units": 16,
                 "learning_rate": 0.002737518746357635},
    "nonstationary_transformer": {"d_model": 48, "num_heads": 2, "ff_dim": 64, "num_layers": 1,
                                  "dropout": 0.006764371568774047, "projector_units": 128, "head_units": 32,
                                  "learning_rate": 0.0018272019459218294},
}
# ablation variant A (no MOBO)
DEFAULT_TIMEXER_CONFIG = {"d_model": 64, "num_heads": 4, "ff_dim": 128, "num_layers": 2, "dropout": 0.05,
                          "patch_len": 2, "head_units": 32, "learning_rate": 1e-3}


# ---------------------------------------------------------------- reproducibility
def seed_everything(seed: int) -> None:
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


# ---------------------------------------------------------------- data on the device
class WindowData:
    """Standardised table on the device; windows are cut on the fly from target-row indices."""

    def __init__(self, data: MotionData):
        self.data = data
        self.feat = torch.from_numpy(np.ascontiguousarray(data.features)).to(DEVICE)
        self.L, self.h = data.lookback, data.horizon
        self.offsets = torch.arange(self.L, device=DEVICE)
        self.n_features = data.features.shape[1]
        self.n_targets = data.n_targets

    def index(self, split: str, maximum) -> torch.Tensor:
        return torch.from_numpy(select_evenly(self.data.split_indices[split], maximum).copy()).to(DEVICE)

    def windows(self, idx: torch.Tensor, columns: torch.Tensor | None = None) -> torch.Tensor:
        starts = idx - self.h - self.L + 1
        x = self.feat[starts[:, None] + self.offsets[None, :]]
        return x if columns is None else x[:, :, columns]

    def targets(self, idx: torch.Tensor, cols=None) -> torch.Tensor:
        return self.feat[idx][:, slice(0, self.n_targets) if cols is None else cols]


# ---------------------------------------------------------------- building blocks
def glorot_(module: nn.Module) -> None:
    """Keras-style init: Glorot kernels, zero biases, orthogonal recurrent kernels, LSTM forget bias 1."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for name, p in m.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(p)
                elif name.startswith("weight_hh"):
                    nn.init.orthogonal_(p)
                elif name.startswith("bias"):
                    nn.init.zeros_(p)
            if isinstance(m, nn.LSTM):
                u = m.hidden_size
                for name, p in m.named_parameters():
                    if name.startswith("bias_ih"):
                        with torch.no_grad():
                            p[u:2 * u].fill_(1.0)


class MHA(nn.Module):
    """Multi-head attention; with tau/delta: de-stationary attention softmax(tau*QK^T/sqrt(d)+delta)."""

    def __init__(self, d: int, h: int, dropout: float, qkv_bias: bool = True):
        super().__init__()
        if d % h:
            raise ValueError("d_model must be divisible by num_heads")
        self.h, self.dk = h, d // h
        self.q, self.k, self.v = (nn.Linear(d, d, bias=qkv_bias) for _ in range(3))
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def reset_keras_mha(self) -> None:
        """Glorot scale of Keras MultiHeadAttention (3-D kernels)."""
        d, h, dk = self.h * self.dk, self.h, self.dk
        a_qkv = math.sqrt(6.0 / (h * d + dk * d))
        a_out = math.sqrt(6.0 / (dk * h + d * h))
        with torch.no_grad():
            for lin in (self.q, self.k, self.v):
                lin.weight.uniform_(-a_qkv, a_qkv)
            self.o.weight.uniform_(-a_out, a_out)

    def forward(self, xq, xkv, tau=None, delta=None):
        B, T, _ = xq.shape
        S = xkv.shape[1]
        q = self.q(xq).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k(xkv).view(B, S, self.h, self.dk).transpose(1, 2)
        v = self.v(xkv).view(B, S, self.h, self.dk).transpose(1, 2)
        sc = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        if tau is not None:
            sc = sc * tau[:, None, None, :] + delta[:, None, None, :]
        w = self.drop(sc.softmax(-1))
        return self.o((w @ v).transpose(1, 2).reshape(B, T, self.h * self.dk))


class ResidualHeads(nn.Module):
    """Eq. 10: per-target GELU layer + linear output, added to the last observation."""

    def __init__(self, in_dim: int, n_targets: int, head_units: int):
        super().__init__()
        self.hidden = nn.ModuleList(nn.Linear(in_dim, head_units) for _ in range(n_targets))
        self.out = nn.ModuleList(nn.Linear(head_units, 1) for _ in range(n_targets))

    def forward(self, ctx, last):
        return last + torch.cat([o(F.gelu(h(ctx))) for h, o in zip(self.hidden, self.out)], dim=1)


def _rnn_input_dropout(x, p: float, training: bool):
    if not training or p <= 0:
        return x
    mask = torch.bernoulli(torch.full((x.shape[0], 1, x.shape[2]), 1.0 - p, device=x.device)) / (1.0 - p)
    return x * mask


# ---------------------------------------------------------------- MOBO-TimeXer
class TimeXer(nn.Module):
    """Shared-encoder TimeXer with task-specific residual heads (Eqs. 4-10).
    ar_bypass: zero-initialised linear AR map of the anchored lags; window_norm: window-wise amplitude
    normalisation with eps = window_norm_eps (Sec. 2.1.2). Both off = original network."""

    def __init__(self, lookback, n_features, n_targets=4, *, d_model=64, num_heads=4, ff_dim=128, num_layers=2,
                 dropout=0.05, patch_len=2, head_units=32, ar_bypass=True, window_norm=True, window_norm_eps=0.02, **_):
        super().__init__()
        if lookback % patch_len:
            raise ValueError("lookback must be divisible by patch_len")
        self.nt = n_targets
        n_patch = lookback // patch_len
        self.patch = nn.Conv1d(n_targets, d_model, kernel_size=patch_len, stride=patch_len)
        self.pos = nn.Parameter(torch.empty(n_patch, d_model))
        self.exo = nn.Linear(lookback, d_model)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict(dict(
                sa=MHA(d_model, num_heads, dropout), n1=nn.LayerNorm(d_model, eps=1e-6),
                ca=MHA(d_model, num_heads, dropout), n2=nn.LayerNorm(d_model, eps=1e-6),
                ff1=nn.Linear(d_model, ff_dim), ff2=nn.Linear(ff_dim, d_model), n3=nn.LayerNorm(d_model, eps=1e-6))))
        self.drop = nn.Dropout(dropout)
        self.heads = ResidualHeads(2 * d_model + n_targets, n_targets, head_units)
        glorot_(self)
        nn.init.xavier_uniform_(self.pos)
        for b in self.blocks:
            b["sa"].reset_keras_mha()
            b["ca"].reset_keras_mha()
        self.use_ar, self.use_rev, self.eps = bool(ar_bypass), bool(window_norm), float(window_norm_eps)
        if self.use_ar:
            self.ar = nn.Linear(n_targets * lookback, n_targets)
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
        ctx = torch.cat([e[:, 0], e[:, -1], z[:, -1]], dim=1)
        inc = self.heads(ctx, torch.zeros_like(raw[:, -1]))
        if self.use_rev:
            inc = inc * s[:, 0, :]
        if self.use_ar:
            inc = inc + self.ar(endo.flatten(1))
        return raw[:, -1] + inc


# ---------------------------------------------------------------- neural benchmarks
class GRUNet(nn.Module):
    def __init__(self, lookback, n_features, n_targets=4, *, units=64, num_layers=2, dropout=0.1, head_units=32, **_):
        super().__init__()
        self.nt, self.p = n_targets, float(dropout)
        self.rnns = nn.ModuleList(nn.GRU(n_features if i == 0 else units, units, batch_first=True)
                                  for i in range(num_layers))
        self.heads = ResidualHeads(units + n_targets, n_targets, head_units)
        glorot_(self)

    def forward(self, x):
        h = x
        for rnn in self.rnns:
            h, _ = rnn(_rnn_input_dropout(h, self.p, self.training))
        last = x[:, -1, : self.nt]
        return self.heads(torch.cat([h[:, -1], last], dim=1), last)


class CNNLSTMNet(nn.Module):
    def __init__(self, lookback, n_features, n_targets=4, *, filters=64, kernel_size=3, lstm_units=64, dropout=0.1,
                 head_units=32, **_):
        super().__init__()
        self.nt, self.k, self.p = n_targets, int(kernel_size), float(dropout)
        self.conv = nn.Conv1d(n_features, filters, self.k)
        self.bn = nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01)
        self.lstm = nn.LSTM(filters, lstm_units, batch_first=True)
        self.heads = ResidualHeads(lstm_units + n_targets, n_targets, head_units)
        glorot_(self)

    def forward(self, x):
        total = self.k - 1
        h = F.pad(x.transpose(1, 2), (total // 2, total - total // 2))
        h = self.bn(F.gelu(self.conv(h)))
        h = F.max_pool1d(h, 2).transpose(1, 2)
        out, _ = self.lstm(_rnn_input_dropout(h, self.p, self.training))
        last = x[:, -1, : self.nt]
        return self.heads(torch.cat([out[:, -1], last], dim=1), last)


class NSTNet(nn.Module):
    def __init__(self, lookback, n_features, n_targets=4, *, d_model=64, num_heads=4, ff_dim=128, num_layers=2,
                 dropout=0.1, projector_units=64, head_units=32, **_):
        super().__init__()
        self.nt = n_targets
        self.proj = nn.Linear(2 * n_features, projector_units)
        self.tau = nn.Linear(projector_units, 1)
        self.delta = nn.Linear(projector_units, lookback)
        self.emb = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.empty(lookback, d_model))
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict(dict(
                att=MHA(d_model, num_heads, dropout, qkv_bias=False), n1=nn.LayerNorm(d_model, eps=1e-6),
                ff1=nn.Linear(d_model, ff_dim), ff2=nn.Linear(ff_dim, d_model), n2=nn.LayerNorm(d_model, eps=1e-6))))
        self.drop = nn.Dropout(dropout)
        self.heads = ResidualHeads(2 * d_model + n_targets, n_targets, head_units)
        glorot_(self)
        nn.init.xavier_uniform_(self.pos)

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = torch.sqrt(x.var(1, unbiased=False, keepdim=True) + 1e-5)
        z = (x - mean) / std
        pr = F.gelu(self.proj(torch.cat([mean.flatten(1), std.flatten(1)], dim=1)))
        tau, delta = F.softplus(self.tau(pr)), self.delta(pr)
        h = self.emb(z) + self.pos
        for b in self.blocks:
            h = b["n1"](h + self.drop(b["att"](h, h, tau, delta)))
            h = b["n2"](h + b["ff2"](self.drop(F.gelu(b["ff1"](h)))))
        last = x[:, -1, : self.nt]
        return self.heads(torch.cat([h.mean(1), h[:, -1], last], dim=1), last)


BUILDERS = {"timexer": TimeXer, "gru": GRUNet, "cnn_lstm": CNNLSTMNet, "nonstationary_transformer": NSTNet}
CUDA_GRAPH_MODELS = (TimeXer, NSTNet)


# ---------------------------------------------------------------- neural training
@torch.no_grad()
def predict_tensor(model, wd: WindowData, idx: torch.Tensor, columns=None, batch: int = 8192) -> torch.Tensor:
    model.eval()
    return torch.cat([model(wd.windows(idx[s: s + batch], columns)) for s in range(0, len(idx), batch)], dim=0)


def fit_torch(builder, wd: WindowData, config: dict, protocol: dict, seed: int, *, columns=None, target_cols=None,
              n_targets: int | None = None, log=None, cuda_graph: bool | None = None):
    """Returns (model, history, seconds, parameter count). cuda_graph replays each training step as a CUDA
    graph (same mathematics, less launch overhead); default on for attention models on GPU."""
    seed_everything(seed)
    n_targets = wd.n_targets if n_targets is None else n_targets
    mcfg = {k: v for k, v in config.items() if k not in ("learning_rate", "task_weights")}
    n_in = len(columns) if columns is not None else wd.n_features
    model = builder(wd.L, n_in, n_targets, **mcfg).to(DEVICE)
    if cuda_graph is None:
        cuda_graph = DEVICE.type == "cuda" and builder in CUDA_GRAPH_MODELS
    cuda_graph = bool(cuda_graph) and DEVICE.type == "cuda"
    w = np.asarray(config.get("task_weights", [1.0] * n_targets), dtype=np.float32)[:n_targets]
    w = torch.tensor(w / w.mean(), device=DEVICE)
    lr0 = float(config.get("learning_rate", 1e-3))
    if cuda_graph:
        optim = torch.optim.Adam(model.parameters(), lr=torch.tensor(lr0, device=DEVICE), eps=1e-7, capturable=True)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=lr0, eps=1e-7, fused=DEVICE.type == "cuda")
    tr, va = wd.index("train", protocol["max_train_samples"]), wd.index("val", protocol["max_val_samples"])
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    bs, n = int(protocol["batch_size"]), len(tr)

    def loss_of(idx):
        return ((model(wd.windows(idx, columns)) - wd.targets(idx, target_cols)) ** 2 * w).mean()

    def eager_step(idx):
        loss = loss_of(idx)
        optim.zero_grad(set_to_none=False)
        loss.backward()
        optim.step()
        return loss.detach()

    graph = None
    if cuda_graph and n >= bs:
        static_idx = tr[:bs].clone()
        p0 = [p.detach().clone() for p in model.parameters()]
        b0 = [b.detach().clone() for b in model.buffers()]
        model.train()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                eager_step(static_idx)
        torch.cuda.current_stream().wait_stream(side)
        optim.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_loss = loss_of(static_idx)
            static_loss.backward()
            optim.step()
        with torch.no_grad():
            for p, v in zip(model.parameters(), p0):
                p.copy_(v)
            for b, v in zip(model.buffers(), b0):
                b.copy_(v)
            for st in optim.state.values():
                for t_ in st.values():
                    if torch.is_tensor(t_):
                        t_.zero_()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        del p0, b0

    def set_lr(v):
        for g in optim.param_groups:
            if torch.is_tensor(g["lr"]):
                g["lr"].fill_(v)
            else:
                g["lr"] = v

    def get_lr():
        v = optim.param_groups[0]["lr"]
        return float(v) if torch.is_tensor(v) else v

    best, wait, best_state = math.inf, 0, None
    lr_best, lr_wait, lr_patience = math.inf, 0, max(2, int(protocol["patience"]) // 4)
    hist = []
    t0 = time.perf_counter()
    for ep in range(int(protocol["epochs"])):
        model.train()
        perm = tr[torch.randperm(n, generator=gen, device=DEVICE)]
        total = torch.zeros((), device=DEVICE)
        for s in range(0, n, bs):
            idx = perm[s: s + bs]
            if graph is not None and len(idx) == bs:
                static_idx.copy_(idx)
                graph.replay()
                total += static_loss.detach() * bs
            else:
                total += eager_step(idx) * len(idx)
        pv = predict_tensor(model, wd, va, columns)
        val_loss = float(((pv - wd.targets(va, target_cols)) ** 2 * w).mean())
        train_loss = float(total / n)
        cur_lr = get_lr()
        hist.append({"epoch": ep + 1, "loss": train_loss, "val_loss": val_loss, "learning_rate": cur_lr,
                     "seconds": time.perf_counter() - t0})
        if log:
            log(f"    epoch {ep + 1:02d} loss {train_loss:.5f} val {val_loss:.5f} lr {cur_lr:.2e}")
        if val_loss < best:
            best, wait = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if val_loss < lr_best - 1e-4:
            lr_best, lr_wait = val_loss, 0
        else:
            lr_wait += 1
            if lr_wait >= lr_patience and cur_lr > 1e-6:
                set_lr(max(cur_lr * 0.5, 1e-6))
                lr_wait = 0
        if wait >= int(protocol["patience"]):
            break
    elapsed = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    params = int(sum(p.numel() for p in model.parameters()))
    del graph
    return model, pd.DataFrame(hist), elapsed, params


# ---------------------------------------------------------------- XGBoost / ELM
class ExtremeLearningMachine:
    """Random hidden layer + closed-form ridge output weights."""

    def __init__(self, input_dim: int, hidden_units: int = 512, alpha: float = 1e-2, activation: str = "tanh",
                 seed: int = 42) -> None:
        self.input_dim, self.hidden_units, self.alpha, self.activation = int(input_dim), int(hidden_units), float(alpha), activation
        rng = np.random.default_rng(seed)
        self.input_weights = rng.normal(0.0, 1.0 / math.sqrt(input_dim), (input_dim, hidden_units)).astype(np.float32)
        self.bias = rng.uniform(-0.5, 0.5, hidden_units).astype(np.float32)
        self.output_weights = None

    def _activate(self, v):
        if self.activation == "relu":
            return np.maximum(v, 0.0)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(v, -30.0, 30.0)))
        return np.tanh(v)

    def fit(self, x, y, batch_size: int = 8192):
        hth = np.zeros((self.hidden_units, self.hidden_units), dtype=np.float64)
        hty = np.zeros((self.hidden_units, y.shape[1]), dtype=np.float64)
        for s in range(0, len(x), batch_size):
            h = self._activate(x[s: s + batch_size] @ self.input_weights + self.bias)
            hth += h.T @ h
            hty += h.T @ y[s: s + batch_size]
        hth.flat[:: self.hidden_units + 1] += self.alpha
        self.output_weights = np.linalg.solve(hth, hty).astype(np.float32)
        return self

    def predict(self, x, batch_size: int = 8192):
        out = []
        for s in range(0, len(x), batch_size):
            out.append(self._activate(x[s: s + batch_size] @ self.input_weights + self.bias) @ self.output_weights)
        return np.concatenate(out, axis=0)

    def save(self, path) -> None:
        np.savez_compressed(path, input_weights=self.input_weights, bias=self.bias, output_weights=self.output_weights,
                            alpha=self.alpha, activation=self.activation)


def fit_xgboost(data: MotionData, config: dict, protocol: dict, seed: int, device: str | None = None, n_jobs=None):
    from xgboost import XGBRegressor
    device = device or ("cuda" if DEVICE.type == "cuda" else "cpu")
    x_tr, y_tr, _ = materialize_windows(data, "train", protocol["max_train_samples"], flatten=True)
    x_va, y_va, _ = materialize_windows(data, "val", protocol["max_val_samples"], flatten=True)
    models = []
    t0 = time.perf_counter()
    for k in range(data.n_targets):
        est = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=seed, device=device,
                           n_jobs=n_jobs or max(1, (os.cpu_count() or 2) - 1), **config)
        est.fit(x_tr, y_tr[:, k], eval_set=[(x_va, y_va[:, k])], verbose=False)
        models.append(est)
    return models, time.perf_counter() - t0, int(sum(e.get_booster().num_boosted_rounds() for e in models))


def fit_elm(data: MotionData, config: dict, protocol: dict, seed: int):
    x_tr, y_tr, _ = materialize_windows(data, "train", protocol["max_train_samples"], flatten=True)
    m = ExtremeLearningMachine(x_tr.shape[1], int(config["hidden_units"]), float(config["alpha"]),
                               str(config["activation"]), seed=seed)
    t0 = time.perf_counter()
    m.fit(x_tr, y_tr, batch_size=int(protocol.get("batch_size", 8192)))
    return m, time.perf_counter() - t0, int(m.input_weights.size + m.bias.size + m.output_weights.size)


def predict_tabular(kind: str, model, data: MotionData, split: str, maximum=0):
    idx = select_evenly(data.split_indices[split], maximum)
    parts = []
    for s in range(0, len(idx), 8192):
        x = history_tensor(data, idx[s: s + 8192])
        flat = x.reshape(len(x), -1)
        res = model.predict(flat) if kind == "elm" else np.column_stack([e.predict(flat) for e in model])
        parts.append(x[:, -1, : data.n_targets] + res)
    return idx, np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------- unified interface
class TrainedModel:

    def __init__(self, key, kind, obj, seconds, params, history=None, columns=None):
        self.key, self.kind, self.obj, self.seconds, self.params = key, kind, obj, seconds, params
        self.history, self.columns = history, columns

    def predict(self, data: MotionData, wd: WindowData | None, split: str, maximum=0, as_tensor=False):
        if self.kind in NEURAL:
            idx = wd.index(split, maximum)
            p = predict_tensor(self.obj, wd, idx, self.columns)
            return (idx, p) if as_tensor else (idx.cpu().numpy(), p.cpu().numpy())
        return predict_tabular(self.kind, self.obj, data, split, maximum)

    def save(self, path_stem) -> str:
        if self.kind in NEURAL:
            torch.save(self.obj.state_dict(), f"{path_stem}.pt")
            return f"{path_stem}.pt"
        if self.kind == "elm":
            self.obj.save(f"{path_stem}_elm.npz")
            return f"{path_stem}_elm.npz"
        for k, e in enumerate(self.obj):
            e.save_model(f"{path_stem}_xgb_target{k}.json")
        return f"{path_stem}_xgb_target*.json"


def train_model(key: str, data: MotionData, wd: WindowData | None, config: dict, protocol: dict, seed: int,
                model_options: dict | None = None, log=None, cuda_graph=None) -> TrainedModel:
    """model_options: fixed settings merged into config, e.g. {"ar_bypass": True, "window_norm": True}."""
    cfg = {**(model_options or {}), **config}
    if key in NEURAL:
        m, hist, sec, params = fit_torch(BUILDERS[key], wd, cfg, protocol, seed, log=log, cuda_graph=cuda_graph)
        return TrainedModel(key, key, m, sec, params, hist)
    seed_everything(seed)
    if key == "xgboost":
        m, sec, params = fit_xgboost(data, cfg, protocol, seed, device=protocol.get("xgb_device"))
    elif key == "elm":
        m, sec, params = fit_elm(data, cfg, protocol, seed)
    else:
        raise KeyError(key)
    return TrainedModel(key, key, m, sec, params)


def train_independent_timexer(data: MotionData, wd: WindowData, config: dict, protocol: dict, seed: int,
                              model_options: dict | None = None, log=None):
    """Ablation C: one single-task TimeXer per target (own history + exogenous channels)."""
    cfg = {k: v for k, v in {**(model_options or {}), **config}.items() if k != "task_weights"}
    out = []
    for k in range(data.n_targets):
        cols = torch.tensor([k] + list(range(data.n_targets, wd.n_features)), device=DEVICE)
        m, hist, sec, params = fit_torch(TimeXer, wd, cfg, protocol, seed, columns=cols, target_cols=slice(k, k + 1),
                                         n_targets=1, log=log)
        hist.insert(0, "target", data.target_names[k])
        out.append(TrainedModel("timexer", "timexer", m, sec, params, hist, columns=cols))
    return out


# ---------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred, train_scale, target_names, target_units, scenario, model_name):
    """Per-target metrics + macro average; NMSE = MSE / training variance (Eq. 22)."""
    rows = []
    for k, target in enumerate(target_names):
        t, p = y_true[:, k].astype(np.float64), y_pred[:, k].astype(np.float64)
        e = p - t
        mse = float(np.mean(e ** 2))
        rows.append({"model": model_name, "scenario": scenario, "target": target, "unit": target_units[k],
                     "n": len(t), "MSE": mse, "RMSE": math.sqrt(mse), "MAE": float(np.mean(np.abs(e))),
                     "NMSE": mse / (float(train_scale[k]) ** 2 + 1e-12),
                     "R2": float(1.0 - np.sum(e ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-30)),
                     "sMAPE_pct": float(200.0 * np.mean(np.abs(e) / (np.abs(t) + np.abs(p) + 1e-6))),
                     "bias": float(np.mean(e))})
    f = pd.DataFrame(rows)
    rows.append({"model": model_name, "scenario": scenario, "target": "average", "unit": "mixed", "n": len(y_true),
                 **{c: float(f[c].mean()) for c in ["MSE", "RMSE", "MAE", "NMSE", "R2", "sMAPE_pct", "bias"]},
                 "NMSE_max": float(f["NMSE"].max()), "R2_min": float(f["R2"].min())})
    return rows
