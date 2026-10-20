"""Synthetic demo data with the structure of the paper's records. Not field data.

    python make_demo_data.py [--days 5] [--out data/demo_platform_motion.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

DT = 2.0


def lowpass_noise(rng, n, tau_s, sd=1.0):
    a = np.exp(-DT / tau_s)
    x = lfilter([np.sqrt(1 - a * a)], [1, -a], rng.standard_normal(n))
    return sd * x


def resonant(x, period_s, damping=0.03):
    w = 2 * np.pi * DT / period_s
    r = np.exp(-damping * w)
    y = lfilter([1.0], [1.0, -2 * r * np.cos(w), r * r], x)
    return y / np.sqrt(np.mean(y[: min(len(y), 20000)] ** 2) + 1e-12) * np.std(x[: min(len(x), 20000)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--start", default="2024-01-01 00:00:00")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "demo_platform_motion.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = int(a.days * 86400 / DT)
    t = pd.date_range(a.start, periods=n, freq="2s")
    hours = np.arange(n) * DT / 3600.0
    storm_peak = (a.days - 1) * 24 + 6.0
    storm = np.exp(-0.5 * ((hours - storm_peak) / 4.0) ** 2) + 0.35 * np.exp(-0.5 * ((hours - (a.days - 2) * 24 - 14) / 5.0) ** 2)

    # environment
    wind_speed = np.clip(6 + 16 * storm + lowpass_noise(rng, n, 600, 1.5) + lowpass_noise(rng, n, 20, 0.8), 0, None)
    wind_dir = (230 + 40 * storm + lowpass_noise(rng, n, 4 * 3600, 20)) % 360
    hs = np.clip(0.6 + 4.2 * storm + lowpass_noise(rng, n, 1800, 0.08), 0.1, None)
    tp = np.clip(7 + 4 * storm + lowpass_noise(rng, n, 3600, 0.4), 3, 20)
    wave_dir = (wind_dir + 15 + lowpass_noise(rng, n, 3600, 5)) % 360
    cur_speed = [np.clip(0.3 - 0.06 * k + 0.4 * storm + lowpass_noise(rng, n, 3 * 3600, 0.05), 0, None) for k in range(4)]
    cur_dir = [(320 - 10 * k + lowpass_noise(rng, n, 6 * 3600, 15)) % 360 for k in range(4)]

    # horizontal slow drift
    fx = 0.004 * wind_speed ** 2 * np.cos(np.radians(wind_dir)) + 2.0 * cur_speed[0] * np.cos(np.radians(cur_dir[0]))
    fy = 0.004 * wind_speed ** 2 * np.sin(np.radians(wind_dir)) + 2.0 * cur_speed[0] * np.sin(np.radians(cur_dir[0]))
    drift = lambda f, s: (lfilter([1 - np.exp(-DT / 300)], [1, -np.exp(-DT / 300)], f)
                          + resonant(rng.standard_normal(n) * (0.3 + 0.8 * storm), 180, 0.08) * (0.4 + 1.5 * storm) + s)
    surge = drift(fx, -1.0) + lowpass_noise(rng, n, 30, 0.05)
    sway = drift(fy, 1.5) + lowpass_noise(rng, n, 30, 0.05)
    total_offset = np.hypot(surge, sway)

    # angular motions
    yaw = 0.45 + 0.004 * (wind_dir - 230) + resonant(rng.standard_normal(n), 80, 0.1) * (0.05 + 0.25 * storm) \
        + lowpass_noise(rng, n, 1200, 0.06)
    wave_exc = rng.standard_normal(n) * hs
    roll = -0.01 + 0.03 * fy + 0.06 * resonant(wave_exc, 10.3, 0.05) + lowpass_noise(rng, n, 900, 0.08) * (1 + storm)
    pitch = -0.23 + 0.02 * fx + 0.05 * resonant(0.7 * wave_exc + 0.7 * rng.standard_normal(n) * hs, 11.4, 0.05) \
        + 0.4 * roll * storm + lowpass_noise(rng, n, 900, 0.07) * (1 + storm)
    roll, pitch, yaw = np.round(roll, 2), np.round(pitch, 2), np.round(yaw, 2)

    # 16 mooring-line tensions
    angles = np.radians(np.repeat([45, 135, 225, 315], 4) + np.tile([-6, -2, 2, 6], 4))
    tensions = {}
    for j, ang in enumerate(angles):
        proj = surge * np.cos(ang) + sway * np.sin(ang)
        tensions[f"tension_{j + 1:02d}"] = np.round(3150 + 60 * proj + 4 * proj ** 2 + lowpass_noise(rng, n, 10, 8)
                                                    + 25 * hs * rng.standard_normal(n) * 0.3)

    df = pd.DataFrame({"timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                       "wind_speed": wind_speed, "wind_dir": wind_dir, "Hs": hs, "Tp": tp, "wave_dir": wave_dir,
                       **{f"current_speed_{k + 1}": cur_speed[k] for k in range(4)},
                       **{f"current_dir_{k + 1}": cur_dir[k] for k in range(4)},
                       **tensions,
                       "yaw_deg": yaw, "roll_deg": roll, "total_offset_m": total_offset, "pitch_deg": pitch,
                       "surge_m": surge, "sway_m": sway})
    # 560-s recording gap
    g0 = int(((a.days - 1) * 24 + 17.5) * 3600 / DT)
    df = df.drop(index=range(g0, g0 + 280)).reset_index(drop=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(5).to_csv(out, index=False)
    print(f"written {out} ({len(df)} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
