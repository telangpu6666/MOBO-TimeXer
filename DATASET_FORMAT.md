# Dataset format

This document describes the data a user must provide to run the MOBO-TimeXer pipeline on their own records
(`run_pipeline.py`). The field data of the paper are confidential; `make_demo_data.py` writes a synthetic file
with exactly the same structure, and `data/example_format_first20rows.csv` shows its first 20 rows.

## 1. One table, one row per time step

| Requirement | Details |
|---|---|
| File type | `.csv` / `.txt` / `.tsv` (any encoding, e.g. `utf-8`, `utf-16`, `gbk`; any separator), `.parquet`, `.xlsx`, `.feather` |
| Header | First row contains the column names. Names can be anything (the paper's file uses `"0"`, `"1"`, …, `"36"`). |
| Time column | One column with the time stamp of every row, e.g. `2022-09-01 11:46:14`. Mixed formats such as `2022/9/25 0:00:00` are accepted. |
| Target columns | The *K* responses to be predicted (paper: K = 4 — yaw deviation, roll, total horizontal offset, pitch). Numeric. |
| Exogenous columns | Any number of numeric driver channels (paper: 29 = 13 wind/wave/current channels + 16 mooring-line tensions). |
| Other columns | Allowed; they are ignored unless listed (or used by a derived-column rule). |
| Sampling | Constant nominal interval (paper: 2 s). Gaps are allowed: no sliding window will span a gap. |
| Units | Physical units. Targets keep their units in all reported metrics (MSE, RMSE, MAE). |

Example (synthetic demo file, abbreviated):

```text
timestamp,wind_speed,wind_dir,Hs,Tp,wave_dir,current_speed_1,...,tension_16,yaw_deg,roll_deg,total_offset_m,pitch_deg,surge_m,sway_m
2024-01-01 00:00:00,5.7591,229.62337,0.60409,6.98595,244.78136,0.30009,...,3054.0,0.45,0.01,1.81574,-0.19,-1.02274,1.50031
2024-01-01 00:00:02,6.24744,229.74185,0.60305,6.98701,244.82325,0.29912,...,3056.0,0.45,0.01,1.80999,-0.19,-0.99953,1.50897
...
```

Column roles in the paper and in the demo file:

| Role | Paper (Deep Sea No.1 record, `configs/paper_config.json`) | Demo file (`configs/demo_config.json`) |
|---|---|---|
| time | `0` | `timestamp` |
| targets (in this order) | `31` yaw deviation (deg), `33` roll (deg), `34` total horizontal offset (m), `36` pitch (deg) | `yaw_deg`, `roll_deg`, `total_offset_m`, `pitch_deg` |
| exogenous | `1` … `29` (wind speed/direction, significant wave height, peak period, current speed/direction at several depths, 16 mooring-line tensions) | `wind_speed`, `wind_dir`, `Hs`, `Tp`, `wave_dir`, `current_speed_1..4`, `current_dir_1..4`, `tension_01..16` |
| auxiliary | `32`, `35` horizontal position components (only used to rebuild `34`) | `surge_m`, `sway_m` (not used) |

## 2. What the pipeline does with the table

1. **Cleaning** – rows with an unparsable time or a non-finite value in a used column are dropped, duplicated
   time stamps are removed (first kept), rows are sorted by time. The numbers of dropped rows are written to
   `results/<run>/data/data_profile.json`. If you prefer interpolation of short gaps, do it before running the pipeline.
2. **Derived / repaired columns** (optional, `derived_columns`) – evaluated in order, optionally only inside a time interval:

   ```json
   "derived_columns": [
     {"name": "34", "op": "hypot", "inputs": ["32", "35"], "from": "2022-09-25 00:00:00"},
     {"name": "Hs_m", "op": "linear", "inputs": ["Hs_cm"], "scale": 0.01, "bias": 0.0},
     {"name": "yaw_dev", "op": "sum", "inputs": ["heading", "const_90"], "weights": [1, -1]}
   ]
   ```
   `hypot` = √(Σx²) (total offset from surge and sway, Eq. 2 of the paper), `linear` = scale·x + bias (unit
   conversion), `sum` = weighted sum. A derived column can be used as a target or exogenous channel.
3. **Chronological split** – either explicit date boundaries (as in the paper, end exclusive, `null` = open end):

   ```json
   "splits": {
     "train":        [null,                  "2022-09-25 00:00:00"],
     "val":          ["2022-09-25 00:00:00", "2022-09-27 00:00:00"],
     "extreme_test": ["2022-09-27 00:00:00", "2022-09-29 00:00:00"],
     "normal_test":  ["2022-09-29 00:00:00", null]
   }
   ```
   or ratios of the record in time order, e.g. `"split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}`.
   `train` and `val` are required; **every other split name is treated as a test segment** and is evaluated
   separately (any number of test segments, any names). Splits may be given in any order and need not be contiguous.
4. **Standardisation** – Z-score statistics are computed on the training rows only and applied to all rows
   (`normalization_statistics.csv`). NMSE is the MSE divided by the training-segment variance of each target (Eq. 22).
5. **Sliding windows** – a sample is `X = rows [t-h-L+1, …, t-h]` (all targets + exogenous channels, standardised),
   `y = targets at row t`, with look-back `L` (`lookback`, paper 16 samples = 32 s) and horizon `h` (`horizon`,
   paper 1 sample = 2 s). A window is used only if all its rows and the target lie in one gap-free segment. A
   sample belongs to the split that contains its target time `t` (the input window of the first validation samples
   may reach back into the training rows, which uses no future information).

## 3. Size and content recommendations

* Several hundred thousand rows are typical (paper: 1,252,934 rows, 1.0 M training windows). The table is kept in
  memory as float32: rows × (targets + exogenous) × 4 bytes (paper: ≈ 165 MB). For very large files set
  `"cache_dir"` so that the cleaned table is cached as a binary `.npz` file after the first read.
* The validation segment drives early stopping and the MOBO objective; it should resemble the conditions you care
  about but must not overlap the test segments.
* Angles should be continuous (no 359° → 0° jumps) when used as **targets**; unwrap them or use a deviation from a
  reference heading (the paper uses yaw deviation = heading − 90°). Direction channels used as exogenous inputs can stay as they are,
  or be replaced by sine/cosine pairs via a preprocessing step.
* Constant channels are allowed (their scale is set to 1).
* The number of targets K is free (K ≥ 1). The task-weight part of the MOBO decision vector then has K dimensions,
  and the report uses your target names.

## 4. Configuration keys of the `"data"` block

| Key | Type | Default | Meaning |
|---|---|---|---|
| `path` | str | – | data file; relative paths are relative to the configuration file |
| `time_column` | str | – | name of the time column |
| `target_columns` | list | – | target column names (prediction order) |
| `target_names` | list | = `target_columns` | short names used in tables and figures |
| `target_units` | list | `""` | units used in tables and figures |
| `exogenous_columns` | list or `"auto"` | `"auto"` | exogenous channels; `"auto"` = every other numeric column except `exclude_columns` |
| `exclude_columns` | list | `[]` | columns ignored when `exogenous_columns` = `"auto"` |
| `derived_columns` | list | `[]` | derived / repaired columns (Section 2) |
| `encoding`, `sep`, `sheet_name` | – | `utf-8`, `,`, `0` | reading options |
| `sample_seconds` | float | median step | nominal sampling interval; any other step is a gap |
| `lookback`, `horizon` | int | 16, 1 | L and h in samples |
| `splits` / `split_ratios` | dict | – | chronological split (Section 2) |
| `cache_dir` | str | none | folder for the binary cache of the cleaned table |
