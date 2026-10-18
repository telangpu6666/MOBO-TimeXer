# MOBO-TimeXer 代码使用说明（中文）

本代码对应论文 *A motion coupling prediction method for ultra-deepwater semi-submersible platforms based on
multi-objective Bayesian optimization*，可在任意多变量时间序列数据上复现论文的完整流程（算法 1）：数据处理、
耦合特性分析、六个模型的多目标贝叶斯优化（MOBO）、膝点配置的多种子训练、消融实验、显著性检验、结果表和图。
英文完整说明见 `README.md`，数据集格式见 `DATASET_FORMAT.md`（下文第 2 节为中文摘要）。

“深海一号”现场监测数据受保密要求限制，不随代码发布；`make_demo_data.py` 可生成结构完全相同的合成数据，用于试运行。

## 1. 文件

| 文件 | 内容 |
|---|---|
| `mobo.py` | **MOBO 全部代码**：六个模型的决策空间（式 12）、解码、任务权重 softmax 参数化、Sobol 初始设计、批量 ParEGO（增广 Chebyshev 标量化，式 15）、Matérn-5/2+白噪声高斯过程（式 16）、EI 采集（式 17）、帕累托集（式 14）、超体积、膝点（式 18）；**每个候选配置用 5 个随机种子分别训练，取 5 个种子验证 NMSE 的平均值作为目标进行评估和迭代**；与模型无关，可断点续跑 |
| `models.py` | **预测模型全部代码**：MOBO-TimeXer（共享编码器 + 任务专属残差头 + 线性 AR 旁路 + 窗口幅值归一化）、GRU、CNN-LSTM、Non-stationary Transformer、XGBoost、ELM；统一训练协议（式 11 加权多任务损失、早停、学习率衰减）；评价指标（式 19–22）；论文表 6 的膝点配置 |
| `data_pipeline.py` | 数据读取、清洗、派生列/修复、按时间划分、只用训练段的标准化、不跨越时间间断的滑动窗口 |
| `run_pipeline.py` | 一键运行全部阶段的命令行入口 |
| `mobo_timexer.py` | **MOBO-TimeXer 完整框架单文件版**（数据处理、TimeXer、5 种子 MOBO、膝点、多种子最终训练、测试指标），不依赖本包其他文件，可单独运行 |
| `make_demo_data.py` | 生成合成演示数据（5 天、2 s 采样、4 个目标、29 个外生通道、1 次风暴过程） |
| `configs/demo_config.json` | 演示数据的配置文件 |
| `configs/paper_config.json` | 论文所用配置（现场数据，路径需指向授权数据文件） |
| `DATASET_FORMAT.md` | 数据集格式说明（英文） |
| `data/example_format_first20rows.csv` | 演示数据前 20 行，用于展示格式 |

## 2. 数据集格式（摘要）

* **一张表，每行一个时刻**：CSV/TXT/TSV（任意编码与分隔符）、Parquet、Excel 或 Feather；第一行为列名。
* **时间列** 1 列（如 `2022-09-01 11:46:14`，混合格式可自动识别）。
* **目标列** K 列（论文 K = 4：艏向偏差、横摇、水平总偏移、纵摇），物理单位。
* **外生列** 任意多列（论文 29 列：13 个风/浪/流通道 + 16 根系泊缆张力）；可写列名列表，或 `"auto"`（其余全部数值列，`exclude_columns` 中的除外）。
* **采样间隔恒定**（论文 2 s），允许有时间间断，滑动窗口不会跨越间断。
* 程序自动：删除无效行与重复时间戳并排序；按 `derived_columns` 生成派生/修复列（如 `hypot` 由纵荡、横荡计算总偏移，`linear` 做单位换算，可限定时间段）；按日期边界（或比例）划分 `train`、`val` 和任意个测试段（除 `train`/`val` 外的名称都视为测试段）；只用训练段计算 Z-score 参数；用 t−h−L+1…t−h 行预测 t 行目标（L = `lookback`，h = `horizon`）。

日期划分示例（左闭右开，`null` 表示不限）：

```json
"splits": {
  "train":        [null, "2022-09-25 00:00:00"],
  "val":          ["2022-09-25 00:00:00", "2022-09-27 00:00:00"],
  "extreme_test": ["2022-09-27 00:00:00", "2022-09-29 00:00:00"],
  "normal_test":  ["2022-09-29 00:00:00", null]
}
```

## 3. 安装

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124     # GPU 版 PyTorch（CPU 版用 pip install torch）
pip install -r requirements.txt
```

已在 Windows 11 + Python 3.11.9 + PyTorch 2.6.0 (CUDA 12.4) + RTX 4070 Laptop GPU，以及 Linux + Python 3.13（仅 CPU）上测试。

## 4. 运行

```bash
python make_demo_data.py                                             # 生成演示数据
python run_pipeline.py --config configs/demo_config.json --quick     # 冒烟测试（所有阶段，CPU 约 7 分钟，数值无意义）
python run_pipeline.py --config configs/demo_config.json             # 完整运行
```

单文件版（只含 MOBO-TimeXer，结果在 `results/demo/mobo_timexer/`）：

```bash
python mobo_timexer.py --config configs/demo_config.json --quick     # 冒烟测试（CPU 约 1–2 分钟）
python mobo_timexer.py --config configs/demo_config.json             # 5 种子 MOBO 寻优 + 10 种子最终训练
python mobo_timexer.py --config configs/demo_config.json --stage final --knee paper    # 跳过寻优，用论文膝点
```

单文件版参数：`--stage all|search|final`、`--knee search|paper|文件.json`、`--mobo-seeds 0-4`（每次寻优评估的种子）、
`--final-seeds 0-9`、`--output 目录`、`--quick`。

用自己的数据：复制 `configs/demo_config.json`，修改 `"data"` 部分（文件路径、时间列、目标列、外生列、采样间隔、划分），
其余设置不写则默认采用论文协议；先用 `--quick` 检查 `results/<run>/data/data_profile.json` 中的行数、间断和各段窗口数，再完整运行。

| 阶段 `--stage` | 内容 | 对应论文 |
|---|---|---|
| `audit` | 划分、各段统计量、标准化参数、持续性基线 | 3 节，表 3 |
| `analysis` | 0.03 Hz 零相位 Butterworth 分频、分频带 Pearson 相关、Welch 功率谱、能量占比与峰值周期 | 4.1 节，图 6–7，表 4 |
| `search` | 各模型 MOBO：Sobol + 批量 ParEGO，低保真训练（多种子平均），帕累托集、超体积、膝点 | 2.2、4.2 节，表 5，图 8–9 |
| `final` | 膝点配置在全部训练窗口上用种子 0–9 训练，所有测试段指标、逐窗口预测、模型权重 | 4.3–4.4 节，表 7–8 |
| `ablation` | A 默认结构+等权、B 膝点结构+等权、C 四个单任务 TimeXer、D MOBO-TimeXer | 4.4.2 节，表 9 |
| `report` | 多种子均值/标准差/CV/95% CI，Welch t 检验（Holm 校正）、Mann–Whitney U、Cohen's d，种子集成，相对最佳对比模型和持续性基线的改进，MOBO 汇总、膝点配置、训练成本、图，Excel 汇总表 | 表 5–9，图 8–15 |
| `all`（默认） | 依次运行以上阶段（`ablation` 需在配置中 `"enabled": true`） | |

常用参数：`--models timexer,xgboost`（只算部分模型）、`--seeds 0-9`、`--knee search|paper|文件.json`
（膝点来源：本次寻优 / 论文表 6 / 自定义）、`--quick`。所有阶段均可断点续跑：已完成的寻优评估和种子会自动跳过。

## 5. 主要可调设置

* `mobo.default`（对比模型：10 Sobol + 8×2 ParEGO、30,000/15,000 窗口、6 轮）与
  `mobo.timexer`（10 Sobol + 10×2 ParEGO、60,000/20,000 窗口、8 轮）。`seeds` 默认 `[0, 1, 2, 3, 4]`：每个候选配置用
  5 个种子分别做低保真训练，高斯过程、帕累托集和膝点都基于 5 个种子的平均验证 NMSE；逐种子结果和标准差保存在
  `mobo_trials_per_seed.csv` / `mobo_trials.csv`。
* 论文已发表的结果用的种子更少（MOBO-TimeXer 3 个种子 0–2，对比模型 1 个种子）；要逐位复现论文数值，设
  `"mobo": {"timexer": {"seeds": [0, 1, 2]}, "default": {"seeds": [42]}}`（单文件版用 `--mobo-seeds 0-2`）。
  **在新数据上做完全对称的比较时，建议所有模型使用同一预算**（把 `timexer` 的设置移到 `default`）。
* `final`：最终训练协议（论文：种子 0–9，最多 60 轮，早停耐心 15，批大小 1024，全部训练/验证窗口）。
* `model_options.timexer`：`ar_bypass`、`window_norm` 控制 2.1.2 节的两个组件，均为 `false` 即原始网络。
* `ablation.network`：`"revised"`（改进网络）或 `"original"`（论文消融所用原始网络）。
* `search_spaces`：替换某个模型的决策空间（格式同 `mobo.SEARCH_SPACES`）。

## 6. 输出

`results/<run>/` 下：`data/`（数据审计）、`analysis/`（耦合分析）、`mobo/<模型>/`（全部评估、逐种子目标值、帕累托前沿、
超体积过程、膝点 `best_config.json`）、`final/<模型>/seed_XX/`（指标、逐窗口预测 CSV、训练过程、模型权重）、
`ablation/`、`report/`（论文式表格、显著性检验、改进幅度、图和 `results_summary.xlsx`）。详见 `README.md` 第 6 节。

## 7. 复现性

* 用 `configs/paper_config.json` 读取论文数据时，训练/验证/极端测试/常规测试窗口数为 1,015,597 / 86,400 / 86,400 / 64,505，
  与论文一致；MOBO-TimeXer 第一个寻优配置（论文所用 3 个种子）的验证 NMSE 与归档结果在小数点后 6 位一致；
  膝点配置种子 0 的最终训练在两个测试段上各目标 NMSE 与归档结果一致（参数量 86,376）。
* 同一种子、同一 GPU 和库版本下训练是确定性的；换 GPU、用 CPU 或换库版本时单个种子的数值会略有差别，结论应以多种子统计为准。
* 运行时间（RTX 4070 Laptop GPU，约 100 万训练窗口）：MOBO-TimeXer 寻优（30 组 × 5 种子）约 12 分钟，单次最终训练约 3–5 分钟，6 个模型 × 10 个种子共需数小时。

## 8. 许可证

代码以 MIT 许可证开源（见 `LICENSE`）；现场监测数据不在许可范围内。
