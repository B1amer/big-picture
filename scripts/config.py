"""Big Picture 数据管道 — 配置与常量"""

import os
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
STATIC_DIR = SCRIPT_DIR / "static"
DATA_DIR.mkdir(exist_ok=True)

# ── 环境变量 ──────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── yfinance Ticker 映射 ──────────────────────────────
TICKERS = {
    "sp500": "^GSPC",
    "nasdaq_comp": "^IXIC",
    "ndx": "^NDX",
    "vix": "^VIX",
    "vxn": "^VXN",
    "qqq": "QQQ",
    "dow": "^DJI",
    "sp500tr": "^SP500TR",
    "m7_members": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
}

# ── M7 等权指数基准日 ─────────────────────────────────
M7_BASE_DATE = "2022-01-03"

# ── 回撤事件中文标注 ──────────────────────────────────
# key=回撤起点日期(YYYY-MM)，用于匹配自动检测到的回撤事件
DRAWDOWN_LABELS = {
    "1929-09": {"category": "narrative_break", "cause": "大萧条"},
    "1937-03": {"category": "policy_shift", "cause": "罗斯福紧缩"},
    "1946-05": {"category": "post_event_repricing", "cause": "战后调整"},
    "1961-12": {"category": "valuation_compression", "cause": "高估值回调"},
    "1968-11": {"category": "policy_shift", "cause": "尼克松政策"},
    "1973-01": {"category": "credit_destruction", "cause": "滞胀危机"},
    "1980-11": {"category": "policy_shift", "cause": "沃尔克加息"},
    "1987-08": {"category": "mechanical_scare", "cause": "黑色星期一"},
    "1990-07": {"category": "exogenous_shock", "cause": "海湾战争"},
    "1998-07": {"category": "contagion", "cause": "LTCM/亚洲金融危机"},
    "2000-03": {"category": "narrative_break", "cause": "互联网泡沫"},
    "2007-10": {"category": "credit_destruction", "cause": "全球金融危机"},
    "2011-04": {"category": "contagion", "cause": "欧债危机"},
    "2015-05": {"category": "mechanical_scare", "cause": "A股闪崩/人民币贬值"},
    "2018-01": {"category": "policy_shift", "cause": "贸易战/加息"},
    "2020-02": {"category": "exogenous_shock", "cause": "新冠疫情"},
    "2022-01": {"category": "policy_shift", "cause": "激进加息/通胀"},
    "2025-02": {"category": "policy_shift", "cause": "关税冲击"},
}

# NDX 回撤事件标注（2000年后共用大部分，补充 NDX 特有）
NDX_DRAWDOWN_LABELS = {
    "1998-07": {"category": "contagion", "cause": "LTCM/亚洲金融危机"},
    "2000-03": {"category": "narrative_break", "cause": "互联网泡沫"},
    "2007-10": {"category": "credit_destruction", "cause": "全球金融危机"},
    "2011-04": {"category": "contagion", "cause": "欧债危机"},
    "2018-01": {"category": "policy_shift", "cause": "贸易战/加息"},
    "2020-02": {"category": "exogenous_shock", "cause": "新冠疫情"},
    "2022-01": {"category": "policy_shift", "cause": "激进加息/通胀"},
    "2025-02": {"category": "policy_shift", "cause": "关税冲击"},
}

# ── 回撤分类 ──────────────────────────────────────────
DRAWDOWN_CATEGORIES = [
    {"id": "bear", "name": "熊市", "threshold": -0.20},
    {"id": "correction", "name": "回调", "threshold": -0.10},
    {"id": "dip", "name": "小幅回撤", "threshold": -0.05},
]

# ── 更新日志 ──────────────────────────────────────────
UPDATE_LOG = []  # 运行时填充，记录每个文件的状态
