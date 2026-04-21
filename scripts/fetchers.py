"""Big Picture 数据管道 — 数据源获取"""

import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from config import DATA_DIR, STATIC_DIR, FRED_API_KEY, TICKERS, UPDATE_LOG


# ── 工具函数 ──────────────────────────────────────────

def save_json(filename: str, data: dict):
    """保存 JSON 到 data/ 目录"""
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False)
    status = f"updated → {path.name}"
    UPDATE_LOG.append(status)
    print(f"  ✓ {status}")


def load_existing(filename: str) -> dict | None:
    """读取已有 JSON，不存在返回 None"""
    path = DATA_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_static(filename: str) -> dict | None:
    """读取 static/ 目录的基准数据"""
    path = STATIC_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ── yfinance 获取 ─────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一 DataFrame 列名为小写简单名（处理 yfinance MultiIndex）"""
    if isinstance(df.columns, pd.MultiIndex):
        # 取第一层（Price 类型名），如 ('Close', 'AAPL') → 'close'
        df.columns = [c[0].lower().replace(" ", "_") for c in df.columns]
    else:
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # 去重（单 ticker 时 Close/High/Low 等可能重复）
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def fetch_yf(ticker: str, period: str = "max", interval: str = "1d",
             retry: int = 3, delay: float = 3.0) -> pd.DataFrame:
    """从 yfinance 获取数据，带重试和限速保护"""
    for attempt in range(retry):
        try:
            time.sleep(delay)
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError(f"yfinance 返回空数据: {ticker}")
            df = _normalize_columns(df)
            return df
        except Exception as e:
            is_rate_limit = "429" in str(e) or "Too Many" in str(e) or "Rate" in str(e)
            if attempt < retry - 1:
                # 限速错误等更久
                wait = (2 ** attempt) * (10 if is_rate_limit else 1)
                print(f"  ⚠ yfinance {ticker} 失败({e})，{wait}s 后重试...")
                time.sleep(wait)
            else:
                raise


def fetch_yf_monthly(ticker: str) -> pd.DataFrame:
    """获取月线数据"""
    return fetch_yf(ticker, interval="1mo")


# ── FRED 获取 ─────────────────────────────────────────

def fetch_fred(series_id: str) -> pd.DataFrame | None:
    """从 FRED API 获取数据（需 FRED_API_KEY）"""
    if not FRED_API_KEY:
        print(f"  ⚠ FRED_API_KEY 未设置，跳过 {series_id}")
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "1900-01-01",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("observations", [])
        df = pd.DataFrame(records)
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).set_index("date")
        return df
    except Exception as e:
        print(f"  ⚠ FRED {series_id} 获取失败: {e}")
        return None


# ── Shiller 数据 ──────────────────────────────────────

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"

def fetch_shiller() -> pd.DataFrame | None:
    """下载 Shiller 历史数据 Excel"""
    try:
        df = pd.read_excel(SHILLER_URL, sheet_name="Data", header=7)
        # Shiller Excel 格式：Date(如 1871.01), SP, Dividend, Earnings, CPI, ...
        df = df.iloc[:, :10]  # 只取前 10 列
        df.columns = ["date", "sp", "dividend", "earnings", "cpi",
                       "date_frac", "gs10", "real_price", "real_div",
                       "real_earn"]
        # 解析日期：1871.01 → 1871-01
        df["date"] = df["date"].apply(_parse_shiller_date)
        df = df.dropna(subset=["date"]).set_index("date")
        return df
    except Exception as e:
        print(f"  ⚠ Shiller 数据获取失败: {e}")
        return None


def _parse_shiller_date(val) -> str | None:
    """解析 Shiller 日期格式 1871.01 → 1871-01"""
    try:
        val = float(val)
        year = int(val)
        month = round((val - year) * 100)
        if month < 1 or month > 12:
            return None
        return f"{year}-{month:02d}"
    except (ValueError, TypeError):
        return None
