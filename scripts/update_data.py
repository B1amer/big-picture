#!/usr/bin/env python3
"""Big Picture 数据管道 — 主入口

用法：
    python update_data.py              # 全量更新
    python update_data.py --only sp500  # 只更新标普500系列
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 确保能 import 同目录模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (DATA_DIR, STATIC_DIR, TICKERS, M7_BASE_DATE,
                    DRAWDOWN_LABELS, NDX_DRAWDOWN_LABELS, UPDATE_LOG,
                    FRED_API_KEY)
from fetchers import (fetch_yf, fetch_yf_monthly, fetch_fred,
                       fetch_shiller, save_json, load_existing,
                       load_static, today_str)
from processors import (calc_drawdown, calc_rolling_vol,
                         calc_monthly_returns, calc_intrayear_dd,
                         calc_annual_returns, calc_return_distribution,
                         detect_drawdown_events, calc_rolling_5y,
                         build_m7_index, calc_qqq_return_details)


# ── 基础价格数据构建 ─────────────────────────────────

def build_price_json(df: pd.DataFrame, ticker: str = "",
                     extra_meta: dict | None = None) -> dict:
    """构建价格 + drawdown JSON"""
    df = df.sort_index()
    close = df["close"]
    dd = calc_drawdown(close)

    series = []
    for date, row in df.iterrows():
        item = {"date": date.strftime("%Y-%m-%d"), "close": round(float(row["close"]), 2)}
        dd_val = dd.get(date)
        if pd.notna(dd_val):
            item["drawdown"] = round(float(dd_val), 6)
        series.append(item)

    result = {
        "updated": today_str(),
        "series": series,
    }
    if ticker:
        result["ticker"] = ticker
        result["latest"] = {
            "date": series[-1]["date"],
            "value": series[-1]["close"],
        }
    if extra_meta:
        result.update(extra_meta)
    return result


def build_century_json(df: pd.DataFrame, ticker: str = "",
                       scale: str = "logarithmic",
                       source: dict | None = None) -> dict:
    """构建百年月线 JSON"""
    df = df.sort_index()
    series = []
    for date, row in df.iterrows():
        val = row.get("close", row.get("value"))
        if pd.notna(val) and val > 0:
            series.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "value": round(float(val), 2),
            })

    if not series:
        return {"series": [], "updated": today_str()}

    result = {
        "updated": today_str(),
        "start": {"date": series[0]["date"], "value": series[0]["value"]},
        "latest": {"date": series[-1]["date"], "value": series[-1]["value"]},
        "cagr": round(float((series[-1]["value"] / series[0]["value"]) ** (1 / ((len(series)) / 12)) - 1), 4)
            if len(series) > 12 else 0,
        "scale": scale,
        "source": source or {},
        "series": series,
    }
    if ticker:
        result["ticker"] = ticker
    return result


def build_volatility_json(df: pd.DataFrame, ticker: str = "") -> dict:
    """构建波动率 JSON"""
    df = df.sort_index()
    close = df["close"]
    vol20 = calc_rolling_vol(close, 20)
    vol60 = calc_rolling_vol(close, 60)

    series = []
    for i, (date, row) in enumerate(df.iterrows()):
        item = {"date": date.strftime("%Y-%m-%d"), "close": round(float(row["close"]), 2)}
        v20 = vol20.iloc[i] if i < len(vol20) else None
        v60 = vol60.iloc[i] if i < len(vol60) else None
        if pd.notna(v20):
            item["vol20"] = round(float(v20), 6)
        if pd.notna(v60):
            item["vol60"] = round(float(v60), 6)
        series.append(item)

    result = {"updated": today_str(), "series": series}
    if ticker:
        result["ticker"] = ticker
    return result


def build_simple_series_json(df: pd.DataFrame, value_col: str = "value",
                              extra: dict | None = None) -> dict:
    """构建简单 value 序列 JSON（VIX/VXN/衰退等）"""
    series = []
    for date, row in df.iterrows():
        val = row[value_col]
        if pd.notna(val):
            series.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "value": round(float(val), 4),
            })

    result = {"updated": today_str(), "series": series}
    if series:
        result["latest"] = {"date": series[-1]["date"], "value": series[-1]["value"]}
    if extra:
        result.update(extra)
    return result


# ── 各模块更新函数 ───────────────────────────────────

def update_sp500_price():
    """更新标普500日频价格 + drawdown"""
    print("\n── 标普500 价格 ──")
    df = fetch_yf(TICKERS["sp500"])
    data = build_price_json(df, ticker="^GSPC")
    save_json("sp500_price.json", data)
    return df


def update_sp500_volatility(sp500_daily: pd.DataFrame = None):
    """更新标普500波动率"""
    print("\n── 标普500 波动率 ──")
    if sp500_daily is None:
        sp500_daily = fetch_yf(TICKERS["sp500"])
    data = build_volatility_json(sp500_daily)
    save_json("sp500_volatility.json", data)


def update_sp500_vix():
    """更新 VIX"""
    print("\n── VIX ──")
    df = fetch_yf(TICKERS["vix"])
    data = build_simple_series_json(df, "close",
                                     extra={"source": {"name": "CBOE", "url": "https://www.cboe.com/vix"}})
    save_json("sp500_vix.json", data)


def update_sp500_century():
    """更新标普500百年月线"""
    print("\n── 标普500 百年月线 ──")
    df = fetch_yf_monthly(TICKERS["sp500"])
    # 重命名为 value 列
    df["value"] = df["close"]
    data = build_century_json(df, source={
        "name": "yfinance ^GSPC monthly",
        "page": "https://finance.yahoo.com/quote/%5EGSPC/"
    })
    save_json("sp500_century.json", data)
    return df


def update_sp500_monthly(sp500_daily: pd.DataFrame = None):
    """更新标普500月度涨跌"""
    print("\n── 标普500 月度涨跌 ──")
    if sp500_daily is None:
        sp500_daily = fetch_yf(TICKERS["sp500"])
    data = calc_monthly_returns(sp500_daily)
    data["updated"] = today_str()
    save_json("sp500_monthly.json", data)


def update_sp500_intrayear_dd(sp500_daily: pd.DataFrame = None):
    """更新标普500年内回撤"""
    print("\n── 标普500 年内回撤 ──")
    if sp500_daily is None:
        sp500_daily = fetch_yf(TICKERS["sp500"])
    result = calc_intrayear_dd(sp500_daily)
    result.update({
        "updated": today_str(),
        "label": "标普500 年内最大回撤 vs 全年总回报",
        "source": "yfinance ^GSPC daily close",
    })
    save_json("sp500_intrayear_dd.json", result)


def update_sp500_annual_returns(sp500_century_df: pd.DataFrame = None):
    """更新标普500年度回报"""
    print("\n── 标普500 年度回报 ──")
    if sp500_century_df is None:
        sp500_century_df = fetch_yf_monthly(TICKERS["sp500"])
        sp500_century_df["value"] = sp500_century_df["close"]
    data = calc_annual_returns(sp500_century_df, source="yfinance ^GSPC monthly")
    save_json("sp500_annual_returns_long.json", data)
    return data


def update_sp500_annual_tr():
    """更新标普500年度总回报分布"""
    print("\n── 标普500 年度总回报分布 ──")
    try:
        df = fetch_yf(TICKERS["sp500tr"], period="max")
        df.index = pd.to_datetime(df.index)
        df["year"] = df.index.year

        annual_series = []
        for year, grp in df.groupby("year"):
            if len(grp) < 2:
                continue
            start = grp["close"].iloc[0]
            end = grp["close"].iloc[-1]
            ret = round(float((end - start) / start * 100), 2)
            annual_series.append({"year": int(year), "value": ret})

        data = calc_return_distribution(annual_series,
                                         source_label="yfinance ^SP500TR",
                                         return_kind="总回报口径（含股息）")
        save_json("sp500_annual_tr.json", data)
    except Exception as e:
        print(f"  ⚠ ^SP500TR 获取失败({e})，尝试从现有文件保留")
        UPDATE_LOG.append(f"sp500_annual_tr.json: skipped ({e})")


def update_sp500_drawdowns(sp500_daily: pd.DataFrame = None):
    """更新标普500回撤事件"""
    print("\n── 标普500 回撤事件 ──")
    if sp500_daily is None:
        sp500_daily = fetch_yf(TICKERS["sp500"])
    events = detect_drawdown_events(sp500_daily, labels=DRAWDOWN_LABELS)
    # 转换格式
    drawdowns = []
    for ev in events:
        drawdowns.append({
            "period": ev.get("period", ""),
            "high": ev["high"],
            "low": ev["low"],
            "days": ev.get("days_to_low", 0),
            "decline": ev["decline"],
            "recovery_days": ev.get("recovery_days"),
            "recovery_date": ev.get("recovery_date"),
            "category": ev.get("category", "unclassified"),
            "cause": ev.get("cause", ""),
            "active": ev.get("active", False),
        })
    save_json("sp500_drawdowns.json", {"drawdowns": drawdowns})


def update_sp500_rolling5y(sp500_century_df: pd.DataFrame = None):
    """更新标普500五年滚动"""
    print("\n── 标普500 五年滚动 ──")
    if sp500_century_df is None:
        sp500_century_df = fetch_yf_monthly(TICKERS["sp500"])
        sp500_century_df["value"] = sp500_century_df["close"]
    series = []
    for date, row in sp500_century_df.iterrows():
        if pd.notna(row.get("value")):
            series.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "value": round(float(row["value"]), 2),
            })
    data = calc_rolling_5y(series)
    data["source"] = "sp500_century.json 月线"
    save_json("sp500_rolling5y.json", data)


# ── 纳指系列 ─────────────────────────────────────────

def update_nasdaq_composite():
    """更新纳斯达克综指"""
    print("\n── 纳斯达克综指 ──")
    df = fetch_yf(TICKERS["nasdaq_comp"])
    df["value"] = df["close"]
    data = build_century_json(df, ticker="^IXIC", source={
        "name": "yfinance ^IXIC daily",
    })
    save_json("nasdaq_composite.json", data)
    return df


def update_ndx_daily():
    """更新纳指100日线"""
    print("\n── 纳指100 日线 ──")
    df = fetch_yf(TICKERS["ndx"])
    series = []
    for date, row in df.iterrows():
        series.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": round(float(row["close"]), 2),
        })
    save_json("ndx_daily.json", {
        "updated": today_str(),
        "ticker": "^NDX",
        "series": series,
    })
    return df


def update_ndx_price():
    """更新纳指100价格+回撤"""
    print("\n── 纳指100 价格 ──")
    df = fetch_yf(TICKERS["ndx"])
    data = build_price_json(df, ticker="^NDX")
    save_json("ndx_price.json", data)
    return df


def update_ndx_volatility(ndx_daily: pd.DataFrame = None):
    """更新纳指100波动率"""
    print("\n── 纳指100 波动率 ──")
    if ndx_daily is None:
        ndx_daily = fetch_yf(TICKERS["ndx"])
    data = build_volatility_json(ndx_daily, ticker="^NDX")
    save_json("ndx_volatility.json", data)


def update_ndx_vxn():
    """更新 VXN"""
    print("\n── VXN ──")
    df = fetch_yf(TICKERS["vxn"])
    data = build_simple_series_json(df, "close", extra={
        "ticker": "^VXN",
        "note": "CBOE Nasdaq-100 Volatility Index",
    })
    save_json("ndx_vxn.json", data)


def update_ndx_monthly(ndx_daily: pd.DataFrame = None):
    """更新纳指100月度涨跌"""
    print("\n── 纳指100 月度涨跌 ──")
    if ndx_daily is None:
        ndx_daily = fetch_yf(TICKERS["ndx"])
    data = calc_monthly_returns(ndx_daily)
    data["updated"] = today_str()
    data["ticker"] = "^NDX"
    save_json("ndx_monthly.json", data)


def update_ndx_intrayear_dd(ndx_daily: pd.DataFrame = None):
    """更新纳指100年内回撤"""
    print("\n── 纳指100 年内回撤 ──")
    if ndx_daily is None:
        ndx_daily = fetch_yf(TICKERS["ndx"])
    result = calc_intrayear_dd(ndx_daily)
    result.update({
        "updated": today_str(),
        "label": "纳指100 年内最大回撤 vs 全年总回报",
        "source": "ndx_daily.json (^NDX 日频)",
        "ticker": "^NDX",
    })
    save_json("ndx_intrayear_dd.json", result)


def update_ndx_drawdowns(ndx_daily: pd.DataFrame = None):
    """更新纳指100回撤事件"""
    print("\n── 纳指100 回撤事件 ──")
    if ndx_daily is None:
        ndx_daily = fetch_yf(TICKERS["ndx"])
    events = detect_drawdown_events(ndx_daily, labels=NDX_DRAWDOWN_LABELS)
    drawdowns = []
    for ev in events:
        drawdowns.append({
            "period": ev.get("period", ""),
            "high": ev["high"],
            "low": ev["low"],
            "days": ev.get("days_to_low", 0),
            "decline": ev["decline"],
            "recovery_days": ev.get("recovery_days"),
            "recovery_date": ev.get("recovery_date"),
            "category": ev.get("category", "unclassified"),
            "cause": ev.get("cause", ""),
            "active": ev.get("active", False),
        })
    save_json("ndx_drawdowns.json", {
        "updated": today_str(),
        "ticker": "^NDX",
        "drawdowns": drawdowns,
    })


def update_ndx_annual_returns():
    """更新纳指100年度回报"""
    print("\n── 纳指100 年度回报 ──")
    df = fetch_yf_monthly(TICKERS["ndx"])
    df["value"] = df["close"]
    data = calc_annual_returns(df, ticker="^NDX",
                                source="yfinance ^NDX 日线，价格回报口径（不含股息）")
    save_json("ndx_annual_returns_long.json", data)
    return data


def update_ndx_annual_tr():
    """更新纳指100年度回报分布"""
    print("\n── 纳指100 年度回报分布 ──")
    try:
        df = fetch_yf_monthly(TICKERS["ndx"])
        df["value"] = df["close"]
        annual_data = calc_annual_returns(df, ticker="^NDX", source="yfinance ^NDX")
        data = calc_return_distribution(annual_data["series"],
                                         source_label="yfinance ^NDX 日线",
                                         return_kind="价格回报口径（不含股息）")
        save_json("ndx_annual_tr.json", data)
    except Exception as e:
        print(f"  ⚠ 失败: {e}")
        UPDATE_LOG.append(f"ndx_annual_tr.json: failed ({e})")


def update_ndx_rolling5y():
    """更新纳指100五年滚动"""
    print("\n── 纳指100 五年滚动 ──")
    df = fetch_yf_monthly(TICKERS["ndx"])
    df["value"] = df["close"]
    series = []
    for date, row in df.iterrows():
        if pd.notna(row.get("value")):
            series.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "value": round(float(row["value"]), 2),
            })
    data = calc_rolling_5y(series)
    data["ticker"] = "^NDX"
    data["source"] = "ndx_daily.json 月线"
    save_json("ndx_rolling5y.json", data)


# ── 其他指数 ─────────────────────────────────────────

def update_dow_century():
    """更新道琼斯百年月线"""
    print("\n── 道琼斯百年月线 ──")
    df = fetch_yf_monthly(TICKERS["dow"])
    df["value"] = df["close"]
    data = build_century_json(df, source={
        "name": "yfinance ^DJI monthly",
        "page": "https://finance.yahoo.com/quote/%5EDJI/"
    })
    save_json("dow_jones_century.json", data)


def update_m7_index():
    """更新M7等权指数"""
    print("\n── M7 等权指数 ──")
    member_data = {}
    for ticker in TICKERS["m7_members"]:
        try:
            df = fetch_yf(ticker, period="5y")
            member_data[ticker] = df
        except Exception as e:
            print(f"  ⚠ {ticker} 获取失败: {e}")
    if len(member_data) < 5:
        print("  ⚠ M7 成员不足5个，跳过")
        return
    data = build_m7_index(member_data, M7_BASE_DATE)
    save_json("m7_index.json", data)


def update_qqq_return_details():
    """更新QQQ回报分解"""
    print("\n── QQQ 回报分解 ──")
    df = fetch_yf(TICKERS["qqq"])
    data = calc_qqq_return_details(df)
    save_json("qqq_return_details.json", data)


# ── TIER 2: 半自动 ──────────────────────────────────

def update_recessions():
    """更新美国衰退周期"""
    print("\n── 美国衰退周期 ──")
    df = fetch_fred("USREC")
    if df is None:
        print("  ⚠ 跳过（需 FRED_API_KEY）")
        return

    # 找衰退区间
    periods = []
    in_recession = False
    start = None
    for date, row in df.iterrows():
        if row["value"] == 1 and not in_recession:
            start = date.strftime("%Y-%m-%d")
            in_recession = True
        elif row["value"] == 0 and in_recession:
            periods.append({"start": start, "end": date.strftime("%Y-%m-%d")})
            in_recession = False
    if in_recession:
        periods.append({"start": start, "end": None})

    series = [{"date": d.strftime("%Y-%m-%d"), "value": int(row["value"])}
              for d, row in df.iterrows()]

    save_json("us_recessions.json", {
        "updated": today_str(),
        "source": {"name": "FRED USREC", "url": "https://fred.stlouisfed.org/series/USREC"},
        "periods": periods,
        "series": series,
    })


def update_shiller_pe_eps():
    """更新 Shiller PE 和 EPS"""
    print("\n── Shiller PE/EPS ──")
    df = fetch_shiller()
    if df is None:
        print("  ⚠ Shiller 数据获取失败，跳过")
        return

    # CAPE = real_price / 10年均 real_earn
    df["earnings_10y_avg"] = df["real_earn"].rolling(120).mean()  # 月度 10 年 = 120 月
    df["cape"] = df["real_price"] / df["earnings_10y_avg"]

    pe_series = []
    eps_series = []
    for date, row in df.iterrows():
        d = date if isinstance(date, str) else date.strftime("%Y-%m-%d")
        if pd.notna(row.get("cape")):
            pe_series.append({"date": d, "value": round(float(row["cape"]), 2)})
        if pd.notna(row.get("real_earn")) and row["real_earn"] > 0:
            eps_series.append({"date": d, "value": round(float(row["real_earn"]), 2)})

    save_json("sp500_pe.json", {
        "updated": today_str(),
        "cape": pe_series,
    })

    save_json("sp500_eps.json", {
        "updated": today_str(),
        "series": eps_series,
    })


# ── TIER 3: 手动文件检查 ─────────────────────────────

MANUAL_FILES = [
    "sp500_constituents.json",
    "sp500_sectors.json",
    "sp500_changes.json",
    "sp500_rules.json",
    "nasdaq100_panels.json",
    "sp500_roe.json",
    "sp500_return_details.json",
    "aiae.json",
]

def check_manual_files():
    """检查手动维护文件的新鲜度"""
    print("\n── 手动文件检查 ──")
    for fname in MANUAL_FILES:
        existing = load_existing(fname)
        if existing is None:
            print(f"  ⚠ {fname} 不存在")
        elif "updated" in existing:
            updated = existing["updated"]
            print(f"  · {fname} 最后更新: {updated}")
        else:
            print(f"  · {fname} 存在（无 updated 字段）")


# ── 主流程 ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Big Picture 数据更新")
    parser.add_argument("--only", help="只更新指定系列: sp500 / ndx / other", default=None)
    args = parser.parse_args()

    print(f"{'='*50}")
    print(f"Big Picture 数据更新 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    only = args.only

    # ── TIER 1: 全自动 ──
    # 每个函数独立 try/except，单个失败不阻断其他
    update_funcs_sp500 = [
        ("sp500_price", lambda: update_sp500_price()),
        ("sp500_volatility", lambda: update_sp500_volatility(sp500_daily)),
        ("sp500_vix", lambda: update_sp500_vix()),
        ("sp500_century", lambda: update_sp500_century()),
        ("sp500_monthly", lambda: update_sp500_monthly(sp500_daily)),
        ("sp500_intrayear_dd", lambda: update_sp500_intrayear_dd(sp500_daily)),
        ("sp500_drawdowns", lambda: update_sp500_drawdowns(sp500_daily)),
        ("sp500_annual_returns", lambda: update_sp500_annual_returns(sp500_century)),
        ("sp500_annual_tr", lambda: update_sp500_annual_tr()),
        ("sp500_rolling5y", lambda: update_sp500_rolling5y(sp500_century)),
    ]

    update_funcs_ndx = [
        ("ndx_daily", lambda: update_ndx_daily()),
        ("ndx_price", lambda: update_ndx_price()),
        ("ndx_volatility", lambda: update_ndx_volatility(ndx_daily)),
        ("ndx_vxn", lambda: update_ndx_vxn()),
        ("ndx_monthly", lambda: update_ndx_monthly(ndx_daily)),
        ("ndx_intrayear_dd", lambda: update_ndx_intrayear_dd(ndx_daily)),
        ("ndx_drawdowns", lambda: update_ndx_drawdowns(ndx_daily)),
        ("ndx_annual_returns", lambda: update_ndx_annual_returns()),
        ("ndx_annual_tr", lambda: update_ndx_annual_tr()),
        ("ndx_rolling5y", lambda: update_ndx_rolling5y()),
    ]

    update_funcs_other = [
        ("nasdaq_composite", lambda: update_nasdaq_composite()),
        ("dow_century", lambda: update_dow_century()),
        ("m7_index", lambda: update_m7_index()),
        ("qqq_return_details", lambda: update_qqq_return_details()),
    ]

    sp500_daily = None
    sp500_century = None
    ndx_daily = None

    if only is None or only == "sp500":
        for name, func in update_funcs_sp500:
            try:
                result = func()
                # 捕获可复用的数据
                if name == "sp500_price":
                    sp500_daily = result
                elif name == "sp500_century":
                    sp500_century = result
            except Exception as e:
                print(f"  ✗ {name} 失败: {e}")
                UPDATE_LOG.append(f"{name}: FAILED ({e})")

    if only is None or only == "ndx":
        for name, func in update_funcs_ndx:
            try:
                result = func()
                if name == "ndx_daily":
                    ndx_daily = result
            except Exception as e:
                print(f"  ✗ {name} 失败: {e}")
                UPDATE_LOG.append(f"{name}: FAILED ({e})")

    if only is None or only == "other":
        for name, func in update_funcs_other:
            try:
                func()
            except Exception as e:
                print(f"  ✗ {name} 失败: {e}")
                UPDATE_LOG.append(f"{name}: FAILED ({e})")

    # ── TIER 2: 半自动 ──
    try:
        update_recessions()
        update_shiller_pe_eps()
    except Exception as e:
        print(f"\n✗ TIER 2 更新出错: {e}")
        traceback.print_exc()

    # ── TIER 3: 手动文件检查 ──
    check_manual_files()

    # ── 报告 ──
    print(f"\n{'='*50}")
    print(f"更新完成，共处理 {len(UPDATE_LOG)} 个文件：")
    for log in UPDATE_LOG:
        print(f"  {log}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
