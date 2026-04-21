"""Big Picture 数据管道 — 计算逻辑"""

import numpy as np
import pandas as pd
from datetime import datetime


def _fmt_date(val) -> str:
    """安全格式化日期"""
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d")
    return str(val)[:10]


def _to_dt(val) -> datetime:
    """转为 datetime"""
    if isinstance(val, datetime):
        return val
    return pd.Timestamp(val).to_pydatetime()


# ── Drawdown ─────────────────────────────────────────

def calc_drawdown(closes: pd.Series) -> pd.Series:
    """计算从高点回撤，返回小数（如 -0.2537）"""
    running_max = closes.cummax()
    return (closes - running_max) / running_max


# ── 滚动波动率 ───────────────────────────────────────

def calc_rolling_vol(closes: pd.Series, window: int) -> pd.Series:
    """计算年化滚动波动率，返回小数（如 0.152）"""
    log_ret = np.log(closes / closes.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


# ── 月度涨跌 ─────────────────────────────────────────

def calc_monthly_returns(daily_df: pd.DataFrame) -> dict:
    """
    从日线数据计算月度涨跌。
    返回 {probability: {1..12: float}, years: [{year, months: {1..12}, annual}]}
    probability 和 months 值存小数（0.025 = 2.5%）
    """
    df = daily_df.copy()
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    df["month"] = df.index.month

    # 按月取最后一个交易日收盘价
    monthly = df.groupby(["year", "month"])["close"].last().reset_index()
    monthly = monthly.sort_values(["year", "month"])

    # 计算月度收益率
    monthly["ret"] = monthly["close"].pct_change()

    years_data = []
    for year, grp in monthly.groupby("year"):
        months = {}
        for _, row in grp.iterrows():
            m = int(row["month"])
            val = row["ret"]
            months[str(m)] = round(float(val), 6) if pd.notna(val) else None
        # 年度收益：最后一个/第一个
        year_close = grp["close"].iloc[-1]
        year_open = grp["close"].iloc[0]
        # 取前一年最后一个月的收盘作为年初价
        annual_ret = None
        if year > monthly["year"].min():
            prev = monthly[monthly["year"] == year - 1]
            if not prev.empty:
                year_open = prev["close"].iloc[-1]
                annual_ret = round(float((year_close - year_open) / year_open), 6)
        years_data.append({"year": int(year), "months": months, "annual": annual_ret})

    # 各月正收益概率
    probability = {}
    for m in range(1, 13):
        vals = [y["months"].get(str(m)) for y in years_data if y["months"].get(str(m)) is not None]
        if vals:
            probability[str(m)] = round(sum(1 for v in vals if v > 0) / len(vals), 4)
        else:
            probability[str(m)] = None

    return {"probability": probability, "years": years_data}


# ── 年内最大回撤 vs 全年涨幅 ─────────────────────────

def calc_intrayear_dd(daily_df: pd.DataFrame) -> dict:
    """
    计算每年年内最大回撤 vs 全年总回报。
    dd/tr 存小数（-0.087）
    """
    df = daily_df.copy()
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year

    annual = []
    for year, grp in df.groupby("year"):
        grp = grp.sort_index()
        close = grp["close"]
        running_max = close.cummax()
        dd_series = (close - running_max) / running_max
        max_dd = float(dd_series.min())

        year_start = close.iloc[0]
        year_end = close.iloc[-1]
        tr = float((year_end - year_start) / year_start)

        # 谷底日期
        trough_idx = dd_series.idxmin()
        trough_date = trough_idx.strftime("%Y-%m-%d")

        # 是否创年内新高
        ath = int(close.iloc[-1] >= close.max() - 0.01)

        annual.append({
            "year": int(year),
            "dd": round(max_dd, 6),
            "tr": round(tr, 6),
            "trough_date": trough_date,
            "year_end_close": round(float(year_end), 2),
            "year_end_date": grp.index[-1].strftime("%Y-%m-%d"),
            "ongoing": int(year) >= pd.Timestamp.now().year,
            "ath": ath,
        })

    # 汇总
    dds = [a["dd"] for a in annual]
    trs = [a["tr"] for a in annual]
    positive = sum(1 for t in trs if t > 0)

    summary = {
        "year_range": f"{annual[0]['year']}-{annual[-1]['year']}" if annual else "",
        "n_years": len(annual),
        "avg_dd": round(float(np.mean(dds)), 4),
        "avg_tr": round(float(np.mean(trs)), 4),
        "positive_years": positive,
        "positive_years_pct": round(positive / len(annual), 4) if annual else 0,
    }

    return {"summary": summary, "annual": annual}


# ── 年度回报 ─────────────────────────────────────────

def calc_annual_returns(monthly_df: pd.DataFrame, ticker: str = "",
                        source: str = "") -> dict:
    """
    从月线数据计算年度价格回报。
    value 存百分比数值（35.5 = +35.5%）
    """
    df = monthly_df.copy()
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year

    series = []
    for year, grp in df.groupby("year"):
        grp = grp.sort_index()
        if len(grp) < 2 and year == df["year"].max():
            # 当前年度 YTD
            start = grp["value"].iloc[0]
            end = grp["value"].iloc[-1]
        else:
            # 取前一年最后一个值作为年初
            prev = df[df["year"] == year - 1]
            if prev.empty:
                start = grp["value"].iloc[0]
            else:
                start = prev["value"].iloc[-1]
            end = grp["value"].iloc[-1]

        if pd.notna(start) and pd.notna(end) and start > 0:
            ret = round(float((end - start) / start * 100), 2)
            series.append({"date": f"{year}-12-31", "value": ret, "year": int(year)})

    if not series:
        return {"series": [], "average": 0, "positiveYears": 0, "negativeYears": 0}

    values = [s["value"] for s in series]
    pos = sum(1 for v in values if v > 0)
    neg = sum(1 for v in values if v < 0)
    best = max(series, key=lambda x: x["value"])
    worst = min(series, key=lambda x: x["value"])

    result = {
        "updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "average": round(float(np.mean(values)), 2),
        "positiveYears": pos,
        "negativeYears": neg,
        "best": {"year": best["year"], "value": best["value"]},
        "worst": {"year": worst["year"], "value": worst["value"]},
        "source": source or "yfinance",
        "series": series,
    }
    if ticker:
        result["ticker"] = ticker
    return result


# ── 年度总回报分布（分桶）──────────────────────────────

def calc_return_distribution(annual_series: list, source_label: str = "",
                              return_kind: str = "总回报口径（含股息）") -> dict:
    """
    将年度回报分桶统计。
    annual_series: [{year, value}] — value 为百分比数值
    """
    BINS = [
        {"key": "lt-40", "label": "<-40%", "min": -999, "max": -40},
        {"key": "-40to-30", "label": "-40%~-30%", "min": -40, "max": -30},
        {"key": "-30to-20", "label": "-30%~-20%", "min": -30, "max": -20},
        {"key": "-20to-10", "label": "-20%~-10%", "min": -20, "max": -10},
        {"key": "-10to0", "label": "-10%~0%", "min": -10, "max": 0},
        {"key": "0to10", "label": "0%~10%", "min": 0, "max": 10},
        {"key": "10to20", "label": "10%~20%", "min": 10, "max": 20},
        {"key": "20to30", "label": "20%~30%", "min": 20, "max": 30},
        {"key": "30to40", "label": "30%~40%", "min": 30, "max": 40},
        {"key": "40to50", "label": "40%~50%", "min": 40, "max": 50},
        {"key": "gt50", "label": ">50%", "min": 50, "max": 9999},
    ]

    buckets = []
    for b in BINS:
        years_in = [s["year"] for s in annual_series
                    if b["min"] <= s["value"] < b["max"]]
        buckets.append({
            "key": b["key"],
            "label": b["label"],
            "min": b["min"],
            "max": b["max"],
            "years": years_in,
        })

    values = [s["value"] for s in annual_series]
    avg = float(np.mean(values)) if values else 0
    positive = sum(1 for v in values if v > 0)
    within2 = sum(1 for v in values if abs(v - avg) <= 2)

    return {
        "updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "source": source_label,
        "returnKind": return_kind,
        "latestYear": annual_series[-1]["year"] if annual_series else None,
        "latestDate": f"{annual_series[-1]['year']}-12-31" if annual_series else None,
        "latestIsYtd": annual_series[-1]["year"] >= pd.Timestamp.now().year if annual_series else False,
        "average": round(avg, 2),
        "positiveYears": positive,
        "totalYears": len(annual_series),
        "withinAvgPlusMinus2": within2,
        "buckets": buckets,
    }


# ── 回撤事件检测 ─────────────────────────────────────

def detect_drawdown_events(daily_df: pd.DataFrame,
                           labels: dict | None = None,
                           min_decline: float = 0.10) -> list:
    """
    从日线数据检测 ≥ min_decline 的回撤事件。
    labels: {YYYY-MM: {category, cause}} 映射表
    decline 存小数（-0.5678）
    """
    df = daily_df.copy()
    df = df.sort_index().reset_index(drop=False)
    dates = df.iloc[:, 0]  # 第一列是日期
    close = df["close"]

    running_max = close.cummax()
    dd = (close - running_max) / running_max

    events = []
    in_drawdown = False
    peak_i = None
    peak_price = None

    for i in range(len(df)):
        if dd.iloc[i] >= 0 and in_drawdown:
            # 回撤结束，创新高
            recovery_date = dates.iloc[i]
            events.append({
                "peak_date": _fmt_date(dates.iloc[peak_i]),
                "recovery_date": _fmt_date(recovery_date),
                "high": round(float(peak_price), 2),
                "low": round(float(trough_price), 2),
                "trough_date": _fmt_date(dates.iloc[trough_i]),
                "decline": round(float(trough_decline), 4),
                "days_to_low": int((_to_dt(dates.iloc[trough_i]) - _to_dt(dates.iloc[peak_i])).days),
                "recovery_days": int((_to_dt(recovery_date) - _to_dt(dates.iloc[trough_i])).days),
            })
            in_drawdown = False

        if dd.iloc[i] < 0 and not in_drawdown:
            # 开始新回撤
            in_drawdown = True
            peak_i = i - 1 if i > 0 else i
            peak_price = close.iloc[peak_i]
            trough_price = close.iloc[i]
            trough_decline = dd.iloc[i]
            trough_i = i

        if in_drawdown and dd.iloc[i] < trough_decline:
            trough_price = close.iloc[i]
            trough_decline = dd.iloc[i]
            trough_i = i

    # 处理进行中的回撤
    if in_drawdown:
        events.append({
            "peak_date": _fmt_date(dates.iloc[peak_i]),
            "recovery_date": None,
            "high": round(float(peak_price), 2),
            "low": round(float(trough_price), 2),
            "trough_date": _fmt_date(dates.iloc[trough_i]),
            "decline": round(float(trough_decline), 4),
            "days_to_low": int((_to_dt(dates.iloc[trough_i]) - _to_dt(dates.iloc[peak_i])).days),
            "recovery_days": None,
        })

    # 过滤小回撤，添加 category/cause 标注
    result = []
    for ev in events:
        if abs(ev["decline"]) < min_decline:
            continue
        period = f"{ev['peak_date']} ~ {ev['trough_date']}"
        ev["period"] = period
        ev["active"] = ev["recovery_date"] is None

        # 查标注
        peak_month = ev["peak_date"][:7]
        label = (labels or {}).get(peak_month, {})
        ev["category"] = label.get("category", "unclassified")
        ev["cause"] = label.get("cause", "")

        # 分类
        for cat in [{"id": "bear", "threshold": -0.20}, {"id": "correction", "threshold": -0.10}]:
            if ev["decline"] <= cat["threshold"]:
                ev["category_id"] = cat["id"]
                break
        else:
            ev["category_id"] = "dip"

        result.append(ev)

    # 按 decline 降序
    result.sort(key=lambda x: x["decline"])
    return result


# ── 5年滚动年化 ──────────────────────────────────────

def calc_rolling_5y(monthly_series: list) -> dict:
    """
    从月度序列计算 5 年滚动年化回报。
    value 存百分比数值
    monthly_series: [{date, value}]
    """
    df = pd.DataFrame(monthly_series)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    series = []
    for i in range(60, len(df)):  # 60 个月 = 5 年
        start_price = df.loc[i - 60, "value"]
        end_price = df.loc[i, "value"]
        start_date = df.loc[i - 60, "date"]
        end_date = df.loc[i, "date"]

        if start_price > 0 and end_price > 0:
            annualized = ((end_price / start_price) ** (1 / 5) - 1) * 100
            series.append({
                "date": end_date.strftime("%Y-%m-%d"),
                "value": round(float(annualized), 2),
                "startDate": start_date.strftime("%Y-%m-%d"),
                "startPrice": round(float(start_price), 2),
                "endPrice": round(float(end_price), 2),
            })

    if not series:
        return {"series": [], "latest": None, "average": 0, "negativePercent": 0}

    values = [s["value"] for s in series]
    neg_pct = round(sum(1 for v in values if v < 0) / len(values), 4)

    return {
        "updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "latest": series[-1]["value"],
        "average": round(float(np.mean(values)), 2),
        "negativePercent": neg_pct,
        "source": "computed from monthly data",
        "series": series,
    }


# ── M7 等权指数 ──────────────────────────────────────

def build_m7_index(member_data: dict, base_date: str) -> dict:
    """
    构建 M7 等权指数。
    member_data: {ticker: DataFrame(close)}
    base_date: 基准日，当天所有成员归一化为 100
    """
    # 对齐到同一日期范围
    dfs = []
    for ticker, df in member_data.items():
        s = df["close"].copy()
        s.name = ticker
        dfs.append(s)

    combined = pd.concat(dfs, axis=1).dropna()
    # 归一化：以 base_date 为基准
    base_idx = combined.index.get_indexer([pd.Timestamp(base_date)], method="nearest")
    if base_idx[0] < 0:
        base_idx[0] = 0

    base_prices = combined.iloc[base_idx[0]]
    normalized = combined / base_prices * 100

    # 等权平均
    index_series = normalized.mean(axis=1)

    members = []
    for ticker in member_data:
        col = normalized[ticker]
        base_price = float(combined[ticker].iloc[base_idx[0]])
        latest_price = float(combined[ticker].iloc[-1])
        ret_pct = round((latest_price / base_price - 1) * 100, 2)

        members.append({
            "ticker": ticker,
            "name": _TICKER_NAMES.get(ticker, ticker),
            "basePrice": round(base_price, 2),
            "returnPct": ret_pct,
            "series": [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
                       for d, v in col.items()],
        })

    index_list = [{"date": d.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
                  for d, v in index_series.items()]

    return {
        "updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "baseDate": base_date,
        "latest": index_list[-1]["value"] if index_list else None,
        "methodology": {
            "description": "按复权收盘价先各自归一化到基准日=100，再做等权平均",
            "members": list(member_data.keys()),
        },
        "source": {"name": "Yahoo Finance", "url": "https://finance.yahoo.com"},
        "members": members,
        "indexSeries": index_list,
    }


_TICKER_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
    "AMZN": "Amazon", "GOOGL": "Alphabet", "META": "Meta Platforms",
    "TSLA": "Tesla",
}


# ── QQQ 回报分解 ─────────────────────────────────────

def calc_qqq_return_details(qqq_df: pd.DataFrame) -> dict:
    """
    从 QQQ 日线数据计算年度回报分解。
    yfinance adjusted close 已含分红复权，用 total_return = adj_close 变化
    priceReturn = close 变化（不含分红）
    dividendReturn ≈ totalReturn - priceReturn（简化处理）
    buybackYield 需要额外数据，这里留 0
    """
    df = qqq_df.copy()
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year

    # yfinance auto_adjust=True 时 close 即复权价
    # 需要同时获取不复权价格来计算 priceReturn
    # 简化：用 close 列作为复权价，另取不复权价
    series = []
    for year, grp in df.groupby("year"):
        if len(grp) < 2:
            continue
        start_close = grp["close"].iloc[0]
        end_close = grp["close"].iloc[-1]
        price_ret = round(float((end_close - start_close) / start_close * 100), 2)
        # 分红贡献简化为 0（QQQ 分红占比极小）
        series.append({
            "year": int(year),
            "priceReturn": price_ret,
            "dividendReturn": 0.0,
            "buybackYield": 0.0,
            "totalReturn": price_ret,
        })

    return {
        "updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "ticker": "QQQ",
        "note": "auto_adjust=True，分红贡献简化为0，buybackYield需手动补充",
        "buybackCoverage": "无（需手动从 S&P Dow Jones Indices 报告获取）",
        "series": series,
    }
