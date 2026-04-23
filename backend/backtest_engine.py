"""
backtest_engine.py
Layer 2 — Livermore 2x2 Factorial Research
Calls the existing /backtest endpoint (app.py) for each stock.
All strategy logic lives in app.py — this file only orchestrates and extends.
"""

import os
import requests
import numpy as np
import pandas as pd

# Backend URL — self-call on same Render service, or override via env var
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:7860")

# 2x2 Stock Universe
STOCK_GROUPS = {
    "Industrial / High Volatility": {
        "tickers":  ["CAT", "XOM"],
        "thesis":   "Cyclical, commodity-driven — closest to Livermore's original 1920s market",
        "color":    "#c9922a",
        "sector":   "industrial",
        "vol_tier": "high"
    },
    "Industrial / Low Volatility": {
        "tickers":  ["JNJ", "KO"],
        "thesis":   "Defensive industrials — steady dividends, few breakout signals",
        "color":    "#2980b9",
        "sector":   "industrial",
        "vol_tier": "low"
    },
    "Tech / High Volatility": {
        "tickers":  ["NVDA", "TSLA"],
        "thesis":   "Modern narrative stocks — emotion-driven, like 1920s cotton futures",
        "color":    "#c0392b",
        "sector":   "tech",
        "vol_tier": "high"
    },
    "Tech / Low Volatility": {
        "tickers":  ["MSFT", "AAPL"],
        "thesis":   "Institutional-grade tech — fundamentals suppress breakout signals",
        "color":    "#27ae60",
        "sector":   "tech",
        "vol_tier": "low"
    }
}


# Derive extra metrics from cumulative return series

def _sharpe(cum_series):
    """Annualised Sharpe from cumulative % series."""
    arr   = np.array(cum_series) / 100.0
    daily = np.diff(arr) / (1 + arr[:-1] + 1e-9)
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return round(float(daily.mean() / daily.std() * np.sqrt(252)), 3)


def _max_drawdown(cum_series):
    """Max drawdown (%) from cumulative % series."""
    arr  = np.array(cum_series) / 100.0 + 1.0
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak
    return round(float(dd.min() * 100), 2)


def _volatility(cum_series):
    """Annualised historical volatility (%) from cumulative % series."""
    arr   = np.array(cum_series) / 100.0
    daily = np.diff(arr) / (1 + arr[:-1] + 1e-9)
    if len(daily) < 2:
        return 0.0
    return round(float(daily.std() * np.sqrt(252) * 100), 2)


def run_single(ticker, start, end):
    """
    Calls app.py's /backtest endpoint and returns enriched metrics.
    Uses identical Livermore strategy logic — no duplication.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/backtest",
            json={"symbol": ticker, "start_date": start, "end_date": end},
            timeout=60
        )
        resp.raise_for_status()
        d = resp.json()

        if "error" in d:
            return {"ticker": ticker, "error": d["error"]}

        strat = d.get("strategy_series", [])
        bh    = d.get("bh_series", [])

        return {
            "ticker":          ticker,
            "strategy_return": d.get("strategy_return", 0),
            "bh_return":       d.get("bh_return", 0),
            "outperformance":  d.get("outperformance", 0),
            "trade_count":     d.get("trade_count", 0),
            # Extended metrics derived from the returned series
            "sharpe":          _sharpe(strat),
            "max_drawdown":    _max_drawdown(strat),
            "avg_volatility":  _volatility(bh),
            "chart": {
                "dates":    d.get("dates", []),
                "strategy": strat,
                "bh":       bh
            }
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def run_comparison(start="2020-01-01", end="2025-01-01"):
    """
    Runs Livermore strategy across all 8 stocks via the existing /backtest endpoint.
    Returns structured results for the 2x2 matrix, heatmap, and hypothesis test.
    """
    stocks = []

    for group_name, info in STOCK_GROUPS.items():
        for ticker in info["tickers"]:
            result = run_single(ticker, start, end)
            if result and "error" not in result:
                result.update({
                    "group":    group_name,
                    "thesis":   info["thesis"],
                    "color":    info["color"],
                    "sector":   info["sector"],
                    "vol_tier": info["vol_tier"]
                })
                stocks.append(result)

    # Group-level averages
    df     = pd.DataFrame(stocks)
    groups = []

    for group_name, info in STOCK_GROUPS.items():
        g = df[df["group"] == group_name]
        if g.empty:
            continue
        groups.append({
            "group":          group_name,
            "thesis":         info["thesis"],
            "color":          info["color"],
            "sector":         info["sector"],
            "vol_tier":       info["vol_tier"],
            "tickers":        g["ticker"].tolist(),
            "avg_outperform": round(g["outperformance"].mean(), 2),
            "avg_sharpe":     round(g["sharpe"].mean(), 3),
            "avg_drawdown":   round(g["max_drawdown"].mean(), 2),
            "avg_volatility": round(g["avg_volatility"].mean(), 2),
        })

    # Hypothesis test: rho(avg_volatility, avg_sharpe) across 4 groups
    # Core claim: Livermore's Sharpe correlates with asset volatility
    gs         = pd.DataFrame(groups)
    hypothesis = {"correlation": None, "supported": None, "interpretation": "Insufficient data"}

    if len(gs) >= 3:
        corr      = float(gs["avg_volatility"].corr(gs["avg_sharpe"]))
        supported = corr > 0.3
        hypothesis = {
            "correlation": round(corr, 3),
            "supported":   supported,
            "interpretation": (
                "Hypothesis SUPPORTED — Higher volatility stocks yield better Livermore Sharpe"
                if supported else
                "Hypothesis NOT SUPPORTED — Volatility alone does not predict strategy effectiveness"
            )
        }

    return {
        "stocks":     stocks,
        "groups":     groups,
        "hypothesis": hypothesis,
        "period":     {"start": start, "end": end}
    }
