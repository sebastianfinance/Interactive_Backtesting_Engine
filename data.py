"""
data.py
-------
Downloads, cleans, and caches historical price data for the three asset
classes used in the dashboard:

    Stocks : SPY     (SPDR S&P 500 ETF Trust)
    Bonds  : VBTIX   (Vanguard Total Bond Market Index Fund, Institutional)
    Gold   : GC=F    (COMEX Gold Futures, continuous front-month contract)

The dashboard always uses the *full* available history for these three
instruments (as far back as data exists for all of them), rather than a
user-selected date range. Because GC=F is a futures continuous contract,
Yahoo Finance's raw feed can occasionally contain data quirks (zero/negative
prints, single-day spikes from contract-roll artifacts, stale holiday
values, etc.). This module cleans those out so the rest of the app can
assume "prices" are trustworthy, gap-free, and aligned across all three
assets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Ticker symbol -> friendly asset-class name shown throughout the UI.
TICKERS = {
    "Stocks": "SPY",
    "Bonds": "VBTIX",
    "Gold": "GC=F",
}

# A single-day move larger than this (in absolute value) is treated as a
# probable bad tick / data artifact rather than a genuine market move, and
# is smoothed out via forward-fill. This matters most for the GC=F futures
# feed, which occasionally has stray prints around contract rollovers.
MAX_PLAUSIBLE_DAILY_MOVE = 0.40  # 40% in a single day


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)  # cache for 12 hours
def _download_raw(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Download the full available price history for each ticker (i.e. as far
    back as Yahoo Finance has data for that instrument).

    Returns
    -------
    pd.DataFrame
        Raw adjusted close prices, one column per ticker. May contain NaNs
        where an individual ticker has no data yet (e.g. before its
        inception) or on days it didn't trade.
    """
    raw = yf.download(
        list(tickers),
        period="max",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if raw is None or raw.empty:
        raise ValueError(
            "No price data was returned from Yahoo Finance. Check your "
            "network connection or the ticker symbols."
        )

    # yfinance returns a MultiIndex (ticker, field) when multiple tickers are
    # requested, but a flat index when only one ticker is requested.
    if isinstance(raw.columns, pd.MultiIndex):
        close_data = {}
        for ticker in tickers:
            if ticker in raw.columns.get_level_values(0):
                close_data[ticker] = raw[ticker]["Close"]
        prices = pd.DataFrame(close_data)
    else:
        prices = pd.DataFrame({tickers[0]: raw["Close"]})

    return prices.sort_index()


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw price data so downstream calculations never see bad ticks or
    gaps:

      1. Replace non-positive prices (0 or negative) with NaN -- these are
         never valid prices and indicate a feed error.
      2. Replace single-day moves larger than MAX_PLAUSIBLE_DAILY_MOVE with
         NaN, since these are almost always bad ticks rather than genuine
         market moves.
      3. Forward-fill any resulting gaps -- this also naturally handles
         holidays where one instrument trades but another doesn't (e.g.
         GC=F futures trading on a day the bond market is closed).
    """
    cleaned = prices.copy()

    # Step 1: kill non-positive prices outright.
    cleaned[cleaned <= 0] = np.nan

    # Step 2: kill implausible single-day jumps, per column.
    for col in cleaned.columns:
        series = cleaned[col]
        daily_return = series.pct_change()
        bad_ticks = daily_return.abs() > MAX_PLAUSIBLE_DAILY_MOVE
        cleaned.loc[bad_ticks, col] = np.nan

    # Step 3: forward-fill remaining gaps (does not fill leading NaNs before
    # an instrument's inception -- that's intentional, see get_asset_prices).
    cleaned = cleaned.ffill()

    return cleaned


def get_asset_prices() -> pd.DataFrame:
    """
    Return a clean, aligned price history for Stocks/Bonds/Gold, trimmed to
    the window where *all three* instruments have valid data -- i.e. the
    intersection of their histories, starting on the date the
    latest-inception instrument (typically GC=F) first has usable data.

    This means the dashboard always uses the longest common history
    available, with no manual date selection required.

    Returns
    -------
    pd.DataFrame
        Columns "Stocks", "Bonds", "Gold"; index = trading dates, fully
        aligned with no missing values.
    """
    tickers = tuple(TICKERS.values())
    raw = _download_raw(tickers)
    cleaned = _clean_prices(raw)

    reverse_map = {v: k for k, v in TICKERS.items()}
    cleaned = cleaned.rename(columns=reverse_map)
    ordered_cols = [name for name in TICKERS.keys() if name in cleaned.columns]
    cleaned = cleaned[ordered_cols]

    missing_tickers = [name for name in TICKERS if name not in cleaned.columns]
    if missing_tickers:
        raise ValueError(f"Could not retrieve data for: {', '.join(missing_tickers)}.")

    # Trim to the common window: drop any leading/trailing rows where any
    # single asset is still missing (e.g. before its inception date, or a
    # trailing gap that couldn't be forward-filled).
    cleaned = cleaned.dropna(how="any")

    if cleaned.empty:
        raise ValueError(
            "No overlapping price history was found across all three tickers."
        )

    # Final safety check: there should be zero missing values left anywhere
    # in the aligned window.
    if cleaned.isna().any().any():
        raise ValueError("Cleaned price data still contains missing values.")

    return cleaned


def get_backtest_window(prices: pd.DataFrame) -> dict:
    """
    Summarize the available backtesting window for display in the UI.

    Returns
    -------
    dict with keys: start (Timestamp), end (Timestamp), years (float)
    """
    start = prices.index[0]
    end = prices.index[-1]
    years = (end - start).days / 365.25
    return {"start": start, "end": end, "years": years}
