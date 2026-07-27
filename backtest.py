"""
backtest.py
-----------
Core portfolio math: turns a set of asset prices + target weights into a
portfolio value time series, honoring the chosen rebalancing frequency.

This module has no Streamlit or plotting dependencies so it can be tested
and reasoned about in isolation.
"""

from __future__ import annotations

import pandas as pd

# Supported rebalancing frequencies, mapped to pandas period codes.
# "Buy & Hold" means weights drift with the market and are never reset.
REBALANCE_FREQUENCIES = {
    "Buy & Hold": None,
    "Monthly": "M",
    "Quarterly": "Q",
    "Annual": "Y",
}


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a price DataFrame into simple daily percentage returns.

    The first row of returns is always NaN (no prior day to compare to) and
    is dropped.
    """
    returns = prices.pct_change().dropna(how="all")
    return returns


def run_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    initial_investment: float,
    rebalance_frequency: str = "Buy & Hold",
) -> pd.DataFrame:
    """
    Simulate a portfolio's value over time given target weights and a
    rebalancing rule.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, columns = asset names, index = trading dates.
    weights : dict
        Target allocation, e.g. {"Stocks": 0.6, "Bonds": 0.3, "Gold": 0.1}.
        Must sum to 1.0 (within floating point tolerance).
    initial_investment : float
        Starting dollar amount.
    rebalance_frequency : str
        One of the keys in REBALANCE_FREQUENCIES.

    Returns
    -------
    pd.DataFrame
        Indexed by date, with columns:
        - one column per asset: dollar value held in that asset
        - "Total": total portfolio value
        - "Return": daily simple return of the total portfolio
    """
    assets = list(weights.keys())
    _validate_weights(weights)

    if prices.empty:
        raise ValueError("Price data is empty; cannot run backtest.")

    returns = compute_daily_returns(prices[assets])
    if returns.empty:
        raise ValueError("Not enough price history to compute returns.")

    dates = returns.index
    freq_code = REBALANCE_FREQUENCIES.get(rebalance_frequency)

    # Precompute which dates trigger a rebalance (the first trading day of
    # each new period, e.g. first day of each new month for "Monthly").
    rebalance_dates = _get_rebalance_dates(dates, freq_code)

    # Dollar value allocated to each asset, updated day by day.
    asset_values = {asset: initial_investment * weights[asset] for asset in assets}

    history = []
    for current_date in dates:
        day_returns = returns.loc[current_date]

        # Grow each asset's value by that day's return.
        for asset in assets:
            r = day_returns[asset]
            if pd.notna(r):
                asset_values[asset] *= (1 + r)

        total_value = sum(asset_values.values())

        # If today is a rebalance date, reset each asset back to its target
        # weight of the (post-growth) total value.
        if current_date in rebalance_dates and total_value > 0:
            for asset in assets:
                asset_values[asset] = total_value * weights[asset]

        row = {asset: asset_values[asset] for asset in assets}
        row["Total"] = total_value
        history.append(row)

    result = pd.DataFrame(history, index=dates)
    result["Return"] = result["Total"].pct_change()

    # Prepend the initial investment as day zero so charts/metrics have a
    # clean starting point.
    first_date = dates[0] - pd.Timedelta(days=1)
    initial_row = {asset: initial_investment * weights[asset] for asset in assets}
    initial_row["Total"] = initial_investment
    initial_row["Return"] = 0.0
    initial_df = pd.DataFrame([initial_row], index=[first_date])

    result = pd.concat([initial_df, result])
    return result


def _validate_weights(weights: dict[str, float]) -> None:
    """Raise a clear error if weights don't sum to ~100%."""
    total = sum(weights.values())
    if not (0.999 <= total <= 1.001):
        raise ValueError(
            f"Allocation weights must sum to 100%, got {total * 100:.1f}%."
        )
    for asset, w in weights.items():
        if w < 0:
            raise ValueError(f"Allocation for {asset} cannot be negative.")


def _get_rebalance_dates(dates: pd.DatetimeIndex, freq_code: str | None) -> set:
    """
    Determine the set of dates on which rebalancing should occur: the first
    trading day available in each new calendar period.

    Buy & Hold (freq_code is None) returns an empty set, meaning weights
    drift freely with market performance and are never reset.
    """
    if freq_code is None:
        return set()

    periods = dates.to_period(freq_code)
    # A rebalance happens whenever the period changes from the previous day.
    period_series = pd.Series(periods, index=dates)
    changed = period_series != period_series.shift(1)
    return set(dates[changed])
