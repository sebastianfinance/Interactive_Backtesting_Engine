"""
metrics.py
----------
Performance analytics computed from a portfolio's daily value/return series.

All functions take plain pandas Series/DataFrames (as produced by
`backtest.run_backtest`) and return plain Python floats or small DataFrames,
so they can be unit-tested without Streamlit or plotting involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # Kept simple/transparent for an educational tool.


def cagr(values: pd.Series) -> float:
    """
    Compound Annual Growth Rate: the constant annual growth rate that would
    take the starting value to the ending value over the elapsed period.
    """
    if len(values) < 2 or values.iloc[0] <= 0:
        return np.nan

    total_return = values.iloc[-1] / values.iloc[0]
    years = (values.index[-1] - values.index[0]).days / 365.25
    if years <= 0:
        return np.nan

    return total_return ** (1 / years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Standard deviation of daily returns, scaled to an annual figure."""
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return np.nan
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns: pd.Series) -> float:
    """
    Risk-adjusted return using total volatility as the risk measure.
    Sharpe = (annualized return - risk-free rate) / annualized volatility.
    """
    daily_returns = daily_returns.dropna()
    vol = annualized_volatility(daily_returns)
    if not vol or np.isnan(vol) or vol == 0:
        return np.nan
    mean_annual_return = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    return (mean_annual_return - RISK_FREE_RATE) / vol


def sortino_ratio(daily_returns: pd.Series) -> float:
    """
    Like Sharpe, but only penalizes downside volatility (negative returns),
    since most investors don't mind upside "volatility".
    """
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return np.nan

    downside = daily_returns[daily_returns < 0]
    if downside.empty:
        return np.nan

    downside_std = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan

    mean_annual_return = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    return (mean_annual_return - RISK_FREE_RATE) / downside_std


def drawdown_series(values: pd.Series) -> pd.Series:
    """
    Percentage drawdown at each point in time relative to the running
    peak value up to that point. Always <= 0.
    """
    running_max = values.cummax()
    return values / running_max - 1


def max_drawdown(values: pd.Series) -> float:
    """The single worst peak-to-trough decline over the period."""
    dd = drawdown_series(values)
    if dd.empty:
        return np.nan
    return dd.min()


def calmar_ratio(values: pd.Series) -> float:
    """CAGR divided by the magnitude of the maximum drawdown."""
    growth = cagr(values)
    mdd = max_drawdown(values)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return np.nan
    return growth / abs(mdd)


def annual_returns(values: pd.Series) -> pd.Series:
    """
    Calendar-year total return for each year present in the series.
    Uses the last available value of the prior year (or the first value in
    the series, for the first year) as the starting point.
    """
    values = values.sort_index()
    yearly_last = values.resample("YE").last()
    yearly_first = values.resample("YE").first()

    # For each year, the "starting" value should be the previous year's
    # ending value where available, else that year's first observed value.
    prev_year_end = yearly_last.shift(1)
    start_values = prev_year_end.combine_first(yearly_first)

    returns = (yearly_last / start_values) - 1
    returns.index = returns.index.year
    returns.name = "Return"
    return returns


def best_worst_year(values: pd.Series) -> tuple[float, float]:
    """Return (best year return, worst year return) as decimals."""
    yearly = annual_returns(values)
    if yearly.empty:
        return np.nan, np.nan
    return yearly.max(), yearly.min()


def summarize(portfolio: pd.DataFrame) -> dict:
    """
    Compute the full metrics bundle used by the dashboard's metric cards.

    Parameters
    ----------
    portfolio : pd.DataFrame
        Output of backtest.run_backtest (must contain "Total" and "Return").

    Returns
    -------
    dict with keys: cagr, volatility, sharpe, sortino, max_drawdown,
    calmar, best_year, worst_year
    """
    values = portfolio["Total"]
    daily_returns = portfolio["Return"]

    best_year, worst_year = best_worst_year(values)

    return {
        "cagr": cagr(values),
        "volatility": annualized_volatility(daily_returns),
        "sharpe": sharpe_ratio(daily_returns),
        "sortino": sortino_ratio(daily_returns),
        "max_drawdown": max_drawdown(values),
        "calmar": calmar_ratio(values),
        "best_year": best_year,
        "worst_year": worst_year,
    }
