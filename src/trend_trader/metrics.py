import numpy as np
import pandas as pd


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    years = len(equity_curve) / periods_per_year
    if years <= 0 or start <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def max_drawdown(equity_curve: pd.Series) -> float:
    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve / cumulative_max - 1
    return drawdown.min()


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return excess.mean() / downside.std() * np.sqrt(periods_per_year)


def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    dd = abs(max_drawdown(equity_curve))
    if dd == 0:
        return 0.0
    return cagr(equity_curve, periods_per_year=periods_per_year) / dd


def summary_stats(
    equity_curve: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    stats = {
        "cagr": cagr(equity_curve, periods_per_year=periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar": calmar_ratio(equity_curve, periods_per_year=periods_per_year),
        "volatility": returns.std() * np.sqrt(periods_per_year),
        "hit_rate": (returns > 0).mean(),
    }
    return stats
