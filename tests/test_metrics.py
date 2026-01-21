import pandas as pd
from trend_trader.metrics import max_drawdown, sharpe_ratio, summary_stats


def test_max_drawdown_simple():
    eq = pd.Series([100, 120, 110, 140, 130])
    dd = max_drawdown(eq)
    assert round(dd, 4) == -0.0833


def test_sharpe_ratio_nonzero():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.0])
    sr = sharpe_ratio(returns, periods_per_year=252)
    assert sr > 0


def test_summary_stats_keys():
    eq = pd.Series([100, 101, 103, 102, 104])
    ret = eq.pct_change().fillna(0)
    stats = summary_stats(eq, ret)
    expected_keys = {"cagr", "sharpe", "sortino", "max_drawdown", "calmar", "volatility", "hit_rate"}
    assert expected_keys.issubset(stats.keys())
