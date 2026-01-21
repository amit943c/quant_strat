import pandas as pd

from trend_trader.data import generate_synthetic_data
from trend_trader.strategy import MovingAverageTrendStrategy
from trend_trader.backtester import Backtester


def test_backtester_runs():
    data = generate_synthetic_data(periods=120, start_price=50)
    strat = MovingAverageTrendStrategy(fast_window=10, slow_window=30)
    signals = strat.generate_signals(data)
    bt = Backtester(initial_capital=50_000, fee_rate=0.0, slippage_rate=0.0)
    result = bt.run(data, signals)
    assert len(result.equity_curve) == len(data)
    assert result.equity_curve.iloc[-1] > 0
    assert result.trades is not None
