# Trend Trader

Python 3.9+ toolkit for a simple moving-average trend-following strategy with backtesting, performance metrics (Sharpe, Sortino, max drawdown, Calmar, CAGR), and a small CLI.

## Features
- Moving-average plus ATR-based trailing stop.
- Vectorized backtester with fee and slippage modeling.
- Performance summary: Sharpe, Sortino, max drawdown, CAGR, Calmar, volatility, hit rate.
- Synthetic data generator so you can try the system without live market data.
- Pytest suite for metrics and backtester basics.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[dev]
```

Generate synthetic data and run a backtest:
```bash
python -m trend_trader.cli --generate-sample
```

Use your own CSV (must include Date, Open, High, Low, Close, Volume):
```bash
python -m trend_trader.cli --data my_prices.csv --fast 10 --slow 40 --atr 14 --atr-mult 2.5 --capital 75000 --fee 0.0003 --slippage 0.0002
```

## Running tests
```bash
pytest
```

## How it works
- `MovingAverageTrendStrategy` builds signals where the fast MA crosses above the slow MA. ATR provides a trailing stop reference.
- `Backtester` applies positions to price returns, deducts fees/slippage on position changes, and produces equity plus trade ledger.
- `metrics.summary_stats` calculates headline risk/return ratios from the equity curve and strategy returns.

## Data notes
- The synthetic generator creates 252 business days of gently trending OHLCV data; use `--generate-sample` to write it to `data/sample_prices.csv`.
- For real data, align columns as noted above; all timestamps are parsed via pandas and sorted ascending.

## Safety
This code is for research and education only. It does not constitute investment advice and has no live trading connectivity.
