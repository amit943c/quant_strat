import argparse
from pathlib import Path
import pandas as pd

from .data import load_price_data, write_synthetic_csv
from .strategy import MovingAverageTrendStrategy
from .backtester import Backtester


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Trend trading backtester")
    parser.add_argument("--data", type=Path, default=Path("data/sample_prices.csv"), help="Path to CSV with Date, Open, High, Low, Close, Volume")
    parser.add_argument("--fast", type=int, default=20, help="Fast moving average window")
    parser.add_argument("--slow", type=int, default=50, help="Slow moving average window")
    parser.add_argument("--atr", type=int, default=14, help="ATR window")
    parser.add_argument("--atr-mult", type=float, default=2.0, help="ATR multiple for stop")
    parser.add_argument("--capital", type=float, default=100000.0, help="Starting capital")
    parser.add_argument("--fee", type=float, default=0.0005, help="Per-trade fee rate")
    parser.add_argument("--slippage", type=float, default=0.0002, help="Per-trade slippage rate")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic data to the provided path if it does not exist")

    args = parser.parse_args()

    if not args.data.exists():
        created = write_synthetic_csv(args.data)
        if not args.generate_sample:
            print(f"No data found; wrote synthetic sample to {created}")

    data = load_price_data(args.data)
    strat = MovingAverageTrendStrategy(
        fast_window=args.fast,
        slow_window=args.slow,
        atr_window=args.atr,
        atr_mult=args.atr_mult,
    )
    signals = strat.generate_signals(data)

    bt = Backtester(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
    )
    result = bt.run(data, signals)

    latest = result.stats
    print("=== Performance ===")
    for k, v in latest.items():
        if k in {"max_drawdown"}:
            print(f"{k:12s}: {v:.2%}")
        else:
            print(f"{k:12s}: {v:.4f}")

    print("\n=== Equity Curve (tail) ===")
    print(result.equity_curve.tail())


if __name__ == "__main__":
    run_cli()
