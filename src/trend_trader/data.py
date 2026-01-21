from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def load_price_data(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data with a Date column, sorted ascending."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Price file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the price file")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    return df


def generate_synthetic_data(
    periods: int = 252,
    start_price: float = 100.0,
    daily_trend: float = 0.0006,
    daily_vol: float = 0.012,
    seed: Optional[int] = 7,
) -> pd.DataFrame:
    """Create a synthetic trending price series for quick experiments."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=daily_trend, scale=daily_vol, size=periods)
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    prices = prices[1:]
    dates = pd.bdate_range("2020-01-02", periods=periods)
    df = pd.DataFrame({
        "Date": dates,
        "Close": prices,
    })
    df["Open"] = df["Close"].shift(1, fill_value=start_price)
    df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + rng.uniform(0.0005, 0.002, periods))
    df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - rng.uniform(0.0005, 0.002, periods))
    df["Volume"] = 100000 + rng.integers(0, 10000, periods)
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    return df[cols]


def write_synthetic_csv(path: str | Path, periods: int = 252) -> Path:
    target = Path(path)
    df = generate_synthetic_data(periods=periods)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target
