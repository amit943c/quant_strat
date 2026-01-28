from dataclasses import dataclass
import pandas as pd

from .metrics import summary_stats


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    stats: dict


class Backtester:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_rate: float = 0.0005,
        slippage_rate: float = 0.0002,
        periods_per_year: int = 252,
    ) -> None:
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.periods_per_year = periods_per_year

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        price = data.merge(signals, on="Date", how="left").fillna({"position": 0})
        price = price.sort_values("Date").reset_index(drop=True)

        raw_returns = price["Close"].pct_change().fillna(0)
        position = price["position"].fillna(0)
        trade_change = position.diff().abs().fillna(0)
        cost = trade_change * (self.fee_rate + self.slippage_rate)
        # Support long (+1), short (-1), flat (0)
        strategy_returns = position.shift(1).fillna(0) * raw_returns - cost

        equity_curve = (1 + strategy_returns).cumprod() * self.initial_capital

        trades = self._extract_trades(price)
        stats = summary_stats(equity_curve, strategy_returns, periods_per_year=self.periods_per_year)

        # Add more stats output
        stats["final_equity"] = equity_curve.iloc[-1]
        stats["total_trades"] = len(trades)
        stats["avg_trade_return"] = trades["return"].mean() if not trades.empty else 0.0
        stats["avg_trade_duration"] = trades["bars_held"].mean() if not trades.empty else 0.0
        stats["win_rate"] = (trades["return"] > 0).mean() if not trades.empty else 0.0
        stats["loss_rate"] = (trades["return"] < 0).mean() if not trades.empty else 0.0
        stats["best_trade"] = trades["return"].max() if not trades.empty else 0.0
        stats["worst_trade"] = trades["return"].min() if not trades.empty else 0.0

        return BacktestResult(equity_curve=equality_safe(equity_curve), returns=strategy_returns, trades=trades, stats=stats)

    def _extract_trades(self, price: pd.DataFrame) -> pd.DataFrame:
        position = price["position"].fillna(0)
        entries = position.diff().fillna(0) > 0
        exits = position.diff().fillna(0) < 0
        trade_list = []
        entry_idx = None

        for idx, is_entry in entries.items():
            if is_entry:
                entry_idx = idx
            if exits.loc[idx] and entry_idx is not None:
                trade = self._build_trade(price, entry_idx, idx)
                trade_list.append(trade)
                entry_idx = None

        if entry_idx is not None:
            trade = self._build_trade(price, entry_idx, len(price) - 1)
            trade_list.append(trade)

        return pd.DataFrame(trade_list) if trade_list else pd.DataFrame(columns=[
            "entry_date", "exit_date", "entry_price", "exit_price", "return"])

    def _build_trade(self, price: pd.DataFrame, entry_idx: int, exit_idx: int) -> dict:
        entry_row = price.iloc[entry_idx]
        exit_row = price.iloc[exit_idx]
        gross_return = exit_row["Close"] / entry_row["Close"] - 1
        duration = exit_idx - entry_idx + 1
        return {
            "entry_date": entry_row["Date"],
            "exit_date": exit_row["Date"],
            "entry_price": entry_row["Close"],
            "exit_price": exit_row["Close"],
            "return": gross_return,
            "bars_held": duration,
        }


def equality_safe(series: pd.Series) -> pd.Series:
    # Avoid returning a view that may be mutated later.
    return pd.Series(series.values, index=series.index, name=series.name)
