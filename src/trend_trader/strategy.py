from dataclasses import dataclass
import pandas as pd


@dataclass
class MovingAverageTrendStrategy:
    fast_window: int = 20
    slow_window: int = 50
    atr_window: int = 14
    atr_mult: float = 2.0

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        price = data.copy()
        price["fast_ma"] = price["Close"].rolling(self.fast_window).mean()
        price["slow_ma"] = price["Close"].rolling(self.slow_window).mean()
        price["atr"] = self._atr(price, window=self.atr_window)
        price["trend"] = (price["fast_ma"] > price["slow_ma"]).astype(int)
        price["position"] = price["trend"].where(price["fast_ma"].notna() & price["slow_ma"].notna(), 0)
        # trailing stop based on ATR
        price["stop"] = price["Close"] - price["atr"] * self.atr_mult
        return price[["Date", "position", "stop", "fast_ma", "slow_ma", "atr"]]

    @staticmethod
    def _atr(df: pd.DataFrame, window: int) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()
