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
        # Long if fast > slow, short if fast < slow, flat if equal
        price["position"] = 0
        price.loc[price["fast_ma"] > price["slow_ma"], "position"] = 1
        price.loc[price["fast_ma"] < price["slow_ma"], "position"] = -1
        price.loc[~(price["fast_ma"].notna() & price["slow_ma"].notna()), "position"] = 0
        # trailing stop based on ATR
        price["stop"] = price["Close"] - price["atr"] * self.atr_mult
        # Add more signal/stat columns
        price["ma_diff"] = price["fast_ma"] - price["slow_ma"]
        price["signal_change"] = price["position"].diff().fillna(0)
        price["volatility_20"] = price["Close"].rolling(20).std()
        return price[["Date", "position", "stop", "fast_ma", "slow_ma", "atr", "ma_diff", "signal_change", "volatility_20"]]

    @staticmethod
    def _atr(df: pd.DataFrame, window: int) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()
