from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base for all backtesting strategies.

    Each concrete strategy receives a full OHLCV DataFrame and returns a
    signals DataFrame.  The contract:

      - signals["entry"][i] = True  →  open long at open[i+1]
      - signals["exit"][i]  = True  →  close long at open[i+1]

    Signals are computed over the full historical dataset at once (vectorised),
    but only backward-looking data is used — no look-ahead bias.
    Extra debug columns (indicator values) may also be included.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short display name shown in reports."""
        ...

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        data : DataFrame
            DatetimeIndex (UTC), columns: open, high, low, close, volume (all float).

        Returns
        -------
        DataFrame with same index, boolean columns 'entry' and 'exit',
        plus any optional indicator columns.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
