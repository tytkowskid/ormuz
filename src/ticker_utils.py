"""
Small shared helper for accepting either a single ticker or multiple
tickers everywhere a `ticker` argument is used (orchestrator pipeline,
baselines, evaluation CLI).

Kept dependency-free (no langchain/pandas/etc.) so it's cheap to import
from anywhere.
"""

from __future__ import annotations

from typing import Iterable, Union

TickerArg = Union[str, Iterable[str]]


def normalize_tickers(ticker: TickerArg) -> list[str]:
    """
    Accepts:
      - a single ticker:            "AAPL"
      - a comma-separated string:   "AAPL,MSFT, TSLA"
      - a list/tuple of tickers:    ["AAPL", "MSFT"]

    Returns a de-duplicated (order-preserving), upper-cased list of tickers.
    Raises ValueError if nothing usable is found.
    """
    if isinstance(ticker, str):
        raw = ticker.split(",")
    else:
        raw = list(ticker)

    seen: set[str] = set()
    tickers: list[str] = []
    for t in raw:
        t = str(t).strip().upper()
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    if not tickers:
        raise ValueError(f"No valid ticker(s) found in: {ticker!r}")

    return tickers
