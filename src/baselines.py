"""
Baseline trading strategies.

These exist to contextualize the LLM agent pipeline's performance:

  * Buy & Hold (S&P 500 index) — passive benchmark. Allocate capital at
    t=0, hold for the entire test period. Represents the market return
    floor.

  * Random Trade Agent — lower-bound control. Executes random buy / sell /
    hold decisions. It is intentionally *not* an intelligent agent (no LLM,
    no signal); any learned/orchestrated system must beat this to justify
    its added complexity.

Both baselines trade through the same simulated broker (server/broker-server.py)
and TradingExecutor as the LLM agent pipeline, each under its own
`portfolio_id`, so all three strategies' Success Metrics
(Cumulative Return, Sharpe Ratio, Max Drawdown, Win Rate, Avg Trade P&L)
are computed identically and are directly comparable. Metrics are recomputed
by the broker after every single trade (see PerformanceTracker.record_fill).
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Optional

from src.trading_executor import TradingExecutor

logger = logging.getLogger("baselines")


def trading_days(start: datetime, n: int):
    """Yield the next n weekday dates starting from (and including) `start`."""
    day, count = start, 0
    while count < n:
        if day.weekday() < 5:
            yield day
            count += 1
        day += timedelta(days=1)


class BuyAndHoldBaseline:
    """
    Passive benchmark. Allocates ~all starting capital to `ticker` on the
    first trading day, then holds for the entire test window (no further
    trades). Use ticker="SPY" (or another S&P 500 tracking instrument /
    local CSV) to represent "the market" itself.

    Because there is only ever one BUY, there are no *closed* trades, so
    Win Rate and Avg Trade P&L are undefined (reported as None/"n/a") for
    this baseline — Cumulative Return, Sharpe Ratio, and Max Drawdown are
    its meaningful metrics.
    """

    def __init__(
        self,
        api_url: str,
        ticker: str = "SPY",
        portfolio_id: str = "buy_and_hold",
        initial_cash: float = 100_000.0,
    ):
        self.ticker = ticker.upper()
        self.executor = TradingExecutor(api_url, portfolio_id=portfolio_id)
        self.initial_cash = initial_cash

    def run(self, start_date: str, num_days: int) -> dict:
        self.executor.reset(self.initial_cash)

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        dates = [d.strftime("%Y-%m-%d") for d in trading_days(start_dt, num_days)]

        entry_price = self.executor.get_price(self.ticker, dates[0])["price"]
        quantity = int(self.initial_cash // entry_price)  # whole shares, long-only

        if quantity > 0:
            logger.info("[BUY&HOLD] Buying %d shares of %s @ %.2f on %s", quantity, self.ticker, entry_price, dates[0])
            self.executor.buy_stock(self.ticker, quantity, dates[0])
        else:
            logger.warning("[BUY&HOLD] Not enough capital for a single share of %s @ %.2f", self.ticker, entry_price)

        for date in dates[1:]:
            price = self.executor.get_price(self.ticker, date)["price"]
            self.executor.mark_to_market(date, {self.ticker: price})

        return self.executor.get_metrics()


class RandomTradeAgent:
    """
    Lower-bound control. Each trading day, randomly chooses BUY / SELL /
    HOLD (with a random position size) — no market data, no learning, no
    signal of any kind. Any learned/orchestrated trading system must beat
    this baseline to justify its complexity.
    """

    def __init__(
        self,
        api_url: str,
        ticker: str,
        portfolio_id: str = "random",
        initial_cash: float = 100_000.0,
        max_trade_fraction: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.ticker = ticker.upper()
        self.executor = TradingExecutor(api_url, portfolio_id=portfolio_id)
        self.initial_cash = initial_cash
        self.max_trade_fraction = max_trade_fraction
        self.rng = random.Random(seed)

    def run(self, start_date: str, num_days: int) -> dict:
        self.executor.reset(self.initial_cash)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        for date in (d.strftime("%Y-%m-%d") for d in trading_days(start_dt, num_days)):
            price = self.executor.get_price(self.ticker, date)["price"]
            action = self.rng.choice(["BUY", "SELL", "HOLD"])
            portfolio = self.executor.get_portfolio()
            cash = portfolio["cash"]
            held = portfolio["stocks"].get(self.ticker, 0)

            if action == "BUY":
                budget = cash * self.rng.uniform(0.0, self.max_trade_fraction)
                quantity = int(budget // price) if price else 0
                if quantity > 0:
                    logger.info("[RANDOM] %s BUY %d %s @ %.2f", date, quantity, self.ticker, price)
                    self.executor.buy_stock(self.ticker, quantity, date)
                    continue

            elif action == "SELL" and held > 0:
                quantity = self.rng.randint(1, int(held))
                logger.info("[RANDOM] %s SELL %d %s @ %.2f", date, quantity, self.ticker, price)
                self.executor.sell_stock(self.ticker, quantity, date)
                continue

            # HOLD (or a BUY/SELL that couldn't execute): just mark equity.
            self.executor.mark_to_market(date, {self.ticker: price})

        return self.executor.get_metrics()
