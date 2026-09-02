"""
Performance measurement for trading strategies (baselines and agent-driven
portfolios alike).

Implements the project's Success Metrics:
    - Cumulative Return : total portfolio gain vs. initial capital
    - Sharpe Ratio       : mean excess return / std of returns (annualized)
    - Maximum Drawdown   : largest peak-to-trough decline in equity
    - Win Rate           : fraction of profitable *closed* trades
    - Avg Trade P&L      : mean profit/loss per closed trade, net of fees

Design goal: metrics are recomputed and snapshotted after every single trade
(`record_fill`), not just at the end of a run, so the evolution of each
strategy's performance can be plotted/compared trade-by-trade. Between
trades, `mark_to_market` can be used to keep the equity curve (and therefore
the Sharpe/drawdown calculations) up to date on a daily basis, which matters
most for buy-and-hold style strategies that place a single trade and then
simply hold.

This module has no dependency on Flask, LangChain, or any specific broker —
it is a plain simulation ledger that anything (an LLM agent, a random
baseline, a buy-and-hold baseline) can report fills to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Fill:
    """A single executed order."""
    date: str
    ticker: str
    action: Literal["BUY", "SELL"]
    quantity: float
    price: float
    fee: float = 0.0

    @property
    def notional(self) -> float:
        return self.quantity * self.price

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "ticker": self.ticker,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "fee": self.fee,
        }


@dataclass
class ClosedTrade:
    """
    A round-trip position (shares bought, later sold), matched FIFO.
    Win Rate and Avg Trade P&L are computed over these, since "trade" in the
    spec means a completed buy->sell (or sell->buy for shorts) cycle, not a
    single order.
    """
    ticker: str
    open_date: str
    close_date: str
    quantity: float
    entry_price: float
    exit_price: float
    fees: float = 0.0

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity - self.fees

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "open_date": self.open_date,
            "close_date": self.close_date,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "fees": self.fees,
            "pnl": self.pnl,
        }


@dataclass
class EquityPoint:
    date: str
    equity: float
    cash: float
    positions_value: float
    event: Literal["trade", "mark"]

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "equity": self.equity,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "event": self.event,
        }


class PerformanceTracker:
    """
    Simulated long-only, single- or multi-ticker portfolio ledger.

    Typical usage:
        tracker = PerformanceTracker(initial_capital=100_000.0)
        snapshot = tracker.record_fill("2019-06-03", "AAPL", "BUY", 10, 185.2)
        # snapshot already contains Cumulative Return / Sharpe / Max DD /
        # Win Rate / Avg Trade P&L computed immediately after this trade.
        ...
        tracker.mark_to_market("2019-06-04", {"AAPL": 187.0})  # no trade, just marks equity
        ...
        final = tracker.summary()
    """

    def __init__(self, initial_capital: float = 100_000.0, fee_bps: float = 0.0):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.fee_bps = fee_bps  # simulated per-trade fee, in basis points of notional

        self.positions: dict[str, float] = {}
        # FIFO queue of open lots per ticker: [[qty, price, open_date], ...]
        self._lots: dict[str, list[list]] = {}

        self.fills: list[Fill] = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve: list[EquityPoint] = []
        self._last_prices: dict[str, float] = {}

        # One metrics snapshot appended after every trade (record_fill call).
        self.metrics_history: list[dict] = []

    # -----------------------------------------------------------------
    # Recording activity
    # -----------------------------------------------------------------

    def record_fill(
        self,
        date: str,
        ticker: str,
        action: str,
        quantity: float,
        price: float,
    ) -> dict:
        """
        Record one executed order, update cash/positions/closed trades,
        mark the equity curve, and return the performance snapshot computed
        immediately after this trade (this is the "calculate after each
        trade" hook the rest of the system should call).
        """
        action = action.upper()
        quantity = float(quantity)
        price = float(price)
        fee = abs(quantity) * price * (self.fee_bps / 10_000.0)

        if action == "BUY":
            self.cash -= quantity * price + fee
            self.positions[ticker] = self.positions.get(ticker, 0.0) + quantity
            self._lots.setdefault(ticker, []).append([quantity, price, date])

        elif action == "SELL":
            self.cash += quantity * price - fee
            self.positions[ticker] = self.positions.get(ticker, 0.0) - quantity
            self._match_fifo_close(ticker, quantity, price, date, fee)

        else:
            raise ValueError(f"Unsupported action: {action!r} (expected BUY or SELL)")

        self.fills.append(Fill(date, ticker, action, quantity, price, fee))
        self._last_prices[ticker] = price
        self._mark(date, event="trade")

        snapshot = self.metrics()
        snapshot["date"] = date
        snapshot["trade_number"] = len(self.fills)
        self.metrics_history.append(snapshot)
        return snapshot

    def mark_to_market(self, date: str, prices: dict[str, float]) -> EquityPoint:
        """
        Update last-known prices and append an equity-curve point WITHOUT a
        trade. Useful for buy-and-hold baselines (a single trade, then daily
        marks) so Sharpe/Max Drawdown reflect the whole holding period.
        """
        self._last_prices.update(prices)
        return self._mark(date, event="mark")

    def _mark(self, date: str, event: str) -> EquityPoint:
        positions_value = sum(
            qty * self._last_prices.get(ticker, 0.0)
            for ticker, qty in self.positions.items()
        )
        equity = self.cash + positions_value
        point = EquityPoint(
            date=date, equity=equity, cash=self.cash,
            positions_value=positions_value, event=event,
        )
        self.equity_curve.append(point)
        return point

    def _match_fifo_close(self, ticker, quantity, exit_price, date, fee):
        remaining = quantity
        lots = self._lots.get(ticker, [])
        total_qty = quantity if quantity else 1.0
        while remaining > 1e-9 and lots:
            lot_qty, lot_price, lot_date = lots[0]
            matched = min(lot_qty, remaining)
            self.closed_trades.append(ClosedTrade(
                ticker=ticker,
                open_date=lot_date,
                close_date=date,
                quantity=matched,
                entry_price=lot_price,
                exit_price=exit_price,
                fees=fee * (matched / total_qty),
            ))
            lot_qty -= matched
            remaining -= matched
            if lot_qty <= 1e-9:
                lots.pop(0)
            else:
                lots[0][0] = lot_qty
        # NOTE: if `remaining` is still > 0 here, more shares were sold than
        # were held/tracked (e.g. a short). This long-only simulator does not
        # model shorts, so any such excess is not turned into a closed trade.

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    @property
    def current_equity(self) -> float:
        if self.equity_curve:
            return self.equity_curve[-1].equity
        return self.cash

    def cumulative_return(self) -> float:
        """Total portfolio gain vs. initial capital over the test window."""
        if self.initial_capital == 0:
            return 0.0
        return (self.current_equity / self.initial_capital) - 1.0

    def _period_returns(self) -> list[float]:
        values = [p.equity for p in self.equity_curve]
        if len(values) < 2:
            return []
        return [
            (values[i] / values[i - 1]) - 1.0
            for i in range(1, len(values))
            if values[i - 1] != 0
        ]

    def sharpe_ratio(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ) -> Optional[float]:
        """Mean excess return / std of returns, annualized. None if undefined."""
        returns = self._period_returns()
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return None
        period_rf = risk_free_rate / periods_per_year
        return ((mean - period_rf) / std) * math.sqrt(periods_per_year)

    def max_drawdown(self) -> float:
        """Largest peak-to-trough decline in equity, as a negative fraction."""
        peak = None
        max_dd = 0.0
        for point in self.equity_curve:
            peak = point.equity if peak is None else max(peak, point.equity)
            if peak > 0:
                dd = (point.equity - peak) / peak
                max_dd = min(max_dd, dd)
        return max_dd

    def win_rate(self) -> Optional[float]:
        """Proportion of profitable *closed* trades out of all closed trades."""
        if not self.closed_trades:
            return None
        wins = sum(1 for t in self.closed_trades if t.pnl > 0)
        return wins / len(self.closed_trades)

    def avg_trade_pnl(self) -> Optional[float]:
        """Mean profit/loss per closed trade, net of simulated fees."""
        if not self.closed_trades:
            return None
        return sum(t.pnl for t in self.closed_trades) / len(self.closed_trades)

    def metrics(self) -> dict:
        """All five Success Metrics, plus a couple of useful counters."""
        return {
            "cumulative_return": self.cumulative_return(),
            "sharpe_ratio": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "win_rate": self.win_rate(),
            "avg_trade_pnl": self.avg_trade_pnl(),
            "equity": self.current_equity,
            "cash": self.cash,
            "num_fills": len(self.fills),
            "num_closed_trades": len(self.closed_trades),
        }

    def summary(self) -> dict:
        """Final-state metrics report, e.g. for end-of-run comparisons."""
        summary = self.metrics()
        summary["initial_capital"] = self.initial_capital
        summary["final_equity"] = self.current_equity
        return summary

    # -----------------------------------------------------------------
    # Serialization helpers (for the Flask broker / JSON responses)
    # -----------------------------------------------------------------

    def fills_as_dicts(self) -> list[dict]:
        return [f.to_dict() for f in self.fills]

    def closed_trades_as_dicts(self) -> list[dict]:
        return [t.to_dict() for t in self.closed_trades]

    def equity_curve_as_dicts(self) -> list[dict]:
        return [p.to_dict() for p in self.equity_curve]
