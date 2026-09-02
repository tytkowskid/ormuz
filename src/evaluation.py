"""
Runs the Baselines (Buy & Hold, Random Trade Agent) — and, optionally, the
full LLM agent pipeline — over the same ticker/date range, then prints a
side-by-side comparison of the Success Metrics.

Requires server/broker-server.py to be running (default http://127.0.0.1:5000).

Usage:
    python -m src.evaluation AAPL 2019-06-01 20
    python -m src.evaluation AAPL 2019-06-01 20 --with-agent
    python -m src.evaluation AAPL 2019-06-01 20 --sp500-ticker SPY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from src.baselines import BuyAndHoldBaseline, RandomTradeAgent
from src.trading_executor import TradingExecutor

logger = logging.getLogger("evaluation")

API_URL = "http://127.0.0.1:5000"
INITIAL_CAPITAL = 100_000.0
SP500_TICKER = "SPY"

# label -> (display name, format string)
METRIC_LABELS = {
    "cumulative_return": ("Cumulative Return", "{:.2%}"),
    "sharpe_ratio": ("Sharpe Ratio", "{:.2f}"),
    "max_drawdown": ("Max Drawdown", "{:.2%}"),
    "win_rate": ("Win Rate", "{:.2%}"),
    "avg_trade_pnl": ("Avg Trade P&L", "${:,.2f}"),
}


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "n/a"
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def print_comparison(results: dict[str, dict]) -> None:
    print("\n=== PERFORMANCE COMPARISON ===")
    col_width = 22
    header = f"{'Metric':<20}" + "".join(f"{name:<{col_width}}" for name in results)
    print(header)
    print("-" * len(header))
    for key, (label, fmt) in METRIC_LABELS.items():
        row = f"{label:<20}"
        for name in results:
            value = results[name]["metrics"].get(key)
            row += f"{_fmt(value, fmt):<{col_width}}"
        print(row)
    print()


async def run_agent_pipeline(ticker: str, start_date: str, num_days: int) -> dict:
    """
    Runs the LLM agent pipeline (src/agents/orchestrator.py), which trades
    through the "agent" portfolio_id bound in src/tools.py, then fetches
    that portfolio's Success Metrics the same way the baselines do.
    """
    from src.agents.orchestrator import run_pipeline

    executor = TradingExecutor(API_URL, portfolio_id="agent")
    executor.reset(INITIAL_CAPITAL)
    await run_pipeline(ticker, start_date, num_days)
    return executor.get_metrics()


def main():
    parser = argparse.ArgumentParser(description="Baselines & performance comparison")
    parser.add_argument("ticker", help="Ticker traded by the Random Trade Agent (and agent pipeline, if enabled)")
    parser.add_argument("start_date", help="YYYY-MM-DD")
    parser.add_argument("num_days", type=int, help="Number of trading days to simulate")
    parser.add_argument("--with-agent", action="store_true", help="Also run the full LLM agent pipeline")
    parser.add_argument(
        "--sp500-ticker", default=SP500_TICKER,
        help="Ticker/CSV used to represent the S&P 500 index for the Buy & Hold baseline",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the Random Trade Agent")
    parser.add_argument("--output", default="performance_comparison.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    results: dict[str, dict] = {}

    print(f"Running Buy & Hold baseline on {args.sp500_ticker} (S&P 500 proxy)...")
    buy_hold = BuyAndHoldBaseline(API_URL, ticker=args.sp500_ticker, initial_cash=INITIAL_CAPITAL)
    results["Buy & Hold (S&P 500)"] = buy_hold.run(args.start_date, args.num_days)

    print(f"Running Random Trade Agent baseline on {args.ticker}...")
    random_agent = RandomTradeAgent(API_URL, ticker=args.ticker, initial_cash=INITIAL_CAPITAL, seed=args.seed)
    results["Random Trade Agent"] = random_agent.run(args.start_date, args.num_days)

    if args.with_agent:
        print(f"Running LLM agent pipeline on {args.ticker}...")
        results["Agent Pipeline"] = asyncio.run(run_agent_pipeline(args.ticker, args.start_date, args.num_days))

    print_comparison(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results (incl. per-trade metric history) to {args.output}")


if __name__ == "__main__":
    main()
