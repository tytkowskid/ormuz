import random
import sys
from pathlib import Path

from flask import Flask, request, jsonify
import pandas as pd

# Allow `from src.performance import PerformanceTracker` when this file is
# run directly as `python server/broker-server.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.performance import PerformanceTracker  # noqa: E402

app = Flask(__name__)

DEFAULT_INITIAL_CASH = 100_000.0

# -----------------------------------------------------------------------
# MULTI-PORTFOLIO STATE
# -----------------------------------------------------------------------
# Each portfolio_id is an independent simulated account (its own cash,
# holdings, and PerformanceTracker). This lets the LLM agent pipeline
# (portfolio_id="agent") and the Buy & Hold / Random Trade baselines
# (portfolio_id="buy_and_hold" / "random") run over the same ticker/date
# range without clobbering each other, so their Success Metrics are
# directly comparable.
portfolios: dict[str, dict] = {}


def _get_or_create_portfolio(portfolio_id: str, initial_cash: float = DEFAULT_INITIAL_CASH) -> dict:
    if portfolio_id not in portfolios:
        portfolios[portfolio_id] = {
            "stocks": {},
            "tracker": PerformanceTracker(initial_capital=initial_cash),
        }
    return portfolios[portfolio_id]


# Fallback quote prices, used only when no historical CSV is available.
prices = {
    "AAPL": 180,
    "MSFT": 420,
    "GOOG": 170,
    "TSLA": 200,
    "SPY": 400,   # S&P 500 ETF proxy, used by the Buy & Hold baseline
}


def _load_historical_price(ticker: str, date: str) -> float | None:
    """Deterministic close price for a ticker/date from local CSV data."""
    path = Path("data/prices") / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    row = df[df["date"] <= pd.to_datetime(date)].dropna(subset=["close"])
    return float(row.iloc[-1]["close"]) if not row.empty else None


def _resolve_price(ticker: str, date: str | None) -> float | None:
    price = _load_historical_price(ticker, date) if date else None
    if price is None:
        price = prices.get(ticker)
    return price


# -----------------------------------------------------------------------
# PORTFOLIO / PRICE ENDPOINTS
# -----------------------------------------------------------------------

@app.route("/portfolio", methods=["GET"])
def get_default_portfolio():
    return _portfolio_view("default")


@app.route("/portfolio/<portfolio_id>", methods=["GET"])
def get_portfolio(portfolio_id):
    return _portfolio_view(portfolio_id)


def _portfolio_view(portfolio_id: str):
    p = _get_or_create_portfolio(portfolio_id)
    tracker = p["tracker"]
    return jsonify({
        "portfolio_id": portfolio_id,
        "cash": tracker.cash,
        "stocks": p["stocks"],
        "equity": tracker.current_equity,
    })


@app.route("/prices/<ticker>", methods=["GET"])
def get_live_price(ticker):
    """Noisy 'live' quote (random walk) — unchanged legacy behaviour."""
    ticker = ticker.upper()
    if ticker not in prices:
        return jsonify({"error": "Unknown ticker"}), 404

    prices[ticker] += random.randint(-5, 5)

    return jsonify({
        "ticker": ticker,
        "price": prices[ticker]
    })


@app.route("/price/<ticker>", methods=["GET"])
def get_historical_price(ticker):
    """
    Deterministic historical close price (optionally as of ?date=YYYY-MM-DD),
    used by baselines and backtests that need a stable price to decide/mark
    trades on, as opposed to the noisy /prices/<ticker> "live" quote above.
    """
    ticker = ticker.upper()
    date = request.args.get("date")
    price = _resolve_price(ticker, date)
    if price is None:
        return jsonify({"error": "Unknown ticker/date"}), 404
    return jsonify({"ticker": ticker, "date": date, "price": price})


# -----------------------------------------------------------------------
# TRADING ENDPOINTS
# -----------------------------------------------------------------------

@app.route("/buy", methods=["POST"])
def buy_stock():
    data = request.json

    ticker = data["ticker"].upper()
    quantity = float(data["quantity"])
    date = data.get("date")
    portfolio_id = data.get("portfolio_id", "default")

    price = _resolve_price(ticker, date)
    if price is None:
        return jsonify({"error": "Unknown ticker"}), 404

    p = _get_or_create_portfolio(portfolio_id)
    tracker = p["tracker"]
    cost = price * quantity

    if tracker.cash < cost:
        return jsonify({"error": "Not enough cash"}), 400

    p["stocks"][ticker] = p["stocks"].get(ticker, 0) + quantity

    # Record the fill and compute Success Metrics immediately after this trade.
    metrics_snapshot = tracker.record_fill(date or "unknown", ticker, "BUY", quantity, price)

    return jsonify({
        "message": "Stock bought",
        "portfolio_id": portfolio_id,
        "fill": {"ticker": ticker, "action": "BUY", "quantity": quantity, "price": price, "date": date},
        "portfolio": {"cash": tracker.cash, "stocks": p["stocks"]},
        "metrics": metrics_snapshot,
    })


@app.route("/sell", methods=["POST"])
def sell_stock():
    data = request.json

    ticker = data["ticker"].upper()
    quantity = float(data["quantity"])
    date = data.get("date")
    portfolio_id = data.get("portfolio_id", "default")

    p = _get_or_create_portfolio(portfolio_id)
    tracker = p["tracker"]
    owned = p["stocks"].get(ticker, 0)

    if owned < quantity:
        return jsonify({"error": "Not enough shares"}), 400

    price = _resolve_price(ticker, date)
    if price is None:
        return jsonify({"error": "Unknown ticker"}), 404

    p["stocks"][ticker] -= quantity

    # Record the fill and compute Success Metrics immediately after this trade.
    metrics_snapshot = tracker.record_fill(date or "unknown", ticker, "SELL", quantity, price)

    return jsonify({
        "message": "Stock sold",
        "portfolio_id": portfolio_id,
        "fill": {"ticker": ticker, "action": "SELL", "quantity": quantity, "price": price, "date": date},
        "portfolio": {"cash": tracker.cash, "stocks": p["stocks"]},
        "metrics": metrics_snapshot,
    })


@app.route("/mark/<portfolio_id>", methods=["POST"])
def mark_portfolio(portfolio_id):
    """
    Mark a portfolio's equity to market WITHOUT executing a trade — e.g. a
    daily close. Needed so buy-and-hold style strategies (one trade, then
    hold) still get a full equity curve for Sharpe / Max Drawdown.
    Body: {"date": "YYYY-MM-DD", "prices": {"TICKER": price, ...}}
    """
    data = request.json
    date = data["date"]
    price_map = {k.upper(): float(v) for k, v in data.get("prices", {}).items()}

    p = _get_or_create_portfolio(portfolio_id)
    point = p["tracker"].mark_to_market(date, price_map)

    return jsonify({"portfolio_id": portfolio_id, "date": date, "equity": point.equity})


# -----------------------------------------------------------------------
# PERFORMANCE / RESET ENDPOINTS
# -----------------------------------------------------------------------

@app.route("/metrics/<portfolio_id>", methods=["GET"])
def get_metrics(portfolio_id):
    """
    Current Success Metrics for a portfolio (Cumulative Return, Sharpe
    Ratio, Max Drawdown, Win Rate, Avg Trade P&L), plus the full
    metrics_history — one snapshot computed after each individual trade —
    so performance-over-time can be plotted per strategy.
    """
    p = _get_or_create_portfolio(portfolio_id)
    tracker = p["tracker"]
    return jsonify({
        "portfolio_id": portfolio_id,
        "metrics": tracker.summary(),
        "metrics_history": tracker.metrics_history,
        "fills": tracker.fills_as_dicts(),
        "closed_trades": tracker.closed_trades_as_dicts(),
        "equity_curve": tracker.equity_curve_as_dicts(),
    })


@app.route("/reset/<portfolio_id>", methods=["POST"])
def reset_portfolio(portfolio_id):
    """(Re)initialize a portfolio with fresh cash and an empty ledger."""
    data = request.json or {}
    initial_cash = float(data.get("cash", DEFAULT_INITIAL_CASH))
    portfolios[portfolio_id] = {
        "stocks": {},
        "tracker": PerformanceTracker(initial_capital=initial_cash),
    }
    return jsonify({"message": "reset", "portfolio_id": portfolio_id, "initial_cash": initial_cash})


# favicon
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')


if __name__ == "__main__":
    app.run(debug=True)
