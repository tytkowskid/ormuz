from flask import Flask, request, jsonify
import random

import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Simple in memory portfolio
portfolio = {
    "cash": 100000000000,
    "stocks": {}
}

prices = {
    "AAPL": 180,
    "MSFT": 420,
    "GOOG": 170,
    "TSLA": 200
}

def _load_price(ticker: str, date: str) -> float | None:
    path = Path("data/prices") / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    row = df[df["date"] <= pd.to_datetime(date)].dropna(subset=["close"])
    return float(row.iloc[-1]["close"]) if not row.empty else None

# GET portfolio
@app.route("/portfolio", methods=["GET"])
def get_portfolio():
    return jsonify(portfolio)

# GET price
@app.route("/prices/<ticker>", methods=["GET"])
def get_price(ticker):
    ticker = ticker.upper()
    if ticker not in prices:
        return jsonify({"error": "Unknown ticker"}), 404

    # simulate small random movement
    prices[ticker] += random.randint(-5, 5)

    return jsonify({
        "ticker": ticker,
        "price": prices[ticker]
    })

# BUY
@app.route("/buy", methods=["POST"])
def buy_stock():
    data = request.json

    ticker = data["ticker"].upper()
    quantity = data["quantity"]

    if ticker not in prices:
        return jsonify({"error": "Unknown ticker"}), 404
    
    date = data.get("date")
    price = _load_price(ticker, date) if date else prices[ticker]

    cost = price * quantity

    if portfolio["cash"] < cost:
        return jsonify({"error": "Not enough cash"}), 400

    portfolio["cash"] -= cost

    if ticker not in portfolio["stocks"]:
        portfolio["stocks"][ticker] = 0

    portfolio["stocks"][ticker] += quantity

    return jsonify({
        "message": "Stock bought",
        "portfolio": portfolio
    })

# SELL
@app.route("/sell", methods=["POST"])
def sell_stock():
    data = request.json

    ticker = data["ticker"].upper()
    quantity = data["quantity"]

    owned = portfolio["stocks"].get(ticker, 0)

    if owned < quantity:
        return jsonify({"error": "Not enough shares"}), 400
    
    date = data.get("date")  # caller must send this
    price = _load_price(ticker, date) if date else prices[ticker]

    revenue = price * quantity

    portfolio["stocks"][ticker] -= quantity
    portfolio["cash"] += revenue

    return jsonify({
        "message": "Stock sold",
        "portfolio": portfolio
    })

# favicon
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

if __name__ == "__main__":
    app.run(debug=True)