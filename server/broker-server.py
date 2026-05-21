from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# Simple in memory portfolio
portfolio = {
    "cash": 10000,
    "stocks": {}
}

prices = {
    "AAPL": 180,
    "MSFT": 420,
    "GOOG": 170,
    "TSLA": 200
}

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

    cost = prices[ticker] * quantity

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

    revenue = prices[ticker] * quantity

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