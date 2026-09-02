import requests


class TradingExecutor:
    """
    Thin client for the simulated broker (server/broker-server.py).

    `portfolio_id` scopes every call to an independent simulated account, so
    the LLM agent pipeline and the Buy & Hold / Random Trade baselines can
    each run their own portfolio over the same ticker/date range without
    interfering with one another, while sharing the exact same execution and
    performance-measurement logic.
    """

    def __init__(self, api_url: str, portfolio_id: str = "default"):
        self.api_url = api_url
        self.portfolio_id = portfolio_id

    def buy_stock(self, ticker: str, quantity: float, date: str):
        response = requests.post(
            f"{self.api_url}/buy",
            json={
                "ticker": ticker,
                "quantity": quantity,
                "date": date,
                "portfolio_id": self.portfolio_id,
            }
        )
        return response.json()

    def sell_stock(self, ticker: str, quantity: float, date: str):
        response = requests.post(
            f"{self.api_url}/sell",
            json={
                "ticker": ticker,
                "quantity": quantity,
                "date": date,
                "portfolio_id": self.portfolio_id,
            }
        )
        return response.json()

    def mark_to_market(self, date: str, prices: dict):
        """
        Update this portfolio's equity curve for a date without trading
        (e.g. a daily close), so metrics like Sharpe/Max Drawdown reflect
        the whole holding period even between trades.
        """
        response = requests.post(
            f"{self.api_url}/mark/{self.portfolio_id}",
            json={"date": date, "prices": prices}
        )
        return response.json()

    def get_price(self, ticker: str, date: str | None = None):
        """Deterministic historical close price, for decisions/marks."""
        params = {"date": date} if date else {}
        response = requests.get(f"{self.api_url}/price/{ticker}", params=params)
        return response.json()

    def get_portfolio(self):
        response = requests.get(f"{self.api_url}/portfolio/{self.portfolio_id}")
        return response.json()

    def get_metrics(self):
        """
        Success Metrics (Cumulative Return, Sharpe Ratio, Max Drawdown, Win
        Rate, Avg Trade P&L) for this portfolio, plus the per-trade history.
        """
        response = requests.get(f"{self.api_url}/metrics/{self.portfolio_id}")
        return response.json()

    def reset(self, initial_cash: float = 100_000.0):
        """(Re)initialize this portfolio with fresh cash and an empty ledger."""
        response = requests.post(
            f"{self.api_url}/reset/{self.portfolio_id}",
            json={"cash": initial_cash}
        )
        return response.json()
