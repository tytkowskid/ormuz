import requests

class TradingExecutor:

    def __init__(self, api_url):
        self.api_url = api_url

    def buy_stock(self, ticker: str, quantity: int):

        response = requests.post(
            f"{self.api_url}/buy",
            json={
                "ticker": ticker,
                "quantity": quantity
            }
        )

        return response.json()

    def sell_stock(self, ticker: str, quantity: int):

        response = requests.post(
            f"{self.api_url}/sell",
            json={
                "ticker": ticker,
                "quantity": quantity
            }
        )

        return response.json()