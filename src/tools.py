from langchain.tools import tool
from src.trading_executor import TradingExecutor

executor = TradingExecutor("http://127.0.0.1:5000")

@tool
def buy_stock(ticker: str, quantity: int):

    """
    Execute a stock purchase.
    """

    return executor.buy_stock(
        ticker,
        quantity
    )


@tool
def sell_stock(ticker: str, quantity: int):

    """
    Execute a stock sale.
    """

    return executor.sell_stock(
        ticker,
        quantity
    )