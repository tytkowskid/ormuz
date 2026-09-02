from langchain.tools import tool
from src.trading_executor import TradingExecutor

# The LLM agent pipeline always trades through the "agent" portfolio, so its
# performance metrics (see src/performance.py) can be fetched with
# TradingExecutor("http://127.0.0.1:5000", portfolio_id="agent").get_metrics()
# and compared directly against the Buy & Hold / Random Trade baselines,
# which trade through their own portfolio_ids (see src/baselines.py).
executor = TradingExecutor("http://127.0.0.1:5000", portfolio_id="agent")

@tool
def buy_stock(ticker: str, quantity: int, date: str):

    """
    Execute a stock purchase.
    """

    return executor.buy_stock(
        ticker,
        quantity,
        date
    )


@tool
def sell_stock(ticker: str, quantity: int, date: str):

    """
    Execute a stock sale.
    """

    return executor.sell_stock(
        ticker,
        quantity,
        date
    )