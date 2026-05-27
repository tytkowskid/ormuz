from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from langchain.agents import create_agent
from langchain.tools import tool


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_DIR = Path("data")

NEWS_DIR = DATA_DIR / "news"
TWITTER_DIR = DATA_DIR / "twitter"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path)


def _filter_by_date(df: pd.DataFrame, date_column: str, start_date, end_date):
    df[date_column] = pd.to_datetime(df[date_column])

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    return df[
        (df[date_column] >= start_dt) &
        (df[date_column] <= end_dt)
    ]


# -------------------------------------------------------------------
# NEWS TOOL
# -------------------------------------------------------------------

@tool
def get_news(
    ticker: str,
    start_date: str,
    end_date: str
) -> list[dict]:
    """
    Returns historical news sentiment data from local CSV files.

    Args:
        ticker: Stock ticker symbol (e.g. TSLA)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of filtered news records.
    """

    ticker = ticker.upper()

    file_path = NEWS_DIR / f"{ticker}_news_2019.csv"

    df = _load_csv(file_path)

    df = _filter_by_date(
        df,
        date_column="time_published",
        start_date=start_date,
        end_date=end_date
    )

    # Optional sorting
    df = df.sort_values("time_published", ascending=False)

    return df.to_dict(orient="records")


# -------------------------------------------------------------------
# TWITTER TOOL
# -------------------------------------------------------------------

@tool
def get_twitter_posts(
    ticker: str,
    start_date: str,
    end_date: str
) -> list[dict]:
    """
    Returns historical Twitter/X posts from local CSV files.

    Args:
        ticker: Stock ticker symbol (e.g. TSLA)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of filtered social media posts.
    """

    ticker = ticker.upper()

    file_path = TWITTER_DIR / f"{ticker}_2019.csv"

    df = _load_csv(file_path)

    df = _filter_by_date(
        df,
        date_column="post_date",
        start_date=start_date,
        end_date=end_date
    )

    # Optional sorting
    df = df.sort_values("post_date", ascending=False)

    return df.to_dict(orient="records")


# -------------------------------------------------------------------
# AGENT CREATION
# -------------------------------------------------------------------

def create_sentiment_analyst_agent(model):

    SYSTEM_PROMPT = """
    You are a financial news and sentiment analyst.

    Your task is to analyze historical news articles and social media
    sentiment for specific stock tickers using local CSV datasets.

    Available tools:
    - get_news(ticker, start_date, end_date)
    - get_twitter_posts(ticker, start_date, end_date)

    Use the data to:
    - Identify bullish or bearish sentiment trends
    - Detect major events or catalysts
    - Summarize market perception
    - Support trading and investment decisions
    - Compare news sentiment with social sentiment

    Provide concise but actionable insights supported by evidence
    from the retrieved data.
    """

    tools = [
        get_news,
        get_twitter_posts,
    ]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    tool_names = ", ".join(tool.name for tool in tools)

    return agent, tool_names