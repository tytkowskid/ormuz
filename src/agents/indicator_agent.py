from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from langchain.agents import create_agent
from langchain.tools import tool


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_DIR = Path("data")
PRICE_DIR = DATA_DIR / "stock"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _load_price_data(symbol: str) -> pd.DataFrame:
    """
    Loads OHLCV stock data from CSV.
    """

    symbol = symbol.upper()

    file_path = PRICE_DIR / f"{symbol}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Price file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Handle malformed multi-header Yahoo Finance export
    # Remove second row if it contains ticker names
    if str(df.iloc[0]["Date"]).upper() == symbol:
        df = df.iloc[1:]

    df.columns = [
        col.strip().replace(" ", "_").lower()
        for col in df.columns
    ]

    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = [
        "adj_close",
        "close",
        "high",
        "low",
        "open",
        "volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date")

    return df


def _filter_dates(
    df: pd.DataFrame,
    curr_date: str,
    look_back_days: int
):
    curr_dt = pd.to_datetime(curr_date)

    start_dt = curr_dt - pd.Timedelta(days=look_back_days)

    return df[
        (df["date"] >= start_dt) &
        (df["date"] <= curr_dt)
    ]


# -------------------------------------------------------------------
# INDICATOR CALCULATIONS
# -------------------------------------------------------------------

def calculate_sma(df: pd.DataFrame, period: int):
    return df["close"].rolling(window=period).mean()


def calculate_ema(df: pd.DataFrame, period: int):
    return df["close"].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14):
    delta = df["close"].diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain_series = pd.Series(gain, index=df.index)
    loss_series = pd.Series(loss, index=df.index)

    avg_gain = gain_series.rolling(period).mean()
    avg_loss = loss_series.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(df: pd.DataFrame):
    ema12 = calculate_ema(df, 12)
    ema26 = calculate_ema(df, 26)

    macd = ema12 - ema26

    signal = macd.ewm(span=9, adjust=False).mean()

    histogram = macd - signal

    return macd, signal, histogram


def calculate_bollinger(df: pd.DataFrame, period: int = 20):
    middle = calculate_sma(df, period)

    std = df["close"].rolling(window=period).std()

    upper = middle + (2 * std)
    lower = middle - (2 * std)

    return middle, upper, lower


def calculate_atr(df: pd.DataFrame, period: int = 14):

    high_low = df["high"] - df["low"]

    high_close = np.abs(df["high"] - df["close"].shift())

    low_close = np.abs(df["low"] - df["close"].shift())

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_vwma(df: pd.DataFrame, period: int = 20):

    pv = df["close"] * df["volume"]

    vwma = (
        pv.rolling(period).sum() /
        df["volume"].rolling(period).sum()
    )

    return vwma


# -------------------------------------------------------------------
# TOOL
# -------------------------------------------------------------------

@tool
def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    time_period: int = 14
) -> str:
    """
    Calculates technical indicators from local historical OHLCV CSV files.

    Args:
        symbol: Ticker symbol (e.g. TSLA)
        indicator: Indicator name
        curr_date: Current trading date YYYY-MM-DD
        look_back_days: Number of days to look back
        time_period: Rolling period for indicator

    Returns:
        String with calculated indicator values.
    """

    supported_indicators = {
        "close_50_sma",
        "close_200_sma",
        "close_10_ema",
        "macds",
        "macdh",
        "rsi",
        "boll",
        "boll_ub",
        "boll_lb",
        "atr",
        "vwma"
    }

    if indicator not in supported_indicators:
        return (
            f"Unsupported indicator: {indicator}\n"
            f"Supported: {supported_indicators}"
        )

    try:

        df = _load_price_data(symbol)

        # -----------------------------------------------------------
        # CALCULATE INDICATORS
        # -----------------------------------------------------------

        if indicator == "close_50_sma":
            df["indicator"] = calculate_sma(df, 50)

        elif indicator == "close_200_sma":
            df["indicator"] = calculate_sma(df, 200)

        elif indicator == "close_10_ema":
            df["indicator"] = calculate_ema(df, 10)

        elif indicator == "rsi":
            df["indicator"] = calculate_rsi(df, time_period)

        elif indicator == "macds":
            _, signal, _ = calculate_macd(df)
            df["indicator"] = signal

        elif indicator == "macdh":
            _, _, hist = calculate_macd(df)
            df["indicator"] = hist

        elif indicator == "boll":
            middle, _, _ = calculate_bollinger(df)
            df["indicator"] = middle

        elif indicator == "boll_ub":
            _, upper, _ = calculate_bollinger(df)
            df["indicator"] = upper

        elif indicator == "boll_lb":
            _, _, lower = calculate_bollinger(df)
            df["indicator"] = lower

        elif indicator == "atr":
            df["indicator"] = calculate_atr(df)

        elif indicator == "vwma":
            df["indicator"] = calculate_vwma(df)

        # -----------------------------------------------------------
        # FILTER DATE RANGE
        # -----------------------------------------------------------

        filtered = _filter_dates(
            df,
            curr_date,
            look_back_days
        )

        filtered = filtered.dropna(subset=["indicator"])

        if filtered.empty:
            return "No indicator data available."

        # -----------------------------------------------------------
        # FORMAT OUTPUT
        # -----------------------------------------------------------

        output = []

        output.append(
            f"## {indicator.upper()} values for {symbol}"
        )

        output.append("")

        for _, row in filtered.iterrows():

            output.append(
                f"{row['date'].strftime('%Y-%m-%d')}: "
                f"{round(row['indicator'], 4)}"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Error calculating indicator: {str(e)}"


# -------------------------------------------------------------------
# AGENT
# -------------------------------------------------------------------

def create_indicators_analyst_agent(model):

    SYSTEM_PROMPT = """
    You are a professional financial indicators research analyst.

    Your role is to select and analyze companys financial indicators computed based on stock market data. Selec indicators in a way that will give you meaningful insight and avoids redundancy.

    Available indicators and their categories:

        Moving Averages:
        - close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
        - close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
        - close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

        MACD Related:
        - macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
        - macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
        - macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

        Momentum Indicators:
        - rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

        Volatility Indicators:
        - boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
        - boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
        - boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
        - atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

        Volume-Based Indicators:
        - vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

    When producing analysis:
    - Use actual financial values
    - Mention trends over time
    - Explain bullish and bearish signals
    - Focus on investment relevance
    - Be concise but data-driven
    """

    tools = [
        get_indicator,
    ]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    tool_names = ", ".join(
        tool.name for tool in tools
    )

    return agent, tool_names