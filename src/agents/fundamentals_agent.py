from pathlib import Path
import pandas as pd

from langchain.agents import create_agent
from langchain.tools import tool


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_DIR = Path("data")
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"

INCOME_DIR = FUNDAMENTALS_DIR / "income_statement"
BALANCE_DIR = FUNDAMENTALS_DIR / "balance_sheet"
CASHFLOW_DIR = FUNDAMENTALS_DIR / "cash_flow"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _load_statement_csv(directory: Path, symbol: str) -> pd.DataFrame:

    symbol = symbol.upper()

    file_path = directory / f"{symbol}_quarterly.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Normalize columns
    df.columns = [
        col.strip().lower()
        for col in df.columns
    ]

    # Try converting date columns
    possible_date_cols = [
        "fiscaldateending",
        "date",
        "reporteddate"
    ]

    for col in possible_date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df


def _filter_period_type(
    df: pd.DataFrame,
    period_type: str
):
    """
    Filters annual or quarterly rows if possible.
    """

    period_type = period_type.lower()

    # Common Alpha Vantage style
    if "reporttype" in df.columns:
        return df[
            df["reporttype"].str.lower() == period_type
        ]

    # Fallback heuristic
    if period_type == "annual":
        return df.iloc[::4]

    return df


def _prepare_output(
    df: pd.DataFrame,
    limit: int = 5
):
    """
    Formats dataframe for LLM readability.
    """

    if df.empty:
        return "No data available."

    df = df.head(limit)

    return df.to_string(index=False)


# -------------------------------------------------------------------
# INCOME STATEMENT TOOL
# -------------------------------------------------------------------

@tool
def get_income_statement(
    symbol: str,
    period_type: str = "annual",
    limit: int = 5
) -> str:
    """
    Returns company income statement data from local CSV files.

    Args:
        symbol: Ticker symbol
        period_type: annual or quarterly
        limit: Number of rows returned

    Returns:
        Formatted income statement data.
    """

    try:

        df = _load_statement_csv(
            INCOME_DIR,
            symbol
        )

        df = _filter_period_type(
            df,
            period_type
        )

        return (
            f"Income Statement ({period_type}) "
            f"for {symbol}\n\n"
            + _prepare_output(df, limit)
        )

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------------------------------------------------
# BALANCE SHEET TOOL
# -------------------------------------------------------------------

@tool
def get_balance_sheet(
    symbol: str,
    period_type: str = "annual",
    limit: int = 5
) -> str:
    """
    Returns company balance sheet data from local CSV files.

    Args:
        symbol: Ticker symbol
        period_type: annual or quarterly
        limit: Number of rows returned

    Returns:
        Formatted balance sheet data.
    """

    try:

        df = _load_statement_csv(
            BALANCE_DIR,
            symbol
        )

        df = _filter_period_type(
            df,
            period_type
        )

        return (
            f"Balance Sheet ({period_type}) "
            f"for {symbol}\n\n"
            + _prepare_output(df, limit)
        )

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------------------------------------------------
# CASH FLOW TOOL
# -------------------------------------------------------------------

@tool
def get_cash_flow(
    symbol: str,
    period_type: str = "annual",
    limit: int = 5
) -> str:
    """
    Returns company cash flow statement data from local CSV files.

    Args:
        symbol: Ticker symbol
        period_type: annual or quarterly
        limit: Number of rows returned

    Returns:
        Formatted cash flow data.
    """

    try:

        df = _load_statement_csv(
            CASHFLOW_DIR,
            symbol
        )

        df = _filter_period_type(
            df,
            period_type
        )

        return (
            f"Cash Flow Statement ({period_type}) "
            f"for {symbol}\n\n"
            + _prepare_output(df, limit)
        )

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------------------------------------------------
# FUNDAMENTAL RATIOS TOOL
# -------------------------------------------------------------------

@tool
def calculate_fundamental_ratios(
    symbol: str,
    period_type: str = "annual"
) -> str:
    """
    Calculates important fundamental ratios using local financial data.

    Ratios:
    - Current Ratio
    - Debt to Equity
    - Net Profit Margin
    - Operating Margin
    - Free Cash Flow Margin

    Args:
        symbol: Ticker symbol
        period_type: annual or quarterly

    Returns:
        Formatted ratio analysis.
    """

    try:

        income_df = _load_statement_csv(
            INCOME_DIR,
            symbol
        )

        balance_df = _load_statement_csv(
            BALANCE_DIR,
            symbol
        )

        cash_df = _load_statement_csv(
            CASHFLOW_DIR,
            symbol
        )

        income_df = _filter_period_type(
            income_df,
            period_type
        )

        balance_df = _filter_period_type(
            balance_df,
            period_type
        )

        cash_df = _filter_period_type(
            cash_df,
            period_type
        )

        # Use latest row
        income = income_df.iloc[0]
        balance = balance_df.iloc[0]
        cash = cash_df.iloc[0]

        def safe_float(value):
            try:
                return float(value)
            except:
                return 0.0

        # -----------------------------------------------------------
        # EXTRACT VALUES
        # -----------------------------------------------------------

        revenue = safe_float(
            income.get("totalrevenue", 0)
        )

        net_income = safe_float(
            income.get("netincome", 0)
        )

        operating_income = safe_float(
            income.get("operatingincome", 0)
        )

        current_assets = safe_float(
            balance.get("totalcurrentassets", 0)
        )

        current_liabilities = safe_float(
            balance.get("totalcurrentliabilities", 0)
        )

        total_liabilities = safe_float(
            balance.get("totalliabilities", 0)
        )

        shareholder_equity = safe_float(
            balance.get("totalshareholderequity", 0)
        )

        operating_cashflow = safe_float(
            cash.get("operatingcashflow", 0)
        )

        capex = abs(
            safe_float(
                cash.get("capitalexpenditures", 0)
            )
        )

        # -----------------------------------------------------------
        # RATIOS
        # -----------------------------------------------------------

        current_ratio = (
            current_assets / current_liabilities
            if current_liabilities else 0
        )

        debt_to_equity = (
            total_liabilities / shareholder_equity
            if shareholder_equity else 0
        )

        net_margin = (
            net_income / revenue
            if revenue else 0
        )

        operating_margin = (
            operating_income / revenue
            if revenue else 0
        )

        free_cash_flow = (
            operating_cashflow - capex
        )

        fcf_margin = (
            free_cash_flow / revenue
            if revenue else 0
        )

        # -----------------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------------

        return f"""
Fundamental Ratio Analysis for {symbol}

Period Type: {period_type}

Liquidity:
- Current Ratio: {current_ratio:.2f}

Leverage:
- Debt to Equity: {debt_to_equity:.2f}

Profitability:
- Net Profit Margin: {net_margin:.2%}
- Operating Margin: {operating_margin:.2%}

Cash Flow:
- Free Cash Flow: {free_cash_flow:,.2f}
- Free Cash Flow Margin: {fcf_margin:.2%}
"""

    except Exception as e:
        return f"Error calculating ratios: {str(e)}"


# -------------------------------------------------------------------
# AGENT
# -------------------------------------------------------------------

def create_fundamentals_analyst_agent(model):

    SYSTEM_PROMPT = """
    You are a professional equity research and fundamentals analyst.

    Your task is to analyze company financial statements
    using local historical financial datasets.

    You have access to:
    - income statements
    - balance sheets
    - cash flow statements
    - calculated financial ratios

    Your responsibilities:
    - Assess company financial health
    - Analyze profitability trends
    - Evaluate liquidity and solvency
    - Examine operational efficiency
    - Detect financial risks
    - Compare annual vs quarterly performance
    - Identify growth or deterioration trends

    When responding:
    - Explain what the numbers mean
    - Highlight strengths and weaknesses
    - Mention important trends
    - Provide actionable investment insights
    - Support claims with actual financial values

    Focus on:
    - revenue growth
    - margins
    - debt levels
    - cash generation
    - operational performance
    - balance sheet quality
    """

    tools = [
        get_income_statement,
        get_balance_sheet,
        get_cash_flow,
        calculate_fundamental_ratios
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