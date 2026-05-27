"""
Risk Management Agent
---------------------
Stateless LLM agent.  Called by the trader agent via run_risk_discussion().

The trader and risk agent exchange messages until risk concludes with:
  RISK_DECISION: APPROVE <qty> | REJECT | RESIZE <qty>
  STOP_LOSS: <price>
  REASON: <one line>
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from langchain.agents import create_agent

logger = logging.getLogger("risk_agent")

# -----------------------------------------------------------------------
# PORTFOLIO CONSTANTS  (single source of truth)
# -----------------------------------------------------------------------

PORTFOLIO_VALUE   = 100_000.0
MAX_EXPOSURE      = 0.35     # max single-position fraction
VOLATILITY_HARD   = 0.50     # reject above this ATR/price ratio
VOLATILITY_RESIZE = 0.30     # halve size between RESIZE and HARD thresholds
STOP_LOSS_PCT     = 0.05     # 5 % below entry price
MAX_DISCUSSION_TURNS = 4     # max trader↔risk back-and-forth rounds


# -----------------------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------------------

RISK_SYSTEM_PROMPT = f"""You are a risk management agent for an algorithmic trading system.

The trader will present a proposed trade. You MUST evaluate it against these rules:
- Hard reject if volatility (ATR/price) > {VOLATILITY_HARD}
- Halve position size if volatility > {VOLATILITY_RESIZE}
- Ensure post-trade exposure ≤ {MAX_EXPOSURE * 100:.0f}% of portfolio value
- Always set stop_loss = entry_price × {1 - STOP_LOSS_PCT}

You and the trader discuss the trade. You may ask for clarification or push back.
When you have reached a conclusion, end your final message with EXACTLY this block:

RISK_DECISION: APPROVE <int_qty> | REJECT | RESIZE <int_qty>
STOP_LOSS: <float>
REASON: <one concise sentence>

Do not include this block until you are ready to conclude. Be direct, no filler.
"""


# -----------------------------------------------------------------------
# AGENT FACTORY
# -----------------------------------------------------------------------

def create_risk_agent(model):
    return create_agent(
        model=model,
        tools=[],                  # risk agent reasons, does not execute
        system_prompt=RISK_SYSTEM_PROMPT,
    )


# -----------------------------------------------------------------------
# DISCUSSION RUNNER  (called from trader_agent tools)
# -----------------------------------------------------------------------

def run_risk_discussion(
    risk_agent,
    initial_trader_message: str,
) -> tuple[str, list[dict]]:
    """
    Run a multi-turn trader↔risk conversation.

    Returns:
        final_risk_message  – the risk agent's concluding message (contains RISK_DECISION block)
        full_history        – list of {"role": ..., "content": ...} dicts for logging
    """
    history: list[dict] = [
        {"role": "user", "content": initial_trader_message}
    ]

    final_message = ""

    for turn in range(MAX_DISCUSSION_TURNS):
        response = risk_agent.invoke({"messages": history})
        risk_reply = response["messages"][-1].content

        logger.info("[RISK turn %d] %s", turn + 1, risk_reply[:200])

        history.append({"role": "assistant", "content": risk_reply})
        final_message = risk_reply

        # Risk agent has concluded
        if "RISK_DECISION:" in risk_reply:
            break

        # Trader pushes back (simple acknowledgement to keep discussion going)
        # In a richer setup the trader agent would generate this reply too;
        # here we let the risk agent drive to conclusion to save tokens.
        history.append({
            "role": "user",
            "content": "Understood. Please provide your final RISK_DECISION block now.",
        })

    if "RISK_DECISION:" not in final_message:
        logger.warning("[RISK] No RISK_DECISION after %d turns – defaulting REJECT", MAX_DISCUSSION_TURNS)
        final_message = (
            "RISK_DECISION: REJECT\n"
            f"STOP_LOSS: 0.0\n"
            "REASON: Risk agent failed to conclude within allowed turns."
        )

    return final_message, history


# -----------------------------------------------------------------------
# PRICE / VOLATILITY HELPERS
# -----------------------------------------------------------------------

def get_price(ticker: str, date: str) -> float:
    path = Path("data") / "stock" / f"{ticker.upper()}.csv"
    if not path.exists():
        logger.warning("Price file not found for %s; using 100.0", ticker)
        return 100.0
    try:
        df = pd.read_csv(path)
        if str(df.iloc[0]["Date"]).upper() == ticker.upper():
            df = df.iloc[1:]
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["date"]  = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        row = df[df["date"] <= pd.to_datetime(date)].dropna(subset=["close"])
        return float(row.iloc[-1]["close"]) if not row.empty else 100.0
    except Exception as exc:
        logger.warning("Price lookup failed (%s); using 100.0", exc)
        return 100.0


def get_volatility(ticker: str, date: str, window: int = 14) -> float:
    path = Path("data") / "prices" / f"{ticker.upper()}.csv"
    if not path.exists():
        return 0.2
    try:
        df = pd.read_csv(path)
        if str(df.iloc[0]["Date"]).upper() == ticker.upper():
            df = df.iloc[1:]
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        for col in ("close", "high", "low"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["date"] <= pd.to_datetime(date)].dropna().tail(window + 1)
        if len(df) < 2:
            return 0.2
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).mean()
        price = df.iloc[-1]["close"]
        return round(float(atr / price), 4) if price else 0.2
    except Exception as exc:
        logger.warning("Volatility lookup failed (%s); using 0.2", exc)
        return 0.2
