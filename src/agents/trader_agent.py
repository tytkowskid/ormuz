"""
Trader Agent
------------
LLM agent that receives the orchestrator's market brief and decides
whether to trade.

If the signal is not HOLD, the trader opens a discussion with the risk
management agent via the `consult_risk_manager` tool. They go back and
forth until risk concludes. The trader then calls `execute_trade` or
closes with no action.

Tools:
  consult_risk_manager(proposal_json) → risk discussion transcript + decision
  buy_stock(ticker, quantity)
  sell_stock(ticker, quantity)
"""

from __future__ import annotations

import json
import logging
from functools import partial

from langchain.agents import create_agent
from langchain.tools import tool

from src.agents.risk_agent import (
    PORTFOLIO_VALUE,
    create_risk_agent,
    get_price,
    get_volatility,
    run_risk_discussion,
)
from src.tools import buy_stock, sell_stock

logger = logging.getLogger("trader_agent")

# -----------------------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------------------

TRADER_SYSTEM_PROMPT = """You are an algorithmic trader agent.

You receive a structured market brief from the orchestrator containing:
- date and ticker
- each analyst's findings and signal (fundamentals, sentiment, indicators)
- orchestrator's verified consensus and confidence score
- current price, volatility, portfolio exposure

Your job:
1. Read the brief carefully.
2. If consensus is HOLD or confidence is too low, do nothing.
3. Otherwise, call consult_risk_manager with a JSON proposal. Engage with
   its responses until it issues a RISK_DECISION block.
4. If risk APPROVES or RESIZES, call buy_stock or sell_stock with the
   approved quantity.
5. If risk REJECTS, do nothing.

Be concise. No filler. Your final message must state what action you took
and why (one short paragraph).
"""


# -----------------------------------------------------------------------
# TOOL: consult_risk_manager
# -----------------------------------------------------------------------

def _make_consult_tool(risk_agent):
    """
    Closes over the risk_agent instance so the tool can invoke it.
    Returns a @tool-decorated function.
    """

    @tool
    def consult_risk_manager(proposal: str) -> str:
        """
        Open a discussion with the risk management agent about a proposed trade.

        Args:
            proposal: JSON string with keys: ticker, action, proposed_quantity,
                      price, volatility, current_exposure, portfolio_value,
                      confidence, analyst_summary.

        Returns:
            Full discussion transcript ending with the RISK_DECISION block.
        """
        logger.info("[TRADER] Opening risk discussion")
        final_msg, history = run_risk_discussion(risk_agent, proposal)

        transcript_lines = []
        for msg in history:
            role = "TRADER" if msg["role"] == "user" else "RISK"
            transcript_lines.append(f"[{role}]: {msg['content']}")
        transcript_lines.append(f"[RISK FINAL]: {final_msg}")

        transcript = "\n\n".join(transcript_lines)
        logger.info("[TRADER] Risk discussion complete:\n%s", transcript[-500:])
        return transcript

    return consult_risk_manager


# -----------------------------------------------------------------------
# AGENT FACTORY
# -----------------------------------------------------------------------

def create_trader_agent(model, risk_agent):
    consult_tool = _make_consult_tool(risk_agent)

    agent = create_agent(
        model=model,
        tools=[consult_tool, buy_stock, sell_stock],
        system_prompt=TRADER_SYSTEM_PROMPT,
    )
    return agent


# -----------------------------------------------------------------------
# PUBLIC ENTRY POINT
# -----------------------------------------------------------------------

def run_trader(
    trader_agent,
    brief: str,
    ticker: str,
    date: str,
) -> dict:
    """
    Pass the orchestrator's brief to the trader agent and return a result dict.
    """
    logger.info("[TRADER] Received brief for %s on %s", ticker, date)

    response = trader_agent.invoke({
        "messages": [{"role": "user", "content": brief}]
    })

    final = response["messages"][-1].content
    logger.info("[TRADER] Final: %s", final)

    return {
        "date": date,
        "ticker": ticker,
        "trader_conclusion": final,
        "message_count": len(response["messages"]),
    }
