"""
Orchestrator Agent
------------------
An LLM agent that:
  1. Dispatches the 3 analyst agents in parallel (no look-ahead)
  2. Receives their raw outputs as tool results
  3. Verifies each response (flags missing signal, bad data, contradictions)
  4. Merges them into a structured brief keeping each analyst's voice distinct
  5. Summarises the overall picture with a consensus recommendation
  6. Passes the brief to the Trader Agent

The orchestrator is itself an LLM agent with one tool: `collect_analyst_reports`.
That tool runs the analysts in parallel and returns their raw outputs.
The LLM then does the verification, merging, and summarisation step.

Entry point:  run_pipeline(ticker, start_date, num_days)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from src.agents.fundamentals_agent import create_fundamentals_analyst_agent
from src.agents.indicator_agent import create_indicators_analyst_agent
from src.agents.sentiment_agent import create_sentiment_analyst_agent
from src.agents.risk_agent import (
    PORTFOLIO_VALUE,
    create_risk_agent,
    get_price,
    get_volatility,
)
from src.agents.trader_agent import create_trader_agent, run_trader
from src.models import get_standard_model
from src.streaming import emit, instrument

load_dotenv()

# -----------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("orchestrator")

# -----------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------

INITIAL_EXPOSURE = 0.0


# -----------------------------------------------------------------------
# ANALYST PROMPT  (identical for all three; each uses only its own tools)
# -----------------------------------------------------------------------

def _analyst_prompt(ticker: str, curr_date: str, tool_names: str) -> dict:
    return {
        "messages": [{
            "role": "user",
            "content": (
                f"Ticker: {ticker} | Date: {curr_date} | Tools: {tool_names}\n\n"
                "Analyse only data up to and including the date above — no look-ahead.\n"
                "Structure your response as:\n"
                "FINDINGS: <2-4 sentences of key data points with actual values>\n"
                "SIGNAL: BUY | SELL | HOLD\n"
                "CONFIDENCE: <0.0-1.0>\n"
                "RATIONALE: <1 sentence tying findings to signal>"
            ),
        }]
    }


# -----------------------------------------------------------------------
# ORCHESTRATOR SYSTEM PROMPT
# -----------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of an algorithmic trading system.

Your only tool is collect_analyst_reports. Call it once with the ticker and date.
It returns the raw outputs of three analyst agents (fundamentals, sentiment, indicators).

After receiving the results you MUST:

1. VERIFY each analyst response:
   - Check it contains FINDINGS, SIGNAL, CONFIDENCE, and RATIONALE sections
   - Flag any response that is missing data, internally inconsistent, or evasive
   - Note contradictions between analysts (e.g. BUY vs SELL)

2. MERGE into a structured brief that preserves each analyst's distinct voice:
   --- FUNDAMENTALS ANALYST ---
   <verified findings and signal>

   --- SENTIMENT ANALYST ---
   <verified findings and signal>

   --- INDICATORS ANALYST ---
   <verified findings and signal>

3. RESOLVE conflicts: if analysts disagree, explain why and whether one view
   should dominate (e.g. weak fundamentals may override bullish sentiment).

4. SUMMARISE with:
   CONSENSUS: BUY | SELL | HOLD
   CONFIDENCE: <0.0-1.0>  (penalise for conflicts or missing data)
   MARKET_BRIEF: <3-5 sentences synthesising all three views>

5. Append portfolio context (price, volatility, exposure) — these are passed
   to you in the tool result.

Output the complete structured brief. The trader agent will receive it verbatim.
No filler, no pleasantries.
"""


# -----------------------------------------------------------------------
# COLLECT_ANALYST_REPORTS TOOL FACTORY
# -----------------------------------------------------------------------

def _make_collect_tool(
    fund_agent, fund_tools,
    sent_agent, sent_tools,
    ind_agent,  ind_tools,
    ticker: str,
    curr_date: str,
    exposure: float,
):
    @tool
    def collect_analyst_reports(query: str) -> str:
        """
        Run the three analyst agents in parallel for the given ticker and date.
        Returns their raw outputs plus portfolio context.

        Args:
            query: Ignored. The ticker and date are already bound to this tool.
        """
        logger.info("[COLLECT] Running 3 analysts in parallel for %s on %s", ticker, curr_date)
        emit("analysts_dispatched", date=curr_date, ticker=ticker)

        async def _run_all():
            return await asyncio.gather(
                fund_agent.ainvoke(_analyst_prompt(ticker, curr_date, fund_tools)),
                sent_agent.ainvoke(_analyst_prompt(ticker, curr_date, sent_tools)),
                ind_agent.ainvoke(_analyst_prompt(ticker, curr_date, ind_tools)),
                return_exceptions=True,
            )

        results = asyncio.run(_run_all())
        names   = ["FUNDAMENTALS", "SENTIMENT", "INDICATORS"]
        parts   = []

        for name, result in zip(names, results):
            if isinstance(result, Exception):
                logger.warning("[%s] agent error: %s", name, result)
                parts.append(f"=== {name} ANALYST ===\nERROR: {result}\n")
                emit("analyst_report", date=curr_date, name=name, error=str(result))
            else:
                raw = result["messages"][-1].content
                logger.info("[%s] raw output (first 200): %s", name, raw[:200])
                parts.append(f"=== {name} ANALYST ===\n{raw}\n")
                emit("analyst_report", date=curr_date, name=name, content=raw)

        price      = get_price(ticker, curr_date)
        volatility = get_volatility(ticker, curr_date)

        portfolio_ctx = (
            f"\n=== PORTFOLIO CONTEXT ===\n"
            f"Ticker: {ticker}\n"
            f"Date: {curr_date}\n"
            f"Current price: {price:.2f}\n"
            f"Volatility (ATR/price): {volatility:.4f}\n"
            f"Current exposure: {exposure:.2%} of portfolio\n"
            f"Portfolio value: ${PORTFOLIO_VALUE:,.0f}\n"
        )

        emit("portfolio_context", date=curr_date, ticker=ticker, price=price,
             volatility=volatility, exposure=exposure, portfolio_value=PORTFOLIO_VALUE)

        return "\n".join(parts) + portfolio_ctx

    return collect_analyst_reports


# -----------------------------------------------------------------------
# SINGLE-DAY PIPELINE
# -----------------------------------------------------------------------

async def _run_day(
    ticker: str,
    curr_date: str,
    fund_agent, fund_tools,
    sent_agent, sent_tools,
    ind_agent,  ind_tools,
    orchestrator_model,
    trader_agent,
    exposure: float,
) -> tuple[str, float]:

    logger.info("=== DATE %s ===", curr_date)
    emit("day_start", date=curr_date, ticker=ticker, exposure=exposure)

    # Build a fresh orchestrator agent with the day's bound tool
    collect_tool = _make_collect_tool(
        fund_agent, fund_tools,
        sent_agent, sent_tools,
        ind_agent,  ind_tools,
        ticker, curr_date, exposure,
    )

    orchestrator = instrument(
        create_agent(
            model=orchestrator_model,
            tools=[collect_tool],
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        ),
        "orchestrator",
    )

    orch_response = orchestrator.invoke({
        "messages": [{
            "role": "user",
            "content": (
                f"Collect and verify analyst reports for {ticker} on {curr_date}. "
                "Produce the full structured brief for the trader."
            ),
        }]
    })

    brief = orch_response["messages"][-1].content
    logger.info("[ORCHESTRATOR] Brief produced (%d chars)", len(brief))
    emit("brief", date=curr_date, content=brief)

    # Pass brief to trader
    result = run_trader(trader_agent, brief, ticker, curr_date)
    conclusion = result["trader_conclusion"]

    # Update exposure estimate from conclusion (heuristic; real system tracks fills)
    new_exposure = exposure
    price = get_price(ticker, curr_date)
    if "buy_stock" in conclusion.lower() or "bought" in conclusion.lower():
        new_exposure = min(1.0, exposure + (10 * price / PORTFOLIO_VALUE))
    elif "sell_stock" in conclusion.lower() or "sold" in conclusion.lower():
        new_exposure = max(0.0, exposure - (10 * price / PORTFOLIO_VALUE))

    emit("day_complete", date=curr_date, conclusion=conclusion, exposure=new_exposure)
    return conclusion, new_exposure


# -----------------------------------------------------------------------
# TRADING DAY ITERATOR
# -----------------------------------------------------------------------

def _trading_days(start: datetime, n: int):
    day, count = start, 0
    while count < n:
        if day.weekday() < 5:
            yield day
            count += 1
        day += timedelta(days=1)


# -----------------------------------------------------------------------
# PUBLIC ENTRY POINT
# -----------------------------------------------------------------------

async def run_pipeline(
    ticker: str,
    start_date: str,
    num_days: int,
    model_name: str = "openrouter:openai/gpt-oss-120b:free",
):
    logger.info("Pipeline start  ticker=%s  from=%s  days=%d", ticker, start_date, num_days)
    emit("pipeline_start", ticker=ticker, start_date=start_date, num_days=num_days, model=model_name)

    analyst_model = init_chat_model(model_name, temperature=0.2)
    orchestrator_model = init_chat_model(model_name, temperature=0.1)

    fund_agent, fund_tools = create_fundamentals_analyst_agent(analyst_model)
    sent_agent, sent_tools = create_sentiment_analyst_agent(analyst_model)
    ind_agent,  ind_tools  = create_indicators_analyst_agent(analyst_model)

    fund_agent = instrument(fund_agent, "fundamentals")
    sent_agent = instrument(sent_agent, "sentiment")
    ind_agent  = instrument(ind_agent,  "indicators")

    risk_agent   = instrument(create_risk_agent(analyst_model), "risk")
    trader_agent = instrument(create_trader_agent(analyst_model, risk_agent), "trader")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    exposure = INITIAL_EXPOSURE
    results  = []

    for day in _trading_days(start_dt, num_days):
        date_str = day.strftime("%Y-%m-%d")
        try:
            conclusion, exposure = await _run_day(
                ticker=ticker,
                curr_date=date_str,
                fund_agent=fund_agent, fund_tools=fund_tools,
                sent_agent=sent_agent, sent_tools=sent_tools,
                ind_agent=ind_agent,   ind_tools=ind_tools,
                orchestrator_model=orchestrator_model,
                trader_agent=trader_agent,
                exposure=exposure,
            )
            results.append({
                "date": date_str,
                "conclusion": conclusion,
                "exposure": round(exposure, 4),
            })
        except Exception as exc:
            logger.error("Day %s failed: %s", date_str, exc, exc_info=True)
            results.append({"date": date_str, "error": str(exc)})

    logger.info("Pipeline complete.")
    for r in results:
        logger.info("  %s  exp=%.2f  conclusion: %s", r["date"], r.get("exposure", 0), r.get("conclusion", r.get("error", ""))[:120])

    emit("pipeline_complete", results=results)
    return results


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    import sys

    if len(sys.argv) == 1:
        ticker     = input("Ticker (e.g. AAPL): ").strip().upper()
        start_date = input("Start date (YYYY-MM-DD): ").strip()
        num_days   = int(input("Number of trading days: ").strip())
    elif len(sys.argv) == 4:
        ticker, start_date, num_days = sys.argv[1], sys.argv[2], int(sys.argv[3])
    else:
        print("Usage: python orchestrator.py [TICKER START_DATE NUM_DAYS]")
        sys.exit(1)

    results = asyncio.run(run_pipeline(ticker, start_date, num_days))

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
