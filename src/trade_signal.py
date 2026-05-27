from pydantic import BaseModel, Field
from typing import Literal


class TradeSignal(BaseModel):
    """
    Lightweight signal emitted by a single analyst agent.
    The orchestrator aggregates multiple TradeSignals into a TradeProposal.
    """

    ticker: str = Field(description="Stock ticker symbol")
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0, description="Agent confidence 0-1")
    agent: str = Field(description="Name of the analyst agent")
    rationale: str = Field(default="", description="One-line justification")
