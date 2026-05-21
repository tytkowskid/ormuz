from pydantic import BaseModel, Field

class TradeSignal(BaseModel):

    ticker: str = Field(
        description="Stock ticker symbol"
    )

    action: str = Field(
        description="buy or sell"
    )

    proposed_quantity: int = Field(
        description="Suggested trade size"
    )

    price: float

    rsi: float

    volatility: float

    current_exposure: float