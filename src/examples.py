import json

from langchain.agents import create_agent
from src.tools import buy_stock, sell_stock
from src.models import get_standard_model
from langchain_core.messages import AIMessage

basic_system_prompt="""
        You are a risk management trading agent.

        Your responsibilities:
        - evaluate incoming trade signals
        - approve, reject, or reduce trades
        - execute trades ONLY if risk is acceptable

        Risk rules:
        - Reject trades if volatility > 0.5
        - Reduce size by 50% if current exposure > 0.4
        - Never exceed safe exposure

        Use buy_stock or sell_stock tools when appropriate.

        Always explain your reasoning.
        """


def example_buy_reasoning():
    print("Creating agent")

    agent = create_agent(get_standard_model(),
                         tools=[buy_stock, sell_stock],
                         system_prompt=basic_system_prompt)
    
    # Structured input
    signal = {
        "ticker": "AAPL",
        "action": "buy",
        "proposed_quantity": 10,
        "price": 185.4,
        "rsi": 28.5,
        "volatility": 0.22,
        "current_exposure": 0.31,
    }

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": json.dumps(signal, indent=2)
            }
        ]
    })
    print("\nHUMAN message:\n")
    print(response["messages"][0].content)

    print("\nAGENT RESPONSE:\n")

    for message in response["messages"]:
        if isinstance(message, AIMessage):
                print("\nAI MESSAGE:")
                print(message.content)