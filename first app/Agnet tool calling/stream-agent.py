import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")


# ===================================================
# TOOLS
# ===================================================

@tool
def hotel_price(city: str, nights: int) -> int:
    """
    Get the total hotel cost in rupees
    for a number of nights in a city.
    """

    per_night = {
        "delhi": 3000,
        "bengaluru": 3500,
        "kolkata": 2800
    }

    return per_night.get(city.lower(), 3200) * nights


@tool
def ticket_price(city: str) -> int:
    """
    Get the flight ticket price in rupees
    from Mumbai to a city.
    """

    prices = {
        "delhi": 4500,
        "bengaluru": 3900,
        "kolkata": 5200
    }

    return prices.get(city.lower(), 6000)


# ===================================================
# AGENT
# ===================================================

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[ticket_price, hotel_price],
    system_prompt=(
        "You plan small trips and always look up "
        "real prices using the available tools."
    ),
)


# ===================================================
# USER QUESTION
# ===================================================

question = (
    "I want to go to Bengaluru for 3 nights "
    "from Mumbai. What is my total cost?"
)


# ===================================================
# STREAM AGENT
# ===================================================

for chunk in agent.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    },
    stream_mode="updates"
):
    for node, update in chunk.items():

        for message in update["messages"]:

            # LLM requested a tool
            if getattr(message, "tool_calls", None):

                print(
                    f"[{node}] wants: "
                    f"{[c['name'] for c in message.tool_calls]}"
                )

            # Tool finished executing
            elif type(message).__name__ == "ToolMessage":

                print(
                    f"[{node}] "
                    f"{message.name} returned: "
                    f"{message.content}"
                )

            # LLM response
            elif message.content:

                print(
                    f"[{node}] says: "
                    f"{message.content}"
                )