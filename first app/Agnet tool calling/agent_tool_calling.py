import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

# Make the console support UTF-8 characters
sys.stdout.reconfigure(encoding="utf-8")


# ----------------------------
# Create tools
# ----------------------------

@tool
def current_time(city: str) -> str:
    """Get the current time in a city."""

    zones = {
        "mumbai": "Asia/Kolkata",
        "london": "Europe/London",
        "new york": "America/New_York",
    }

    zone = zones.get(city.lower())

    if zone is None:
        return f"I do not know the timezone for {city}."

    return datetime.now(
        ZoneInfo(zone)
    ).strftime("%d %B %Y, %I:%M %p")


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the exact result."""

    return a * b


# ----------------------------
# Create Agent
# ----------------------------

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[multiply, current_time],
    system_prompt=(
        "You are a helpful assistant. "
        "Use the tools when they fit."
    ),
)


# ----------------------------
# Invoke Agent
# ----------------------------

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "What time is it in Mumbai, and what is 98765 times 43210? and when india was indepedent"
                ),
            }
        ]
    }
)


#for message  in result["messages"]:
#here you can check what happend behind the scene


# ----------------------------
# Print final response
# ----------------------------

print("Final answer")
print(result["messages"][-1].content) #last message in entire conversation