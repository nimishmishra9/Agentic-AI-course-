import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

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

tools = [current_time, multiply]

tools_by_name = {
    tool.name: tool
    for tool in tools
}

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
).bind_tools(tools)


messages = [
    SystemMessage(
        content="You are a helpful assistant. Use the tools when they fit."
    ),
    HumanMessage(
        content=(
            "What time is it in Mumbai, and what is "
            "98765 times 43210?"
        )
    ),
]

# Tool-calling loop
step = 1
while True:
    # Ask the LLM
    response = model.invoke(messages)

    print(response)

    # Add the AI response to conversation history
    messages.append(response)

    # If no tool calls, we have the final answer
    if not response.tool_calls:
        print("\nFinal response:")
        print(response.content)
        break

    print(
        f"\nStep {step}: The model asked for "
        f"{len(response.tool_calls)} tool call(s)"
    )

    # Execute every tool requested by the LLM
    for call in response.tool_calls:
        
        # Get the requested tool
        tool_to_run = tools_by_name[call["name"]]

        # Execute the tool with the arguments provided by the LLM
        tool_message = tool_to_run.invoke(call)

        print(
            f"  {call['name']}({call['args']}) "
            f"-> {tool_message.content}"
        )

        # Add tool result back to the conversation
        messages.append(tool_message)

    # Move to the next iteration
    step += 1


    #still here everything is manul we can do this by automate using agent