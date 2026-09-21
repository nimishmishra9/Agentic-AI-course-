import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


load_dotenv()

# Make console output UTF-8
sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# CUSTOM STATE
# ============================================================
#
# Normal AgentState already contains:
#
# {
#     "messages": [...]
# }
#
# We are adding our own application-specific state:
#
# {
#     "tool_uses": 0,
#     "refund_started": False
# }
#
# ============================================================

class SupportState(AgentState):
    """The normal agent state plus custom support-related state."""

    tool_uses: int
    refund_started: bool


# ============================================================
# TOOLS
# ============================================================

@tool
def order_status(order_id: str) -> str:
    """Get the status of an order using its ID."""
    print(f"[tool] order_status called with: {order_id}")

    return f"{order_id.upper()}: packed, ships tomorrow"


@tool
def delivery_estimate(pin_code: str) -> str:
    """Estimate delivery days for an Indian PIN code."""
    print(f"[tool] delivery_estimate called with: {pin_code}")

    return "5 days"


@tool
def start_refund(order_id: str, reason: str) -> str:
    """Start a refund for an order."""
    print(
        f"[tool] start_refund called with: "
        f"order_id={order_id}, reason={reason}"
    )

    return (
        f"Refund started for {order_id.upper()}, "
        f"reason: {reason}"
    )


# ============================================================
# CUSTOM MIDDLEWARE
# ============================================================

class TrackWork(AgentMiddleware):
    """
    Tracks tool calls and detects when the refund tool
    has been requested by the model.
    """

    # Tell LangChain that this middleware works with
    # our custom SupportState.
    state_schema = SupportState

    def after_model(self, state, runtime):
        """
        Runs after the model produces an AI message.

        At this point, the model may have generated
        one or more tool calls.
        """

        # Get the latest message
        last_message = state["messages"][-1]

        # Extract tool calls from the AI message
        tool_calls = getattr(last_message, "tool_calls", []) or []

        # If the model didn't request any tools,
        # there is nothing to update.
        if not tool_calls:
            return None

        # Count how many tools the model requested
        current_tool_uses = state.get("tool_uses", 0)

        update = {
            "tool_uses": current_tool_uses + len(tool_calls)
        }

        # Check whether start_refund was requested
        if any(
            call["name"] == "start_refund"
            for call in tool_calls
        ):
            update["refund_started"] = True

        print(
            "[state] "
            f"tool_uses={update['tool_uses']} "
            f"refund_started="
            f"{update.get('refund_started', state.get('refund_started'))}"
        )

        # Returning a dictionary updates the agent state
        return update


# ============================================================
# DYNAMIC PROMPT
# ============================================================

@dynamic_prompt
def prompt_with_budget(request):
    """
    Dynamically creates the system prompt using
    the current custom state.
    """

    # Read our custom state
    tool_uses = request.state.get("tool_uses", 0)

    prompt = (
        "You are a helpful support agent. "
        "Use the available tools to look up information. "
        "Never guess."
    )

    # Change the instruction when the tool budget is reached
    if tool_uses >= 3:
        prompt += (
            " You have already used three tools. "
            "Do not make additional tool calls. "
            "Answer the customer using the information "
            "you already have."
        )

    return prompt


# ============================================================
# MODEL
# ============================================================

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# ============================================================
# CREATE AGENT
# ============================================================

agent = create_agent(
    model=model,

    tools=[
        order_status,
        delivery_estimate,
        start_refund,
    ],

    middleware=[
        TrackWork(),
        prompt_with_budget,
    ],

    state_schema=SupportState,
)


# ============================================================
# INITIAL STATE
# ============================================================

initial_state = {
    "messages": [
        {
            "role": "user",
            "content": (
                "Where is ORD-1002, "
                "when does it reach pin 560034, "
                "and refund it because it is too late."
            ),
        }
    ],

    # Our custom state
    "tool_uses": 0,
    "refund_started": False,
}


# ============================================================
# RUN AGENT
# ============================================================

result = agent.invoke(initial_state)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)

print("Answer:")
print(result["messages"][-1].content)

print()

print("Custom State:")
print("tool_uses      :", result["tool_uses"])
print("refund_started :", result["refund_started"])

print("=" * 60)


if result["refund_started"]:
    print(
        "Refund has been started, "
        "please check your email for confirmation."
    )