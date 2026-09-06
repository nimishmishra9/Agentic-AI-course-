import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")


@tool
def book_seat(name: str, seat: str) -> str:
    """Book a seat in the class for a person."""
    return f"Seat {seat} booked for {name}."


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[book_seat],
    system_prompt="You help learners book a seat in the LangChain class.",
    checkpointer=InMemorySaver(),  # keeps conversation memory
)


def ask(question, thread_id):

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    print(f"[{thread_id}] Q: {question}")
    print(f"[{thread_id}] A: {result['messages'][-1].content}")


ask("My name is Vinay", thread_id="vinay")

ask("Book me a seat A12", thread_id="vinay")

ask("waht is my name and what is my seat number ", thread_id="vinay")

ask("waht is my name and what is my seat number ", thread_id="nimish") #here is has been changed 