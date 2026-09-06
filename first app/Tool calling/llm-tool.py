import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

model=ChatOpenAI(model="gpt-4o-mini")


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the exact result."""
    return a * b

@tool
def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())

model_with_tools= model.bind_tools([
    multiply,word_count
])

response =model_with_tools.invoke("what is 90854 multipled by 2345")
print("complete respone from LLM")
print(response)
print("           ")
print("===========")
print(response.content)
print("===========")
print(response.tool_calls)

print("_________Tool details _______")
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(tool_call["name"])
    print(tool_call["args"])

    tools = {
        "multiply" : multiply,
        "word_count" : word_count
    }

    selected_tools  = tools[tool_call["name"]]
    tool_result  = selected_tools.invoke(tool_call["args"])

    print("tools result")
    print(tool_result)
else : 
    print("No tools requested")

    #here everything is manual we can make it though agent
