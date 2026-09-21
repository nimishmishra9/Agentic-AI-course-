from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import sys
from pathlib import Path
import asyncio

load_dotenv()

SERVER_PATH = Path(__file__).parent / "server.py"

async def main():
    client = MultiServerMCPClient({
        'greeting': {
            'transport':'stdio',
            'command': sys.executable,
            "args": [str(SERVER_PATH)]
        }
    })

    tools = await client.get_tools() # to get the tools list

    # For LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools(tools=tools)

    messages = [HumanMessage(content="Print Akshay in unique way default times")]

    response = await llm_with_tools.ainvoke(messages)

    messages.append(AIMessage(content=response.content, tool_calls=response.tool_calls))

    #tools all the avialble tools 
    #tools_by_name name of the tools from  avilable tolls 

    tools_by_name = {tool.name: tool for tool in tools}
    for tool_call in response.tool_calls:
        print(tool_call)
        tool = tools_by_name[tool_call["name"]]
        result = await tool.ainvoke(tool_call['args']) # to call the tool
        tool_output = result[0]['text']
        print(tool_output)

        messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call['id']))

    final_response = await llm_with_tools.ainvoke(messages)
    print(final_response.content)

    


if __name__ == "__main__":
    asyncio.run(main())