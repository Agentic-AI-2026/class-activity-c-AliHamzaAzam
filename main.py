import sys
import asyncio
import nest_asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

from graph import create_graph, AgentState

nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()

# Configure MCP servers based on the provided Class Code setup
mcp = MultiServerMCPClient({
    "math": {
        "command": sys.executable,
        "args": ["Tools/math_server.py"],
        "transport": "stdio",
    },
    "search": {
        "command": sys.executable,
        "args": ["Tools/search_server.py"],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    }
})

async def get_mcp_tools(servers: list) -> tuple:
    tools = []
    print("Getting Tools...")
    for server in servers:
        tool = await mcp.get_tools(server_name=server)
        tools.extend(tool)
    t_map  = {t.name: t for t in tools}
    print(f"MCP tools loaded: {list(t_map.keys())}\n")
    return tools, t_map

async def run():
    print("Initializing LLM and Tools...")
    llm = ChatOllama(model="minimax-m2.5:cloud", temperature=0)

    # We need search, math, and weather tools for the query execution
    try:
        tools, tools_map = await get_mcp_tools(["search", "math", "weather"])
    except Exception as e:
        print(f"Could not load tools properly (is weather server running remotely?): {e}")
        return

    # Bind tools to our LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Compile the LangGraph app
    app = create_graph(llm_with_tools, tools_map)

    while True:
        try:
            query = input("\nEnter your query (or 'quit' to exit): ")
            if query.lower() in ['quit', 'exit', 'q']:
                break
        except (KeyboardInterrupt, EOFError):
            break
            
        if not query.strip():
            continue
            
        print(f"\nUser Query: {query}\n")

        # Initialize the required state
        initial_state = {
            "input": query,
            "agent_scratchpad": [],
            "final_answer": ""
        }
        
        # Run the graph and stream intermediate steps
        print("--- Running Graph ---")
        async for event in app.astream(initial_state, stream_mode="updates"):
            for node, state_update in event.items():
                print(f"[{node}] execution completed.")
                if "agent_scratchpad" in state_update:
                    last_msg = state_update["agent_scratchpad"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        print(f"   Thought: {last_msg.content}")

                if "final_answer" in state_update and state_update["final_answer"]:
                    print(f"\n=============================\nFinal Answer:\n{state_update['final_answer']}")

if __name__ == "__main__":
    asyncio.run(run())
