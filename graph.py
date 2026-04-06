import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END

# --- 1. Define State ---
class AgentState(TypedDict):
    input: str
    agent_scratchpad: Annotated[List[BaseMessage], operator.add]
    final_answer: str

# ReAct prompt 
REACT_SYSTEM = """You are a ReAct agent. Strictly follow this loop:
Thought → Action (tool call) → Observation → Thought → ...

RULES:
1. ALWAYS use a tool for factual information — never answer from memory expect the founding years of companies.
2. For multi-part questions, make one tool call per fact.
3. ALWAYS use calculator for any arithmetic — never compute in your head.
4. Only give Final Answer AFTER all required tool calls are complete."""

# --- Graph Creation ---
def create_graph(llm_with_tools, tools_map):

    # --- 2. ReAct Node ---
    async def react_node(state: AgentState):
        messages = [SystemMessage(content=REACT_SYSTEM)]
        if state.get("input"):
            messages.append(HumanMessage(content=state["input"]))
            
        if state.get("agent_scratchpad"):
            messages.extend(state["agent_scratchpad"])
            
        response = await llm_with_tools.ainvoke(messages)
        
        final_answer = ""
        if not response.tool_calls:
            final_answer = response.content
            
        return {
            "agent_scratchpad": [response],
            "final_answer": final_answer
        }

    # --- 3. Tool Execution Node ---
    async def tool_node(state: AgentState):
        last_message = state["agent_scratchpad"][-1]
        
        tool_messages = []
        for tc in last_message.tool_calls:
            print(f"   | Executing Tool [{tc['name']}] | Args: {tc['args']}")
            try:
                result = await tools_map[tc["name"]].ainvoke(tc["args"])
            except Exception as e:
                result = f"Error: {str(e)}"
                
            print(f"      Observation: {str(result)}")

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            
        return {"agent_scratchpad": tool_messages}

    # --- 5. Conditional Routing ---
    def should_continue(state: AgentState):
        if state.get("final_answer"):
            return "is_final"
        else:
            return "is_action"

    # --- 4. Graph Flow ---
    workflow = StateGraph(AgentState)
    
    workflow.add_node("react_node", react_node)
    workflow.add_node("tool_node", tool_node)
    
    workflow.set_entry_point("react_node")
    
    # START -> react_node -> Conditional Edge
    workflow.add_conditional_edges(
        "react_node",
        should_continue,
        {
            "is_action": "tool_node",
            "is_final": END
        }
    )
    
    # If Action -> tool_node -> react_node
    workflow.add_edge("tool_node", "react_node")
    
    return workflow.compile()
