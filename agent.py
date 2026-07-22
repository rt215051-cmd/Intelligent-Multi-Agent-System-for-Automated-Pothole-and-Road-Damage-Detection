"""
Simple LangGraph AI Agent for Pothole Detection
"""
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: list
    detection_context: dict

def create_agent():
    """Create a simple LangGraph agent for pothole detection chat"""
    
    # Initialize LLM with Ollama
    llm = ChatOllama(
        model="gemma4:31b-cloud",
        temperature=0.7
    )
    
    def chat_node(state: AgentState):
        """Simple chat node that responds to user messages"""
        messages = state["messages"]
        detection_context = state.get("detection_context", {})
        
        # Add context about pothole detection
        system_prompt = """You are a helpful AI assistant for a pothole detection system. 
You can answer questions about road damage analysis, pothole severity, and detection results.
Be concise and helpful."""
        
        # Build message list with system prompt
        chat_messages = [{"role": "system", "content": system_prompt}]
        
        # Add detection context if available
        if detection_context:
            context_str = f"Current detection data: {detection_context}"
            chat_messages.append({"role": "system", "content": context_str})
        
        # Add conversation history
        for msg in messages:
            if isinstance(msg, HumanMessage):
                chat_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_messages.append({"role": "assistant", "content": msg.content})
        
        # Get response
        response = llm.invoke(chat_messages)
        
        return {"messages": [response]}
    
    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("chat", chat_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)
    
    return workflow.compile()

def chat_with_agent(message: str, detection_context: dict = None):
    """Simple function to chat with the agent"""
    try:
        agent = create_agent()
        
        state = {
            "messages": [HumanMessage(content=message)],
            "detection_context": detection_context or {}
        }
        
        result = agent.invoke(state)
        response = result["messages"][-1].content
        return response
    except Exception as e:
        return f"Error: {str(e)}. Please make sure Ollama is running and the model is installed."
