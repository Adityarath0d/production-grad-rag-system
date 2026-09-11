from typing import Literal

from pydantic import BaseModel, Field
import logfire
from langchain_groq import ChatGroq

from app.agents.state import AgentState
from app.config import settings


# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
# llm = get_langchain_llm(feature="planner")
# model initialization
llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    api_key = settings.GROQ_API_KEY)


class PlannerDecision(BaseModel):
    """Validated routing decision produced by the planner."""

    route: Literal["conversational", "retrieval"] = Field(
        description="Whether to answer from the conversation or retrieve documentation."
    )
    query: str | None = Field(
        description="A refined documentation search query, or null for conversational requests.",
    )


planner_llm = llm.with_structured_output(
    PlannerDecision,
    method="json_schema",
    strict=True,
)


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg.type == "human" else "Assistant"
        history += f"{role}: {msg.content}\n"
    
    user_message = state["messages"][-1].content if state["messages"] else ""
    
    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. Choose route="conversational" if the latest message is a greeting (hi, hello), general question or can be answered using ONLY the conversation history above (e.g., "what is my name"). In this case, set query to null.
    2. Choose route="retrieval" for a technical question about Kubernetes, Intel, or Networking that requires fresh documentation. In this case, provide a refined search query.

    Return the requested structured response only.
    """
    
    with logfire.span("🧠 Planner Decision"):
        decision = planner_llm.invoke(prompt)
        logfire.info(f"Intent identified: {decision.route}")
    
    if decision.route == "conversational":
        return {
            "route": "conversational",
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    query = decision.query
    if not query:
        raise ValueError("Planner selected retrieval without a search query.")

    return {
        "route": "retrieval",
        "current_query": query,
        "status": f"Technical research needed. Searching for: {query}",
        "plan": ["Intent: Technical", f"Search Term: {query}"]
    }
