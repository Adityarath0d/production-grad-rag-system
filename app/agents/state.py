from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    route : str
    current_query : str
    documents : list[Document]
    plan : list[str]
    status : str
    final_answer : str
