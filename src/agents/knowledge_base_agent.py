import asyncio
import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph

from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


class AgentState(TypedDict):
    """The state of the agent."""

    messages: List[BaseMessage]
    documents: Optional[List[Dict[str, Any]]]


async def load_chroma_db() -> Chroma:
    """Load the Chroma vector database."""
    # Create the embedding function for the project description database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Load the stored vector database
    chroma_db = Chroma(
        collection_name="company_handbook",
        embedding_function=embeddings,
        persist_directory="./data/chroma_db",
    )
    return chroma_db


async def search_documents(query: str) -> List[Dict[str, Any]]:
    """Search the knowledge base for relevant documents."""
    chroma_db = await load_chroma_db()
    # Get the chroma retriever
    retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
    # Search for relevant documents
    docs = retriever.invoke(query)
    # Format documents into a list of dicts
    documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    return documents


async def agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """The main agent node."""
    messages = state["messages"]
    
    # Get the latest user message
    latest_message = messages[-1]
    if isinstance(latest_message, HumanMessage):
        query = latest_message.content
        
        # Search for relevant documents
        documents = await search_documents(query)
        
        # Create a system message with the retrieved documents
        if documents:
            doc_content = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(documents)])
            system_message = SystemMessage(
                content=f"You are a helpful assistant. Answer the user's question based on the following documents from the knowledge base:\n\n{doc_content}"
            )
        else:
            system_message = SystemMessage(
                content="You are a helpful assistant. No relevant documents were found in the knowledge base for this query."
            )
        
        # Add the system message to the beginning of the conversation
        messages = [system_message] + messages
    
    # Get the model response
    response = await model.ainvoke(messages)
    
    # Return the updated state
    return {"messages": messages + [response], "documents": documents if 'documents' in locals() else None}


# Create the graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.set_finish_point("agent")

knowledge_base_agent = graph.compile()
