from langchain_core.tools import tool
from agent.rag.retriever import retrieve


@tool
def rag_retrieval_tool(query: str) -> str:
    """
    Retrieves relevant passages from the pedagogical knowledge base.
    Use this tool to look up best practices on assessment design, Bloom's taxonomy,
    discrimination index interpretation, learning gap remediation, and psychometric principles.
    Input should be a natural language question about assessment or pedagogy.
    """
    return retrieve(query)
