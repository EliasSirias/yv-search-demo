import json
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from rag_core import search_docs


def build_support_agent(
    vectorstore: Any,
    knowledge_base_filter: str = "All",
    k: int = 2,
    min_hit_count: int = 1,
    max_distance: float = 1.25,
):
    @tool
    def search_documentation(query: str) -> str:
        """Search the support knowledge base for documented troubleshooting steps and procedures."""

        results = search_docs(
            query=query,
            vectorstore=vectorstore,
            knowledge_base_filter=knowledge_base_filter,
            k=k,
            min_hit_count=min_hit_count,
            max_distance=max_distance,
        )

        return json.dumps(results)

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    return create_agent(
        model=model,
        tools=[search_documentation],
        system_prompt=(
            "You are a technical support assistant. "
            "Use the search_documentation tool whenever the user asks about "
            "support procedures, troubleshooting, configuration, reports, or how-to steps. "
            "Answer only from retrieved documentation. "
            "If the tool does not find adequate documentation, say that the information "
            "is not available in the current knowledge base. "
            "Include the source names used in the answer."
        ),
    )
