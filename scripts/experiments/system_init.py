"""Shared system initialization and query helpers for experiments."""

from __future__ import annotations

import os
from typing import Any, Dict


def init_graphrag():
    from src.rag.graph.querier import CVGraphRAGSystem
    return CVGraphRAGSystem()


def init_naiverag(querier: bool = True):
    if querier:
        from src.rag.naive.querier import NaiveRAGQuerier
        return NaiveRAGQuerier()

    # fallback to the demo system (creates vector store if missing)
    from scripts.demo_naive_rag import NaiveRAGSystem

    system = NaiveRAGSystem()
    if not system.initialize_system():
        raise RuntimeError("Failed to initialize Naive RAG system")
    return system


def init_agentrag(graph_rag, naive_rag):
    from langchain_openai import ChatOpenAI
    from src.rag.agent.workflow import CVGraphAgent

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return CVGraphAgent(llm=llm, graph_rag=graph_rag, naive_rag=naive_rag)


def run_graph_query(graph_rag, question: str) -> Dict[str, Any]:
    res = graph_rag.query_graph(question)
    return {
        "answer": res.get("answer", ""),
        "contexts": res.get("context", []),
        "cypher_query": res.get("cypher_query", ""),
        "success": res.get("success", False)
    }


def run_naive_query(naive_rag, question: str) -> Dict[str, Any]:
    res = naive_rag.query(question)
    return {
        "answer": res.get("answer", ""),
        "contexts": res.get("contexts", []),
        "success": res.get("success", False)
    }


def run_agent_query(agent_rag, question: str) -> Dict[str, Any]:
    res = agent_rag.query(question)
    return {
        "answer": res.get("answer", ""),
        "contexts": res.get("contexts", []),
        "cypher_query": res.get("cypher_query", ""),
        "success": res.get("success", False)
    }
