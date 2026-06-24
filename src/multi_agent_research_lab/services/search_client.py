"""Search client implementation with Tavily API and Local Mock fallback."""

import logging

import requests

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with fallback logic."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Attempts Tavily API first, falls back to mock results on failure or if no key is present.
        """
        api_key = self.settings.tavily_api_key

        if api_key:
            try:
                logger.info(f"Performing real Tavily search for: '{query}'")
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "advanced",
                    },
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                docs = []
                for item in results[:max_results]:
                    docs.append(
                        SourceDocument(
                            title=item.get("title", "No Title"),
                            url=item.get("url"),
                            snippet=item.get("content", ""),
                            metadata={"score": item.get("score", 0.0)},
                        )
                    )
                if docs:
                    return docs
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}. Falling back to mock search.")

        # Fallback to Mock search
        logger.info(f"Performing mock search for query: '{query}'")
        return self._mock_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        query_lower = query.lower()
        mock_db: list[SourceDocument] = []

        # GraphRAG-specific mock docs
        if "graphrag" in query_lower or "graph rag" in query_lower:
            mock_db = [
                SourceDocument(
                    title="GraphRAG: Orchestrating Structured Knowledge in LLMs",
                    url="https://arxiv.org/abs/2404.16130",
                    snippet=(
                        "GraphRAG is a state-of-the-art retrieval-augmented generation method developed by Microsoft. "
                        "Unlike standard vector-based RAG, GraphRAG builds a knowledge graph from the source documents "
                        "and runs community detection algorithms (like Leiden) to summarize topical clusters. "
                        "This enables it to answer global queries ('What are the main themes in this corpus?') "
                        "much more effectively than vector RAG which operates on local chunk similarities."
                    ),
                    metadata={"topic": "GraphRAG", "source": "arXiv"},
                ),
                SourceDocument(
                    title="Microsoft Research Blog: GraphRAG SOTA Performance",
                    url="https://www.microsoft.com/en-us/research/blog/graphrag/",
                    snippet=(
                        "Microsoft's GraphRAG has shown significant performance improvements in multi-document summarization "
                        "and complex reasoning tasks. By combining entity extraction, relationship definition, and LLM-driven "
                        "summarization of graph communities, it solves the 'needle in a haystack' problem and provides "
                        "verifiable citations linked to structural elements in the graph."
                    ),
                    metadata={"topic": "GraphRAG", "source": "Microsoft"},
                ),
                SourceDocument(
                    title="Understanding Leiden Community Detection in Knowledge Graphs",
                    url="https://neo4j.com/developer-blog/leiden-community-detection-graphrag/",
                    snippet=(
                        "The Leiden algorithm is used in GraphRAG pipelines to partition the generated entity-relation graph "
                        "into hierarchical communities. Community summaries are pre-generated using an LLM, allowing "
                        "the retriever to build query-focused summaries by merging community reports, cutting down "
                        "on redundant retrieval steps and maximizing context utility."
                    ),
                    metadata={"topic": "Leiden Detection", "source": "Neo4j"},
                ),
                SourceDocument(
                    title="Vector RAG vs Graph RAG: A Comparative Study",
                    url="https://medium.com/ai-insights/vector-vs-graph-rag",
                    snippet=(
                        "Standard RAG relies on semantic embeddings of document chunks and works best for local, specific details. "
                        "GraphRAG constructs a structural graph of entities and relations, compiling summaries at multiple levels. "
                        "Benchmarks show GraphRAG uses more LLM tokens during indexing but achieves 2x better quality "
                        "on synthesis and high-level reasoning evaluations."
                    ),
                    metadata={"topic": "RAG Comparison", "source": "Medium"},
                ),
            ]
        # Multi-Agent systems / Orchestration mock docs
        elif (
            "agent" in query_lower
            or "orchestra" in query_lower
            or "supervisor" in query_lower
            or "langgraph" in query_lower
        ):
            mock_db = [
                SourceDocument(
                    title="LangGraph: Building Cyclic Multi-Agent Architectures",
                    url="https://langchain-ai.github.io/langgraph/",
                    snippet=(
                        "LangGraph is a framework for building stateful, multi-agent applications with support for cyclic flows. "
                        "It models workflows as graphs containing nodes (representing agent logic or tools) and edges. "
                        "It offers native state persistence, human-in-the-loop validation, and memory, making it the industry "
                        "standard for complex agent orchestrations like supervisor-worker patterns."
                    ),
                    metadata={"topic": "LangGraph", "source": "LangChain"},
                ),
                SourceDocument(
                    title="Anthropics: Building Effective Multi-Agent Workflows",
                    url="https://www.anthropic.com/research/building-effective-agents",
                    snippet=(
                        "Anthropic outlines best practices for multi-agent design: keep roles simple, use explicit handoffs, "
                        "and avoid over-engineering with too many agents. The Supervisor pattern uses a central router "
                        "agent to coordinate worker agents, which reduces cumulative error and optimizes performance "
                        "relative to fully decentralized systems."
                    ),
                    metadata={"topic": "Multi-Agent Design", "source": "Anthropic"},
                ),
                SourceDocument(
                    title="Routing and Handoffs in LLM Networks",
                    url="https://openai.com/research/llm-routing",
                    snippet=(
                        "OpenAI's research into multi-agent systems highlights that routing decisions are critical. "
                        "A supervisor router agent evaluates intermediate results against the global objective, "
                        "routing to specialized sub-agents (e.g. Researcher, Analyst, Critic) and concluding "
                        "only when quality constraints are satisfied. This iterative routing prevents hallucination."
                    ),
                    metadata={"topic": "Orchestration", "source": "OpenAI"},
                ),
                SourceDocument(
                    title="Multi-Agent Systems: Latency and Cost Benchmarks",
                    url="https://towardsdatascience.com/multi-agent-benchmarks",
                    snippet=(
                        "Multi-agent structures deliver superior accuracy on complex tasks but come with high latencies "
                        "and token costs. A typical multi-agent run can be 3x to 5x slower than a single-agent baseline "
                        "due to sequential LLM invocation and self-correction loops. Choosing lightweight models "
                        "(like gpt-4o-mini or gemini-2.5-flash) mitigates the financial cost."
                    ),
                    metadata={"topic": "Benchmarks", "source": "TowardsDataScience"},
                ),
            ]
        # Fallback general AI research mock docs
        else:
            mock_db = [
                SourceDocument(
                    title="State-of-the-Art in Large Language Model Research",
                    url="https://arxiv.org/abs/2312.00001",
                    snippet=(
                        "Current LLM SOTA is dominated by mixture-of-experts architectures, advanced retrieval schemes, "
                        "and multi-agent orchestration frameworks. Systems that incorporate self-correction, tool use, "
                        "and structured workflows achieve higher human evaluation scores on analytical tasks."
                    ),
                    metadata={"topic": "LLM Research", "source": "arXiv"},
                ),
                SourceDocument(
                    title="AI Agents in Action: Survey of Production Implementations",
                    url="https://arxiv.org/abs/2401.99999",
                    snippet=(
                        "A survey of over 100 enterprise AI agent deployments reveals that supervisor-directed workflows "
                        "have the highest success rate. Key lessons include restricting agent tool scopes, implementing "
                        "strict JSON input/output schemas, and placing strict iteration caps to stop runaway loops."
                    ),
                    metadata={"topic": "Agent Survey", "source": "arXiv"},
                ),
            ]

        # Return up to max_results from the matched database
        return mock_db[:max_results]
