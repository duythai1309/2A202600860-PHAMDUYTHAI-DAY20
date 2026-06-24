"""Researcher agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = AgentName.RESEARCHER.value

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.search_client = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        logger.info("ResearcherAgent starting research process.")
        state.add_trace_event("agent_start", {"agent": self.name})

        try:
            # Step 1: Generate search query using LLM
            sys_prompt = "You are a research query optimizer. Generate a single, concise search query based on the user's research request."
            user_prompt = (
                f"Original Query: {state.request.query}\nAudience: {state.request.audience}"
            )
            llm_query_response = self.llm.complete(sys_prompt, user_prompt)

            search_query = llm_query_response.content.strip().strip('"').strip("'")
            logger.info(f"Generated search query: '{search_query}'")

            # Add LLM cost to trace
            state.add_trace_event(
                "llm_call",
                {
                    "agent": self.name,
                    "type": "query_generation",
                    "cost_usd": llm_query_response.cost_usd or 0.0,
                },
            )

            # Step 2: Search for documents
            found_docs = self.search_client.search(
                search_query, max_results=state.request.max_sources
            )

            # Add unique sources
            existing_urls = {doc.url for doc in state.sources if doc.url}
            added_docs = []
            for doc in found_docs:
                if doc.url not in existing_urls:
                    state.sources.append(doc)
                    added_docs.append(doc)

            logger.info(
                f"Retrieved {len(found_docs)} documents. Added {len(added_docs)} new sources."
            )

            # Step 3: Summarize / write research notes using LLM
            context_list = []
            for idx, doc in enumerate(state.sources, 1):
                context_list.append(
                    f"[{idx}] Title: {doc.title}\nURL: {doc.url or 'No URL'}\nContent: {doc.snippet}"
                )
            context_str = "\n\n".join(context_list)

            sys_notes_prompt = (
                "You are an expert researcher. Compile comprehensive, detailed research notes based on the provided source materials "
                "to answer the query. Group facts by sub-topics, highlight definitions, key claims, and cite sources "
                "explicitly using bracketed numbers corresponding to the source list (e.g. [1], [2])."
            )

            user_notes_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n"
                f"Sources:\n{context_str}\n\n"
                "Please synthesize clean research notes."
            )

            llm_notes_response = self.llm.complete(sys_notes_prompt, user_notes_prompt)
            state.research_notes = llm_notes_response.content

            # Add LLM cost to trace
            state.add_trace_event(
                "llm_call",
                {
                    "agent": self.name,
                    "type": "notes_generation",
                    "cost_usd": llm_notes_response.cost_usd or 0.0,
                },
            )

            # Save agent result
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=state.research_notes,
                    metadata={"search_query": search_query, "sources_added": len(added_docs)},
                )
            )

            state.add_trace_event("agent_end", {"agent": self.name, "status": "success"})
        except Exception as e:
            logger.error(f"Error in ResearcherAgent: {e}")
            state.errors.append(f"ResearcherAgent error: {str(e)}")
            state.add_trace_event(
                "agent_end", {"agent": self.name, "status": "error", "error": str(e)}
            )

        return state
