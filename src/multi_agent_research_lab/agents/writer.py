"""Writer agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = AgentName.WRITER.value

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        logger.info("WriterAgent starting writing process.")
        state.add_trace_event("agent_start", {"agent": self.name})

        research_notes = state.research_notes or "No research notes available."
        analysis_notes = state.analysis_notes or "No analysis notes available."

        # Format references to present to the LLM
        references_list = []
        for idx, doc in enumerate(state.sources, 1):
            references_list.append(f"[{idx}] {doc.title} - {doc.url or 'No URL'}")
        references_str = "\n".join(references_list)

        try:
            sys_prompt = (
                "You are an expert technical writer. Synthesize the provided research notes and analysis notes "
                "into a comprehensive, professional, and well-structured final answer to the original query. "
                "Ensure your writing is tailored to the target audience.\n\n"
                "Requirements:\n"
                "- Write a detailed, complete answer (aim for a thorough, high-quality article/report).\n"
                "- Use clear headings, bullet points, and tables if appropriate.\n"
                "- You MUST use inline citations (e.g., [1], [2]) to reference facts and statements back to the source documents.\n"
                "- Include a 'References' section at the very end listing the source documents by their bracketed number indices."
            )

            user_prompt = (
                f"Original Query: {state.request.query}\n"
                f"Target Audience: {state.request.audience}\n\n"
                f"Research Notes:\n{research_notes}\n\n"
                f"Analysis Notes:\n{analysis_notes}\n\n"
                f"Source References:\n{references_str}\n\n"
                "Write the final report:"
            )

            llm_response = self.llm.complete(sys_prompt, user_prompt)
            state.final_answer = llm_response.content

            # Add LLM cost to trace
            state.add_trace_event(
                "llm_call",
                {
                    "agent": self.name,
                    "type": "final_writeup",
                    "cost_usd": llm_response.cost_usd or 0.0,
                },
            )

            # Save agent result
            state.agent_results.append(
                AgentResult(agent=AgentName.WRITER, content=state.final_answer, metadata={})
            )

            state.add_trace_event("agent_end", {"agent": self.name, "status": "success"})
        except Exception as e:
            logger.error(f"Error in WriterAgent: {e}")
            state.errors.append(f"WriterAgent error: {str(e)}")
            state.add_trace_event(
                "agent_end", {"agent": self.name, "status": "error", "error": str(e)}
            )

        return state
