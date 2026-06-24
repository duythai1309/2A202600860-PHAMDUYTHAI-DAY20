"""Analyst agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = AgentName.ANALYST.value

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        logger.info("AnalystAgent starting analysis process.")
        state.add_trace_event("agent_start", {"agent": self.name})

        research_notes = state.research_notes or "No research notes available."

        try:
            sys_prompt = (
                "You are an expert research analyst. Your task is to critique and analyze research findings. "
                "Structure your output into: \n"
                "1. Core Claims & Evidence: Extract the main claims and evaluate their supporting evidence.\n"
                "2. Synthesized Insights: Connect different viewpoints, identifying areas of consensus or conflict.\n"
                "3. Gap Analysis: Highlight weak evidence, logical gaps, or missing critical data in the research notes."
            )

            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Research Notes:\n{research_notes}\n\n"
                "Provide a structured analysis."
            )

            llm_response = self.llm.complete(sys_prompt, user_prompt)
            state.analysis_notes = llm_response.content

            # Add LLM cost to trace
            state.add_trace_event(
                "llm_call",
                {
                    "agent": self.name,
                    "type": "analysis_generation",
                    "cost_usd": llm_response.cost_usd or 0.0,
                },
            )

            # Save agent result
            state.agent_results.append(
                AgentResult(agent=AgentName.ANALYST, content=state.analysis_notes, metadata={})
            )

            state.add_trace_event("agent_end", {"agent": self.name, "status": "success"})
        except Exception as e:
            logger.error(f"Error in AnalystAgent: {e}")
            state.errors.append(f"AnalystAgent error: {str(e)}")
            state.add_trace_event(
                "agent_end", {"agent": self.name, "status": "error", "error": str(e)}
            )

        return state
