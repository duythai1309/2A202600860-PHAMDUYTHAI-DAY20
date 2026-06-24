"""Critic agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Fact-checking and safety-review agent."""

    name = AgentName.CRITIC.value

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        logger.info("CriticAgent starting critique process.")
        state.add_trace_event("agent_start", {"agent": self.name})

        final_answer = state.final_answer or "No final answer available to review."
        research_notes = state.research_notes or "No research notes available."

        references_list = []
        for idx, doc in enumerate(state.sources, 1):
            references_list.append(f"[{idx}] {doc.title} - {doc.url or 'No URL'}")
        references_str = "\n".join(references_list)

        try:
            sys_prompt = (
                "You are an expert editor and fact-checker. Review the proposed Final Report against the original Research Notes and Sources.\n"
                "Your objective is to verify:\n"
                "1. Accuracy: Does the report make claims not backed by the research notes or sources? (Hallucinations)\n"
                "2. Citation Coverage: Are critical claims appropriately cited with bracketed numbers referencing the sources?\n"
                "3. Formatting & Clarity: Is the writing professional and clear?\n\n"
                "You must output your evaluation structured exactly as follows:\n"
                "Verdict: [APPROVED] or [NEEDS_REVISION]\n"
                "Feedback: Provide specific, bulleted feedback detailing what must be fixed or why it is approved."
            )

            user_prompt = (
                f"Proposed Final Report:\n{final_answer}\n\n"
                f"Research Notes:\n{research_notes}\n\n"
                f"Source Materials:\n{references_str}\n\n"
                "Analyze the report and output your Verdict and Feedback:"
            )

            llm_response = self.llm.complete(sys_prompt, user_prompt)
            content = llm_response.content

            # Parse approval status from LLM response
            approved = "APPROVED" in content and "NEEDS_REVISION" not in content
            verdict = "APPROVED" if approved else "NEEDS_REVISION"
            logger.info(f"Critic Verdict: {verdict}")

            # Add LLM cost to trace
            state.add_trace_event(
                "llm_call",
                {
                    "agent": self.name,
                    "type": "fact_check",
                    "cost_usd": llm_response.cost_usd or 0.0,
                },
            )

            # Save agent result
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=content,
                    metadata={"approved": approved, "verdict": verdict},
                )
            )

            state.add_trace_event(
                "agent_end", {"agent": self.name, "status": "success", "verdict": verdict}
            )
        except Exception as e:
            logger.error(f"Error in CriticAgent: {e}")
            state.errors.append(f"CriticAgent error: {str(e)}")
            state.add_trace_event(
                "agent_end", {"agent": self.name, "status": "error", "error": str(e)}
            )

        return state
