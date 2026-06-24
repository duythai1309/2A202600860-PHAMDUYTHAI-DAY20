"""Supervisor / router implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = AgentName.SUPERVISOR.value

    def __init__(self) -> None:
        self.settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        logger.info(f"Supervisor started. Current iteration: {state.iteration}")
        state.add_trace_event("agent_start", {"agent": self.name})

        # Safeguard: Max iterations limit
        max_iter = self.settings.max_iterations
        if state.iteration >= max_iter:
            logger.warning(
                f"Iteration limit reached ({state.iteration} >= {max_iter}). Routing to done."
            )
            next_route = "done"
            state.record_route(next_route)
            state.add_trace_event("agent_end", {"agent": self.name, "next_route": next_route})
            return state

        # Determine the next route based on the current state contents
        if not state.research_notes:
            next_route = AgentName.RESEARCHER.value
        elif not state.analysis_notes:
            next_route = AgentName.ANALYST.value
        elif not state.final_answer:
            next_route = AgentName.WRITER.value
        else:
            # We have a final answer. Let's see if we should run the critic.
            # Find the last runner agent in route history
            history_without_supervisor = [
                r for r in state.route_history if r != AgentName.SUPERVISOR.value
            ]

            if not history_without_supervisor:
                # Fallback: if somehow history is empty, run critic
                next_route = AgentName.CRITIC.value
            elif history_without_supervisor[-1] == AgentName.WRITER.value:
                # Just finished writing the answer, run critic next
                next_route = AgentName.CRITIC.value
            elif history_without_supervisor[-1] == AgentName.CRITIC.value:
                # We just ran the critic. Let's inspect the verdict.
                critic_results = [r for r in state.agent_results if r.agent == AgentName.CRITIC]
                if critic_results:
                    last_critic = critic_results[-1]
                    approved = last_critic.metadata.get("approved", False)

                    if approved:
                        logger.info("Critic approved the report. Routing to done.")
                        next_route = "done"
                    else:
                        logger.info("Critic requested revision.")
                        # Check how many times the critic has run to prevent infinite loop
                        critic_count = sum(
                            1 for r in history_without_supervisor if r == AgentName.CRITIC.value
                        )
                        if critic_count >= 2:
                            logger.warning(
                                "Critic has already run 2 times. Forcing done to avoid loop."
                            )
                            next_route = "done"
                        else:
                            # Re-run researcher/analyst/writer to improve
                            # Here we clean up previous notes to trigger re-execution
                            state.research_notes = None
                            state.analysis_notes = None
                            state.final_answer = None
                            next_route = AgentName.RESEARCHER.value
                else:
                    next_route = "done"
            else:
                # Default fallback
                next_route = AgentName.CRITIC.value

        logger.info(f"Supervisor decision: Routing to '{next_route}'")
        state.record_route(next_route)
        state.add_trace_event("agent_end", {"agent": self.name, "next_route": next_route})

        return state
