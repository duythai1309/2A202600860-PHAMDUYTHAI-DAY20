"""LangGraph workflow implementation."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self.graph = self.build()

    def build(self) -> Any:
        """Create and compile a LangGraph graph."""
        logger.info("Building multi-agent LangGraph workflow.")
        workflow = StateGraph(ResearchState)

        # Register nodes
        workflow.add_node("supervisor", lambda state: self.supervisor.run(state))
        workflow.add_node("researcher", lambda state: self.researcher.run(state))
        workflow.add_node("analyst", lambda state: self.analyst.run(state))
        workflow.add_node("writer", lambda state: self.writer.run(state))
        workflow.add_node("critic", lambda state: self.critic.run(state))

        # Add edges
        workflow.add_edge(START, "supervisor")

        # Routing function from supervisor
        def route_next(state: ResearchState) -> str:
            if not state.route_history:
                logger.warning("route_history empty in router. Falling back to Done.")
                return END
            next_agent = state.route_history[-1]
            if next_agent == "done":
                return END
            return next_agent

        workflow.add_conditional_edges(
            "supervisor",
            route_next,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )

        # Loop back edges from workers to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        logger.info("Executing LangGraph multi-agent research workflow.")

        # Invoke compiled graph
        result = self.graph.invoke(state)

        # Ensure we return a ResearchState Pydantic model
        if isinstance(result, ResearchState):
            return result
        elif isinstance(result, dict):
            return ResearchState(**result)
        else:
            logger.warning(
                f"Unexpected invocation result type: {type(result)}. Returning input state."
            )
            return state
