from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routing_policy() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    # Initial state (no notes) -> routes to researcher
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == AgentName.RESEARCHER.value

    # Research notes added -> routes to analyst
    state.research_notes = "Raw research notes."
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == AgentName.ANALYST.value

    # Analysis notes added -> routes to writer
    state.analysis_notes = "Analyst insights."
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == AgentName.WRITER.value

    # Writer output added -> routes to critic
    state.final_answer = "Polished report."
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == AgentName.CRITIC.value
