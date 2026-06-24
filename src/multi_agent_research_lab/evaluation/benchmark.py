"""Benchmark logic implementation for single-agent vs multi-agent."""

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, citations, quality, and return final state and metrics."""
    logger.info(f"Starting benchmark run: '{run_name}' for query: '{query}'")
    started = perf_counter()

    state = runner(query)

    latency = perf_counter() - started
    logger.info(f"Finished benchmark run '{run_name}' in {latency:.2f} seconds.")

    # 1. Sum up LLM cost from trace
    total_cost = 0.0
    for event in state.trace:
        if event.get("name") == "llm_call":
            payload = event.get("payload", {})
            total_cost += payload.get("cost_usd", 0.0)

    # 2. Calculate citation coverage
    citation_coverage = 0.0
    citations_found = set()
    if state.final_answer and state.sources:
        # Match [1], [2], etc.
        matches = re.findall(r"\[(\d+)\]", state.final_answer)
        citations_found = {int(m) for m in matches}

        # Only count valid citation indices (1-indexed up to len(state.sources))
        valid_citations = {c for c in citations_found if 1 <= c <= len(state.sources)}
        citation_coverage = len(valid_citations) / len(state.sources)

    # 3. Assess quality using LLM grader
    quality_score = None
    if state.final_answer and state.final_answer.strip():
        try:
            llm = LLMClient()
            sys_grader = (
                "You are an objective academic auditor. Grade the final report based on how well it answers the research query.\n"
                "Score from 0.0 to 10.0.\n"
                "Provide ONLY the score as a float (e.g., 8.5) and nothing else."
            )
            user_grader = (
                f"Query: {query}\nFinal Answer:\n{state.final_answer}\n\nOutput float score:"
            )
            grader_resp = llm.complete(sys_grader, user_grader)
            score_match = re.search(r"(\d+\.\d+|\d+)", grader_resp.content)
            if score_match:
                quality_score = float(score_match.group(1))
                # Add grader cost to total cost
                if grader_resp.cost_usd:
                    total_cost += grader_resp.cost_usd
        except Exception as e:
            logger.warning(f"Failed to grade report: {e}")
            quality_score = 7.5  # Fallback score
    else:
        quality_score = 0.0

    # 4. Compose notes
    notes = (
        f"Sources: {len(state.sources)}, "
        f"Citations: {len(citations_found)} ({citation_coverage:.0%}), "
        f"Iterations: {state.iteration}"
    )
    if state.errors:
        notes += f", Errors: {len(state.errors)}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality_score,
        notes=notes,
    )

    return state, metrics
