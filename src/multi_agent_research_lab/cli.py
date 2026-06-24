"""Command-line entrypoint for the multi-agent research lab."""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()
logger = logging.getLogger(__name__)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def run_baseline(query: str) -> ResearchState:
    """Run a single-agent baseline using single-turn retrieval augmented generation."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    llm = LLMClient()
    search = SearchClient()

    state.add_trace_event("agent_start", {"agent": "baseline"})

    try:
        # Search once
        docs = search.search(query, max_results=request.max_sources)
        state.sources = docs

        context_list = []
        for idx, doc in enumerate(state.sources, 1):
            context_list.append(
                f"[{idx}] Title: {doc.title}\nURL: {doc.url or 'No URL'}\nContent: {doc.snippet}"
            )
        context_str = "\n\n".join(context_list)

        sys_prompt = (
            "You are a helpful research assistant. Answer the user query based on the sources provided. "
            "Use inline citations (e.g. [1], [2]) and list a References section at the end matching these numbers."
        )
        user_prompt = f"Query: {query}\nSources:\n{context_str}"

        resp = llm.complete(sys_prompt, user_prompt)
        state.final_answer = resp.content

        state.add_trace_event(
            "llm_call",
            {"agent": "baseline", "type": "completions", "cost_usd": resp.cost_usd or 0.0},
        )
        state.add_trace_event("agent_end", {"agent": "baseline", "status": "success"})
    except Exception as e:
        logger.error(f"Error in baseline: {e}")
        state.errors.append(str(e))
        state.add_trace_event(
            "agent_end", {"agent": "baseline", "status": "error", "error": str(e)}
        )

    return state


def run_multi(query: str) -> ResearchState:
    """Run the multi-agent LangGraph workflow."""
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline implementation."""
    _init()
    console.print(f"[bold green]Running Baseline Agent for: '{query}'[/bold green]")
    state, metrics = run_benchmark("baseline", query, run_baseline)

    console.print(Panel.fit(state.final_answer or "No response", title="Baseline Final Answer"))
    console.print(f"[bold cyan]Latency:[/bold cyan] {metrics.latency_seconds:.2f}s")
    console.print(f"[bold cyan]Cost (USD):[/bold cyan] ${metrics.estimated_cost_usd or 0.0:.6f}")
    console.print(f"[bold cyan]Quality score:[/bold cyan] {metrics.quality_score or 0.0:.1f}/10")
    console.print(f"[bold cyan]Details:[/bold cyan] {metrics.notes}")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    console.print(f"[bold green]Running Multi-Agent Workflow for: '{query}'[/bold green]")
    state, metrics = run_benchmark("multi-agent", query, run_multi)

    console.print(Panel.fit(state.final_answer or "No response", title="Multi-Agent Final Answer"))
    console.print(f"[bold cyan]Latency:[/bold cyan] {metrics.latency_seconds:.2f}s")
    console.print(f"[bold cyan]Cost (USD):[/bold cyan] ${metrics.estimated_cost_usd or 0.0:.6f}")
    console.print(f"[bold cyan]Quality score:[/bold cyan] {metrics.quality_score or 0.0:.1f}/10")
    console.print(f"[bold cyan]Details:[/bold cyan] {metrics.notes}")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run both baseline and multi-agent, compare them, and output a benchmark report."""
    _init()

    console.print("[bold yellow]========================================[/bold yellow]")
    console.print("[bold yellow]       RUNNING SINGLE-AGENT BASELINE   [/bold yellow]")
    console.print("[bold yellow]========================================[/bold yellow]")
    state_base, metrics_base = run_benchmark("baseline", query, run_baseline)

    console.print("\n[bold yellow]========================================[/bold yellow]")
    console.print("[bold yellow]       RUNNING MULTI-AGENT WORKFLOW    [/bold yellow]")
    console.print("[bold yellow]========================================[/bold yellow]")
    state_multi, metrics_multi = run_benchmark("multi-agent", query, run_multi)

    # Generate report
    report_md = render_markdown_report([metrics_base, metrics_multi])

    # Append detailed output sample
    report_md += "\n\n## Summary of Answers\n\n"
    report_md += "### 1. Single-Agent Baseline Answer\n\n"
    report_md += f"{state_base.final_answer}\n\n"
    report_md += "### 2. Multi-Agent System Answer\n\n"
    report_md += f"{state_multi.final_answer}\n\n"

    # Save to report directory
    store = LocalArtifactStore()
    report_path = store.write_text("benchmark_report.md", report_md)

    console.print("\n[bold yellow]========================================[/bold yellow]")
    console.print("[bold yellow]           BENCHMARK COMPLETED         [/bold yellow]")
    console.print("[bold yellow]========================================[/bold yellow]")
    console.print(Panel.fit(report_md, title="Benchmark Summary"))
    console.print(
        f"\n[bold green]Success! Full benchmark report saved to: {report_path}[/bold green]"
    )


if __name__ == "__main__":
    app()
