# Design Template

## Problem

Complex research tasks (e.g. comparing SOTA models or describing architectural patterns) require multi-stage workflows: generating search queries, finding articles, assessing their validity, identifying gaps/conflicts, draft-writing, checking for hallucinations, and formatting references. A single-agent system struggles to accomplish all these steps reliably in a single generation, often leading to hallucinations, loose arguments, or lack of proper citations.

## Why multi-agent?

A multi-agent approach breaks this monolithic task into specialized roles with independent instructions and boundaries:
1. **Researcher** focuses only on search query generation and source parsing, reducing information overload.
2. **Analyst** critiques research findings objectively without writing the final prose.
3. **Writer** synthesizes clear prose tailored to the audience, focusing on citation layout.
4. **Critic** serves as an independent gatekeeper to factcheck the writer's report.
5. **Supervisor** directs coordination dynamically without hardcoding state dependencies in the workers.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| **Supervisor** | Orchestrates flow, manages iteration budget, and routes to next worker | `ResearchState` | Updated `route_history` | Runaway routing loop (mitigated by max iteration caps) |
| **Researcher** | Optimizes search queries, retrieves docs, and drafts raw research notes | Query, source list | `research_notes`, unique `sources` | API timeout or empty search returns (mitigated by mock database fallback) |
| **Analyst** | Extracts core claims, synthesizes viewpoints, and reports research gaps | `research_notes` | `analysis_notes` | Summarization hallucination (mitigated by strict formatting prompt) |
| **Critic** | Evaluates final report for accuracy, formatting, and citation coverage | `final_answer`, sources | Verdict (`APPROVED` / `NEEDS_REVISION`) & feedback | Infinite critique rejection loop (mitigated by maximum 2 critic runs cap) |
| **Writer** | Compiles research and analysis notes into a structured report with citations | Notes, sources | `final_answer` | Missing inline citations or reference keys (mitigated by post-fact-check edit loop) |

## Shared state

We use `ResearchState` (inherits from Pydantic `BaseModel`) which contains:
- `request`: Config details (query, audience, limits).
- `iteration` / `route_history`: Track execution steps and detect loops.
- `sources`: Shared list of `SourceDocument` objects containing raw snippets and URLs.
- `research_notes` & `analysis_notes`: Intermediate files exchanged between researcher, analyst, and writer.
- `final_answer`: Polished markdown output of the Writer.
- `agent_results`: Output contents from each agent for history check.
- `trace` / `errors`: Span durations, LLM API costs, and exceptions.

## Routing policy

Our workflow is implemented as a cycle controlled by the Supervisor:
```text
[START] -> Supervisor (Router)
             |
             +---> Researcher -> Supervisor
             |
             +---> Analyst -> Supervisor
             |
             +---> Writer -> Supervisor
             |
             +---> Critic -> Supervisor (NEEDS_REVISION loops back to Researcher)
             |
             +---> [END] (done / iteration limit)
```

## Guardrails

- **Max iterations**: Enforced in `SupervisorAgent` by comparing `state.iteration` against settings (`MAX_ITERATIONS=6`).
- **Timeout**: Enforced on Tavily Search requests (10 seconds timeout) and LLM API requests.
- **Retry**: Decorated on `LLMClient.complete` using `tenacity.retry` with exponential backoff and 3 attempts.
- **Fallback**:
  - `SearchClient` automatically falls back to a high-quality local mock search engine matching topic keywords if no Tavily API Key is found or if search fails.
  - Critic agent rejections are capped at `2` runs; any subsequent rejection is bypassed and routed to `"done"`.
- **Validation**: Critic parses output verdict carefully and Supervisor validates routing history contents to avoid invalid transitions.

## Benchmark plan

- **Queries**:
  - `Research GraphRAG state-of-the-art and write a 500-word summary`
- **Metrics**:
  - Latency: Wall-clock execution time (seconds).
  - Cost: Estimated token usage cost based on prompt & completion pricing.
  - Citation coverage: Percentage of search sources referenced with inline brackets in the final report.
  - Quality score: Auditor LLM grade on a 0.0 to 10.0 scale.
