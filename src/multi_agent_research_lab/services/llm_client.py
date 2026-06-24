"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client implementation."""

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.openai_api_key
        base_url = None

        # Check for Gemini key prefix to auto-route to the OpenAI compatibility endpoint
        if api_key and api_key.startswith("AQ."):
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            logger.info(
                "Gemini API key detected; using Google Gemini OpenAI compatibility endpoint."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = settings.openai_model

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Keep retry, timeout, and token logging here rather than inside agents.
        """
        logger.debug(f"Calling LLM ({self.model}) with system prompt length {len(system_prompt)}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        input_tokens = None
        output_tokens = None
        cost_usd = None

        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            # Calculate estimated cost
            # Prices per 1,000,000 tokens:
            # gpt-4o-mini: $0.150 input, $0.600 output
            # gemini-2.5-flash: $0.075 input, $0.300 output
            if "gemini" in self.model.lower():
                cost_usd = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.300 / 1_000_000)
            else:
                cost_usd = (input_tokens * 0.150 / 1_000_000) + (output_tokens * 0.600 / 1_000_000)

            logger.debug(
                f"LLM usage: input_tokens={input_tokens}, output_tokens={output_tokens}, cost_usd={cost_usd:.6f}"
            )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
