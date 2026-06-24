"""Tracing hooks implementation."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Span context manager that logs duration and metadata."""
    logger.info(f"Starting trace span: '{name}'")
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.info(f"Trace span '{name}' finished in {span['duration_seconds']:.4f} seconds.")
