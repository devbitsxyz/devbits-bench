"""Portable discovery of inference engines already installed on the host."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .base import Engine, EngineNotReadyError
from .ollama import OllamaEngine
from .mlx_lm import MLXLMEngine


class EngineState(str, Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class EngineCandidate:
    id: str
    display_name: str
    factory: Callable[[], Engine]


@dataclass(frozen=True)
class DiscoveredEngine:
    candidate: EngineCandidate
    state: EngineState
    engine: Engine | None = None
    detail: str = ""


def supported_engine_candidates() -> tuple[EngineCandidate, ...]:
    return (
        EngineCandidate("ollama", "Ollama", OllamaEngine),
        EngineCandidate("mlx-lm", "MLX-LM", MLXLMEngine),
    )


def discover_engines(diagnostic: Callable[[str], None] | None = None) -> list[DiscoveredEngine]:
    """Return detected engines; discovery never installs software."""
    log = diagnostic or (lambda _message: None)
    found: list[DiscoveredEngine] = []
    for candidate in supported_engine_candidates():
        log(f"Engine discovery: checking {candidate.display_name}")
        engine = candidate.factory()
        available = engine.available(diagnostic=log) if isinstance(engine, MLXLMEngine) else engine.available()
        if available:
            try:
                detail = engine.version()
            except EngineNotReadyError as exc:
                log(f"Engine discovery: {candidate.display_name} detected but not ready: {exc}")
                found.append(DiscoveredEngine(candidate, EngineState.UNAVAILABLE, engine, str(exc)))
                continue
            log(f"Engine discovery: {candidate.display_name} ready ({detail})")
            found.append(DiscoveredEngine(candidate, EngineState.READY, engine, detail))
        else:
            log(f"Engine discovery: {candidate.display_name} not detected")
    return found


def discover_available_engines() -> list[tuple[EngineCandidate, Engine]]:
    return [(item.candidate, item.engine) for item in discover_engines()
            if item.state is EngineState.READY and item.engine is not None]
