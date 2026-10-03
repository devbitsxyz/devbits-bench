"""Engine-neutral runtime contract; times are seconds, counts are engine tokens."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class EngineNotReadyError(RuntimeError):
    """Expected runtime unavailability, with an adapter-supplied user message."""


@dataclass(frozen=True)
class EngineInfo:
    name: str
    version: str


@dataclass(frozen=True)
class ModelInfo:
    engine: str
    id: str
    display_name: str
    architecture: str | None = None
    quantization: str | None = None
    parameter_count: str | None = None
    advertised_context: int | None = None
    configured_context: int | None = None
    size_bytes: int | None = None
    # Opaque, engine-advertised choices, not a universal reasoning-level enum.
    reasoning_modes: tuple[str, ...] = ()
    default_reasoning: str = "false"


@dataclass(frozen=True)
class PreparedModel:
    """Handle whose lifecycle belongs to the engine that prepared it.

    ``id`` is an opaque runtime identifier retained in compatibility reports.
    Configured context is a request window, never a synonym for a KV cache limit.
    """
    model: ModelInfo
    id: str
    configured_context: int


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_output_tokens: int
    reasoning: str
    seed: int = 42
    temperature: float = 0
    stream: bool = True


@dataclass(frozen=True)
class GenerationMetrics:
    """Missing measurements remain None; adapters identify their provenance.

    Prompt throughput uses uncached tokens for the current Ollama protocol.
    Native request total and client-observed total are deliberately distinct.
    """
    prompt_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    output_tokens: int | None = None
    load_s: float | None = None
    prompt_s: float | None = None
    total_s: float | None = None
    prompt_tps: float | None = None
    generation_tps: float | None = None
    ttft_s: float | None = None
    answer_ttft_s: float | None = None
    client_total_s: float | None = None
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    metrics: GenerationMetrics
    thinking_text: str = ""
    answer_text: str = ""
    requested_reasoning: str | None = None
    # None means the engine did not establish an effective mode.
    effective_reasoning: str | None = None
    native_metrics: dict[str, Any] = field(default_factory=dict)


class Engine(Protocol):
    """Adapters own discovery, preparation, residency and native API translation.

    available() checks client installation, not runtime readiness. Discovery may
    raise EngineNotReadyError when an installed runtime cannot be reached.
    reset() establishes the engine's cold state. For Ollama this unloads all
    running models, as in v0.1. release() disposes preparation resources; it does
    not imply an extra unload. Generation errors are raised to the runner.
    """
    @property
    def name(self) -> str: ...
    def available(self) -> bool: ...
    def version(self) -> str: ...
    def list_models(self) -> list[ModelInfo]: ...
    def inspect_model(self, model_id: str) -> ModelInfo: ...
    def prepare_model(self, model: ModelInfo, context: int) -> PreparedModel: ...
    def release(self, prepared: PreparedModel) -> None: ...
    def reset(self, prepared: PreparedModel) -> None: ...
    def generate(self, prepared: PreparedModel, request: GenerationRequest,
                 on_event: Callable[[str], None] | None = None) -> GenerationResult: ...
