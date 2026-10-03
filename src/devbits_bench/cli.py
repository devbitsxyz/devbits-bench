"""Devbits Bench CLI.

v0.2 structural pass: preserves the v0.1 benchmark methodology while moving
runtime ownership behind an engine boundary. MLX-LM support is intentionally not
implemented in this pass.
"""
from __future__ import annotations
from .engines import OllamaEngine
from .engines.base import EngineNotReadyError
from . import legacy


def main() -> None:
    # v0.2-dev currently exposes the proven Ollama engine only. The package/engine
    # contract must be qualified for MLX-LM before adding another adapter.
    engine = OllamaEngine()
    legacy.set_engine(engine)
    try:
        legacy.main()
    except EngineNotReadyError as exc:
        raise SystemExit(str(exc)) from None


def entrypoint() -> None:
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark cancelled by user.")
        raise SystemExit(130)
