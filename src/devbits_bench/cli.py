"""Devbits Bench CLI composition root."""
from __future__ import annotations

from . import legacy
from .engines.base import EngineNotReadyError
from .engines.discovery import EngineState, discover_engines
from .ui import terminal as ui
from .ui.terminal import B, C, col


def _select_engine(requested: str | None = None, assume_yes: bool = False):
    engines = discover_engines(diagnostic=ui.ui_verbose)
    if requested:
        key = requested.strip().lower()
        matches = [item for item in engines if key in (item.candidate.id.lower(), item.candidate.display_name.lower())]
        if not matches:
            raise SystemExit(f"engine is not available on this platform: {requested}")
        selected = matches[0]
        if selected.state is EngineState.READY:
            return selected.engine
        raise SystemExit(f"engine is not available: {requested}")

    if not engines:
        raise SystemExit("no supported benchmark engine is available")

    print(col("Available engines", B))
    print()
    for index, item in enumerate(engines, 1):
        detail = item.detail if item.detail else item.state.value
        if item.state is EngineState.UNAVAILABLE:
            detail = "Detected • not ready"
        print(f"  {col(str(index).rjust(2), C)}  {item.candidate.display_name:<12} {detail}")
    print()
    raw = input("Select engine [1]: ").strip() or "1"
    print()
    try:
        index = int(raw)
    except ValueError:
        raise SystemExit("Invalid engine selection") from None
    if not 1 <= index <= len(engines):
        raise SystemExit("Invalid engine selection")
    selected = engines[index - 1]
    if selected.state is EngineState.READY:
        return selected.engine
    if selected.detail:
        raise SystemExit(selected.detail)
    raise SystemExit(f"engine is not ready: {selected.candidate.display_name}")


def main() -> None:
    try:
        legacy.main(engine_selector=_select_engine)
    except EngineNotReadyError as exc:
        raise SystemExit(str(exc)) from None


def entrypoint() -> None:
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark cancelled by user.")
        raise SystemExit(130)
