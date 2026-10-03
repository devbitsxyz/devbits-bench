from __future__ import annotations
from dataclasses import dataclass
import statistics

@dataclass
class Result:
    base_model: str
    model: str
    context: int
    mode: str
    run: int
    thinking: str
    requested_fill: int
    prompt_tokens: int
    cached_prompt_tokens: int
    output_tokens: int
    load_s: float
    prompt_s: float
    total_s: float
    prompt_tps: float
    generation_tps: float
    memory_before: dict
    memory_after: dict
    ttft_s: float | None = None
    answer_ttft_s: float | None = None
    client_total_s: float | None = None
    phase: str = "Manual"
    measured: bool = True
    isolated: bool = False
    stabilization_s: float = 0.0
    checkpoint_pass: bool | None = None
    checkpoint_hits: int = 0
    checkpoint_total: int = 0
    suite_memory_start: dict | None = None
    benchmark_thinking: str | None = None
    error: str = ""

def summarize_trials(trials, field: str) -> dict[str, float]:
    vals = [getattr(x, field) for x in trials if not x.error and getattr(x, field) is not None]
    if not vals:
        return {}
    return {
        "median": statistics.median(vals),
        "mean": statistics.mean(vals),
        "min": min(vals),
        "max": max(vals),
        "stddev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }
