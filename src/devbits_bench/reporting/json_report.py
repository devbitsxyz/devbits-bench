"""JSON report serialization with compatibility-preserving multi-engine provenance."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable, Any

from ..benchmark.models import Result


def _result_document(result: Result, *, include_provenance: bool) -> dict[str, Any]:
    row = asdict(result)
    if include_provenance:
        provenance = getattr(result, "_measurement_provenance", None)
        native = getattr(result, "_native_metrics", None)
        requested = getattr(result, "_requested_reasoning", None)
        effective = getattr(result, "_effective_reasoning", None)
        if provenance:
            row["measurement_provenance"] = dict(provenance)
        if native:
            row["native_metrics"] = dict(native)
        if requested is not None:
            row["requested_reasoning"] = requested
        if effective is not None:
            row["effective_reasoning"] = effective
    return row


def json_document(*, protocol: str, system: dict[str, Any], engine_name: str,
                  engine_version: str, results: Iterable[Result], suite: str | None = None,
                  corpus: str | None = None, corpus_seed: int | None = None,
                  memory_pressure_policy: str | None = None,
                  output_budget_tokens: int | None = None,
                  context_integrity_check: bool | None = None,
                  engine_metadata: dict[str, Any] | None = None,
                  model_metadata: dict[str, Any] | None = None,
                  suite_status: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a report while preserving the historical Ollama compatibility shape.

    Non-Ollama engines may add structured provenance. The legacy dynamic engine
    version key remains present so existing readers continue to work.
    """
    doc: dict[str, Any] = {"protocol": protocol}
    if suite is not None:
        doc["suite"] = suite
    if suite_status is not None:
        doc["suite_status"] = dict(suite_status)
    if memory_pressure_policy is not None:
        doc["memory_pressure_policy"] = memory_pressure_policy
    if corpus is not None:
        doc["corpus"] = corpus
    if corpus_seed is not None:
        doc["corpus_seed"] = corpus_seed
    if output_budget_tokens is not None:
        doc["output_budget_tokens"] = output_budget_tokens
    if context_integrity_check is not None:
        doc["context_integrity_check"] = context_integrity_check
    doc["system"] = system
    doc[engine_name] = engine_version
    include_provenance = engine_name.lower() != "ollama"
    if include_provenance:
        doc["report_schema"] = "devbits-report-v2"
    if include_provenance and engine_metadata:
        doc["engine"] = engine_metadata
    if include_provenance and model_metadata:
        doc["models"] = model_metadata
    doc["results"] = [_result_document(x, include_provenance=include_provenance) for x in results]
    return doc


def write_json_report(path: Path, **kwargs: Any) -> None:
    path.write_text(json.dumps(json_document(**kwargs), indent=2))
