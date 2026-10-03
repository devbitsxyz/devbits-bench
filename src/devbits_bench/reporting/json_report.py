"""JSON report serialization.

This module deliberately preserves the v0.1 report shape while report schema v2 is
being designed. Keeping serialization outside protocol runners prevents engine APIs
from leaking into reporting without changing benchmark comparability.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable, Any

from ..benchmark.models import Result


def json_document(*, protocol: str, system: dict[str, Any], engine_name: str,
                  engine_version: str, results: Iterable[Result], suite: str | None = None,
                  corpus: str | None = None, corpus_seed: int | None = None,
                  memory_pressure_policy: str | None = None,
                  output_budget_tokens: int | None = None,
                  context_integrity_check: bool | None = None) -> dict[str, Any]:
    """Build the compatibility JSON document used by v0.1/v0.2-dev.

    ``engine_name`` remains a dynamic key (``ollama`` today) so extracting this
    module does not silently change existing report consumers. A versioned generic
    engine schema will be introduced separately when multi-engine reporting lands.
    """
    doc: dict[str, Any] = {"protocol": protocol}
    if suite is not None:
        doc["suite"] = suite
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
    doc["results"] = [asdict(x) for x in results]
    return doc


def write_json_report(path: Path, **kwargs: Any) -> None:
    path.write_text(json.dumps(json_document(**kwargs), indent=2))
