"""Versioned JSON-lines protocol shared with the optional MLX-LM worker."""
from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 1


class WorkerProtocolError(RuntimeError):
    """The MLX worker violated the Devbits worker protocol."""


def encode_message(message: dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"


def decode_message(line: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise WorkerProtocolError(f"invalid worker JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise WorkerProtocolError("worker message must be a JSON object")
    return value


def request(request_id: str, op: str, **fields: Any) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "id": request_id, "op": op, **fields}


def validate_response(message: dict[str, Any], request_id: str) -> dict[str, Any]:
    if message.get("v") != PROTOCOL_VERSION:
        raise WorkerProtocolError(
            f"unsupported worker protocol version: {message.get('v')!r}"
        )
    if message.get("id") != request_id:
        raise WorkerProtocolError("worker response id does not match request")
    if message.get("event") == "error":
        detail = message.get("message") or "MLX worker reported an error"
        raise WorkerProtocolError(str(detail))
    return message
