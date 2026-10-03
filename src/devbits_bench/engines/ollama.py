"""Ollama translation and lifecycle; benchmark protocols use the generic contract."""
from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .base import EngineNotReadyError, GenerationMetrics, GenerationRequest, GenerationResult, ModelInfo, PreparedModel


class OllamaEngine:
    name = "ollama"

    def __init__(self, api_url: str = "http://localhost:11434"):
        self.api_url = api_url.rstrip("/")
        self._listed_models: dict[str, ModelInfo] = {}
        # Keep live handles by identity: an equal or foreign handle owns nothing.
        self._prepared: dict[int, PreparedModel] = {}
        self._variant_handles: dict[int, tuple[str, int]] = {}
        self._variants: dict[tuple[str, int], str] = {}

    def available(self) -> bool:
        return shutil.which("ollama") is not None

    def version(self) -> str:
        p = subprocess.run(["ollama", "--version"], text=True, capture_output=True, check=True)
        # A stopped server makes the CLI print warnings to stdout, including its
        # useful client version. Keep those warnings out of the system summary.
        version = "\n".join(line for line in p.stdout.splitlines()
                            if not line.lstrip().startswith("Warning:")).strip()
        if version:
            return version
        client = re.search(r"(?m)^\s*Warning: client version is (\S+)\s*$", p.stdout + "\n" + p.stderr)
        return f"client version is {client.group(1)}" if client else "version unavailable"

    def list_models(self) -> list[ModelInfo]:
        try:
            response = self._api("GET", "/api/tags")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            # HTTP errors mean a server answered. Invalid URLs, malformed data
            # and unexpected adapter failures must remain diagnosable.
            if isinstance(exc, urllib.error.HTTPError) or not isinstance(reason, (ConnectionError, TimeoutError, socket.gaierror)):
                raise
            raise EngineNotReadyError(
                "✗ Ollama is installed, but its local server is not running or cannot be reached.\n\n"
                f"  Could not connect to {self.api_url}.\n\n"
                "  Start Ollama, then run the benchmark again.\n\n"
                "  Homebrew service:\n"
                "    brew services start ollama\n\n"
                "  Or run it manually in another terminal:\n"
                "    ollama serve\n\n"
                "  Devbits Bench never starts background services automatically."
            ) from exc
        models = [self._model_info(item.get("name") or item["model"], item)
                  for item in response.get("models", [])]
        self._listed_models = {model.id: model for model in models}
        return models

    def inspect_model(self, model_id: str) -> ModelInfo:
        info = self._api("POST", "/api/show", {"model": model_id})
        return self._model_info(model_id, info)

    def prepare_model(self, model: ModelInfo, context: int) -> PreparedModel:
        if model.engine != self.name:
            raise ValueError(f"Cannot prepare a {model.engine} model with {self.name}")
        key = (model.id, context)
        if model.configured_context == context:
            runtime_id = model.id
        else:
            if key not in self._variants:
                self._variants[key] = self._create_context_variant(model.id, context)
            runtime_id = self._variants[key]
        prepared = PreparedModel(model, runtime_id, context)
        self._prepared[id(prepared)] = prepared
        if runtime_id != model.id:
            self._variant_handles[id(prepared)] = key
        return prepared

    def release(self, prepared: PreparedModel) -> None:
        """Dispose only owned temporary variants, after their final live handle."""
        handle_id = id(prepared)
        if self._prepared.get(handle_id) is not prepared:
            return
        key = self._variant_handles.get(handle_id)
        if key is not None and sum(value == key for value in self._variant_handles.values()) == 1:
            # Retain ownership if the CLI cannot even be launched, allowing the
            # caller to retry cleanup. A completed nonzero exit remains ignored
            # just as the historical best-effort `ollama rm` operation was.
            self._remove_context_variant(self._variants[key])
            del self._variants[key]
        del self._prepared[handle_id]
        self._variant_handles.pop(handle_id, None)

    def reset(self, prepared: PreparedModel) -> None:
        # Historical cold means unload ALL resident models, not just this handle.
        self._unload_all()

    def generate(self, prepared: PreparedModel, request: GenerationRequest,
                 on_event: Callable[[str], None] | None = None) -> GenerationResult:
        payload = {
            "model": prepared.id,
            "prompt": request.prompt,
            "think": False if request.reasoning == "false" else request.reasoning,
            "keep_alive": "5m",
            "options": {
                "seed": request.seed,
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        if request.stream:
            final, ttft, answer_ttft, client_total, thinking, answer = self._stream_generate(payload, on_event)
        else:
            payload["stream"] = False
            final = self._api("POST", "/api/generate", payload)
            ttft = answer_ttft = client_total = None
            thinking = final.get("thinking") or ""
            answer = final.get("response") or ""
        return GenerationResult(
            metrics=self._metrics(final, ttft, answer_ttft, client_total),
            thinking_text=thinking,
            answer_text=answer,
            requested_reasoning=request.reasoning,
            effective_reasoning=None,
            native_metrics=final,
        )

    def _model_info(self, model_id: str, info: dict[str, Any]) -> ModelInfo:
        details = info.get("details") or {}
        native = info.get("model_info") or {}
        advertised = max([int(value) for key, value in native.items()
                          if key.endswith(".context_length") and isinstance(value, (int, float))] or [0])
        thinking = info.get("thinking")
        default = thinking.get("default") if isinstance(thinking, dict) else False
        modes = (thinking.get("values") or []) if isinstance(thinking, dict) else []
        listed = self._listed_models.get(model_id)
        return ModelInfo(
            engine=self.name,
            id=model_id,
            display_name=model_id,
            architecture=details.get("family") or native.get("general.architecture"),
            quantization=details.get("quantization_level"),
            parameter_count=details.get("parameter_size"),
            advertised_context=advertised or None,
            configured_context=self._configured_context(info) or None,
            size_bytes=info.get("size", listed.size_bytes if listed else None),
            reasoning_modes=tuple(str(mode).lower() if isinstance(mode, bool) else str(mode) for mode in modes),
            default_reasoning=str(default).lower(),
        )

    @staticmethod
    def _metrics(final: dict[str, Any], ttft: float | None, answer_ttft: float | None,
                 client_total: float | None) -> GenerationMetrics:
        # Missing native fields intentionally keep the v0.1 zero defaults. The
        # generic contract permits None for engines without these measurements.
        def seconds(value):
            return (value or 0) / 1e9

        def rate(count, duration):
            return (count or 0) / seconds(duration) if duration else 0.0

        prompt_tokens = final.get("prompt_eval_count", 0)
        cached = final.get("prompt_eval_cached_count", 0)
        uncached = max(0, prompt_tokens - cached)
        fields = {
            "prompt_tokens": "prompt_eval_count",
            "cached_prompt_tokens": "prompt_eval_cached_count",
            "output_tokens": "eval_count",
            "load_s": "load_duration",
            "prompt_s": "prompt_eval_duration",
            "total_s": "total_duration",
        }
        provenance = {metric: f"ollama:{native}" + (" (legacy default: 0)" if native not in final else "")
                      for metric, native in fields.items()}
        provenance.update({
            "prompt_tps": "derived: max(0, prompt_eval_count - prompt_eval_cached_count) / prompt_eval_duration_s",
            "generation_tps": "derived: eval_count / eval_duration_s",
        })
        if client_total is not None:
            provenance.update({
                "ttft_s": "client: request start to first thinking or answer chunk",
                "answer_ttft_s": "client: request start to first answer chunk",
                "client_total_s": "client: request start to stream close",
            })
        return GenerationMetrics(
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached,
            output_tokens=final.get("eval_count", 0),
            load_s=seconds(final.get("load_duration")),
            prompt_s=seconds(final.get("prompt_eval_duration")),
            total_s=seconds(final.get("total_duration")),
            prompt_tps=rate(uncached, final.get("prompt_eval_duration")),
            generation_tps=rate(final.get("eval_count"), final.get("eval_duration")),
            ttft_s=ttft,
            answer_ttft_s=answer_ttft,
            client_total_s=client_total,
            provenance=provenance,
        )

    def _api(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 1800):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.api_url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    @staticmethod
    def _configured_context(info: dict[str, Any]) -> int:
        params = info.get("parameters") or ""
        m = re.search(r"(?:^|\n)\s*num_ctx\s+(\d+)", params)
        if m:
            return int(m.group(1))
        details = info.get("details") or {}
        for key in ("context_length", "context_window"):
            if isinstance(details.get(key), (int, float)):
                return int(details[key])
        return 0

    def _create_context_variant(self, base: str, context: int) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", base.replace(":", "-")).strip("-").lower()
        tag = f"devbits-bench-{safe}-{context}"
        with tempfile.TemporaryDirectory() as td:
            mf = Path(td) / "Modelfile"
            mf.write_text(f"FROM {base}\nPARAMETER num_ctx {context}\n")
            subprocess.run(["ollama", "create", tag, "-f", str(mf)], text=True, capture_output=True, check=True)
        return tag

    def _remove_context_variant(self, tag: str) -> None:
        subprocess.run(["ollama", "rm", tag], text=True, capture_output=True, check=False)

    def _unload_all(self) -> None:
        try:
            running = self._api("GET", "/api/ps").get("models", [])
        except Exception:
            return
        for model in running:
            name = model.get("name") or model.get("model")
            if name:
                try:
                    self._api("POST", "/api/generate", {"model": name, "keep_alive": 0})
                except Exception:
                    pass

    def _stream_generate(self, payload: dict[str, Any], on_event: Callable[[str], None] | None = None):
        """Preserve all six v0.1 observations and progress-event clock boundaries."""
        body = dict(payload)
        body["stream"] = True
        req = urllib.request.Request(
            self.api_url + "/api/generate",
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        first_generated = None
        first_answer = None
        final = {}
        thoughts = []
        answers = []
        if on_event:
            on_event("request_sent")
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                if not raw.strip():
                    continue
                event = json.loads(raw)
                now = time.perf_counter()
                if first_generated is None and (event.get("thinking") or event.get("response")):
                    first_generated = now
                    if on_event:
                        on_event("generation_started")
                if first_answer is None and event.get("response"):
                    first_answer = now
                    if on_event:
                        on_event("answer_started")
                if event.get("thinking"):
                    thoughts.append(event["thinking"])
                if event.get("response"):
                    answers.append(event["response"])
                if event.get("done"):
                    final = event
        t1 = time.perf_counter()
        return (
            final,
            (first_generated - t0) if first_generated else None,
            (first_answer - t0) if first_answer else None,
            t1 - t0,
            "".join(thoughts),
            "".join(answers),
        )
