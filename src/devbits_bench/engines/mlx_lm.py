"""Native MLX-LM adapter boundary (core side, stdlib only)."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Callable

from .base import (
    EngineNotReadyError, GenerationMetrics, GenerationRequest, GenerationResult,
    ModelInfo, PreparedModel,
)
from ..runtimes.mlx import MLXRuntimeError, discover_runtime
from .mlx_worker_protocol import PROTOCOL_VERSION, WorkerProtocolError, decode_message, encode_message, request, validate_response


class MLXWorkerUnavailableError(RuntimeError):
    """The optional MLX worker environment cannot be started or contacted."""


@dataclass(frozen=True)
class MLXWorkerInfo:
    worker_version: str
    python: str
    python_version: str
    platform: str
    mlx_version: str | None
    mlx_lm_version: str | None
    capabilities: tuple[str, ...]


class MLXWorkerClient:
    def __init__(self, python_executable: str | None = None):
        if python_executable is None:
            try:
                runtime = discover_runtime()
            except MLXRuntimeError as exc:
                raise MLXWorkerUnavailableError(str(exc)) from exc
            if runtime is None:
                raise MLXWorkerUnavailableError("MLX-LM runtime is not ready")
            python_executable = str(runtime.python)
        self.python_executable = python_executable
        self._process: subprocess.Popen[str] | None = None
        self._next_id = 1

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> MLXWorkerInfo:
        if self.running:
            return self.hello()
        executable = Path(self.python_executable).expanduser()
        if not executable.is_file():
            raise MLXWorkerUnavailableError(f"MLX Python interpreter not found: {self.python_executable}")
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = source_root if not existing else source_root + os.pathsep + existing
        try:
            self._process = subprocess.Popen(
                [str(executable), "-m", "devbits_bench.workers.mlx_lm_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=env,
            )
        except OSError as exc:
            raise MLXWorkerUnavailableError(f"could not start MLX worker: {exc}") from exc
        try:
            return self.hello()
        except Exception:
            self.terminate()
            raise

    def hello(self) -> MLXWorkerInfo:
        r = self._round_trip("hello")
        deps = r.get("dependencies") or {}
        return MLXWorkerInfo(
            str(r.get("worker_version", "")), str(r.get("python", "")),
            str(r.get("python_version", "")), str(r.get("platform", "")), deps.get("mlx"),
            deps.get("mlx_lm"), tuple(str(x) for x in (r.get("capabilities") or [])),
        )

    def list_models(self) -> list[dict]:
        return list(self._round_trip("list_models").get("models") or [])

    def inspect_model(self, model_id: str) -> dict:
        return dict(self._round_trip("inspect_model", model_id=model_id).get("model") or {})

    def load_model(self, model_id: str) -> dict:
        return dict(self._round_trip("load_model", model_id=model_id).get("model") or {})

    def release_model(self) -> None:
        self._round_trip("release_model")

    def generate(self, *, prompt: str, max_output_tokens: int, reasoning: str,
                 seed: int, temperature: float, fresh_cache: bool = True,
                 on_event: Callable[[dict], None] | None = None) -> dict:
        """Stream one generation and return its terminal complete event.

        A callback exception is cancellation from the core's point of view. The
        worker is terminated because generation is synchronous inside the worker;
        this guarantees no orphaned native generation continues in the background.
        """
        try:
            return self._stream_round_trip(
                "generate", on_event=on_event, prompt=prompt,
                max_output_tokens=max_output_tokens, reasoning=reasoning,
                seed=seed, temperature=temperature, fresh_cache=fresh_cache,
            )
        except BaseException:
            self.terminate()
            raise

    def shutdown(self) -> None:
        if not self.running:
            self._close_process()
            return
        try:
            r = self._round_trip("shutdown")
            if r.get("event") != "shutdown":
                raise WorkerProtocolError("worker did not acknowledge shutdown")
            assert self._process is not None
            self._process.wait(timeout=2)
        except Exception:
            self.terminate()
            raise
        finally:
            self._close_process()

    def terminate(self) -> None:
        p = self._process
        if p is None:
            return
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=2)
        self._close_process()

    def _begin(self, op: str, **fields) -> tuple[str, subprocess.Popen[str]]:
        p = self._process
        if p is None or p.poll() is not None or p.stdin is None or p.stdout is None:
            raise MLXWorkerUnavailableError("MLX worker is not running")
        rid = str(self._next_id); self._next_id += 1
        try:
            p.stdin.write(encode_message(request(rid, op, **fields))); p.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MLXWorkerUnavailableError("MLX worker communication failed") from exc
        return rid, p

    def _read(self, p: subprocess.Popen[str], rid: str) -> dict:
        assert p.stdout is not None
        line = p.stdout.readline()
        if not line:
            detail = p.stderr.read().strip() if p.stderr is not None else ""
            raise MLXWorkerUnavailableError("MLX worker exited unexpectedly" + (f": {detail}" if detail else ""))
        return validate_response(decode_message(line), rid)

    def _round_trip(self, op: str, **fields) -> dict:
        rid, p = self._begin(op, **fields)
        return self._read(p, rid)

    def _stream_round_trip(self, op: str, on_event=None, **fields) -> dict:
        rid, p = self._begin(op, **fields)
        while True:
            message = self._read(p, rid)
            event = message.get("event")
            if event in {"reasoning_delta", "answer_delta"} and on_event is not None:
                on_event(message)
            if event == "complete":
                return message
            if event not in {"reasoning_delta", "answer_delta", "progress"}:
                raise WorkerProtocolError(f"unexpected streaming event: {event!r}")

    def _close_process(self):
        p = self._process; self._process = None
        if p:
            for s in (p.stdin, p.stdout, p.stderr):
                if s: s.close()

    def __enter__(self): self.start(); return self
    def __exit__(self, exc_type, exc, tb): self.shutdown() if exc_type is None else self.terminate()


class MLXLMEngine:
    """MLX-LM engine whose worker owns native model residency and generation."""
    name = "mlx-lm"

    def __init__(self, python_executable: str | None = None):
        self.python_executable = python_executable
        self._models: dict[str, ModelInfo] = {}
        self._worker: MLXWorkerClient | None = None
        self._prepared: PreparedModel | None = None
        self._resident = False

    def available(self, diagnostic=None) -> bool:
        if self.python_executable:
            return Path(self.python_executable).expanduser().is_file()
        try: return discover_runtime(diagnostic=diagnostic) is not None
        except MLXRuntimeError as exc:
            if diagnostic: diagnostic(f"MLX runtime: {exc}")
            return False

    def report_model_metadata(self, model_ids) -> dict:
        """Return already-qualified model metadata without reloading model weights."""
        out = {}
        for model_id in model_ids:
            info = self._models.get(model_id)
            if info is None:
                continue
            out[model_id] = {
                "architecture": info.architecture,
                "quantization": info.quantization,
                "parameter_count": info.parameter_count,
                "advertised_context": info.advertised_context,
                "reasoning_modes": list(info.reasoning_modes),
                "default_reasoning": info.default_reasoning,
            }
        return out

    def report_metadata(self) -> dict:
        """Structured runtime provenance for reports; never exposes user paths."""
        runtime = discover_runtime() if self.python_executable is None else None
        with MLXWorkerClient(self.python_executable) as c:
            info = c.hello()
        return {
            "id": self.name,
            "display_name": "MLX-LM",
            "mlx_lm_version": info.mlx_lm_version,
            "mlx_version": info.mlx_version,
            "backend": runtime.backend if runtime else None,
            "worker_protocol": PROTOCOL_VERSION,
            "worker_version": info.worker_version,
            "measurement_contract": {
                "prompt_tokens": "MLX-LM native",
                "output_tokens": "MLX-LM native",
                "prompt_tps": "MLX-LM native",
                "generation_tps": "MLX-LM native",
                "peak_memory_gb": "MLX-LM native",
                "ttft_s": "Devbits client monotonic",
                "answer_ttft_s": "Devbits client monotonic",
                "client_total_s": "Devbits client monotonic",
                "prompt_s": "derived from native prompt_tokens / prompt_tps",
                "host_memory": "Devbits before/after snapshots; not peak engine memory",
            },
        }

    def version(self) -> str:
        try:
            runtime = discover_runtime() if self.python_executable is None else None
            with MLXWorkerClient(self.python_executable) as c: i = c.hello()
            backend = runtime.backend if runtime else None
            detail = f"mlx-lm {i.mlx_lm_version or '?'} • mlx {i.mlx_version or '?'}"
            return detail + (f" • {backend}" if backend else "")
        except (MLXRuntimeError, MLXWorkerUnavailableError, WorkerProtocolError) as exc:
            raise EngineNotReadyError(
                "✗ MLX-LM was detected, but Devbits Bench could not start its worker.\n\n"
                f"  {exc}\n\n  Check the MLX-LM environment or run with --verbose for diagnostics."
            ) from exc

    @staticmethod
    def _info(raw: dict) -> ModelInfo:
        modes = tuple(str(x) for x in (raw.get("reasoning_modes") or []))
        return ModelInfo(
            engine="mlx-lm", id=str(raw["id"]), display_name=str(raw.get("display_name") or raw["id"]),
            architecture=raw.get("architecture"), quantization=raw.get("quantization"),
            parameter_count=raw.get("parameter_count"), advertised_context=raw.get("advertised_context"),
            configured_context=None, size_bytes=raw.get("size_bytes"), reasoning_modes=modes,
            default_reasoning=str(raw.get("default_reasoning") or "false"),
        )

    def list_models(self) -> list[ModelInfo]:
        try:
            with MLXWorkerClient(self.python_executable) as c: raw = c.list_models()
        except (MLXRuntimeError, MLXWorkerUnavailableError, WorkerProtocolError) as exc:
            raise EngineNotReadyError("✗ MLX-LM is installed, but its worker is not ready.\n\n  " + str(exc)) from exc
        models = [self._info(x) for x in raw]; self._models = {m.id: m for m in models}; return models

    def inspect_model(self, model_id: str) -> ModelInfo:
        try:
            with MLXWorkerClient(self.python_executable) as c: raw = c.inspect_model(model_id)
        except (MLXRuntimeError, MLXWorkerUnavailableError, WorkerProtocolError) as exc:
            raise EngineNotReadyError(f"✗ MLX-LM model inspection failed.\n\n  {exc}") from exc
        info = self._info(raw); self._models[info.id] = info; return info

    def prepare_model(self, model: ModelInfo, context: int) -> PreparedModel:
        if model.engine != self.name:
            raise ValueError(f"cannot prepare {model.engine!r} model with MLX-LM")
        if context <= 0:
            raise ValueError("configured context must be positive")
        if model.advertised_context is not None and context > model.advertised_context:
            raise ValueError(f"requested context {context} exceeds advertised model context {model.advertised_context}")
        self.release(self._prepared) if self._prepared is not None else None
        worker = MLXWorkerClient(self.python_executable)
        try:
            worker.start(); worker.load_model(model.id)
        except Exception:
            worker.terminate(); raise
        prepared = PreparedModel(model=model, id=model.id, configured_context=context)
        self._worker = worker; self._prepared = prepared; self._resident = True
        return prepared

    def release(self, prepared: PreparedModel | None) -> None:
        worker = self._worker
        self._worker = None; self._prepared = None; self._resident = False
        if worker is None: return
        try:
            if worker.running: worker.release_model(); worker.shutdown()
        except Exception:
            worker.terminate()
            raise

    def reset(self, prepared: PreparedModel) -> None:
        # Quick cold semantics for MLX-LM: keep the worker/process alive, but
        # release the loaded model and clear MLX caches. The next generate()
        # performs a measured deferred reload. Prompt/KV state is independently
        # fresh for every request.
        worker = self._worker
        if prepared != self._prepared or worker is None or not worker.running:
            raise EngineNotReadyError("MLX-LM prepared model is not available")
        if self._resident:
            worker.release_model()
            self._resident = False

    def isolated_generation_keeps_residency(self) -> bool:
        """Each generation gets a fresh native prompt cache while weights stay resident."""
        return True

    def generate(self, prepared: PreparedModel, request: GenerationRequest,
                 on_event: Callable[[str], None] | None = None) -> GenerationResult:
        worker = self._worker
        if prepared != self._prepared or worker is None or not worker.running:
            raise EngineNotReadyError("MLX-LM prepared model is not available")
        started = time.monotonic()
        load_s = 0.0
        if not self._resident:
            load_started = time.monotonic()
            worker.load_model(prepared.model.id)
            load_s = time.monotonic() - load_started
            self._resident = True
        if request.reasoning not in prepared.model.reasoning_modes and request.reasoning != "false":
            raise ValueError(f"unsupported MLX-LM reasoning mode: {request.reasoning}")
        first_generated = None; first_answer = None
        generation_notified = False; answer_notified = False
        thinking: list[str] = []; answer: list[str] = []

        def event(message: dict) -> None:
            nonlocal first_generated, first_answer, generation_notified, answer_notified
            text = str(message.get("text") or "")
            if text and first_generated is None:
                first_generated = time.monotonic()
                if on_event and not generation_notified:
                    on_event("generation_started"); generation_notified = True
            if message.get("event") == "reasoning_delta":
                thinking.append(text)
            elif message.get("event") == "answer_delta":
                answer.append(text)
                if text.strip() and first_answer is None:
                    first_answer = time.monotonic()
                    if on_event and not answer_notified:
                        on_event("answer_started"); answer_notified = True

        complete = worker.generate(
            prompt=request.prompt, max_output_tokens=request.max_output_tokens,
            reasoning=request.reasoning, seed=request.seed, temperature=request.temperature,
            fresh_cache=True, on_event=event,
        )
        ended = time.monotonic(); native = dict(complete.get("metrics") or {})
        provenance = {
            "prompt_tokens": "mlx-lm native", "output_tokens": "mlx-lm native",
            "prompt_tps": "mlx-lm native", "generation_tps": "mlx-lm native",
            "ttft_s": "devbits client monotonic", "answer_ttft_s": "devbits client monotonic",
            "client_total_s": "devbits client monotonic",
        }
        prompt_tokens = native.get("prompt_tokens")
        prompt_tps = native.get("prompt_tps")
        prompt_s = (prompt_tokens / prompt_tps) if prompt_tokens is not None and prompt_tps else None
        client_total = ended - started
        provenance.update({
            "cached_prompt_tokens": "devbits adapter: fresh prompt cache => 0 reused prompt tokens",
            "load_s": "devbits client monotonic around native model reload",
            "prompt_s": "derived: mlx-lm native prompt_tokens / native prompt_tps",
            "total_s": "devbits client monotonic",
        })
        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens, cached_prompt_tokens=0, output_tokens=native.get("generation_tokens"),
            load_s=load_s, prompt_s=prompt_s, total_s=client_total,
            prompt_tps=prompt_tps, generation_tps=native.get("generation_tps"),
            ttft_s=(first_generated-started) if first_generated is not None else None,
            answer_ttft_s=(first_answer-started) if first_answer is not None else None,
            client_total_s=client_total, provenance=provenance,
        )
        return GenerationResult(
            metrics=metrics, thinking_text="".join(thinking), answer_text="".join(answer),
            requested_reasoning=request.reasoning, effective_reasoning=str(complete.get("effective_reasoning") or request.reasoning),
            native_metrics={**native, "fresh_prompt_cache": True, "configured_context": prepared.configured_context},
        )
