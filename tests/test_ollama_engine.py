from pathlib import Path
import subprocess
import unittest
from unittest.mock import call, patch

from devbits_bench.engines.base import GenerationRequest, ModelInfo, PreparedModel
from devbits_bench.engines.ollama import OllamaEngine


class OllamaEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = OllamaEngine()
        self.model = ModelInfo("ollama", "org/Model:Q4_K_M", "org/Model:Q4_K_M", configured_context=4096)
        self.prepared = self.engine.prepare_model(self.model, 4096)

    def test_discovery_and_inspection_separate_configured_and_advertised_context(self):
        details = {"family": "qwen", "parameter_size": "27B", "quantization_level": "Q4_K_M"}
        show = {
            "details": dict(details, context_length=8192),
            "parameters": "temperature 0.7\nnum_ctx 32768\n",
            "model_info": {"qwen.context_length": 131072, "other.context_length": 65536},
            "thinking": {"default": "Medium", "values": [False, "low", "medium", "high"]},
        }
        with patch.object(self.engine, "_api", side_effect=[
            {"models": [{"name": self.model.id, "size": 100, "details": details}]}, show,
        ]) as api:
            listed = self.engine.list_models()
            info = self.engine.inspect_model(self.model.id)
        self.assertEqual(listed[0].id, self.model.id)
        self.assertEqual(listed[0].architecture, "qwen")
        self.assertEqual(listed[0].quantization, "Q4_K_M")
        self.assertEqual(listed[0].parameter_count, "27B")
        self.assertEqual(info.size_bytes, 100)
        self.assertEqual(info.advertised_context, 131072)
        self.assertEqual(info.configured_context, 32768)
        self.assertEqual(info.reasoning_modes, ("false", "low", "medium", "high"))
        self.assertEqual(info.default_reasoning, "medium")
        self.assertEqual(api.call_args_list, [call("GET", "/api/tags"), call("POST", "/api/show", {"model": self.model.id})])

    def test_metadata_omissions_and_historical_context_fallbacks(self):
        with patch.object(self.engine, "_api", return_value={"model_info": {"x.context_length": 8192}}):
            info = self.engine.inspect_model("minimal")
        self.assertEqual(info.advertised_context, 8192)
        self.assertIsNone(info.configured_context)
        self.assertIsNone(info.quantization)
        self.assertIsNone(info.architecture)
        self.assertIsNone(info.size_bytes)
        self.assertEqual(info.default_reasoning, "false")
        self.assertEqual(info.reasoning_modes, ())
        with patch.object(self.engine, "_api", return_value={"thinking": {"values": None}}):
            info = self.engine.inspect_model("minimal")
        self.assertEqual(info.reasoning_modes, ())
        self.assertEqual(info.default_reasoning, "none")
        for details, expected in [({"context_length": 8192, "context_window": 4096}, 8192),
                                  ({"context_window": 4096.0}, 4096), ({"context_length": "8192"}, 0)]:
            with self.subTest(details=details):
                self.assertEqual(self.engine._configured_context({"details": details}), expected)

    def test_direct_preparation_and_release_do_not_touch_runtime(self):
        with patch.object(self.engine, "_api") as api, patch("devbits_bench.engines.ollama.subprocess.run") as run:
            prepared = self.engine.prepare_model(self.model, 4096)
            self.assertEqual(prepared.id, self.model.id)
            self.engine.release(prepared)
            self.engine.release(prepared)
        api.assert_not_called()
        run.assert_not_called()

    def test_shared_variant_lifecycle_and_foreign_handles(self):
        with patch.object(self.engine, "_create_context_variant", return_value="variant") as create, \
             patch.object(self.engine, "_remove_context_variant") as remove, \
             patch.object(self.engine, "_unload_all") as unload:
            first = self.engine.prepare_model(self.model, 8192)
            second = self.engine.prepare_model(self.model, 8192)
            create.assert_called_once_with(self.model.id, 8192)
            self.assertEqual(first.id, "variant")
            self.assertEqual(first.configured_context, 8192)
            self.engine.release(PreparedModel(self.model, "variant", 8192))
            OllamaEngine().release(first)
            remove.assert_not_called()
            self.engine.release(first)
            self.engine.release(first)
            remove.assert_not_called()
            self.engine.release(second)
            self.engine.release(second)
            remove.assert_called_once_with("variant")
            unload.assert_not_called()
            third = self.engine.prepare_model(self.model, 8192)
            self.assertEqual(create.call_count, 2)
            self.engine.release(third)

    def test_native_variant_creation_preserves_modelfile_and_tag(self):
        observed = []
        def capture(args, **kwargs):
            observed.append((args[:4], Path(args[4]).read_text(), kwargs))
        with patch("devbits_bench.engines.ollama.subprocess.run", side_effect=capture):
            prepared = self.engine.prepare_model(self.model, 8192)
        self.assertEqual(prepared.id, "devbits-bench-org-model-q4_k_m-8192")
        self.assertEqual(observed, [(
            ["ollama", "create", prepared.id, "-f"],
            "FROM org/Model:Q4_K_M\nPARAMETER num_ctx 8192\n",
            {"text": True, "capture_output": True, "check": True},
        )])
        with patch("devbits_bench.engines.ollama.subprocess.run") as run:
            self.engine.release(prepared)
        run.assert_called_once_with(["ollama", "rm", prepared.id], text=True, capture_output=True, check=False)

    def test_failed_preparation_does_not_register_a_variant(self):
        with patch.object(self.engine, "_create_context_variant", side_effect=subprocess.CalledProcessError(1, "ollama")), \
             patch.object(self.engine, "_remove_context_variant") as remove:
            with self.assertRaises(subprocess.CalledProcessError):
                self.engine.prepare_model(self.model, 8192)
            self.engine.release(PreparedModel(self.model, "variant", 8192))
            remove.assert_not_called()
        self.assertEqual(self.engine._variants, {})

    def test_failed_cleanup_can_be_retried_without_losing_ownership(self):
        with patch.object(self.engine, "_create_context_variant", return_value="variant"):
            prepared = self.engine.prepare_model(self.model, 8192)
        with patch.object(self.engine, "_remove_context_variant", side_effect=[OSError("CLI missing"), None]) as remove:
            with self.assertRaisesRegex(OSError, "CLI missing"):
                self.engine.release(prepared)
            self.engine.release(prepared)
            self.assertEqual(remove.call_args_list, [call("variant"), call("variant")])
        self.assertEqual(self.engine._variants, {})

    def test_cold_reset_unloads_every_running_model_before_generation(self):
        with patch.object(self.engine, "_api", side_effect=[
            {"models": [{"name": "first"}, {"model": "second"}, {}]}, {}, {}, {},
        ]) as api:
            self.engine.reset(self.prepared)
            self.engine.generate(self.prepared, GenerationRequest("p", 256, "false", stream=False))
        self.assertEqual(api.call_args_list[:3], [
            call("GET", "/api/ps"),
            call("POST", "/api/generate", {"model": "first", "keep_alive": 0}),
            call("POST", "/api/generate", {"model": "second", "keep_alive": 0}),
        ])
        self.assertEqual(api.call_args_list[3].args[2]["prompt"], "p")

    def test_reset_preserves_best_effort_unload_failures(self):
        with patch.object(self.engine, "_api", side_effect=OSError("ps failed")):
            self.engine.reset(self.prepared)
        with patch.object(self.engine, "_api", side_effect=[
            {"models": [{"name": "first"}, {"name": "second"}]}, OSError("unload failed"), {},
        ]) as api:
            self.engine.reset(self.prepared)
        self.assertEqual(api.call_count, 3)

    def test_nonstream_payload_preserves_native_reasoning_and_caps(self):
        for reasoning, native in [("false", False), ("False", "False"), ("low", "low"), ("true", "true")]:
            for cap in (64, 256, 512):
                with self.subTest(reasoning=reasoning, cap=cap), patch.object(self.engine, "_api", return_value={}) as api:
                    result = self.engine.generate(self.prepared, GenerationRequest("prompt", cap, reasoning, stream=False))
                    api.assert_called_once_with("POST", "/api/generate", {
                        "model": self.model.id, "prompt": "prompt", "think": native, "keep_alive": "5m",
                        "options": {"seed": 42, "temperature": 0, "num_predict": cap}, "stream": False,
                    })
                    self.assertIsNone(result.metrics.client_total_s)
                    self.assertEqual(result.requested_reasoning, reasoning)
                    self.assertIsNone(result.effective_reasoning)

    def test_native_metrics_normalize_exactly_and_preserve_source(self):
        native = {
            "prompt_eval_count": 100, "prompt_eval_cached_count": 20, "eval_count": 12,
            "load_duration": 125_000_000, "prompt_eval_duration": 2_000_000_000,
            "eval_duration": 3_000_000_000, "total_duration": 5_125_000_000,
            "thinking": "thought", "response": "answer", "extra_native_field": {"x": 42},
        }
        with patch.object(self.engine, "_api", return_value=native):
            result = self.engine.generate(self.prepared, GenerationRequest("p", 256, "medium", stream=False))
        metrics = result.metrics
        self.assertEqual((metrics.prompt_tokens, metrics.cached_prompt_tokens, metrics.output_tokens), (100, 20, 12))
        self.assertEqual((metrics.load_s, metrics.prompt_s, metrics.total_s), (0.125, 2, 5.125))
        self.assertEqual((metrics.prompt_tps, metrics.generation_tps), (40, 4))
        self.assertEqual(metrics.provenance["total_s"], "ollama:total_duration")
        self.assertIn("derived:", metrics.provenance["prompt_tps"])
        self.assertIs(result.native_metrics, native)
        self.assertEqual((result.thinking_text, result.answer_text), ("thought", "answer"))

    def test_missing_metrics_and_cache_overflow_keep_historical_zero_math(self):
        for native in ({}, {"prompt_eval_count": 10, "prompt_eval_cached_count": 20, "prompt_eval_duration": 1_000_000_000}):
            with self.subTest(native=native), patch.object(self.engine, "_api", return_value=native):
                metrics = self.engine.generate(self.prepared, GenerationRequest("p", 256, "false", stream=False)).metrics
                self.assertEqual(metrics.prompt_tps, 0)
                self.assertEqual(metrics.generation_tps, 0)
                self.assertEqual(metrics.output_tokens, 0)
                self.assertEqual(metrics.load_s, 0)
                self.assertEqual(metrics.total_s, 0)
                self.assertIsNone(metrics.ttft_s)
                self.assertIsNone(metrics.answer_ttft_s)
                self.assertIn("legacy default: 0", metrics.provenance["total_s"])

    def test_nonstream_errors_propagate_to_runner(self):
        with patch.object(self.engine, "_api", side_effect=OSError("generation failed")):
            with self.assertRaisesRegex(OSError, "generation failed"):
                self.engine.generate(self.prepared, GenerationRequest("p", 256, "false", stream=False))


if __name__ == "__main__":
    unittest.main()
