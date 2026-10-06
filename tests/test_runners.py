"""Characterize the preserved protocols through an engine-neutral fake runtime."""
from contextlib import ExitStack, contextmanager, redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from devbits_bench import legacy
from devbits_bench.benchmark import common, context, quick
from devbits_bench.benchmark.models import Result
from devbits_bench.engines.base import (
    GenerationMetrics, GenerationResult, ModelInfo, PreparedModel,
)
from devbits_bench.ui import terminal
from devbits_bench.reporting import json_document


ANSWER = "ORCHID-7291 COBALT-4812 LANTERN-5538 HARBOR-1904"
FIXTURE = Path(__file__).parent / "fixtures" / "pass3_runner_behavior.json"


def prepared(context_size, identifier="fixture-prepared"):
    model = ModelInfo(
        engine="fixture", id="fixture-model", display_name="Fixture model",
        configured_context=context_size,
    )
    return PreparedModel(model, identifier, context_size)


class QuietStatus:
    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, message):
        pass


class FakeEngine:
    """Only the neutral reset/generate methods used by runners are provided.

    In particular, there is no HTTP API, raw response dictionary, context-variant
    method, or Ollama streaming compatibility method for a runner to rely on.
    """
    def __init__(self, outcomes=None, memory_values=None):
        self.trace = []
        self.requests = []
        self.memories = 0
        self.outcomes = outcomes or {}
        self.memory_values = memory_values or {}

    def memory(self):
        self.memories += 1
        self.trace.append(["memory", self.memories])
        snapshot = {
            "platform": "darwin", "available_percent": 80 - self.memories,
            "free_percent": 80 - self.memories,
            "swap_used_mb": 100 + self.memories,
            "compressed_mb": 200 + self.memories,
            "psi_memory_some_avg10": None, "psi_memory_full_avg10": None,
        }
        snapshot.update(self.memory_values)
        return snapshot

    def sleep(self, seconds):
        self.trace.append(["sleep", seconds])

    def reset(self, model):
        self.trace.append(["reset"])

    def generate(self, model, request, on_event=None):
        self.requests.append((model, request))
        n = len(self.requests)
        self.trace.append(["generate", {
            "model": model.id,
            "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
            "prompt_length": len(request.prompt),
            "max_output_tokens": request.max_output_tokens,
            "reasoning": request.reasoning,
            "seed": request.seed,
            "temperature": request.temperature,
            "stream": request.stream,
        }])
        outcome = self.outcomes.get(n, {})
        if isinstance(outcome, Exception):
            raise outcome
        if on_event:
            for event in ("request_sent", "generation_started", "answer_started"):
                on_event(event)
        prompt_tokens = outcome.get("prompt_tokens", 1000 + n * 37)
        cached_tokens = outcome.get("cached_prompt_tokens", 0)
        output_tokens = 32 + n
        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens, cached_prompt_tokens=cached_tokens,
            output_tokens=output_tokens, load_s=0.125, prompt_s=2.0,
            total_s=(3500000000 + n * 100000000) / 1e9,
            prompt_tps=max(0, prompt_tokens - cached_tokens) / 2,
            generation_tps=output_tokens / 1.0,
            ttft_s=outcome.get("ttft_s", n / 10),
            answer_ttft_s=outcome.get("answer_ttft_s", (n + 1) / 10),
            client_total_s=4 + n / 10,
            provenance={"tokens": "fixture", "timing": "fixture"},
        )
        return GenerationResult(
            metrics=metrics,
            thinking_text=outcome.get("thinking_text", "thoughts"),
            answer_text=outcome.get("answer_text", ANSWER),
            requested_reasoning=request.reasoning,
        )


@contextmanager
def observed_run(engine, responses=None):
    with ExitStack() as stack:
        for module in (common, context, quick):
            if hasattr(module, "memory_snapshot"):
                stack.enter_context(patch.object(module, "memory_snapshot", engine.memory))
            if hasattr(module, "Spinner"):
                stack.enter_context(patch.object(module, "Spinner", QuietStatus))
        stack.enter_context(patch.object(terminal, "Spinner", QuietStatus))
        stack.enter_context(patch("time.sleep", engine.sleep))
        reader = stack.enter_context(patch(
            "builtins.input",
            side_effect=responses if responses is not None else AssertionError("Unexpected prompt"),
        ))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield reader


class RunnerCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(FIXTURE.read_text())

    def assert_baseline(self, suite, engine, results):
        self.assertEqual([asdict(result) for result in results], self.baseline[suite]["results"])
        self.assertEqual(engine.trace, self.baseline[suite]["trace"])

    def test_quick_matches_pass3_results_requests_and_lifecycle(self):
        engine = FakeEngine()
        with observed_run(engine):
            results = quick.quick_bench(engine, prepared(32768), "medium")
        self.assert_baseline("quick", engine, results)

    def test_practical_matches_pass3_results_requests_and_lifecycle(self):
        engine = FakeEngine()
        with observed_run(engine):
            results = context.practical_bench(engine, prepared(16385), "medium")
        self.assert_baseline("practical", engine, results)

    def test_stress_matches_pass3_results_requests_and_lifecycle(self):
        engine = FakeEngine()
        with observed_run(engine):
            results = context.full_bench(engine, prepared(8192), "medium")
        self.assert_baseline("stress", engine, results)

    def test_custom_matches_pass3_results_requests_and_lifecycle(self):
        engine = FakeEngine()
        workloads = [(prepared(8192), 4096), (prepared(16384, "fixture-prepared-2"), 8192)]
        with observed_run(engine):
            results = context.custom_bench(engine, "fixture-model", workloads, "medium")
        self.assert_baseline("custom", engine, results)

    def test_quick_cache_limit_only_rejects_measured_trials_above_five_percent(self):
        engine = FakeEngine({
            1: {"prompt_tokens": 2000, "cached_prompt_tokens": 200},
            2: {"prompt_tokens": 2000, "cached_prompt_tokens": 200},
            3: {"prompt_tokens": 2000, "cached_prompt_tokens": 100},
            4: {"prompt_tokens": 2000, "cached_prompt_tokens": 101},
        })
        with observed_run(engine):
            results = quick.quick_bench(engine, prepared(32768), "false")
        self.assertEqual([bool(result.error) for result in results], [False, False, False, True, False])
        self.assertEqual(results[3].error, "prompt cache contamination: 5.05%")
        self.assertEqual(results[1].measured, False)
        self.assertEqual(results[2].prompt_tps, 950)

    def test_isolated_cache_limit_preserves_practical_and_stress_error_text(self):
        for cached, rejected in ((100, False), (101, True)):
            for suite in ("practical", "stress"):
                with self.subTest(cached=cached, suite=suite):
                    engine = FakeEngine({1: {"prompt_tokens": 2000, "cached_prompt_tokens": cached}})
                    with observed_run(engine):
                        if suite == "practical":
                            result, _ = context.practical_single(engine, prepared(8192), "false", 1024, 1)
                        else:
                            result, _ = context.full_single(
                                engine, prepared(8192), "medium", 1024, "Long Prefill", 101, 96,
                            )
                    self.assertEqual(bool(result.error), rejected)
                    if rejected:
                        expected = ("prompt cache contamination: 101/2000 tokens (5.05%) exceeds 5% limit"
                                    if suite == "practical" else "cache 5.1% exceeds 5%")
                        self.assertEqual(result.error, expected)

    def test_integrity_uses_visible_answer_and_preserves_presence_only_check(self):
        for answer, hits in (("", 0), ("HARBOR-1904 LANTERN-5538 COBALT-4812 ORCHID-7291", 4)):
            with self.subTest(answer=answer):
                engine = FakeEngine({1: {"thinking_text": ANSWER, "answer_text": answer}})
                with observed_run(engine):
                    result, _ = context.full_single(
                        engine, prepared(8192), "medium", 1024, "Long Prefill", 101, 96,
                    )
                self.assertEqual(result.checkpoint_hits, hits)
                self.assertEqual(result.checkpoint_pass, hits == 4)
                self.assertEqual(result.benchmark_thinking, "medium")
                self.assertEqual(result.thinking, "false")
                self.assertEqual(engine.requests[0][1].max_output_tokens, 64)

    def test_practical_stage_equal_to_configured_context_is_skipped(self):
        engine = FakeEngine()
        with observed_run(engine):
            results = context.practical_bench(engine, prepared(8192), "false")
        self.assertEqual([result.requested_fill for result in results], [4096])

    def test_missing_stream_timing_remains_optional_in_quick(self):
        engine = FakeEngine({n: {"ttft_s": None, "answer_ttft_s": None} for n in range(1, 6)})
        with observed_run(engine):
            results = quick.quick_bench(engine, prepared(32768), "false")
        self.assertEqual(len(results), 5)
        self.assertTrue(all(result.ttft_s is None and result.answer_ttft_s is None for result in results))
        self.assertTrue(all(not result.error for result in results))

    def test_quick_error_omits_failed_row_but_continues_schedule(self):
        engine = FakeEngine({2: RuntimeError("fixture generation failed")})
        with observed_run(engine):
            results = quick.quick_bench(engine, prepared(32768), "false")
        self.assertEqual(len(engine.requests), 5)
        self.assertEqual([result.phase for result in results], ["Cold", "Measured", "Measured", "Measured"])

    def test_practical_generation_error_stops_remaining_stages(self):
        engine = FakeEngine({2: RuntimeError("fixture generation failed")})
        with observed_run(engine):
            results = context.practical_bench(engine, prepared(65536), "false")
        self.assertEqual(len(engine.requests), 2)
        self.assertEqual([result.requested_fill for result in results], [4096])

    def test_stress_decode_error_preserves_completed_integrity_result(self):
        engine = FakeEngine({7: RuntimeError("fixture generation failed")})
        with observed_run(engine):
            results = context.full_bench(engine, prepared(8192), "medium")
        self.assertEqual(len(engine.requests), 7)
        self.assertEqual(len(results), 6)
        self.assertEqual(results[-1].phase, "Long Prefill")

    def test_custom_generation_error_continues_next_workload(self):
        engine = FakeEngine({1: RuntimeError("fixture generation failed")})
        workloads = [(prepared(8192), 4096), (prepared(16384), 8192)]
        with observed_run(engine):
            results = context.custom_bench(engine, "fixture-model", workloads, "medium")
        self.assertEqual(len(engine.requests), 2)
        self.assertEqual([(result.phase, result.run) for result in results], [("Custom", 2)])

    def test_practical_memory_pressure_decline_preserves_completed_result(self):
        engine = FakeEngine(memory_values={"available_percent": 5})
        with observed_run(engine, ["n"]) as reader:
            results = context.practical_bench(engine, prepared(65536), "false")
        self.assertEqual(len(results), 1)
        self.assertEqual(len(engine.requests), 1)
        reader.assert_called_once()

    def test_stress_memory_pressure_decline_preserves_completed_stage(self):
        engine = FakeEngine(memory_values={"available_percent": 5})
        with observed_run(engine, ["n"]) as reader:
            results = context.full_bench(engine, prepared(8192), "medium")
        self.assertEqual(len(results), 7)
        self.assertEqual([result.phase for result in results[-2:]], ["Long Prefill", "Long Decode"])
        reader.assert_called_once()

    def test_custom_memory_pressure_is_advisory_without_prompt(self):
        engine = FakeEngine(memory_values={"available_percent": 5})
        workloads = [(prepared(8192), 4096), (prepared(16384), 8192)]
        with observed_run(engine) as reader:
            results = context.custom_bench(engine, "fixture-model", workloads, "medium")
        self.assertEqual(len(results), 2)
        reader.assert_not_called()


class PressurePolicyCompatibilityTests(unittest.TestCase):
    def test_macos_pressure_thresholds_are_inclusive(self):
        before = {"platform": "darwin", "swap_used_mb": 100, "compressed_mb": 200}
        after = {"available_percent": 8, "swap_used_mb": 612, "compressed_mb": 1736}
        self.assertEqual(common.pressure_warning(before, after), [
            "macOS available memory fell to 8%",
            "macOS swap increased by 512 MB",
            "macOS compressed memory increased by 1536 MB",
        ])
        after = {"available_percent": 8.1, "swap_used_mb": 611.9, "compressed_mb": 1735.9}
        self.assertEqual(common.pressure_warning(before, after), [])

    def test_linux_pressure_thresholds_are_inclusive(self):
        before = {"platform": "linux", "swap_used_mb": 100}
        after = {
            "available_percent": 8, "swap_used_mb": 612,
            "psi_memory_full_avg10": 1.0, "psi_memory_some_avg10": 10.0,
        }
        self.assertEqual(common.pressure_warning(before, after), [
            "Linux memory available fell to 8.0%",
            "Linux swap increased by 512 MB",
            "Linux severe memory stalls reached 1.00% over the recent 10-second window",
            "Linux memory stalls reached 10.00% over the recent 10-second window",
        ])
        after = {
            "available_percent": 8.1, "swap_used_mb": 611.9,
            "psi_memory_full_avg10": 0.99, "psi_memory_some_avg10": 9.99,
        }
        self.assertEqual(common.pressure_warning(before, after), [])

    def test_unknown_memory_observations_do_not_invent_pressure(self):
        self.assertEqual(common.pressure_warning({}, {}), [])


class ReportCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runners = json.loads(FIXTURE.read_text())
        cls.reports = json.loads((FIXTURE.parent / "pass3_report_behavior.json").read_text())

    def test_complete_json_documents_match_pass3_for_all_suites(self):
        for suite, expected in self.reports.items():
            with self.subTest(suite=suite):
                results = [Result(**row) for row in self.runners[suite]["results"]]
                actual = json_document(results=results, **expected["metadata"])
                self.assertEqual(actual, expected["json"])
                # Preserve serialization order/number representation, too.
                self.assertEqual(json.dumps(actual, indent=2), json.dumps(expected["json"], indent=2))

    def test_markdown_matches_pass3_for_all_suites(self):
        class MetadataEngine:
            def inspect_model(self, model_id):
                return ModelInfo("fixture", model_id, model_id, quantization="Q4_K_M")

        with patch.object(legacy, "_engine", return_value=MetadataEngine()):
            for suite, expected in self.reports.items():
                with self.subTest(suite=suite):
                    results = [Result(**row) for row in self.runners[suite]["results"]]
                    renderer = legacy.report if suite == "custom" else legacy.quick_report
                    actual = renderer(expected["metadata"]["system"], "fixture-version", results, "Ollama")
                    self.assertEqual(actual, expected["markdown"])


class MultiEnginePresentationTests(unittest.TestCase):
    def test_nvfp4_guide_is_engine_neutral(self):
        row = Result(base_model='mlx-community/model', model='model', context=32768, mode='Measured', run=1,
                     thinking='false', requested_fill=0, prompt_tokens=10, cached_prompt_tokens=0,
                     output_tokens=5, load_s=0, prompt_s=1, total_s=2, prompt_tps=10,
                     generation_tps=5, memory_before={}, memory_after={}, phase='Measured')
        class MetadataEngine:
            def inspect_model(self, model_id):
                return ModelInfo('mlx-lm', model_id, model_id, quantization='NVFP4')
        with patch.object(legacy, '_engine', return_value=MetadataEngine()):
            text = '\n'.join(legacy.markdown_model_terms([row]))
        self.assertIn('4-bit floating-point quantization', text)
        self.assertNotIn('Ollama', text)


    def test_quick_progress_block_keeps_historical_compact_geometry(self):
        out = io.StringIO()
        with redirect_stdout(out):
            initial = terminal.quick_progress_block([], 0, 5)
        initial_lines = out.getvalue().splitlines()
        out = io.StringIO()
        with redirect_stdout(out):
            final = terminal.quick_progress_block([f"stage {i}" for i in range(5)], 5, 5)
        final_lines = out.getvalue().splitlines()
        self.assertEqual(initial, 2)
        self.assertEqual(final, 7)
        self.assertEqual(len(initial_lines), 2)
        self.assertEqual(len(final_lines), 7)
        self.assertIn('0/5', initial_lines[0])
        self.assertIn('5/5', final_lines[0])

    def test_quick_reuses_historical_clear_and_repaint_ownership(self):
        engine = FakeEngine()
        clears=[]
        real_clear=quick.clear_terminal_lines
        with observed_run(engine), patch.object(quick, 'clear_terminal_lines', side_effect=lambda n: clears.append(n)):
            with redirect_stdout(io.StringIO()):
                quick.quick_bench(engine, prepared(32768), 'false')
        # One repaint after every completed stage. Geometry grows with completed
        # rows exactly like the original Ollama Quick renderer: 2,3,4,5,6.
        self.assertEqual(clears, [2, 3, 4, 5, 6])

    def test_mlx_quick_report_methodology_is_engine_neutral(self):
        row = Result(base_model='mlx-community/model', model='model', context=32768, mode='Measured', run=1,
                     thinking='false', requested_fill=0, prompt_tokens=10, cached_prompt_tokens=0,
                     output_tokens=5, load_s=0, prompt_s=1, total_s=2, prompt_tps=10,
                     generation_tps=5, memory_before={}, memory_after={}, phase='Measured', measured=True,
                     ttft_s=1.0, answer_ttft_s=1.0)
        with patch.object(legacy, 'markdown_model_terms', return_value=[]):
            text = legacy.quick_report({'machine':'m','chip':'c','memory':'1 GB','gpu':'g','os':'o'},
                                       'mlx-lm 0.32.0 • mlx 0.32.3 • Metal', [row], 'MLX-LM')
        self.assertIn('MLX-LM', text)
        self.assertNotIn('Ollama prompt evaluation duration', text)
        self.assertNotIn('Ollama output-token count', text)
        self.assertIn('selected engine adapter', text)

    def test_quick_verbose_emits_cold_warm_evidence(self):
        engine = FakeEngine()
        out = io.StringIO()
        with observed_run(engine), patch.object(terminal, 'VERBOSE', True), redirect_stdout(out):
            quick.quick_bench(engine, prepared(32768), 'false')
        text = out.getvalue()
        self.assertIn('Cold baseline evidence', text)
        self.assertIn('Warmup evidence', text)
        self.assertIn('Measured 3/3 evidence', text)
        self.assertIn('cached_prompt_tokens=', text)
        self.assertIn('fresh_prompt_cache=', text)


if __name__ == "__main__":
    unittest.main()

class MLXReportProvenancePresentationTests(unittest.TestCase):
    def test_mlx_markdown_exposes_native_vs_devbits_provenance(self):
        row = Result(base_model='mlx-community/model', model='model', context=32768, mode='Measured', run=1,
                     thinking='medium', requested_fill=0, prompt_tokens=10, cached_prompt_tokens=0,
                     output_tokens=5, load_s=0, prompt_s=1, total_s=2, prompt_tps=10,
                     generation_tps=5, memory_before={}, memory_after={}, ttft_s=.5,
                     answer_ttft_s=1.0, client_total_s=2.0, phase='Measured')
        row._measurement_provenance={'prompt_tps':'mlx-lm native','generation_tps':'mlx-lm native',
                                     'ttft_s':'devbits client monotonic','answer_ttft_s':'devbits client monotonic',
                                     'prompt_s':'derived: native prompt_tokens / native prompt_tps'}
        row._native_metrics={'peak_memory_gb':15.4,'finish_reason':'stop','fresh_prompt_cache':True}
        row._requested_reasoning='medium'; row._effective_reasoning='medium'
        class MetadataEngine:
            def inspect_model(self, model_id):
                return ModelInfo('mlx-lm',model_id,model_id,quantization='NVFP4')
        with patch.object(legacy,'_engine',return_value=MetadataEngine()):
            text=legacy.quick_report({'machine':'m','chip':'c','memory':'1 GB','gpu':'g','os':'o'},'mlx-lm 0.32.0 • mlx 0.32.3 • Metal',[row],'MLX-LM')
        self.assertIn('## Measurement provenance',text)
        self.assertIn('mlx-lm native',text)
        self.assertIn('devbits client monotonic',text)
        self.assertIn('15.400 GB',text)
        self.assertIn('fresh prompt cache confirmed',text)
        self.assertIn('requested medium; effective medium',text)
