"""Exercise the real CLI and runners without HTTP, subprocesses or inference."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from devbits_bench import legacy
from devbits_bench.engines.base import GenerationMetrics, GenerationResult, ModelInfo, PreparedModel
from devbits_bench.ui import terminal


HARDWARE = dict(machine='Test machine', chip='Test chip', memory='32 GB', gpu='Test GPU', os='Test OS')
MEMORY = dict(platform='test', available_percent=80, free_percent=80, swap_used_mb=0, compressed_mb=0)


class FakeEngine:
    name = 'test-runtime'

    def __init__(self, context=16384):
        self.info = ModelInfo(self.name, 'test-model', 'Test model', quantization='Q4_K_M',
                              advertised_context=32768, configured_context=context)
        self.prepared = []
        self.released = []
        self.requests = []
        self.resets = []
        self.interrupt = False
        self.fail_prepare_at = None

    def available(self):
        return True

    def version(self):
        return 'test-version'

    def list_models(self):
        return [self.info]

    def inspect_model(self, model_id):
        assert model_id == self.info.id
        return self.info

    def prepare_model(self, model, context):
        if len(self.prepared) == self.fail_prepare_at:
            raise RuntimeError('preparation failed')
        handle = PreparedModel(model, f'opaque-handle-{len(self.prepared)}', context)
        self.prepared.append(handle)
        return handle

    def release(self, prepared):
        self.released.append(prepared)

    def reset(self, prepared):
        self.resets.append(prepared)

    def generate(self, prepared, request, on_event=None):
        if self.interrupt:
            raise KeyboardInterrupt
        self.requests.append((prepared, request))
        if on_event:
            for event in ('request_sent', 'generation_started', 'answer_started'):
                on_event(event)
        return GenerationResult(
            GenerationMetrics(prompt_tokens=1000, cached_prompt_tokens=0, output_tokens=32,
                              load_s=.1, prompt_s=1.0, total_s=3.0, prompt_tps=1000,
                              generation_tps=16, ttft_s=1.1, answer_ttft_s=1.2, client_total_s=3.2),
            answer_text='ORCHID-7291 COBALT-4812 LANTERN-5538 HARBOR-1904',
        )


class CliTests(unittest.TestCase):
    def invoke(self, engine, args, inputs=()):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(legacy, 'ENGINE', engine))
            stack.enter_context(patch.object(legacy, 'hardware', return_value=HARDWARE))
            stack.enter_context(patch.object(legacy, 'Path', side_effect=lambda name: Path(directory) / name))
            stack.enter_context(patch('sys.argv', ['devbits-bench', *args, '--no-ansi']))
            stack.enter_context(patch('builtins.input', side_effect=inputs))
            stack.enter_context(patch('devbits_bench.benchmark.common.time.sleep'))
            stack.enter_context(patch.object(terminal, 'NO_ANSI', True))
            stack.enter_context(patch.object(terminal, 'VERBOSE', False))
            for module in ('quick', 'context', 'common'):
                stack.enter_context(patch(f'devbits_bench.benchmark.{module}.memory_snapshot', return_value=MEMORY.copy()))
            legacy.main()
            paths = list(Path(directory).glob('*.json'))
            self.assertEqual(len(paths), 1)
            self.assertEqual(len(list(Path(directory).glob('*.md'))), 1)
            return json.loads(paths[0].read_text())

    def test_all_modes_run_through_neutral_contract_and_write_compatibility_reports(self):
        for mode, rows, resets in [('quick', 5, 1), ('practical', 2, 2), ('stress', 13, 9), ('custom', 2, 2)]:
            with self.subTest(mode=mode):
                engine = FakeEngine()
                args = ['--mode', mode, '--models', 'test-model', '--thinking', 'false', '--yes']
                if mode == 'stress':
                    args += ['--contexts', '16k']
                if mode == 'custom':
                    args += ['--contexts', '8k,16k']
                doc = self.invoke(engine, args)
                self.assertEqual(doc['test-runtime'], 'test-version')
                self.assertNotIn('engine', doc)
                self.assertEqual(len(doc['results']), rows)
                self.assertEqual(len(engine.resets), resets)
                self.assertEqual(engine.released, list(reversed(engine.prepared)))
                self.assertTrue(all(r['model'].startswith('opaque-handle-') for r in doc['results']))
                self.assertTrue(all(r['base_model'] == 'test-model' for r in doc['results']))
                self.assertTrue(all('native_metrics' not in r for r in doc['results']))
                if mode == 'practical':
                    self.assertEqual(doc['protocol'], 'devbits-practical-v1')
                    self.assertEqual(doc['output_budget_tokens'], 512)
                    self.assertFalse(doc['context_integrity_check'])
                else:
                    self.assertEqual(doc['protocol'], 'devbits-bench-v1')
                if mode == 'custom':
                    self.assertEqual([r['requested_fill'] for r in doc['results']], [7372, 14745])
                    self.assertEqual([r.max_output_tokens for _, r in engine.requests], [512, 512])

    def test_interactive_model_selection_and_default_reasoning(self):
        engine = FakeEngine()
        doc = self.invoke(engine, ['--mode', 'quick'], ['all', ''])
        self.assertEqual(len(doc['results']), 5)
        self.assertEqual({request.reasoning for _, request in engine.requests}, {'false'})

    def test_quick_unknown_context_prepares_historical_32k_fallback(self):
        engine = FakeEngine(context=None)
        self.invoke(engine, ['--mode', 'quick', '--models', 'test-model', '--thinking', 'false'], ['y'])
        self.assertEqual(engine.prepared[0].configured_context, 32768)
        self.assertEqual(engine.released, engine.prepared)

    def test_yes_accepts_quick_unknown_context_without_prompt(self):
        engine = FakeEngine(context=None)
        self.invoke(engine, ['--mode', 'quick', '--models', 'test-model', '--thinking', 'false', '--yes'])
        self.assertEqual(engine.prepared[0].configured_context, 32768)
        self.assertEqual(engine.released, engine.prepared)

    def test_quick_and_stress_declining_preparation_skip_work(self):
        for mode in ('quick', 'stress'):
            with self.subTest(mode=mode):
                engine = FakeEngine(context=None)
                doc = self.invoke(engine, ['--mode', mode, '--models', 'test-model', '--thinking', 'false', '--contexts', '32k'], ['n'])
                self.assertEqual(doc['results'], [])
                self.assertEqual(engine.prepared, [])

    def test_practical_unknown_configured_context_skips_instead_of_using_advertised(self):
        engine = FakeEngine(context=None)
        doc = self.invoke(engine, ['--mode', 'practical', '--models', 'test-model', '--thinking', 'false'])
        self.assertEqual(doc['results'], [])
        self.assertEqual(engine.prepared, [])

    def test_custom_releases_previous_handles_when_later_preparation_fails(self):
        engine = FakeEngine()
        engine.fail_prepare_at = 1
        with self.assertRaisesRegex(RuntimeError, 'preparation failed'):
            self.invoke(engine, ['--mode', 'custom', '--models', 'test-model', '--thinking', 'false', '--contexts', '8k,16k', '--yes'])
        self.assertEqual(engine.released, engine.prepared)

    def test_every_mode_releases_handles_on_interrupt(self):
        for mode in ('quick', 'practical', 'stress', 'custom'):
            with self.subTest(mode=mode):
                engine = FakeEngine()
                engine.interrupt = True
                with self.assertRaises(KeyboardInterrupt):
                    self.invoke(engine, ['--mode', mode, '--models', 'test-model', '--thinking', 'false', '--contexts', '16k', '--yes'])
                self.assertEqual(engine.released, list(reversed(engine.prepared)))
                self.assertTrue(engine.released)

    def test_engine_display_name_preserves_mlx_lm_branding(self):
        self.assertEqual(legacy.engine_display_name('mlx-lm'), 'MLX-LM')
        self.assertEqual(legacy.engine_display_name('ollama'), 'Ollama')

    def test_recovered_size_parser_retains_released_k_m_and_bare_value_semantics(self):
        self.assertEqual(legacy.parsevals('32,64k,131072,1m'), [32768,65536,131072,1048576])
        with self.assertRaises(SystemExit):
            legacy.parsevals('broken')

    def test_mlx_smoke_uses_production_contract_without_benchmark_methodology(self):
        engine = FakeEngine()
        engine.name = 'mlx-lm'
        engine.info = ModelInfo('mlx-lm', 'test-model', 'Test model', quantization='NVFP4',
                                advertised_context=131072, configured_context=None,
                                reasoning_modes=('false','low','medium','xhigh'))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = io.StringIO()
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(patch.object(legacy, 'ENGINE', engine))
            stack.enter_context(patch.object(legacy, 'hardware', return_value=HARDWARE))
            stack.enter_context(patch.object(legacy, 'Path', side_effect=lambda name: Path(directory) / name))
            stack.enter_context(patch('sys.argv', ['devbits-bench', '--engine', 'mlx-lm', '--mlx-smoke',
                                                   '--models', 'test-model', '--no-ansi']))
            stack.enter_context(patch.object(terminal, 'NO_ANSI', True))
            stack.enter_context(patch.object(terminal, 'VERBOSE', False))
            legacy.main()
            self.assertEqual(len(engine.prepared), 1)
            self.assertEqual(engine.prepared[0].configured_context, 32768)
            self.assertEqual(len(engine.requests), 1)
            request = engine.requests[0][1]
            self.assertEqual(request.prompt, 'Reply with exactly: DEVBITS MLX OK')
            self.assertEqual(request.max_output_tokens, 32)
            self.assertEqual(request.reasoning, 'false')
            self.assertEqual(request.seed, 42)
            self.assertEqual(request.temperature, 0)
            self.assertEqual(engine.resets, [])
            self.assertEqual(engine.released, engine.prepared)
            self.assertEqual(list(Path(directory).glob('*.json')), [])
            self.assertEqual(list(Path(directory).glob('*.md')), [])
            self.assertIn('5B-3A acceptance generation completed.', output.getvalue())

    def test_mlx_controls_uses_one_resident_model_and_control_matrix(self):
        engine = FakeEngine()
        engine.name = 'mlx-lm'
        engine.info = ModelInfo('mlx-lm', 'test-model', 'Test model', quantization='NVFP4',
                                advertised_context=131072, configured_context=None,
                                reasoning_modes=('false','low','medium','xhigh'))
        # Supply the native invariants consumed by the acceptance harness.
        original_generate = engine.generate
        def generate(prepared, request, on_event=None):
            result = original_generate(prepared, request, on_event)
            return GenerationResult(result.metrics, thinking_text=('reason' if request.reasoning != 'false' else ''),
                                    answer_text=('42' if request.reasoning != 'false' else 'DEVBITS CONTROL OK'),
                                    requested_reasoning=request.reasoning, effective_reasoning=request.reasoning,
                                    native_metrics={'fresh_prompt_cache': True, 'finish_reason': 'length' if request.max_output_tokens == 1 else 'stop'})
        engine.generate = generate
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = io.StringIO(); stack.enter_context(redirect_stdout(output))
            stack.enter_context(patch.object(legacy, 'ENGINE', engine))
            stack.enter_context(patch.object(legacy, 'hardware', return_value=HARDWARE))
            stack.enter_context(patch.object(legacy, 'Path', side_effect=lambda name: Path(directory) / name))
            stack.enter_context(patch('sys.argv', ['devbits-bench', '--engine', 'mlx-lm', '--mlx-controls', '--models', 'test-model', '--no-ansi']))
            stack.enter_context(patch.object(terminal, 'NO_ANSI', True)); stack.enter_context(patch.object(terminal, 'VERBOSE', False))
            legacy.main()
        self.assertEqual(len(engine.prepared), 1)
        self.assertEqual(len(engine.requests), 6)
        self.assertEqual([r.reasoning for _,r in engine.requests], ['false','false','false','low','medium','xhigh'])
        self.assertEqual([r.max_output_tokens for _,r in engine.requests], [32,32,1,96,96,128])
        self.assertTrue(all(r.seed == 42 and r.temperature == 0 for _,r in engine.requests))
        self.assertEqual(engine.released, engine.prepared)
        self.assertIn('Deterministic replay   True', output.getvalue())
        self.assertIn('5B-3B generation controls completed.', output.getvalue())

    def test_mlx_smoke_rejects_non_mlx_engine(self):
        engine = FakeEngine()
        with patch.object(legacy, 'ENGINE', engine), \
             patch.object(legacy, 'hardware', return_value=HARDWARE), \
             patch('sys.argv', ['devbits-bench', '--mlx-smoke', '--models', 'test-model', '--no-ansi']), \
             patch.object(terminal, 'NO_ANSI', True), patch.object(terminal, 'VERBOSE', False):
            with self.assertRaisesRegex(SystemExit, 'requires --engine mlx-lm'):
                legacy.main()

if __name__ == '__main__':
    unittest.main()
