import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from devbits_bench.engines.mlx_lm import MLXWorkerClient, MLXWorkerUnavailableError
from devbits_bench.engines.mlx_worker_protocol import WorkerProtocolError


class MLXWorkerTests(unittest.TestCase):
    def test_handshake_and_shutdown_with_current_python(self):
        client = MLXWorkerClient(sys.executable)
        info = client.start()
        self.assertEqual(info.python, sys.executable)
        self.assertIn("handshake", info.capabilities)
        self.assertIn("shutdown", info.capabilities)
        self.assertTrue(client.running)
        client.shutdown()
        self.assertFalse(client.running)

    @patch("devbits_bench.engines.mlx_lm.discover_runtime")
    def test_discovered_runtime_python_is_used(self, discover):
        from devbits_bench.runtimes.mlx import MLXRuntime
        discover.return_value = MLXRuntime(Path(sys.executable), None, False, "0.32.3", "0.32.0")
        client = MLXWorkerClient()
        self.assertEqual(client.python_executable, sys.executable)

    def test_missing_interpreter_is_actionable(self):
        client = MLXWorkerClient("/definitely/not/a/python")
        with self.assertRaisesRegex(MLXWorkerUnavailableError, "interpreter not found"):
            client.start()

    def test_worker_rejects_wrong_protocol_version(self):
        proc = subprocess.Popen(
            [sys.executable, "-m", "devbits_bench.workers.mlx_lm_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write('{"v":999,"id":"x","op":"hello"}\n')
        proc.stdin.flush()
        line = proc.stdout.readline()
        self.assertIn('"event":"error"', line)
        proc.terminate()
        proc.wait(timeout=2)
        proc.stdin.close()
        proc.stdout.close()

    def test_context_manager_terminates_on_exception(self):
        client = MLXWorkerClient(sys.executable)
        with self.assertRaises(RuntimeError):
            with client:
                self.assertTrue(client.running)
                raise RuntimeError("boom")
        self.assertFalse(client.running)



class MLXPracticalLifecycleTests(unittest.TestCase):
    @patch('devbits_bench.benchmark.common.time.sleep')
    @patch('devbits_bench.benchmark.common.memory_snapshot', return_value={'platform':'darwin'})
    def test_long_test_stabilization_keeps_resident_mlx_model(self, snapshot, sleep):
        from unittest.mock import MagicMock
        from devbits_bench.benchmark.common import stabilize_for_long_test
        engine=MagicMock()
        engine.isolated_generation_keeps_residency.return_value=True
        prepared=object()
        result=stabilize_for_long_test(engine,prepared,3.0)
        engine.reset.assert_not_called()
        sleep.assert_called_once_with(3.0)
        self.assertEqual(result,{'platform':'darwin'})

    @patch('devbits_bench.benchmark.common.time.sleep')
    @patch('devbits_bench.benchmark.common.memory_snapshot', return_value={'platform':'darwin'})
    def test_long_test_stabilization_preserves_historical_reset_for_other_engines(self, snapshot, sleep):
        from unittest.mock import MagicMock
        from devbits_bench.benchmark.common import stabilize_for_long_test
        engine=MagicMock(spec=['reset'])
        prepared=object()
        stabilize_for_long_test(engine,prepared,3.0)
        engine.reset.assert_called_once_with(prepared)

    @patch('devbits_bench.engines.mlx_lm.MLXWorkerClient')
    def test_mlx_advertises_isolated_generation_with_residency(self, worker_cls):
        from devbits_bench.engines.mlx_lm import MLXLMEngine
        self.assertTrue(MLXLMEngine(sys.executable).isolated_generation_keeps_residency())


class PracticalPresentationTests(unittest.TestCase):
    def test_practical_markdown_has_practical_identity_and_no_quick_methodology(self):
        from devbits_bench.legacy import practical_report, demo_result
        r=demo_result('m',33792,'Practical',1,3800,20.0,200.0,20.0,target=4096)
        text=practical_report(
            {'machine':'M','chip':'C','memory':'36 GB','gpu':'G','os':'macOS'},
            'mlx-lm 0.32.0 • mlx 0.32.3 • Metal',[r],'MLX-LM')
        self.assertIn('# Devbits Bench — Practical Results',text)
        self.assertIn('`devbits-practical-v1`',text)
        self.assertIn('512-token output cap',text)
        self.assertNotIn('One warmup is discarded',text)
        self.assertNotIn('Quick summary',text)


class StressDemoAndReportTests(unittest.TestCase):
    def test_stress_report_has_stress_identity_and_methodology(self):
        from devbits_bench.legacy import stress_report, demo_result
        pre=demo_result('m',131072,'Long Prefill',1,31768,20,180,22,target=32768,hits=4,total=4,passed=True)
        dec=demo_result('m',131072,'Long Decode',1,31768,19,180,20,target=32768)
        text=stress_report({'machine':'M','chip':'C','memory':'36 GB','gpu':'G','os':'macOS'},
                           'mlx-lm 0.32.0 • mlx 0.32.3 • Metal',[pre,dec],'MLX-LM')
        self.assertIn('# Devbits Bench — Stress Results',text)
        self.assertIn('25%, 50%, 75% and 90%',text)
        self.assertIn('integrity/prefill interaction',text)
        self.assertNotIn('Quick summary',text)
        self.assertNotIn('One warmup is discarded',text)

    def test_demo_help_is_engine_neutral(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/legacy.py').read_text()
        self.assertIn('No model inference requests will be made.',source)
        self.assertNotIn('No Ollama inference requests will be made.',source)
        self.assertIn("'stress-mlx'",source)


class StressPressurePresentationTests(unittest.TestCase):
    def test_demo_pressure_warning_does_not_commit_completed_rows_to_history(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/legacy.py').read_text()
        start=source.index("  if pressure and i in (2,3):")
        end=source.index(" if rendered: clear_terminal_lines(rendered)",start)
        block=source[start:end]
        self.assertNotIn("for line in completed: print(line)",block)
        self.assertIn("rendered=stress_progress_block(completed,done,4)",block)
        self.assertIn("clear_terminal_lines(rendered)",block)


class MLXRealStressIntegrationTests(unittest.TestCase):
    def test_real_stress_pressure_path_does_not_duplicate_completed_rows(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/benchmark/context.py').read_text()
        start=source.index("  if reasons and si<len(stages):")
        end=source.index("\n if rendered_lines:",start)
        block=source[start:end]
        self.assertNotIn("for line in completed: print(line)",block)
        self.assertIn("rendered_lines=stress_progress_block(completed,completed_count,len(stages))",block)

    def test_real_stress_queues_fresh_cache_evidence(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/benchmark/context.py').read_text()
        self.assertIn("integrity evidence",source)
        self.assertIn("decode evidence",source)
        self.assertIn('fresh_prompt_cache=',source)

    def test_mlx_stress_default_is_bounded_128k_not_implicit_32k(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/legacy.py').read_text()
        self.assertIn("ctx=min(info.advertised_context,128*1024)",source)


class StressInterruptionReportingTests(unittest.TestCase):
    def test_json_supports_explicit_suite_status(self):
        from devbits_bench.reporting.json_report import json_document
        doc=json_document(protocol='p',system={},engine_name='mlx-lm',engine_version='v',results=[],
                          suite='stress',suite_status={'complete':False,'interrupted':True,'completed_stages':2,'total_stages':4})
        self.assertEqual(doc['suite_status']['completed_stages'],2)
        self.assertTrue(doc['suite_status']['interrupted'])

    def test_stress_markdown_labels_interrupted_suite(self):
        from devbits_bench.legacy import stress_report
        text=stress_report({'machine':'M','chip':'C','memory':'36 GB','gpu':'G','os':'macOS'},'v',[],'MLX-LM',
                           {'complete':False,'interrupted':True,'completed_stages':2,'total_stages':4,'stop_reason':'keyboard_interrupt'})
        self.assertIn('**State:** interrupted',text)
        self.assertIn('**Completed Stress stages:** 2/4',text)
        self.assertIn('keyboard_interrupt',text)

    def test_full_stress_catches_keyboard_interrupt_atomically(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/benchmark/context.py').read_text()
        self.assertIn('except KeyboardInterrupt:',source)
        self.assertIn('Completed results preserved',source)
        self.assertIn("'keyboard_interrupt'",source)


class MLXCustomIntegrationTests(unittest.TestCase):
    def test_custom_mlx_demo_is_available_and_synthetic(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/legacy.py').read_text()
        self.assertIn("'custom-mlx'",source)
        self.assertIn("Synthetic native MLX-LM Custom presentation fixture",source)
        self.assertIn("mlx-community/Qwen3.8-27B-nvfp4",source)

    def test_custom_rejects_context_beyond_advertised_capacity(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/legacy.py').read_text()
        self.assertIn("exceeds advertised context",source)
        self.assertIn("workload skipped",source)

    def test_custom_queues_native_cache_evidence_outside_live_surface(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/benchmark/context.py').read_text()
        start=source.index('def custom_bench')
        end=source.index('def full_single',start)
        block=source[start:end]
        self.assertIn('fresh_prompt_cache=',block)
        self.assertIn('verbose_evidence',block)
        self.assertIn('clear_terminal_lines(rendered)',block)


class MLXCustomDemoRenderTests(unittest.TestCase):
    def test_custom_mlx_demo_renders_results_without_configured_engine(self):
        import contextlib
        import io
        from devbits_bench import legacy
        previous_speed=legacy.DEMO_SPEED
        previous_engine=legacy.ENGINE
        try:
            legacy.DEMO_SPEED='normal'
            legacy.ENGINE=None
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                legacy.demo_custom_mlx(render_results=True)
            rendered=out.getvalue()
            self.assertIn('Devbits Bench — Custom Results',rendered)
            self.assertIn('adapter-qualified residency/cache semantics apply',rendered)
            self.assertIn('mlx-community/Qwen3.8-27B-nvfp4',rendered)
        finally:
            legacy.DEMO_SPEED=previous_speed
            legacy.ENGINE=previous_engine


class MLXCustomSemanticWordingTests(unittest.TestCase):
    def test_custom_results_context_guide_is_engine_neutral(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/legacy.py').read_text()
        self.assertIn('CTX     Benchmark request context window.',source)
        self.assertNotIn('CTX     Configured maximum context window.',source)

    def test_mlx_custom_does_not_claim_kv_context_reconfiguration(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'src/devbits_bench/legacy.py').read_text()
        self.assertIn("MLX-LM will use {ctx//1024}K as the benchmark request envelope.",source)
        self.assertIn("no engine KV/context limit will be changed.",source)
        self.assertIn("Proceed with this workload? [Y/n]",source)

if __name__ == "__main__":
    unittest.main()

class MLXModelDiscoveryTests(unittest.TestCase):
    def _fake_cache(self, root: Path):
        repo=root/'models--mlx-community--Qwen3.8-27B-nvfp4'
        snap=repo/'snapshots'/'abc';snap.mkdir(parents=True)
        (repo/'refs').mkdir();(repo/'refs'/'main').write_text('abc')
        (snap/'config.json').write_text('{"architectures":["Qwen3_5ForConditionalGeneration"],"max_position_embeddings":262144,"quantization":{"mode":"nvfp4"}}')
        (snap/'tokenizer_config.json').write_text('{"chat_template":"{{ enable_thinking }} {{ reasoning_effort }}","model_max_length":262144}')
        (snap/'model-00001-of-00001.safetensors').write_bytes(b'x'*32)

    def test_worker_discovers_cached_model_without_network(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'hub';root.mkdir();self._fake_cache(root)
            with patch.dict(os.environ,{"HUGGINGFACE_HUB_CACHE":str(root)}):
                client=MLXWorkerClient(sys.executable);client.start()
                try: models=client.list_models()
                finally: client.shutdown()
        self.assertEqual(models[0]['id'],'mlx-community/Qwen3.8-27B-nvfp4')
        self.assertEqual(models[0]['quantization'],'NVFP4')
        self.assertEqual(models[0]['advertised_context'],262144)
        self.assertEqual(models[0]['reasoning_modes'],['false','low','medium','xhigh'])

    def test_worker_filters_non_generative_huggingface_models(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'hub';root.mkdir();self._fake_cache(root)
            repo=root/'models--sentence-transformers--all-MiniLM-L6-v2'; snap=repo/'snapshots'/'abc';snap.mkdir(parents=True)
            (repo/'refs').mkdir();(repo/'refs'/'main').write_text('abc')
            (snap/'config.json').write_text('{"architectures":["BertModel"],"model_type":"bert"}')
            (snap/'model.safetensors').write_bytes(b'x')
            with patch.dict(os.environ,{"HUGGINGFACE_HUB_CACHE":str(root)}):
                client=MLXWorkerClient(sys.executable);client.start()
                try: models=client.list_models()
                finally: client.shutdown()
        self.assertEqual([m['id'] for m in models],['mlx-community/Qwen3.8-27B-nvfp4'])

    def test_engine_model_info_does_not_equate_context_with_kv_bound(self):
        from devbits_bench.engines.mlx_lm import MLXLMEngine
        info=MLXLMEngine._info({'id':'x','advertised_context':262144,'reasoning_modes':['false','medium'],'default_reasoning':'medium'})
        self.assertEqual(info.advertised_context,262144)
        self.assertIsNone(info.configured_context)

class MLXTinyGenerationTests(unittest.TestCase):
    def test_reasoning_mapping_is_model_qualified(self):
        from devbits_bench.workers.mlx_lm_worker import _template_kwargs
        self.assertEqual(_template_kwargs('false'), {'enable_thinking': False})
        self.assertEqual(_template_kwargs('low'), {'enable_thinking': True, 'reasoning_effort': 'low'})
        self.assertEqual(_template_kwargs('medium'), {'enable_thinking': True, 'reasoning_effort': 'medium'})
        self.assertEqual(_template_kwargs('xhigh'), {'enable_thinking': True, 'reasoning_effort': 'xhigh'})
        with self.assertRaisesRegex(ValueError, 'unsupported reasoning mode'):
            _template_kwargs('high')

    @patch('devbits_bench.engines.mlx_lm.MLXWorkerClient')
    def test_engine_keeps_worker_resident_and_maps_native_metrics(self, worker_cls):
        from devbits_bench.engines.base import GenerationRequest, ModelInfo
        from devbits_bench.engines.mlx_lm import MLXLMEngine
        worker=worker_cls.return_value; worker.running=True
        worker.generate.side_effect=lambda **kw: (
            kw['on_event']({'event':'reasoning_delta','text':'think '}),
            kw['on_event']({'event':'answer_delta','text':'answer'}),
            {'event':'complete','effective_reasoning':'medium','metrics':{
                'prompt_tokens':11,'generation_tokens':7,'prompt_tps':123.0,
                'generation_tps':45.0,'peak_memory_gb':12.5,'finish_reason':'stop'}},
        )[-1]
        model=ModelInfo(engine='mlx-lm',id='m',display_name='m',advertised_context=32768,
                        reasoning_modes=('false','medium'),default_reasoning='medium')
        engine=MLXLMEngine(sys.executable); prepared=engine.prepare_model(model,8192)
        result=engine.generate(prepared,GenerationRequest('hello',32,'medium'))
        self.assertIs(engine._worker,worker)
        self.assertEqual(result.thinking_text,'think ');self.assertEqual(result.answer_text,'answer')
        self.assertEqual(result.metrics.prompt_tokens,11);self.assertEqual(result.metrics.output_tokens,7)
        self.assertEqual(result.metrics.prompt_tps,123.0);self.assertEqual(result.metrics.generation_tps,45.0)
        self.assertEqual(result.native_metrics['peak_memory_gb'],12.5)
        self.assertTrue(result.native_metrics['fresh_prompt_cache'])
        worker.generate.assert_called_once()
        self.assertTrue(worker.generate.call_args.kwargs['fresh_cache'])
        engine.release(prepared);worker.release_model.assert_called_once();worker.shutdown.assert_called_once()

    @patch('devbits_bench.engines.mlx_lm.MLXWorkerClient')
    def test_engine_rejects_context_beyond_advertised_capacity(self, worker_cls):
        from devbits_bench.engines.base import ModelInfo
        from devbits_bench.engines.mlx_lm import MLXLMEngine
        model=ModelInfo(engine='mlx-lm',id='m',display_name='m',advertised_context=4096)
        with self.assertRaisesRegex(ValueError,'exceeds advertised'):
            MLXLMEngine(sys.executable).prepare_model(model,8192)
        worker_cls.assert_not_called()

    def test_stream_callback_exception_terminates_worker(self):
        from unittest.mock import MagicMock
        client=MLXWorkerClient(sys.executable)
        client._stream_round_trip=MagicMock(side_effect=KeyboardInterrupt())
        client.terminate=MagicMock()
        with self.assertRaises(KeyboardInterrupt):
            client.generate(prompt='x',max_output_tokens=1,reasoning='false',seed=42,temperature=0)
        client.terminate.assert_called_once()

class MLXReasoningCapabilityQualificationTests(unittest.TestCase):
    def test_runtime_template_qualifies_full_reasoning_effort_ladder(self):
        from devbits_bench.workers.mlx_lm_worker import _runtime_reasoning
        class T:
            has_thinking=True
            chat_template='{{ enable_thinking }} {{ reasoning_effort }}'
        self.assertEqual(_runtime_reasoning(T(), ['false','true'], 'true'),
                         (['false','low','medium','xhigh'],'xhigh'))

    def test_runtime_binary_thinking_does_not_invent_effort_levels(self):
        from devbits_bench.workers.mlx_lm_worker import _runtime_reasoning
        class T:
            has_thinking=True
            chat_template='{{ enable_thinking }}'
        self.assertEqual(_runtime_reasoning(T(), [], 'false'), (['false','true'],'true'))

    @patch('devbits_bench.engines.mlx_lm.MLXWorkerClient')
    def test_engine_accepts_every_advertised_mode_and_rejects_unadvertised(self, worker_cls):
        from devbits_bench.engines.base import GenerationRequest, ModelInfo
        from devbits_bench.engines.mlx_lm import MLXLMEngine
        worker=worker_cls.return_value;worker.running=True
        worker.generate.return_value={'event':'complete','effective_reasoning':'false','metrics':{}}
        model=ModelInfo(engine='mlx-lm',id='m',display_name='m',advertised_context=32768,
                        reasoning_modes=('false','low','medium','xhigh'),default_reasoning='xhigh')
        engine=MLXLMEngine(sys.executable);prepared=engine.prepare_model(model,8192)
        for mode in ('false','low','medium','xhigh'):
            worker.generate.return_value={'event':'complete','effective_reasoning':mode,'metrics':{}}
            engine.generate(prepared,GenerationRequest('x',1,mode))
        with self.assertRaisesRegex(ValueError,'unsupported MLX-LM reasoning mode'):
            engine.generate(prepared,GenerationRequest('x',1,'high'))
        engine.release(prepared)

class MLXQuickLifecycleTests(unittest.TestCase):
    @patch('devbits_bench.engines.mlx_lm.time.monotonic', side_effect=[10.0, 10.0, 12.0, 13.0, 14.0, 15.0])
    @patch('devbits_bench.engines.mlx_lm.MLXWorkerClient')
    def test_reset_unloads_and_next_generation_deferred_reloads(self, worker_cls, monotonic):
        from devbits_bench.engines.base import GenerationRequest, ModelInfo
        from devbits_bench.engines.mlx_lm import MLXLMEngine
        worker=worker_cls.return_value; worker.running=True
        worker.generate.side_effect=lambda **kw: (
            kw['on_event']({'event':'answer_delta','text':'ok'}),
            {'event':'complete','effective_reasoning':'false','metrics':{
                'prompt_tokens':20,'generation_tokens':2,'prompt_tps':10.0,
                'generation_tps':20.0,'peak_memory_gb':1.0,'finish_reason':'stop'}},
        )[-1]
        model=ModelInfo(engine='mlx-lm',id='m',display_name='m',advertised_context=32768,
                        reasoning_modes=('false',),default_reasoning='false')
        engine=MLXLMEngine(sys.executable); prepared=engine.prepare_model(model,8192)
        worker.reset_mock()
        engine.reset(prepared)
        self.assertFalse(engine._resident)
        worker.release_model.assert_called_once()
        result=engine.generate(prepared,GenerationRequest('hello',32,'false'))
        worker.load_model.assert_called_once_with('m')
        self.assertTrue(engine._resident)
        self.assertEqual(result.metrics.cached_prompt_tokens,0)
        self.assertEqual(result.metrics.load_s,2.0)
        self.assertEqual(result.metrics.prompt_s,2.0)
        self.assertEqual(result.metrics.total_s,5.0)
        self.assertEqual(result.metrics.ttft_s,3.0)
        self.assertEqual(result.metrics.answer_ttft_s,4.0)
        self.assertIn('fresh prompt cache',result.metrics.provenance['cached_prompt_tokens'])

    @patch('devbits_bench.engines.mlx_lm.MLXWorkerClient')
    def test_warm_generation_does_not_reload_resident_model(self, worker_cls):
        from devbits_bench.engines.base import GenerationRequest, ModelInfo
        from devbits_bench.engines.mlx_lm import MLXLMEngine
        worker=worker_cls.return_value; worker.running=True
        worker.generate.return_value={'event':'complete','effective_reasoning':'false','metrics':{
            'prompt_tokens':10,'generation_tokens':1,'prompt_tps':10.0,'generation_tps':10.0}}
        model=ModelInfo(engine='mlx-lm',id='m',display_name='m',advertised_context=32768,
                        reasoning_modes=('false',),default_reasoning='false')
        engine=MLXLMEngine(sys.executable); prepared=engine.prepare_model(model,8192)
        worker.reset_mock()
        result=engine.generate(prepared,GenerationRequest('hello',1,'false'))
        worker.load_model.assert_not_called()
        self.assertEqual(result.metrics.load_s,0.0)
        self.assertEqual(result.metrics.cached_prompt_tokens,0)
