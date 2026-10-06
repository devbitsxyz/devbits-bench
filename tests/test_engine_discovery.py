import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from devbits_bench import cli
from devbits_bench.engines.discovery import DiscoveredEngine, EngineCandidate, EngineState, discover_available_engines

class FakeEngine:
    def __init__(self,name,available=True,version='1.0'):
        self.name=name; self._available=available; self._version=version
    def available(self): return self._available
    def version(self): return self._version

class EngineDiscoveryTests(unittest.TestCase):
    def test_available_helper_returns_ready_engines(self):
        ready=FakeEngine('ready')
        items=[DiscoveredEngine(EngineCandidate('ready','Ready',lambda:ready),EngineState.READY,ready,'1.0')]
        with patch('devbits_bench.engines.discovery.discover_engines',return_value=items):
            found=discover_available_engines()
        self.assertEqual([(c.id,e.name) for c,e in found],[('ready','ready')])

    def test_selector_lists_only_detected_ready_engines(self):
        ollama=FakeEngine('ollama',version='0.34.4'); mlx=FakeEngine('mlx-lm',version='0.32.0')
        items=[DiscoveredEngine(EngineCandidate('ollama','Ollama',lambda:ollama),EngineState.READY,ollama,'0.34.4'),
               DiscoveredEngine(EngineCandidate('mlx-lm','MLX-LM',lambda:mlx),EngineState.READY,mlx,'0.32.0')]
        out=io.StringIO()
        with patch.object(cli,'discover_engines',return_value=items), patch('builtins.input',return_value='2'), redirect_stdout(out):
            selected=cli._select_engine(None)
        self.assertIs(selected,mlx); self.assertIn('Ollama',out.getvalue()); self.assertIn('MLX-LM',out.getvalue())
        self.assertNotIn('Setup available',out.getvalue())

    def test_explicit_ready_engine_skips_prompt(self):
        engine=FakeEngine('ollama')
        item=DiscoveredEngine(EngineCandidate('ollama','Ollama',lambda:engine),EngineState.READY,engine,'1.0')
        with patch.object(cli,'discover_engines',return_value=[item]), patch('builtins.input') as prompt:
            self.assertIs(cli._select_engine('ollama'),engine); prompt.assert_not_called()

class EngineReadinessSelectionTests(unittest.TestCase):
    def test_detected_but_not_ready_engine_is_visible_and_actionable(self):
        engine=FakeEngine('mlx-lm')
        item=DiscoveredEngine(EngineCandidate('mlx-lm','MLX-LM',lambda:engine),EngineState.UNAVAILABLE,engine,'MLX worker failed')
        out=io.StringIO()
        with patch.object(cli,'discover_engines',return_value=[item]), patch('builtins.input',return_value='1'), redirect_stdout(out):
            with self.assertRaisesRegex(SystemExit,'MLX worker failed'):
                cli._select_engine(None)
        self.assertIn('Detected • not ready',out.getvalue())

    def test_unexpected_discovery_programming_error_is_not_hidden(self):
        with patch.object(cli,'discover_engines',side_effect=TypeError('bug')):
            with self.assertRaisesRegex(TypeError,'bug'):
                cli._select_engine(None)

class VerboseDiscoveryTests(unittest.TestCase):
    def test_selector_passes_verbose_sink_into_engine_discovery(self):
        engine=FakeEngine('ollama')
        item=DiscoveredEngine(EngineCandidate('ollama','Ollama',lambda:engine),EngineState.READY,engine,'1.0')
        seen={}
        def discover(*, diagnostic=None):
            seen['diagnostic']=diagnostic
            diagnostic('Engine discovery diagnostic')
            return [item]
        out=io.StringIO()
        old=cli.ui.VERBOSE
        cli.ui.VERBOSE=True
        try:
            with patch.object(cli,'discover_engines',side_effect=discover), patch('builtins.input',return_value='1'), redirect_stdout(out):
                cli._select_engine(None)
        finally:
            cli.ui.VERBOSE=old
        self.assertIsNotNone(seen.get('diagnostic'))
        self.assertIn('Engine discovery diagnostic',out.getvalue())

