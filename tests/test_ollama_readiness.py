"""Startup qualification: installation and API readiness are separate facts."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch
import urllib.error

from devbits_bench import cli, legacy
from devbits_bench.engines.base import EngineNotReadyError
from devbits_bench.engines.ollama import OllamaEngine
from devbits_bench.ui import terminal


ROOT = Path(__file__).resolve().parents[1]
DOWN_VERSION = ('Warning: could not connect to a running Ollama instance\n'
                'Warning: client version is 0.34.4\n')
HARDWARE = dict(machine='Test', chip='Test', memory='32 GB', gpu='Test', os='Test')


class Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.data).encode()


@contextmanager
def startup(installed=True, transport_error=None, reachable=False):
    """Run real startup and adapter discovery; prevent all inference and files."""
    engine = OllamaEngine()
    output = io.StringIO()
    requests = []

    def transport(request, **kwargs):
        requests.append(request.full_url)
        if transport_error is not None:
            raise transport_error
        if request.full_url.endswith('/api/tags'):
            return Response({'models': [{'name': 'test-model'}]})
        if request.full_url.endswith('/api/show'):
            return Response({'parameters': 'num_ctx 8192\n'})
        raise AssertionError('Unexpected API request')

    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(output))
        stack.enter_context(redirect_stderr(output))
        stack.enter_context(patch.object(legacy, 'ENGINE', None))
        stack.enter_context(patch.object(terminal, 'NO_ANSI', True))
        stack.enter_context(patch.object(terminal, 'VERBOSE', False))
        stack.enter_context(patch.object(cli, 'OllamaEngine', return_value=engine))
        which = stack.enter_context(patch('devbits_bench.engines.ollama.shutil.which', return_value='/test/ollama' if installed else None))
        version = stack.enter_context(patch('devbits_bench.engines.ollama.subprocess.run', return_value=subprocess.CompletedProcess(
            ['ollama', '--version'], 0, stdout='ollama version is 0.34.4\n' if reachable else DOWN_VERSION, stderr='')))
        urlopen = stack.enter_context(patch('devbits_bench.engines.ollama.urllib.request.urlopen', side_effect=transport))
        stack.enter_context(patch.object(legacy, 'hardware', return_value=HARDWARE))
        choose = stack.enter_context(patch.object(legacy, 'choose', return_value=['test-model']))
        runners = {name: stack.enter_context(patch.object(legacy, name, return_value=[]))
                   for name in ('quick_bench', 'practical_bench', 'full_bench', 'custom_bench')}
        stack.enter_context(patch.object(legacy, 'Path'))
        stack.enter_context(patch('sys.argv', ['devbits-bench', '--thinking', 'false', '--no-ansi']))
        prompt = stack.enter_context(patch('builtins.input', return_value=''))
        yield engine, output, requests, which, version, urlopen, choose, runners, prompt


class StartupCliTests(unittest.TestCase):
    def test_installed_server_down_stops_before_model_selection_and_runners(self):
        refusal = urllib.error.URLError(ConnectionRefusedError(61, 'Connection refused'))
        with startup(transport_error=refusal) as state:
            engine, output, requests, which, version, urlopen, choose, runners, prompt = state
            self.assertTrue(engine.available())  # Installation is still available.
            with self.assertRaises(SystemExit) as exit:
                cli.entrypoint()
            message = str(exit.exception)
            self.assertIn('Ollama is installed, but its local server is not running', message)
            self.assertIn('http://localhost:11434', message)
            self.assertIn('brew services start ollama', message)
            self.assertIn('ollama serve', message)
            self.assertIn('Devbits Bench never starts background services automatically.', message)
            self.assertNotEqual(exit.exception.code, 0)
            self.assertNotIn('Warning:', output.getvalue())
            self.assertIn('client version is 0.34.4', output.getvalue())
            self.assertEqual(requests, ['http://localhost:11434/api/tags'])
            choose.assert_not_called()
            for runner in runners.values():
                runner.assert_not_called()
            prompt.assert_called_once_with('Select mode [1]: ')
            version.assert_called_once_with(['ollama', '--version'], text=True, capture_output=True, check=True)

    def test_missing_executable_keeps_existing_message_and_never_queries_api(self):
        with startup(installed=False) as state:
            _, output, requests, which, version, urlopen, choose, runners, _ = state
            with self.assertRaises(SystemExit) as exit:
                cli.entrypoint()
            self.assertEqual(exit.exception.code, 'ollama engine is not available')
            which.assert_called_once_with('ollama')
            version.assert_not_called()
            urlopen.assert_not_called()
            choose.assert_not_called()
            for runner in runners.values():
                runner.assert_not_called()
            self.assertEqual(requests, [])
            self.assertNotIn('server is not running', output.getvalue())

    def test_reachable_server_preserves_discovery_selection_and_dispatch(self):
        with startup(reachable=True) as state:
            engine, output, requests, _, version, _, choose, runners, _ = state
            cli.entrypoint()
            self.assertEqual(requests, ['http://localhost:11434/api/tags',
                                        'http://localhost:11434/api/show',
                                        'http://localhost:11434/api/show'])
            model = choose.call_args.args[0][0]
            self.assertEqual((model.engine, model.id), ('ollama', 'test-model'))
            runners['quick_bench'].assert_called_once()
            passed_engine, prepared, reasoning = runners['quick_bench'].call_args.args
            self.assertIs(passed_engine, engine)
            self.assertEqual((prepared.id, prepared.configured_context, reasoning), ('test-model', 8192, 'false'))
            self.assertIn('Ollama  : ollama version is 0.34.4', output.getvalue())
            self.assertNotIn('Warning:', output.getvalue())
            version.assert_called_once_with(['ollama', '--version'], text=True, capture_output=True, check=True)

    def test_unexpected_adapter_error_is_not_mislabeled(self):
        bug = RuntimeError('adapter programming failure')
        with startup(transport_error=bug) as state:
            _, output, _, _, _, _, choose, runners, _ = state
            with self.assertRaises(RuntimeError) as caught:
                cli.entrypoint()
            self.assertIs(caught.exception, bug)
            self.assertNotIn('server is not running', output.getvalue())
            choose.assert_not_called()
            for runner in runners.values():
                runner.assert_not_called()

    def test_launcher_and_module_exit_nonzero_without_tracebacks(self):
        # Use fresh processes so SystemExit rendering and actual exit status are
        # verified for both supported invocation paths. All transport is mocked.
        bootstrap = '''
import runpy, subprocess, sys, urllib.error
from unittest.mock import patch
from devbits_bench import legacy
installed = sys.argv[1] == 'installed'
launch = sys.argv[2]
sys.argv = ['devbits-bench', '--no-ansi']
def forbidden(*args, **kwargs):
    raise AssertionError('Startup must stop before selection or inference')
with patch('devbits_bench.engines.ollama.shutil.which', return_value='/test/ollama' if installed else None), \\
     patch('devbits_bench.engines.ollama.subprocess.run', return_value=subprocess.CompletedProcess([], 0, 'Warning: could not connect to a running Ollama instance\\nWarning: client version is 0.34.4\\n', '')), \\
     patch('devbits_bench.engines.ollama.urllib.request.urlopen', side_effect=urllib.error.URLError(ConnectionRefusedError(61, 'Connection refused'))), \\
     patch.object(legacy, 'hardware', return_value={}), \\
     patch.object(legacy, 'choose', side_effect=forbidden), \\
     patch.object(legacy, 'quick_bench', side_effect=forbidden), \\
     patch('builtins.input', return_value=''):
    if launch == 'module':
        runpy.run_module('devbits_bench', run_name='__main__')
    else:
        runpy.run_path('devbits-bench', run_name='__main__')
'''
        for installed in ('installed', 'missing'):
            for launch in ('launcher', 'module'):
                with self.subTest(installed=installed, launch=launch):
                    result = subprocess.run([sys.executable, '-c', bootstrap, installed, launch],
                                            cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
                                            text=True, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 1)
                    self.assertNotIn('Traceback', result.stdout + result.stderr)
                    self.assertNotIn('Warning:', result.stdout + result.stderr)
                    if installed == 'installed':
                        self.assertIn('Ollama is installed, but its local server is not running', result.stderr)
                        self.assertIn('brew services start ollama', result.stderr)
                        self.assertIn('ollama serve', result.stderr)
                    else:
                        self.assertEqual(result.stderr.strip(), 'ollama engine is not available')


class AdapterReadinessTests(unittest.TestCase):
    def test_connectivity_failures_become_typed_readiness_errors(self):
        for failure in (urllib.error.URLError(ConnectionRefusedError(61, 'refused')),
                        urllib.error.URLError(TimeoutError('timed out')),
                        urllib.error.URLError(socket.gaierror(-2, 'name not known')),
                        ConnectionResetError('connection reset'), TimeoutError('timed out')):
            with self.subTest(failure=failure):
                engine = OllamaEngine('http://127.0.0.1:12345/')
                with patch.object(engine, '_api', side_effect=failure):
                    with self.assertRaises(EngineNotReadyError) as caught:
                        engine.list_models()
                self.assertIn('http://127.0.0.1:12345.', str(caught.exception))
                self.assertIs(caught.exception.__cause__, failure)

    def test_http_invalid_url_data_and_programming_errors_remain_diagnosable(self):
        for failure in (urllib.error.HTTPError('http://localhost:11434/api/tags', 503, 'Service unavailable', {}, None),
                        urllib.error.URLError('unknown url type: broken'),
                        json.JSONDecodeError('invalid response', '', 0),
                        ValueError('adapter bug'), OSError('unexpected filesystem error')):
            if isinstance(failure, urllib.error.HTTPError):
                self.addCleanup(failure.close)
            with self.subTest(failure=failure), patch.object(OllamaEngine, '_api', side_effect=failure):
                with self.assertRaises(type(failure)) as caught:
                    OllamaEngine().list_models()
                self.assertIs(caught.exception, failure)

    def test_metadata_programming_error_is_not_a_readiness_error(self):
        with patch.object(OllamaEngine, '_api', return_value={'models': [{}]}):
            with self.assertRaises(KeyError):
                OllamaEngine().list_models()

    def test_version_recovers_client_information_without_printing_warnings(self):
        cases = [
            ('ollama version is 0.34.4\n', '', 'ollama version is 0.34.4'),
            (DOWN_VERSION, '', 'client version is 0.34.4'),
            ('Warning: could not connect to a running Ollama instance\n', 'Warning: client version is 0.34.4\n', 'client version is 0.34.4'),
            ('', DOWN_VERSION, 'client version is 0.34.4'),
            ('Warning: could not connect to a running Ollama instance\n', '', 'version unavailable'),
        ]
        for stdout, stderr, expected in cases:
            with self.subTest(stdout=stdout, stderr=stderr), patch(
                'devbits_bench.engines.ollama.subprocess.run',
                return_value=subprocess.CompletedProcess([], 0, stdout, stderr),
            ):
                self.assertEqual(OllamaEngine().version(), expected)

    def test_unexpected_version_subprocess_failure_propagates(self):
        failure = subprocess.CalledProcessError(1, ['ollama', '--version'])
        with patch('devbits_bench.engines.ollama.subprocess.run', side_effect=failure):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                OllamaEngine().version()
        self.assertIs(caught.exception, failure)


if __name__ == '__main__':
    unittest.main()
