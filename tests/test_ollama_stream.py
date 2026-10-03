import json
import unittest
from unittest.mock import patch

from devbits_bench.engines.base import GenerationRequest, ModelInfo
from devbits_bench.engines.ollama import OllamaEngine


class _Response:
    def __init__(self, events):
        self._lines = [event if isinstance(event, bytes) else json.dumps(event).encode() + b"\n"
                       for event in events]

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, *args):
        return False


class OllamaStreamContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = OllamaEngine()
        self.prepared = self.engine.prepare_model(ModelInfo("ollama", "x", "x", configured_context=4096), 4096)

    @patch("devbits_bench.engines.ollama.time.perf_counter", side_effect=[10, 11, 12, 13, 14])
    @patch("devbits_bench.engines.ollama.urllib.request.urlopen")
    def test_stream_preserves_benchmark_contract(self, urlopen, clock):
        final = {"done": True, "eval_count": 1}
        urlopen.return_value = _Response([{"thinking": "h"}, {"response": "answer"}, final])
        events = []
        result = self.engine.generate(self.prepared, GenerationRequest("p", 256, "false"), events.append)
        self.assertEqual(result.thinking_text, "h")
        self.assertEqual(result.answer_text, "answer")
        self.assertEqual(result.native_metrics, final)
        self.assertEqual(events, ["request_sent", "generation_started", "answer_started"])
        self.assertEqual(result.metrics.ttft_s, 1)
        self.assertEqual(result.metrics.answer_ttft_s, 2)
        self.assertEqual(result.metrics.client_total_s, 4)
        self.assertEqual(result.requested_reasoning, "false")
        self.assertIsNone(result.effective_reasoning)
        req = urlopen.call_args.args[0]
        self.assertEqual(json.loads(req.data), {
            "model": "x", "prompt": "p", "think": False, "keep_alive": "5m",
            "options": {"seed": 42, "temperature": 0, "num_predict": 256}, "stream": True,
        })
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 3600})

    @patch("devbits_bench.engines.ollama.time.perf_counter", side_effect=[10, 11, 12, 13, 14, 15])
    @patch("devbits_bench.engines.ollama.urllib.request.urlopen")
    def test_blank_chunks_and_done_content_preserve_all_observations(self, urlopen, clock):
        final = {"done": True, "thinking": "last thought", "response": "last answer"}
        urlopen.return_value = _Response([
            b"\n", {}, {"thinking": "first thought"}, {"response": "first answer"}, final,
        ])
        events = []
        result = self.engine.generate(self.prepared, GenerationRequest("p", 512, "medium"), events.append)
        self.assertEqual(result.thinking_text, "first thoughtlast thought")
        self.assertEqual(result.answer_text, "first answerlast answer")
        self.assertEqual(result.native_metrics, final)
        self.assertEqual(result.metrics.ttft_s, 2)
        self.assertEqual(result.metrics.answer_ttft_s, 3)
        self.assertEqual(result.metrics.client_total_s, 5)
        self.assertEqual(events, ["request_sent", "generation_started", "answer_started"])

    @patch("devbits_bench.engines.ollama.time.perf_counter", side_effect=[10, 11, 12, 13])
    @patch("devbits_bench.engines.ollama.urllib.request.urlopen")
    def test_reasoning_only_has_no_answer_ttft(self, urlopen, clock):
        urlopen.return_value = _Response([{"thinking": "h"}, {"done": True}])
        result = self.engine.generate(self.prepared, GenerationRequest("p", 64, "high"))
        self.assertEqual(result.metrics.ttft_s, 1)
        self.assertIsNone(result.metrics.answer_ttft_s)
        self.assertEqual(result.answer_text, "")
        self.assertEqual(result.metrics.client_total_s, 3)

    @patch("devbits_bench.engines.ollama.time.perf_counter", side_effect=[10, 11, 12])
    @patch("devbits_bench.engines.ollama.urllib.request.urlopen")
    def test_empty_generation_has_no_token_latencies(self, urlopen, clock):
        urlopen.return_value = _Response([{"done": True}])
        events = []
        result = self.engine.generate(self.prepared, GenerationRequest("p", 64, "false"), events.append)
        self.assertEqual(events, ["request_sent"])
        self.assertIsNone(result.metrics.ttft_s)
        self.assertIsNone(result.metrics.answer_ttft_s)
        self.assertEqual(result.metrics.client_total_s, 2)

    @patch("devbits_bench.engines.ollama.urllib.request.urlopen")
    def test_transport_and_parse_errors_propagate_to_runner(self, urlopen):
        urlopen.side_effect = OSError("runtime unavailable")
        with self.assertRaisesRegex(OSError, "runtime unavailable"):
            self.engine.generate(self.prepared, GenerationRequest("p", 64, "false"))
        urlopen.side_effect = None
        urlopen.return_value = _Response([b"invalid json\n"])
        with self.assertRaises(json.JSONDecodeError):
            self.engine.generate(self.prepared, GenerationRequest("p", 64, "false"))


if __name__ == "__main__":
    unittest.main()
