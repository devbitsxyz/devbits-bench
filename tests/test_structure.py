import unittest
from unittest.mock import patch

from devbits_bench.engines.ollama import OllamaEngine


class StructureTests(unittest.TestCase):
    def test_ollama_default_endpoint(self):
        self.assertEqual(OllamaEngine().api_url, "http://localhost:11434")

    def test_context_from_parameters(self):
        engine = OllamaEngine()
        with patch.object(engine, "_api", return_value={"parameters": "temperature 0.7\nnum_ctx 32768\n"}):
            self.assertEqual(engine.inspect_model("model").configured_context, 32768)

    def test_context_missing_is_unknown(self):
        engine = OllamaEngine()
        with patch.object(engine, "_api", return_value={}):
            self.assertIsNone(engine.inspect_model("model").configured_context)
        # Preserve the adapter's historical parser fallback; the public contract
        # explicitly represents unavailable metadata as None.
        self.assertEqual(engine._configured_context({}), 0)


if __name__ == "__main__":
    unittest.main()
