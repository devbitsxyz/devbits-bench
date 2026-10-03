import unittest
from devbits_bench.benchmark.models import Result
from devbits_bench.reporting import json_document

class ReportingTests(unittest.TestCase):
    def test_compatibility_shape_keeps_engine_key(self):
        r = Result('m','m',4096,'Warm',1,'false',100,100,0,10,0.1,0.2,1.0,500.0,20.0,{}, {})
        doc = json_document(protocol='p', system={'chip':'x'}, engine_name='ollama', engine_version='v1', results=[r], corpus='c', corpus_seed=7)
        self.assertEqual(doc['ollama'], 'v1')
        self.assertEqual(doc['protocol'], 'p')
        self.assertEqual(doc['corpus'], 'c')
        self.assertEqual(doc['results'][0]['base_model'], 'm')
        self.assertNotIn('engine', doc)

if __name__ == '__main__': unittest.main()
