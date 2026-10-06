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

class MultiEngineProvenanceTests(unittest.TestCase):
    def _result(self):
        r = Result('mlx-community/model','mlx-community/model',32768,'Measured',1,'false',0,
                   100,0,20,0.0,0.5,2.0,200.0,20.0,{}, {},ttft_s=.6,answer_ttft_s=.6,client_total_s=2.0,
                   phase='Measured',measured=True)
        r._measurement_provenance = {
            'prompt_tps':'mlx-lm native','generation_tps':'mlx-lm native',
            'ttft_s':'devbits client monotonic','answer_ttft_s':'devbits client monotonic',
            'prompt_s':'derived: native tokens / native tps',
        }
        r._native_metrics = {'peak_memory_gb':15.4,'finish_reason':'stop','fresh_prompt_cache':True}
        r._requested_reasoning='false'; r._effective_reasoning='false'
        return r

    def test_mlx_json_adds_structured_engine_model_and_trial_provenance(self):
        r=self._result()
        doc=json_document(protocol='p',system={'chip':'x'},engine_name='mlx-lm',engine_version='v',results=[r],
                          engine_metadata={'id':'mlx-lm','backend':'Metal'},
                          model_metadata={r.base_model:{'quantization':'NVFP4'}})
        self.assertEqual(doc['mlx-lm'],'v')
        self.assertEqual(doc['engine']['backend'],'Metal')
        self.assertEqual(doc['models'][r.base_model]['quantization'],'NVFP4')
        row=doc['results'][0]
        self.assertEqual(row['measurement_provenance']['prompt_tps'],'mlx-lm native')
        self.assertEqual(row['native_metrics']['peak_memory_gb'],15.4)
        self.assertEqual(row['effective_reasoning'],'false')

    def test_ollama_compatibility_shape_strips_additive_sidecar_evidence(self):
        r=self._result()
        doc=json_document(protocol='p',system={'chip':'x'},engine_name='ollama',engine_version='v',results=[r],
                          engine_metadata={'id':'ollama'},model_metadata={'m':{}})
        self.assertNotIn('engine',doc); self.assertNotIn('models',doc)
        self.assertNotIn('measurement_provenance',doc['results'][0])
        self.assertNotIn('native_metrics',doc['results'][0])

class ReportSchemaVersionTests(unittest.TestCase):
    def test_non_ollama_reports_identify_additive_v2_schema(self):
        r=Result('m','m',4096,'Warm',1,'false',0,1,0,1,0,1,1,1,1,{}, {})
        doc=json_document(protocol='p',system={},engine_name='mlx-lm',engine_version='v',results=[r])
        self.assertEqual(doc['report_schema'],'devbits-report-v2')

    def test_ollama_reports_do_not_gain_schema_marker(self):
        r=Result('m','m',4096,'Warm',1,'false',0,1,0,1,0,1,1,1,1,{}, {})
        doc=json_document(protocol='p',system={},engine_name='ollama',engine_version='v',results=[r])
        self.assertNotIn('report_schema',doc)
