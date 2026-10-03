import hashlib
import unittest
from devbits_bench.benchmark.models import Result, summarize_trials
from devbits_bench.benchmark.workloads import filler, long_corpus, practical_corpus, standard_corpus

class BenchmarkCoreCompatibilityTests(unittest.TestCase):
    def assert_sha(self, text, expected):
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest()[:16], expected)

    def test_standard_corpus_v01_compatibility(self):
        self.assert_sha(standard_corpus(5000, 3), "1aa213fdfa4c1ba4")

    def test_practical_corpus_v01_compatibility(self):
        self.assert_sha(practical_corpus(12000, 2), "af984b2dd817802c")

    def test_long_corpus_v01_compatibility(self):
        self.assert_sha(long_corpus(15000, 4), "e8dc7c4b5a277bd6")

    def test_filler_v01_compatibility(self):
        self.assert_sha(filler(4096), "50bdc1fed353825a")

    def test_trial_summary(self):
        def r(v):
            return Result("b","m",1,"Warm",1,"false",0,0,0,0,0,0,0,v,0,{}, {})
        summary=summarize_trials([r(10),r(20),r(30)], "prompt_tps")
        self.assertEqual(summary["median"], 20)
        self.assertEqual(summary["mean"], 20)

if __name__ == "__main__": unittest.main()
