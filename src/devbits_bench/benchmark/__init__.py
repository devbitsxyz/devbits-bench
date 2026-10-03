"""Engine-independent benchmark contracts and deterministic workloads."""
from .models import Result, summarize_trials
from .workloads import filler, long_corpus, practical_corpus, standard_corpus

__all__ = ["Result", "summarize_trials", "filler", "long_corpus", "practical_corpus", "standard_corpus"]
