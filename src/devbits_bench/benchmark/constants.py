"""Versioned benchmark methodology constants.

Changing these values can change benchmark comparability and should be treated as a
protocol change rather than a routine refactor.
"""
VERSION = "0.2.0-dev"
PROTOCOL = "devbits-bench-v1"
PRACTICAL_PROTOCOL = "devbits-practical-v1"
CORPUS = "devbits-standard-context-v1"
CORPUS_SEED = 20261001
CACHE_ACCEPT_RATIO = 0.05
PRACTICAL_STAGES = (4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024)
PROMPT = (
    "A farmer has 17 sheep. All but 9 run away. How many sheep remain? "
    "Explain your reasoning briefly, then give one sentence explaining why the common wrong answer is wrong."
)
