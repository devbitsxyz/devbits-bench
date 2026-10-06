# Devbits Bench architecture

Devbits Bench 0.2.0 is a Python application for reproducible local-model and
inference-engine benchmarking. The current production architecture supports
**Ollama** and **native MLX-LM** while keeping benchmark methodology separate
from engine-specific lifecycle and measurement semantics.

## Responsibility boundaries

```text
src/devbits_bench/
├── cli.py                  # composition root and engine discovery
├── legacy.py               # interactive shell, terminal reports, Markdown, demos
├── benchmark/
│   ├── constants.py        # versioned methodology constants
│   ├── models.py           # normalized benchmark results/statistics
│   ├── workloads.py        # deterministic corpus/fill generation
│   ├── common.py           # isolation and pressure policy
│   ├── quick.py            # Quick protocol
│   └── context.py          # Practical / Stress / Custom protocols
├── engines/
│   ├── base.py             # engine-neutral contract
│   ├── ollama.py           # Ollama adapter
│   └── mlx_lm.py           # core-side MLX-LM adapter
├── runtimes/
│   └── mlx.py              # non-mutating MLX runtime discovery
├── workers/
│   └── mlx_lm_worker.py    # native MLX process boundary
├── reporting/
│   └── json_report.py      # structured report serialization
├── system/
│   └── metrics.py          # host observations
└── ui/
    └── terminal.py         # live progress and formatting
```

Dependency direction is composition → protocol runners → engine contract →
adapter. Protocol runners do not construct Ollama HTTP payloads or import MLX.

## Engine contract

The neutral contract carries model identity/capabilities, prepared-model handles,
generation requests/events/results, lifecycle operations, and measurement
provenance.

Adapters own engine-specific translation:

- model discovery/inspection;
- request preparation;
- reasoning translation;
- generation/streaming;
- prompt-cache and model-residency behavior;
- native metric extraction;
- cancellation/errors;
- release/cleanup.

Unknown or incomparable measurements remain unavailable rather than being
invented from another engine's semantics.

## Ollama lifecycle

Ollama uses the local service. Its adapter preserves the qualified cold/warm and
isolation behavior, including configured request context and native Ollama
measurement fields.

Quick uses a cold baseline, discarded warmup, and measured warm trials. Isolated
context workloads reset according to the qualified Ollama lifecycle contract.

## MLX-LM lifecycle

The Devbits core has no MLX dependency. It launches a versioned native worker
using a validated MLX Python interpreter:

```text
Devbits core
    |
    | structured IPC
    v
MLX-LM worker
    ├── load model/tokenizer
    ├── apply native template/reasoning controls
    ├── create fresh prompt state where isolation requires it
    ├── stream native generation
    ├── return native metrics/provenance
    └── shutdown / terminate on cleanup
```

Weights may remain resident across interactions when the benchmark protocol
requires residency. That is distinct from prompt/KV-state reuse. Existing
isolated workloads use fresh prompt state.

Runtime discovery is non-mutating and bounded. It can use active environments,
PATH-visible commands, supported environment-manager registries, conventional
venv roots, or `DEVBITS_MLX_PYTHON`. It does not recursively scan the home
directory and does not install MLX-LM.

## Context semantics

Keep these concepts separate:

| Concept | Meaning |
| --- | --- |
| Request context | Benchmark envelope selected by Devbits. |
| Approximate fill | Synthetic occupied-context target. |
| Actual prompt tokens | Authoritative adapter/engine token count after formatting/tokenization. |
| Advertised context | Model/runtime capability metadata when available. |
| KV/cache limit | Engine-specific mechanism; not assumed equivalent across engines. |

In particular, an MLX request envelope is **not** presented as an Ollama-style
configured KV/context limit.

## Reasoning semantics

Reasoning labels represent requested benchmark intent, not universal engine
implementation. Adapters expose only qualified choices and retain
requested/effective reasoning provenance where it can be established.

The qualified MLX Qwen path supports `false`, `low`, `medium`, and `xhigh`.
Other models/runtimes may expose different capabilities.

## Measurement provenance

Normalized reports preserve the origin of measurements.

For qualified MLX-LM runs:

- prompt/output token counts: MLX-LM native;
- prompt/decode throughput: MLX-LM native;
- native peak memory: MLX-LM native;
- TTFT/answer-TTFT/client total: Devbits monotonic observation;
- host memory: Devbits before/after snapshots.

Ollama retains its qualified Ollama-native timing/count/throughput semantics.
Host memory snapshots and engine-native peak memory are not interchangeable.

## Benchmark protocols

Methodology constants are versioned. Changes to corpus bytes, schedules, output
budgets, cache/isolation rules, timing definitions, integrity checks, or other
comparability semantics require explicit protocol review.

- **Quick:** cold baseline → discarded warmup → three measured warm trials.
- **Practical:** isolated ~4K/~8K/~16K/~32K workloads where supported.
- **Stress:** calibration plus 25%/50%/75%/90% integrity+decode stages.
- **Custom:** isolated user-selected context/fill workloads.

Stress preserves fully completed stages on interruption and records explicit
suite completion/interruption state.

## Memory pressure

Host memory is contextual. macOS uses available-memory plus swap/compression
deltas; Linux uses its qualified host-pressure observations. Stress may warn
before an already-constrained long run and between expensive stages.

Warnings are advisory. Devbits does not silently shrink the requested benchmark.

## Reports

Markdown is the human-readable report. JSON is the structured provenance record.
Reports retain engine/runtime/model identity and benchmark configuration so
published numbers can be interpreted in context.

Cross-engine equality is never inferred merely because two fields share a name.

## Validation

Release validation includes:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src tests
PYTHONPATH=src python -m devbits_bench --help
PYTHONPATH=src python -m devbits_bench --version
```

Synthetic demos validate presentation without inference. Live-engine
qualification remains separate evidence.

## Qualified engines

- **Ollama** — supported.
- **MLX-LM** — ACCEPTED / FROZEN through native generation, controls, Quick,
  provenance, Practical, Stress, Custom, runtime portability, regression, and
  real-machine release smoke.

Future engines must qualify identity, context/cache semantics, reasoning,
measurements, lifecycle, cancellation, errors, and provenance before being
presented as comparable benchmark engines.
