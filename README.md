# Devbits Bench

Practical benchmarking for local AI models and inference engines.

**Status:** `0.2.0-dev`, engine-neutral refactor in progress. Ollama is the only
implemented runtime. Native MLX-LM is planned after its execution and measurement
semantics are qualified; llama.cpp is a later candidate.

The project began as Devbits Ollama Bench. Its new name reflects the goal of
comparing hardware, model, weight representation, inference engine, context, and
workload using reproducible protocols.

## Run

Python 3.10 or newer is required. Run from the source tree:

```bash
./devbits-bench
```

Or use the package directly:

```bash
PYTHONPATH=src python -m devbits_bench
```

For real benchmarks, install Ollama, run its service, and make the selected models
available locally. The interactive CLI offers Quick, Practical, Stress, and
Custom modes. For example, replacing `MODEL_NAME` with your installed model:

```bash
./devbits-bench --mode quick --models MODEL_NAME --thinking false
```

Use `--help` for options. To preview the terminal interface with synthetic results
without running inference:

```bash
./devbits-bench --demo quick --no-ansi
```

## Install

```bash
python -m pip install -e .
devbits-bench
```

Core and Ollama support have no third-party Python dependencies. The existing
optional extra can install MLX-LM:

```bash
python -m pip install -e '.[mlx]'
```

**Native MLX-LM execution is not implemented.** Installing that extra does not
add an engine choice, and Ollama users do not need it.

## Protocols and compatibility

| Mode | Workload |
| --- | --- |
| Quick | One cold baseline, one discarded warmup, and three measured warm trials; 256-token output cap. |
| Practical | Isolated 4K, 8K, 16K, and 32K targets that fit below the configured context; 512-token output cap. |
| Stress | Quick calibration followed by 25%, 50%, 75%, and 90% context targets, each with separate integrity and decode requests. |
| Custom | One isolated Practical-style interaction per chosen context/target; 90% fill by default and a 512-token output cap. |

Context targets are approximate workload sizes. Reports distinguish requested
fill from the actual engine-reported prompt count. Ollama cold/isolation resets
unload all running models; warm Quick trials reuse the resident model. Host
memory is sampled before and after requests rather than continuously.

Pass 4 separates engine-neutral requests/results and protocol runners while
preserving deterministic corpus bytes, schedules, output budgets, timing
calculations, pressure behavior, and the historical JSON schema. Engine-native
and client-observed measurements remain distinct inside the adapter contract;
new provenance fields are not added to compatibility JSON in this pass.

Some older presentation limitations are intentionally retained: Practical and
Stress Markdown use the Quick report template, Custom Markdown still describes
synthetic fill and a 256-token cap, and Quick summaries combine selected models.
See [known limitations](ARCHITECTURE.md#known-preexisting-limitations-deliberately-preserved)
before interpreting those reports. Correcting these requires an explicit
reporting or methodology change.

## Development structure

```text
src/devbits_bench/
├── cli.py                  # engine composition
├── legacy.py               # interactive shell, tables, Markdown, demos
├── benchmark/
│   ├── constants.py        # versioned methodology
│   ├── models.py           # historical Result and statistics
│   ├── workloads.py        # deterministic corpora
│   ├── common.py           # mapping, isolation, pressure
│   ├── quick.py            # Quick runner
│   └── context.py          # shared Practical / Stress / Custom runners
├── engines/
│   ├── base.py             # neutral runtime contract
│   └── ollama.py           # Ollama implementation
├── reporting/
│   └── json_report.py      # compatibility JSON
├── system/
│   └── metrics.py          # host observations
└── ui/
    └── terminal.py         # progress and formatting
```

Read [ARCHITECTURE.md](ARCHITECTURE.md) for the responsibility inventory, context
and reasoning semantics, preserved limitations, and multi-engine roadmap.

Run automated checks with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src tests
```

Automated contract tests and UI demos do not replace live-engine qualification.

## License

MIT.
