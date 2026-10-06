# Devbits Bench

**Practical, reproducible benchmarking for local AI models and inference engines.**

Devbits Bench runs the same benchmark protocols across supported local inference
engines while preserving engine-specific lifecycle and measurement semantics.

**Supported engines:** **Ollama** and **MLX-LM** (native worker integration;
`mlx_lm.server` is not required).

| Mode | Use it when you want to know… |
| --- | --- |
| **Quick** | How fast and consistent is this model for a short repeatable workload? |
| **Practical** | How does everyday usability change from roughly 4K through 32K input? |
| **Stress** | How far can this configuration be pushed before latency, integrity, or memory pressure becomes unacceptable? |
| **Custom** | How does the model behave at context sizes or fill targets that you choose? |

Devbits Bench writes human-readable Markdown and structured JSON reports so
results can be inspected, compared, archived, or shared with their provenance.

![Devbits Bench Practical benchmark](imgs/ollama-bench-practical.webp)

## Requirements

- Python **3.10+**
- At least one supported inference engine
- Enough RAM / unified memory for the model and context you intend to test

Devbits Bench does not automatically download models, install inference engines,
or modify an MLX Python environment.

## Install

From a checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install .
devbits-bench --help
```

For development use `python -m pip install -e .`. You can also run directly from
the source tree with `./devbits-bench`.

## Engine setup

### Ollama

Install and start Ollama normally, then make the models you want to benchmark
available locally.

```bash
ollama pull MODEL_NAME
devbits-bench --engine ollama
```

Devbits Bench talks to the local Ollama service. Cold/isolation operations use
Ollama's lifecycle controls; warm Quick trials reuse the loaded model.

### MLX-LM

MLX-LM runs in its **own Python environment**. Devbits Bench launches a native
worker using that environment; it does not require or use `mlx_lm.server`.

```bash
python3 -m venv ~/.venvs/mlx-lm
source ~/.venvs/mlx-lm/bin/activate
python -m pip install -U pip mlx-lm
devbits-bench --engine mlx-lm
```

Devbits can discover the active virtualenv/Conda environment, MLX-LM commands on
`PATH`, supported Homebrew/uv/pipx/Conda/pyenv environments, and venvs directly
under `~/.venvs`, `~/.virtualenvs`, `~/venvs`, or `~/envs`. It deliberately does
**not** recursively crawl your home directory.

For a manually located environment outside those locations:

```bash
DEVBITS_MLX_PYTHON=/path/to/mlx-venv/bin/python \
  devbits-bench --engine mlx-lm
```

The override is validated before use. Devbits never installs into or mutates the
selected MLX environment.

## First benchmark

The easiest path is interactive:

```bash
devbits-bench
```

Devbits detects the system and available engines, then asks for engine, mode,
model, context where applicable, and reasoning level.

A repeatable non-interactive Quick example:

```bash
devbits-bench \
  --engine mlx-lm \
  --mode quick \
  --models mlx-community/Qwen3.8-27B-nvfp4 \
  --thinking false \
  --yes
```

Replace the engine/model with your local configuration.

## Benchmark modes

### Quick

Quick is the default repeatable baseline: **one cold baseline, one discarded
warmup, and three measured warm trials**. Use it for short-workload decode speed,
TTFT, prompt throughput, and run-to-run consistency.

```bash
devbits-bench --engine ollama --mode quick --models MODEL_NAME --thinking false --yes
```

### Practical

Practical measures isolated interactions at approximately **4K, 8K, 16K, and
32K** when they fit the selected request envelope. Each workload starts with
fresh prompt state.

Use it to see how TTFT, prompt processing, decode speed, and host memory change
as useful context grows.

```bash
devbits-bench --engine mlx-lm --mode practical --models MODEL_NAME --thinking medium --yes
```

### Stress

Stress is intentionally expensive. It performs a short calibration and then
tests approximately **25%, 50%, 75%, and 90%** of the selected context envelope,
with separate integrity and decode workloads.

Devbits watches host memory pressure. It can warn before an already-constrained
long run and ask before a more expensive stage when swap/compression/available
memory indicates pressure.

You can safely press **Ctrl-C**: fully completed Stress stages are preserved and
the reports are written as interrupted; a partially executing stage is not
reported as complete.

Start conservatively:

```bash
devbits-bench \
  --engine mlx-lm \
  --mode stress \
  --models MODEL_NAME \
  --contexts 32k \
  --thinking false \
  --verbose
```

### Custom

Custom lets you choose the benchmark request window and optional synthetic fill.

```bash
devbits-bench \
  --engine mlx-lm \
  --mode custom \
  --models MODEL_NAME \
  --contexts 32k \
  --fill 24k \
  --thinking medium
```

Without `--fill`, Custom targets roughly 90% of the request window.

For MLX-LM, the Devbits **request context window is a benchmark envelope**, not
an engine KV-cache configuration. Advertised model capacity and the benchmark
request envelope remain distinct.

## Reasoning / thinking

`--thinking` expresses benchmark intent. Qualified models may expose `false`,
`low`, `medium`, and `xhigh`. Support is model- and engine-dependent.

For comparisons, use the same requested reasoning setting, but do not assume
equal labels imply identical internal computation across engines. Reports retain
requested/effective reasoning where the adapter can establish it.

## Context and token counts

These can differ:

- **Request context** — benchmark window selected for the run.
- **Approximate fill** — synthetic workload target Devbits attempts to occupy.
- **Actual prompt tokens** — authoritative count reported by the engine adapter
  after tokenization/template processing.

Compare actual prompt counts when interpreting throughput or cross-engine results.

## Reading results

- **TTFT** — time to first generated token/event under the qualified adapter contract.
- **Prompt tok/s** — prompt/prefill throughput.
- **Decode tok/s** — generated-output throughput.
- **Load** — setup/reload time when the adapter performs one.
- **Memory** — host before/after observations; not automatically engine peak memory.

MLX-LM preserves native MLX prompt/output counts, throughput, and native peak
memory where available; Devbits records client-observed monotonic timings
separately. Ollama retains its qualified Ollama-native semantics.

JSON reports retain engine/runtime/model identity, reasoning, cache/isolation
information, measurement provenance, and normalized results. Unavailable metrics
are not treated as zero.

## Fair-comparison checklist

1. Use the same model family and as-equivalent-as-possible weights/quantization.
2. Use the same mode, request context, fill target, and reasoning intent.
3. Compare **actual prompt tokens**, not only the `4K`/`32K` target label.
4. Do not compare cold and warm runs as though they measure the same lifecycle.
5. Close unrelated memory-heavy workloads for large-context tests.
6. Record memory pressure and swap; heavy swapping measures host contention too.
7. Keep the Markdown/JSON report with published numbers.
8. Treat cross-engine results as qualified measurements, not proof of identical
   caching, reasoning, or timing internals.

Results are local measurements, not universal performance claims.

## Reports

Devbits writes timestamped reports in the current directory:

```text
devbits-bench-YYYYMMDD-HHMMSS.md
devbits-bench-YYYYMMDD-HHMMSS.json
```

Markdown is the human-readable report; JSON is the structured provenance record.
Generated reports are ignored by the repository `.gitignore`.

## UI demos

Preview terminal states without loading a model or making inference requests:

```bash
devbits-bench --demo quick
devbits-bench --demo practical
devbits-bench --demo stress-mlx
devbits-bench --demo custom-mlx
```

For screenshot/review work:

```bash
devbits-bench --demo custom-mlx --demo-speed step
```

Demo values are synthetic fixtures and **must not be published as benchmark results**.

## Useful CLI

```bash
devbits-bench --help
devbits-bench --version
devbits-bench --verbose
devbits-bench --no-ansi
devbits-bench --yes
devbits-bench --mode custom --contexts 32k,64k
devbits-bench --mode custom --contexts 32k --fill 24k
```

## Troubleshooting

### `MLX-LM not detected`

Verify the MLX environment itself:

```bash
source /path/to/mlx-venv/bin/activate

python - <<'PY'
from importlib.metadata import version
print("mlx-lm:", version("mlx-lm"))
print("mlx:", version("mlx"))
PY
```

Then run `devbits-bench --verbose`. If the environment is valid but outside the
bounded discovery locations, set:

```bash
DEVBITS_MLX_PYTHON=/path/to/mlx-venv/bin/python \
  devbits-bench --engine mlx-lm --verbose
```

Do **not** create a new venv merely to activate an existing MLX installation; a
new environment is empty until packages are installed into it.

### Stress warns about memory pressure

The warning is advisory and deliberate. Large-context benchmarking can consume
substantial RAM/unified memory and trigger compression or swap.

If the machine is already under significant pressure, stop other heavy workloads
or restart before collecting results you intend to compare or publish. If
pressure develops between stages, declining the next stage preserves completed work.

### Ollama is installed but unavailable

Make sure the Ollama service is running and the requested model is present
locally. `--verbose` prints engine-discovery diagnostics.

## Methodology and architecture

Benchmark methodology is versioned. Changes to corpus bytes, trial schedules,
output budgets, cache/isolation rules, timing definitions, or other comparability
semantics are protocol changes rather than ordinary refactors.

Public project documentation:

- [Architecture](ARCHITECTURE.md) — engine boundaries, lifecycle, context semantics, measurement provenance, and benchmark protocols.
- [Changelog](CHANGELOG.md) — user-facing changes by release.
- [Release checklist](RELEASE_CHECKLIST.md) — maintainer validation before publishing a release.

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src tests
```

Automated tests and UI demos complement rather than replace live-engine qualification.

## License

MIT.

---

**Devbits:** devbits.xyz · X: @devbits · Reddit: /r/devbits/
