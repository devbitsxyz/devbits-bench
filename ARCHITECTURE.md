# Devbits Bench architecture

Devbits Bench is a small Python application for reproducible local model and
inference-engine benchmarks. Version `0.2.0-dev` continues the v0.1 methodology
while separating workloads, runners, runtime adapters, and presentation.
Ollama is the only implemented engine. Native MLX-LM is the next candidate;
llama.cpp is a later candidate.

## Pass 4 scope and baseline

Pass 4 replaces the Ollama-shaped runtime interface and extracts protocol
orchestration. It preserves workload bytes, trial schedules, output budgets,
cache acceptance, timing formulas, memory sampling, and compatibility reports.
Source-file length is not an acceptance criterion.

The working checkout matched the supplied Pass 3 description and its ten tests.
The handoff names a Pass 3 ZIP, but no ZIP was supplied for this pass; the
available checkout was the source baseline. The supplied handoff ends at the
`PASS 4E` heading, so this work does not assume missing instructions after it.

## Responsibility boundaries

```text
src/devbits_bench/
├── cli.py                  # composition root: select Ollama and start shell
├── legacy.py               # interactive CLI, result tables, Markdown, demos
├── benchmark/
│   ├── constants.py        # versioned methodology constants
│   ├── models.py           # compatibility Result and trial statistics
│   ├── workloads.py        # deterministic corpus and fill generation
│   ├── common.py           # result mapping, isolation, pressure policy
│   ├── quick.py            # Quick schedule and retained manual measurement
│   └── context.py          # Practical, Stress, and Custom orchestration
├── engines/
│   ├── base.py             # neutral dataclasses and Engine protocol
│   └── ollama.py           # native API, metadata, preparation, lifecycle
├── reporting/
│   └── json_report.py      # unchanged compatibility JSON serialization
├── system/
│   └── metrics.py          # hardware and host memory observations
└── ui/
    └── terminal.py         # shared terminal progress and formatting
```

The dependency direction is CLI → runners → engine contract → adapter. Runners
receive an engine and a prepared model explicitly. They do not import
`OllamaEngine`, construct HTTP payloads, or inspect runtime dictionaries.
The CLI chooses the adapter and owns preparation/release around each suite.
`legacy.py` remains a transitional shell because terminal result rendering,
Markdown reporting, and demos can move independently of protocol execution.

Practical and Custom use the same isolated interaction implementation, and
Stress shares reset, pressure, streaming, and context-workload behavior with
them. Keeping these in `context.py` reflects that actual shared behavior;
four protocol names do not require four separate files. The runners still
render progress through shared terminal helpers and prompt when the historical
pressure policy requires it. A fully headless event-driven runner is future work.

### Inventory of the Pass 3 legacy module

This inventory records the responsibilities inspected before extraction,
including helpers that were dormant or only used by demos.

| Original functions/classes | Responsibility and Pass 4 destination |
| --- | --- |
| `set_engine`, `_engine` | Explicit engine ownership retained in the CLI shell; runners receive the engine as an argument. |
| `api`, `models`, `show`, `configured_context`, `create`, `remove_variant`, `unload`, `stream_generate` | Ollama-shaped forwarding removed from orchestration; neutral discovery/inspection/preparation/reset/generation/release replace it. Native work belongs to `engines/ollama.py`. |
| `sh`, `ns`, `rate` | The shell's unused subprocess helper is removed; native time-unit conversion and throughput calculations belong to the adapter. |
| `bench`, `quick_bench` | Manual single measurement and Quick schedule move to `benchmark/quick.py`; `bench` is retained but not selected by the current CLI modes. |
| `practical_single`, `practical_bench`, `custom_bench`, `full_single`, `full_bench` | Shared context interaction and suite orchestration move to `benchmark/context.py`; Stress embeds Quick calibration. |
| `stabilize_for_long_test`, `memory_delta`, `pressure_warning`, `pressure_policy_description` | Isolation, memory comparison, and pressure policy move to `benchmark/common.py`. Host observations stay in `system/metrics.py`. |
| `ui_verbose`, `ui_clear`, `active_progress`, `Spinner`, `progress_bar`, `LiveRegion`, `progress_lines`, `clear_terminal_lines`, `quick_progress_block`, `stress_progress_block`, `stress_stage_summary`, `custom_summary`, `col`, `human_duration`, `hb`, `banner` | Shared progress, color, status, branding, and formatting helpers belong to `ui/terminal.py`. |
| `practical_table`, `full_table`, `table`, `quick_table` | Terminal result presentation remains in `legacy.py`. |
| `benchmark_formats`, `result_format_label`, `result_identity`, `print_model_terms`, `markdown_model_terms` | Model identity and format-guide presentation remain in `legacy.py`, using normalized metadata where inspection is needed. |
| `quick_report`, `report` | Markdown rendering remains in `legacy.py`; JSON was already extracted in Pass 3. |
| `context_advisory`, `main` | User configuration, advisories, dispatch, report paths, and preparation cleanup remain in the shell. |
| `demo_memory`, `demo_result`, `demo_pause`, `demo_step_wait`, `demo_step_state`, `demo_checkpoint`, `demo_quick`, `demo_practical`, `demo_stress`, `demo_custom`, `demo_multi`, `run_demo` | Synthetic demo fixtures and playback remain in `legacy.py`. They are presentation examples, not runtime validation. |

The original runtime coupling covered every execution path: manual generation
called the raw API; Quick streamed raw dictionaries and reset residency; isolated
context interactions reset before streaming; the CLI read raw model/thinking
metadata and managed context variants; format guides queried raw model details.
Runners also computed rates from Ollama nanosecond fields. Those translations
now belong to the adapter or the compatibility result mapper.

## Minimal engine contract

`engines/base.py` defines simple dataclasses and a structural `Engine` protocol:

| Type | Purpose |
| --- | --- |
| `ModelInfo` | Engine/model identity, display name, optional architecture, quantization, parameter count, advertised/configured context, size, and advertised reasoning choices. Unknown metadata remains absent. |
| `PreparedModel` | An adapter-owned handle for one configuration, including the source model, opaque runtime identifier, and configured context. |
| `GenerationRequest` | Prompt, output cap, requested reasoning, seed, temperature, and streaming choice. |
| `GenerationMetrics` | Optional token counts, durations, throughputs, TTFT/TTA/client total, and per-measurement provenance. |
| `GenerationResult` | Metrics, separate thinking/answer text, requested/effective reasoning, and retained engine-native measurements. |

`EngineInfo` is a retained identity container. There is no speculative capability
hierarchy or universal engine-configuration bag. Available reasoning choices
come from model inspection; fields not established by an engine stay optional.

The operations are `available`, `version`, `list_models`, `inspect_model`,
`prepare_model`, `reset`, `generate`, and `release`. Preparation is the adapter's
responsibility: Ollama may reuse the selected model or create a temporary
context variant. HTTP paths, Modelfile syntax, command invocations, temporary
names, nanosecond response fields, and native `think` serialization are private
adapter details. A prepared identifier is opaque to runners even when retained
in a historical result's `model` field.

Installation and readiness are distinct: `available()` checks client installation.
For Ollama, the existing discovery request establishes API reachability. Only
connection failures, timeouts, and DNS failures during discovery become
`EngineNotReadyError`, carrying an adapter-supplied endpoint and startup guidance.
The shared CLI composition translates that error into a clean nonzero exit for
both the launcher and `python -m devbits_bench`. HTTP responses indicating errors,
invalid URLs/data, and programming exceptions remain diagnosable. No extra health
request is added, and Devbits Bench never starts background services automatically.
Ollama version display retains useful client-version information when the CLI
reports a stopped server, without inserting its warning text into the summary.

Generation errors propagate to runners. Their existing behavior is preserved:
Quick continues after a failed trial, Practical/Stress stop the affected suite,
and Custom continues to the next configured workload. Successful results retain
the same cache-rejection fields rather than introducing a new error schema.

## Context, residency, and reasoning semantics

| Concept | Current meaning |
| --- | --- |
| `advertised_context` | Context capacity advertised by model metadata; not proof that a requested workload will fit. |
| `configured_context` | The adapter's configured request window, carried by the prepared model and historical `Result.context`. Ollama derives its existing setting from `num_ctx` or the historical metadata fallback. |
| `requested_fill` | Approximate workload target in the compatibility result, not a measured token count. Quick uses zero; context suites convert their target to roughly four characters per token before appending instructions. |
| `prompt_tokens` | Actual engine-reported input token count, which can differ from the requested target because of tokenization, record sizing, instructions, or runtime formatting. |
| KV cache limit | Not implemented in the contract. A future MLX `max_kv_size` must not be treated as equivalent to Ollama `num_ctx` or the configured request window. |

For Ollama, `reset(prepared)` preserves the historical **unload all running
models** behavior. Quick resets for its cold baseline and reuses residency for
warmup and measured trials. Practical, each Stress integrity/decode interaction,
and each Custom workload reset and then wait three seconds before the memory
baseline. This does not claim that operating-system file caches are cold.

`release(prepared)` disposes preparation resources such as a temporary context
variant. It does not add a new unload after a warm measurement. Preparation,
reset, generation, and cleanup are distinct lifecycle steps. Future MLX worker
startup and model loading require their own measured definitions before reuse
of the word “cold.”

Reasoning is an opaque requested mode with engine-advertised choices, not a
universal `false/low/medium/high` enum. Ollama translates the existing string
`false` to its native boolean and otherwise preserves the requested value.
Requested reasoning is recorded; `effective_reasoning` remains unknown unless
the adapter can establish it. Stress integrity requests still ask for reasoning
off, while `benchmark_thinking` retains the suite's selected setting. No claim
is made that different runtimes or chat templates honor these modes identically.

## Timing, observations, and report compatibility

All normalized durations are seconds. Ollama prompt/output counts and native
load, prompt-evaluation, evaluation, and total durations retain their native
provenance. Prompt throughput still uses uncached prompt tokens divided by
prompt-evaluation duration; decode throughput uses output tokens divided by
evaluation duration.

Generation TTFT is client wall time to the first streamed thinking or answer
content. Answer TTFT is client wall time to the first visible answer content
and can be absent. Client total covers the streamed request. Engine-native total
is separate: neither value is renamed to imply identical timing boundaries.
The native final response remains available within the adapter result. Host
memory snapshots before/after a request are not peak engine memory.

`GenerationMetrics.provenance` and `GenerationResult.native_metrics` retain
measurement distinctions internally. The Ollama adapter preserves its existing
zero defaults for missing native response fields and identifies those defaults
in provenance. `benchmark/common.py` maps to the unchanged historical `Result`
and rejects missing required metrics from other adapters; it does not invent
measurements to make an unqualified engine fit this protocol. The compatibility
JSON does not gain provenance, reasoning-effectiveness, model-metadata, or engine-startup fields in this pass.
It retains the dynamic `ollama` version key and the previous result fields.
Exporting the richer contract requires a separately versioned schema decision.

## Methodology guardrails

`benchmark/constants.py`, `benchmark/workloads.py`, and the existing result and
statistics model are methodology-sensitive. Structural cleanup must not change
their outputs or formulas. In particular:

- Quick remains one cold baseline, one discarded warmup, and three measured
  warm trials, with deterministic trial-specific prompts and a 256-token cap.
- Practical uses 4K/8K/16K/32K targets strictly below configured context,
  isolated requests, and a 512-token cap, without scoring retrieval markers.
- Stress embeds the Quick baseline, then 25%/50%/75%/90% context targets with
  separately isolated integrity and decode requests. Integrity retains its
  64-token cap; decode retains its 256-token cap.
- Custom uses one isolated Practical-style interaction per configured workload,
  defaulting to a 90% target with a 512-token cap.
- All existing prompt instructions, trial IDs, seed 42, temperature zero,
  cache-rejection threshold, memory sampling positions, pressure thresholds,
  and continue/stop behavior are retained.

### Known preexisting limitations, deliberately preserved

These findings need separate methodology or reporting review. They are not
silently corrected by this refactor:

- Practical and Stress still write Markdown through `quick_report`, so titles
  and methodology prose describe Quick; Stress's Quick summary describes its
  calibration trials. JSON retains the existing suite distinctions.
- Custom Markdown says synthetic fill and a 256-token cap, but the active Custom
  runner uses the Practical corpus and a 512-token cap. The terminal column guide
  also retains the older synthetic-fill wording.
- `result_format_label` infers `NVFP4` and `MLX` from an `mlx` substring in a
  model name when quantization is absent. This is presentation inference, not
  authoritative metadata or evidence that native MLX-LM was benchmarked.
- Quick tables and Markdown aggregate accepted measured trials across all
  selected models instead of grouping by model. The identity shown is the first
  accepted model's identity.
- Stress asks for codes in order, but integrity scoring checks only membership
  of each expected code in visible answer text. It does not validate order.
- Workload sizing approximates tokens using characters; it is not tokenizer
  calibration or a guarantee that input plus output fits a context window.
- Quick and Practical ignore the CLI `--contexts` value; Stress uses only the
  first supplied context. The hidden `--warm-runs` option does not alter Quick's
  fixed schedule.
- Stress calibration presentation assumes at least one accepted measured TTFT;
  an all-missing TTFT stream can fail its summary, although standalone Quick
  retains missing TTFT values.
- Ollama temporary variant tags are deterministic, preserving existing report
  identities, and can collide with an existing tag. Removal retains the historical
  best-effort policy for nonzero CLI exits. Naming and cleanup policy changes need
  separate review.
- The large-context advisory's existing memory-text regex can miss ordinary
  values such as `128 GB`, omitting the detected-memory display line.

Pass 3 also contained operational missing-name failures: `choose`, `parsevals`,
`print_pressure_warning`, and the `API` name referenced by generation diagnostics
were undefined. Pass 4 repairs the missing UI helpers and removes endpoint
diagnostics from generic runners. These are execution-path repairs, documented
separately from changes to benchmark measurements; no threshold, workload,
output budget, or calculation is changed to repair them.

## Validation and next steps

The original ten tests cover four fixed workload hashes inherited from the v0.1
comparison, trial summary statistics, Ollama streaming event/timing behavior,
JSON shape, endpoint configuration, and configured-context parsing. Hash fixtures
cover fixed inputs; they are not a proof for all possible inputs or live models.

Pass 4 validation: **56 automated tests passed**, source/test compilation passed,
`--version` returned `0.2.0-dev`, and all UI demos completed. Tests cover exact
Pass 3 runner/result/request/lifecycle fixtures, complete JSON and Markdown
compatibility, neutral CLI execution in every mode, interruption/preparation
cleanup, and deterministic Ollama transport/metadata/lifecycle behavior. The five
core methodology, result/statistics, JSON serializer, and system observation
modules remain byte-identical to the supplied checkout baseline.

Tests and UI demos use synthetic data or mocked transport; they do not establish live
Ollama throughput, context capacity, or native MLX-LM support. Reproducible checks
from the source tree are:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src tests
./devbits-bench --version
./devbits-bench --demo quick --no-ansi
```

The next work should proceed in this order:

1. Qualify the extracted Ollama path with explicitly selected local models and
   compare real requests/results against the existing protocol baseline. Keep
   architecture checks and live-inference evidence distinct.
2. Review the documented report/methodology discrepancies as explicit changes,
   with compatibility and protocol-version decisions where needed. Continue
   extracting Markdown and result presentation independently.
3. **Before implementing native MLX-LM**, qualify its model identity and weight
   representation, prompt/chat-template application, reasoning controls,
   advertised/requested context versus KV cache limits, token counts, timing
   boundaries, cancellation/errors, worker lifecycle, memory observations, and
   cold/warm definitions. Document unsupported or incomparable metrics.
4. Implement an optional MLX-LM adapter or worker environment only after that
   qualification, using the same workloads and runner schedules. Revisit the
   minimal contract only with concrete, tested requirements; keep core/Ollama
   installation free of MLX dependencies.
5. Version multi-engine report evolution explicitly, including provenance,
   configuration, and unavailable measurements, then qualify llama.cpp through
   the same process.

The existing `mlx` packaging extra is unchanged and only installs the optional
dependency. Installing it does not enable an unimplemented engine.
