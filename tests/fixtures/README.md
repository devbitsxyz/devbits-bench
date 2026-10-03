# Pass 3 runner compatibility fixture

`pass3_runner_behavior.json` was captured from the untouched Pass 3 source
snapshot before runner extraction. It records complete serialized `Result` rows
and the ordered reset, stabilization sleep, memory sampling, and generation
operations for Quick, Practical, Stress, and Custom.

Generation entries record full SHA-256 digests and character lengths of the exact
submitted prompts, along with output budget, reasoning, seed, temperature, model
identifier, and streaming mode. The capturing adapter also asserted the original
five-minute keep-alive setting and exact native payload/options key sets. Native
payload translation is subsequently tested at the Ollama adapter boundary.

The capture replaced inference, memory observations, sleeping, and terminal
animation with deterministic fakes. The missing `legacy.API` logging symbol was
injected as `fixture://engine` to allow the original Practical and Stress paths to
execute; no workload, orchestration, arithmetic, or result logic was changed.

The fake returns distinct token counts/timings and memory snapshots for each
request. This makes ordering, warmup labeling, suite baselines, trial IDs, and
result field changes visible. Practical uses a configured context of 16,385 to
exercise its skipped 32K stage; Stress uses 8,192; Custom runs two contexts.

The existing v0.1 corpus digest tests remain separate. These runner fixtures prove
preservation of Pass 3 behavior and do not claim an independently executed v0.1
inference comparison.

`pass3_report_behavior.json` captures complete Markdown and JSON documents from
the same original snapshot and result rows. It includes a deterministic Q4_K_M
model-inspection response to exercise the format guide. Historical reporting
limitations, including the Quick Markdown renderer used for Practical/Stress and
the outdated Custom output-budget description, are intentionally preserved by
these snapshots rather than silently rewritten in a structural refactor.
