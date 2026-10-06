"""Quick calibration and the historical manual request runner."""
from __future__ import annotations
from .constants import CACHE_ACCEPT_RATIO, PROMPT
from .models import Result, summarize_trials
from .workloads import filler, standard_corpus
from .common import result_fields, attach_generation_evidence
from ..engines.base import Engine, GenerationRequest, PreparedModel
from ..system.metrics import memory_snapshot
from ..ui.terminal import (B, C, D, G, X, Y, Spinner, col, ui_verbose,
    clear_terminal_lines, quick_progress_block, human_duration)

def bench(engine: Engine, prepared: PreparedModel, think, mode, n, fill):
 base, model, ctx = prepared.model.id, prepared.id, prepared.configured_context
 print(col('\n  ▶ ',G)+f'Model — {base}')
 print(col('  ↳ ',D)+f'Variant — {model} • context={ctx//1024}K')
 if mode=='Cold':print(col('  ❄ ',C)+'Unloading running models…'); engine.reset(prepared)
 print(col('  ◐ ',Y)+f'{mode} run ({n}) • thinking={think}'+(f' • ~{fill:,} fill tokens' if fill else ''))
 request=GenerationRequest(prompt=filler(fill)+PROMPT,max_output_tokens=256,reasoning=think,stream=False)
 ui_verbose(f'  Generation • thinking={think} • stream=false')
 print(col('  ▶ ',G)+'Running benchmark…')
 before=memory_snapshot()
 try:
  generation=engine.generate(prepared,request)
  after=memory_snapshot()
  z=Result(base_model=base,model=model,context=ctx,mode=mode,run=n,thinking=think,
   requested_fill=fill,**result_fields(generation),
   memory_before=before,memory_after=after)
  attach_generation_evidence(z, generation)
  print(col('  ✓ ',G)+f'{z.generation_tps:.2f} tok/s decode • {z.prompt_tps:.2f} tok/s prompt • {z.total_s:.2f}s total')
  return z
 except Exception as e:
  after=memory_snapshot()
  print(col('  ✗ ',X)+str(e))
  return Result(base_model=base,model=model,context=ctx,mode=mode,run=n,thinking=think,
   requested_fill=fill,prompt_tokens=0,cached_prompt_tokens=0,output_tokens=0,
   load_s=0,prompt_s=0,total_s=0,prompt_tps=0,generation_tps=0,
   memory_before=before,memory_after=after,error=str(e))


def quick_bench(engine: Engine, prepared: PreparedModel, think, embedded=False):
 base, model, ctx = prepared.model.id, prepared.id, prepared.configured_context
 if not embedded:
  print('\n'+col('Quick benchmark',B)); print()
  print(f'  {base} • thinking={think}')
  print('  1 cold baseline • 1 warmup • 3 measured trials')
  print()
 else:
  print('\n  Baseline calibration')
  print('  1 cold baseline • 1 warmup • 3 measured trials')
  print()

 trials=[]
 completed=[]
 schedule=[('Cold',1,True),('Warmup',1,False),('Measured',1,True),('Measured',2,True),('Measured',3,True)]
 rendered=0
 verbose_evidence=[]
 if not embedded:
  rendered=quick_progress_block(completed,0,len(schedule))

 for trial_id,(phase,n,measured) in enumerate(schedule,1):

  if phase=='Cold':
   # Do not print diagnostics while the live Quick surface owns the terminal.
   # The historical Ollama renderer assumes only its spinner occupies the row
   # immediately below the progress block. Queue lifecycle evidence instead.
   engine.reset(prepared)

  corpus=standard_corpus(trial_id=trial_id)
  request=GenerationRequest(
   prompt=corpus+'\n\nWrite a concise 256-token technical explanation of why deterministic benchmarks improve reproducibility.',
   max_output_tokens=256, reasoning=think,
  )
  before=memory_snapshot()
  label='Cold baseline' if phase=='Cold' else ('Warmup' if phase=='Warmup' else f'Measured {n}/3')
  live_label=(f'Baseline calibration • {label}') if embedded else label

  try:
   with Spinner(f'{live_label} • processing input…') as status:
    def on_event(event):
     if event=='generation_started':
      status.update(f'{live_label} • generating tokens…')
     elif event=='answer_started':
      status.update(f'{live_label} • receiving answer…')
    generation=engine.generate(prepared,request,on_event=on_event)

   after=memory_snapshot()
   fresh_cache = generation.native_metrics.get('fresh_prompt_cache')
   verbose_evidence.append(
    f'  {label} evidence • load={generation.metrics.load_s or 0:.3f}s'
    f' • TTFT={generation.metrics.ttft_s if generation.metrics.ttft_s is not None else float("nan"):.3f}s'
    f' • answer_TTFT={generation.metrics.answer_ttft_s if generation.metrics.answer_ttft_s is not None else float("nan"):.3f}s'
    f' • client_total={generation.metrics.client_total_s if generation.metrics.client_total_s is not None else float("nan"):.3f}s'
    f' • cached_prompt_tokens={generation.metrics.cached_prompt_tokens}'
    f' • fresh_prompt_cache={fresh_cache if fresh_cache is not None else "n/a"}'
   )
   z=Result(
    base_model=base,model=model,context=ctx,mode=phase,run=n,thinking=think,requested_fill=0,
    **result_fields(generation),
    memory_before=before,memory_after=after,
    phase=phase,measured=measured,
   )
   attach_generation_evidence(z, generation)

   ratio=(z.cached_prompt_tokens/z.prompt_tokens) if z.prompt_tokens else 0
   if measured and phase=='Measured' and ratio>CACHE_ACCEPT_RATIO:
    z.error=f'prompt cache contamination: {ratio*100:.2f}%'

   trials.append(z)

   if not embedded:
    if phase=='Measured':
     suffix='rejected' if z.error else ''
     line=(
      f'  ✓ Measured {n}/3   '
      f'TTFT {human_duration(z.ttft_s):<8} '
      f'Decode {z.generation_tps:>5.0f} t/s'
     )
     if suffix:
      line+=f'   {suffix}'
     completed.append(line)
    else:
     completed.append(f'  ✓ {label}')

  except Exception as e:
   if embedded:
    print(col('  ✗ ',X)+f'Baseline calibration • {label} • {e}')
   else:
    completed.append(col('  ✗ ',X)+f'{label} • {e}')

  if not embedded:
   # Restore the proven Ollama Quick ownership model: the spinner owns only the
   # transient row below the compact progress block. Once it exits, erase the
   # previously rendered block and repaint it in place with the new stage.
   clear_terminal_lines(rendered)
   rendered=quick_progress_block(completed,trial_id,len(schedule))

 if not embedded:
  # The last stage repaint above is already the canonical 5/5 block. Diagnostics
  # are emitted only after the live surface has finished, so they cannot disturb
  # cursor ownership during the benchmark.
  for line in verbose_evidence:
   ui_verbose(line)
 else:
  for line in verbose_evidence:
   ui_verbose(line)
  valid=[r for r in trials if r.phase=='Measured' and r.measured and not r.error]
  if valid:
   tt=summarize_trials(valid,'ttft_s')
   dec=summarize_trials(valid,'generation_tps')
   print(
    f'  ✓ Baseline calibrated   TTFT {tt["median"]:.1f}s   '
    f'Decode {dec["median"]:.0f} t/s'
   )
 return trials


