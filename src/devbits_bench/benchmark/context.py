"""Practical, Stress and Custom share isolated long-context interactions."""
from __future__ import annotations
import sys

from .constants import CACHE_ACCEPT_RATIO, PRACTICAL_STAGES
from .models import Result
from .workloads import practical_corpus, long_corpus
from .common import result_fields, stabilize_for_long_test, pressure_warning, pressure_policy_description
from ..engines.base import Engine, GenerationRequest, PreparedModel
from ..system.metrics import memory_snapshot
from ..ui import terminal as ui
from ..ui.terminal import (B, C, D, G, X, Y, Spinner, col, ui_verbose,
    clear_terminal_lines, quick_progress_block, stress_progress_block,
    stress_stage_summary, custom_summary, human_duration, print_pressure_warning)

from .quick import quick_bench

def practical_single(engine: Engine, prepared: PreparedModel, think, target_tokens, stage_number,progress_done=0,progress_total=4):
 """Measure one realistic isolated interaction at a fixed practical context depth."""
 base, model, ctx = prepared.model.id, prepared.id, prepared.configured_context
 before = stabilize_for_long_test(engine, prepared, 3.0)
 target_chars = max(2200, target_tokens * 4)
 corpus = practical_corpus(target_chars, trial_id=300 + stage_number)

 instruction = """
You are reviewing the supplied project context.

Write a concise technical assessment of the project records. Identify two
recurring operational patterns and suggest one concrete engineering improvement.
Keep the complete response within the available output budget.
""".strip()

 request = GenerationRequest(prompt=corpus + '\n\n' + instruction, max_output_tokens=512, reasoning=think)
 ui_verbose(f'  Generation • stream=true • target_input≈{target_tokens:,} • output_cap=512 • thinking={think}')

 with Spinner(f'Processing ~{target_tokens // 1024}K input…') as status:
  def on_event(event):
   if event=='request_sent':
    status.update(f'Processing ~{target_tokens // 1024}K input…')
   elif event=='generation_started':
    status.update('Generating tokens…')
   elif event=='answer_started':
    status.update('Receiving visible answer…')
  generation = engine.generate(prepared,request,on_event=on_event)
 status.update('Collecting results…')
 after = memory_snapshot()

 fields = result_fields(generation)
 cached = fields['cached_prompt_tokens']
 prompt_tokens = fields['prompt_tokens']

 result = Result(
  base_model=base,
  model=model,
  context=ctx,
  mode='Practical',
  run=stage_number,
  thinking=think,
  requested_fill=target_tokens,
  **fields,
  memory_before=before,
  memory_after=after,
  phase='Practical',
  measured=True,
  isolated=True,
  stabilization_s=3.0,
  checkpoint_pass=None,
  checkpoint_hits=0,
  checkpoint_total=0,
  benchmark_thinking=think,
 )

 cache_ratio = (cached / prompt_tokens) if prompt_tokens else 0.0
 if cache_ratio > CACHE_ACCEPT_RATIO:
  result.error = (
   f'prompt cache contamination: {cached}/{prompt_tokens} tokens '
   f'({cache_ratio * 100:.2f}%) exceeds {CACHE_ACCEPT_RATIO * 100:.0f}% limit'
  )

 return result, pressure_warning(before, after)


def practical_bench(engine: Engine, prepared: PreparedModel, think):
 """Run fixed practical workloads with Quick-style persistent progress."""
 base, model, ctx = prepared.model.id, prepared.id, prepared.configured_context
 print('\n'+col('Practical benchmark',B)); print()
 print(f'  4K → 8K → 16K → 32K • isolated interactions • thinking={think}')
 suite_start=memory_snapshot()
 ui_verbose(col('  i ',C)+f"Suite memory baseline captured ({suite_start.get('platform')}).")
 results=[]; completed_count=0; completed=[]; rendered_lines=0
 print()

 for stage_number,target in enumerate(PRACTICAL_STAGES,1):
  if target>=ctx:
   completed.append(f'  – {target//1024:>2}K   skipped (configured context {ctx//1024}K)')
   completed_count+=1
   continue

  if rendered_lines:
   clear_terminal_lines(rendered_lines)
  rendered_lines=quick_progress_block(completed,completed_count,len(PRACTICAL_STAGES))

  try:
   result,reasons=practical_single(
    engine,prepared,think,target,stage_number,
    progress_done=completed_count,progress_total=len(PRACTICAL_STAGES)
   )
  except Exception as exc:
   clear_terminal_lines(rendered_lines)
   completed.append(col('  ✗ ',X)+str(exc))
   quick_progress_block(completed,completed_count,len(PRACTICAL_STAGES))
   rendered_lines=0
   break

  result.suite_memory_start=suite_start; results.append(result); completed_count+=1
  ui_verbose(f'  actual input {result.prompt_tokens:,} • output {result.output_tokens} tokens')
  completed.append(
   f'  ✓ {target//1024:>2}K   TTFT {human_duration(result.ttft_s):<9} '
   f'Prompt {result.prompt_tps:>5.0f} t/s   Decode {result.generation_tps:>4.0f} t/s'
  )

  if reasons and stage_number<len(PRACTICAL_STAGES):
   clear_terminal_lines(rendered_lines)
   rendered_lines=quick_progress_block(completed,completed_count,len(PRACTICAL_STAGES))
   print(); print(col('  ⚠ Memory-pressure advisory',Y)); print()
   latest=result.memory_after
   for key,label,unit in [('available_percent','Memory available','%'),('swap_used_mb','Swap',' MB'),('compressed_mb','Compressed',' MB')]:
    start_value=suite_start.get(key); current_value=latest.get(key)
    if start_value is not None and current_value is not None:
     print(f'    {label:<18} {start_value:.0f}{unit} → {current_value:.0f}{unit}')
   next_target=PRACTICAL_STAGES[stage_number]
   print(f'\n    Next practical target: ~{next_target//1024}K input.')
   if input('    Continue? [y/N] ').strip().lower() not in ('y','yes'):
    print(col('    Stopped by user; completed results are preserved.',D)); return results
   print()
   rendered_lines=0

 if rendered_lines:
  clear_terminal_lines(rendered_lines)
 quick_progress_block(completed,completed_count,len(PRACTICAL_STAGES))
 print()
 if completed_count == len(PRACTICAL_STAGES):
  print('  ✓ Practical benchmark complete')
 else:
  print(f'  Practical benchmark stopped • {completed_count}/{len(PRACTICAL_STAGES)} workloads complete')
 return results


def custom_bench(engine: Engine, base, model_contexts, think):
 """Run one isolated user-selected workload per configured context."""
 print('\n'+col('Custom benchmark',B)); print()
 print(f'  {base} • thinking={think}')
 print('  Workloads  ' + ' · '.join(f'~{fill//1024}K @ {prepared.configured_context//1024}K' for prepared,fill in model_contexts))
 print('  One isolated interaction per workload • 10% headroom by default')
 print()
 completed=[]; results=[]; done=0
 rendered=quick_progress_block(completed,done,len(model_contexts))
 for stage,(prepared,fill) in enumerate(model_contexts,1):
  clear_terminal_lines(rendered)
  rendered=quick_progress_block(completed,done,len(model_contexts))
  try:
   result,warning=practical_single(engine,prepared,think,fill,stage,progress_done=done,progress_total=len(model_contexts))
   result.mode='Custom'; result.phase='Custom'
   results.append(result)
   done+=1; completed.append(custom_summary(result))
   clear_terminal_lines(rendered)
   rendered=quick_progress_block(completed,done,len(model_contexts))
   if warning:
    print_pressure_warning(warning)
  except Exception as e:
   clear_terminal_lines(rendered)
   print(col('  ✗ ',X)+f'Custom workload ~{fill//1024}K failed: {e}')
   rendered=quick_progress_block(completed,done,len(model_contexts))
 quick_progress_block(completed,done,len(model_contexts)) if not sys.stdout.isatty() or ui.NO_ANSI else None
 print('\n'+col('✓ Custom benchmark complete',G))
 return results


def full_single(engine: Engine, prepared: PreparedModel, think,target_tokens,phase,trial_id,output_tokens,stabilization_s=3.0,progress_done=0,progress_total=4):
 base, model, ctx = prepared.model.id, prepared.id, prepared.configured_context
 # Every long-context workload starts from an unloaded/stabilized state so Prefill
 # and Decode do not inherit each other's resident-model/KV/memory state.
 before=stabilize_for_long_test(engine,prepared,stabilization_s)
 # ~4 chars/token is only a generator target; the engine prompt count is authoritative.
 target_chars=max(2200,target_tokens*4)
 corpus=long_corpus(target_chars,trial_id)
 if phase=='Long Prefill':
  # This phase validates context accessibility, not reasoning performance.
  request_thinking='false'
  instruction="\n\nReturn only the four verification codes for checkpoints ALPHA, BRAVO, CHARLIE and DELTA, in that order. Do not explain your answer."
  outcap=64
 else:
  request_thinking=think
  instruction="\n\nWrite a concise technical explanation of deterministic benchmarking, using approximately the available output budget."
  outcap=output_tokens
 request=GenerationRequest(prompt=corpus+instruction,max_output_tokens=outcap,reasoning=request_thinking)
 ui_verbose(f'  Generation • stream=true • target_input≈{target_tokens:,} • output_cap={outcap}')
 phase_label='Integrity check' if phase=='Long Prefill' else 'Decode test'
 with Spinner(f'{phase_label} • processing ~{target_tokens//1024}K input…') as status:
  def on_event(event):
   if event=='generation_started':status.update(f'{phase_label} • generating tokens…')
   elif event=='answer_started':status.update(f'{phase_label} • receiving answer…')
  generation=engine.generate(prepared,request,on_event=on_event)
 after=memory_snapshot()
 fields=result_fields(generation)
 expected=['ORCHID-7291','COBALT-4812','LANTERN-5538','HARBOR-1904']
 # Only visible answer text can satisfy the integrity check.
 hits=sum(code in generation.answer_text for code in expected) if phase=='Long Prefill' else 0
 total=4 if phase=='Long Prefill' else 0
 z=Result(base_model=base,model=model,context=ctx,mode=phase,run=trial_id,thinking=request_thinking,requested_fill=target_tokens,
  **fields,memory_before=before,memory_after=after,
  phase=phase,measured=True,isolated=True,stabilization_s=stabilization_s,checkpoint_pass=(hits==total) if total else None,checkpoint_hits=hits,checkpoint_total=total,benchmark_thinking=think)
 cr=(z.cached_prompt_tokens/z.prompt_tokens) if z.prompt_tokens else 0
 if cr>CACHE_ACCEPT_RATIO:z.error=f'cache {cr*100:.1f}% exceeds {CACHE_ACCEPT_RATIO*100:.0f}%'
 return z,pressure_warning(before,after)


def full_bench(engine: Engine, prepared: PreparedModel, think):
 base, model, ctx = prepared.model.id, prepared.id, prepared.configured_context
 print('\n'+col('Stress benchmark',B)); print()
 print(f'  {base} • context={ctx//1024}K • thinking={think}')
 print('  25% → 50% → 75% → 90% • isolated integrity + decode workloads')
 results=[]; suite_start=memory_snapshot()
 ui_verbose(col('  i ',C)+'Memory pressure policy: '+pressure_policy_description(sys.platform))
 results.extend(quick_bench(engine,prepared,think,embedded=True))

 stages=[.25,.50,.75,.90]; completed_count=0; completed=[]; rendered_lines=0
 print()

 for si,fraction in enumerate(stages,1):
  target=int(ctx*fraction)
  if rendered_lines:
   clear_terminal_lines(rendered_lines)
  rendered_lines=stress_progress_block(completed,completed_count,len(stages))
  print(f'  Stress stage {si}/{len(stages)} • ~{target//1024}K ({fraction*100:.0f}% of configured context)')
  print()
  stage_extra=2
  try:
   pre,reasons=full_single(engine,prepared,think,target,'Long Prefill',100+si,96,
                           progress_done=completed_count,progress_total=len(stages))
   pre.suite_memory_start=suite_start; results.append(pre)
   dec,reasons2=full_single(engine,prepared,think,target,'Long Decode',200+si,256,
                            progress_done=completed_count,progress_total=len(stages))
   dec.suite_memory_start=suite_start; results.append(dec)
   reasons=list(dict.fromkeys(reasons+reasons2)); completed_count+=1
   completed.append(stress_stage_summary(target,pre,dec))
  except Exception as e:
   clear_terminal_lines(rendered_lines+stage_extra)
   completed.append(col('  ✗ ',X)+str(e))
   stress_progress_block(completed,completed_count,len(stages))
   rendered_lines=0
   break

  # Remove the previous progress block + stage heading, then redraw the canonical
  # progress surface with the newly completed stage.
  clear_terminal_lines(rendered_lines+stage_extra)
  rendered_lines=stress_progress_block(completed,completed_count,len(stages))

  if reasons and si<len(stages):
   # A blocking warning ends the live progress region. Preserve completed rows as
   # ordinary history so a second progress bar is never left on screen.
   if rendered_lines:
    clear_terminal_lines(rendered_lines)
    for line in completed: print(line)
    rendered_lines=0
   print(); print(col('  ⚠ Memory pressure detected',Y)); print()
   latest=dec.memory_after
   for k,label,unit in [('available_percent','Available memory','%'),('swap_used_mb','Swap',' MB'),('compressed_mb','Compressed',' MB')]:
    x=suite_start.get(k); y=latest.get(k)
    if x is not None and y is not None: print(f'    {label:<18} {x:.0f}{unit} → {y:.0f}{unit}')
   if latest.get('psi_memory_some_avg10') is not None: print(f"    Memory stalls      {latest['psi_memory_some_avg10']:.2f}%")
   if latest.get('psi_memory_full_avg10') is not None: print(f"    Severe stalls      {latest['psi_memory_full_avg10']:.2f}%")
   print(f'\n    Next workload: ~{int(ctx*stages[si])//1024}K')
   if input('    Continue? [y/N] ').strip().lower() not in ('y','yes'):
    print('\n    Stopped by user; completed results are preserved.'); rendered_lines=0; break
   print(); rendered_lines=0

 if rendered_lines:
  clear_terminal_lines(rendered_lines)
 stress_progress_block(completed,completed_count,len(stages))
 print()
 if completed_count == len(stages):
  print('  ✓ Stress benchmark complete')
 else:
  print(f'  Stress benchmark stopped • {completed_count}/{len(stages)} stages complete')
 return results


