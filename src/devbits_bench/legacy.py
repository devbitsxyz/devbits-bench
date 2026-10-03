"""Transitional interactive shell, demos and historical report presentation."""
from __future__ import annotations
import argparse
import re
import sys
import time
from pathlib import Path

from .benchmark.constants import CACHE_ACCEPT_RATIO, CORPUS, CORPUS_SEED, PRACTICAL_PROTOCOL, PRACTICAL_STAGES, PROTOCOL, VERSION
from .benchmark.models import Result, summarize_trials
from .benchmark.quick import quick_bench
from .benchmark.context import practical_bench, full_bench, custom_bench
from .benchmark.common import pressure_policy_description
from .system.metrics import hardware
from .reporting import write_json_report
from .ui import terminal as ui
from .ui.terminal import (B, C, D, G, X, Y, Spinner, banner, col, hb,
    clear_terminal_lines, quick_progress_block, stress_progress_block,
    stress_stage_summary, human_duration)

ENGINE=None
DEMO_SPEED='normal'

def set_engine(engine):
 global ENGINE
 ENGINE=engine


def _engine():
 if ENGINE is None:
  raise RuntimeError('benchmark engine is not configured')
 return ENGINE


def practical_table(results):
 banner()
 print(col('Devbits Bench — Practical Results',B))
 if not results:
  print('\n  No completed Practical measurements.'); return
 first=results[0]
 print()
 for line in result_identity(first): print(line)
 print()
 header=f"{'CONTEXT':>9} {'INPUT':>9} {'TTFT':>11} {'PROMPT':>12} {'DECODE':>12} {'MEMORY':>11}"
 print(header); print('─'*len(header))
 for result in results:
  before=result.memory_before.get('available_percent'); after=result.memory_after.get('available_percent')
  memory=f'{before:.0f}→{after:.0f}%' if before is not None and after is not None else 'n/a'
  print(f"{('~'+str(result.requested_fill//1024)+'K'):>9} {result.prompt_tokens:>9,} {human_duration(result.ttft_s):>11} {result.prompt_tps:>9.0f} t/s {result.generation_tps:>9.0f} t/s {memory:>11}")
 print_model_terms(results)


def full_table(rs):
 banner(); print(col('Devbits Bench — Stress Results',B))
 long=[r for r in rs if r.phase in ('Long Prefill','Long Decode')]
 if not long:
  print('\n  No completed Stress stages.'); return
 models=[]
 for r in long:
  if r.base_model not in models: models.append(r.base_model)

 if len(models)==1:
  first=long[0]
  print()
  for line in result_identity(first): print(line)
  print()
  h=f"{'TARGET':>9} {'INTEGRITY':>12} {'TTFT':>11} {'PROMPT':>12} {'DECODE':>12}"
  print(h); print('─'*len(h))
  for target in sorted(set(r.requested_fill for r in long)):
   pre=next((r for r in long if r.requested_fill==target and r.phase=='Long Prefill'),None)
   dec=next((r for r in long if r.requested_fill==target and r.phase=='Long Decode'),None)
   integrity=f'{pre.checkpoint_hits}/{pre.checkpoint_total} PASS' if pre and pre.checkpoint_pass else ('FAIL' if pre else '-')
   tt=human_duration(dec.ttft_s if dec else (pre.ttft_s if pre else None))
   prompt=f'{pre.prompt_tps:.0f} t/s' if pre else '-'; decode=f'{dec.generation_tps:.0f} t/s' if dec else '-'
   print(f"{('~'+str(target//1024)+'K'):>9} {integrity:>12} {tt:>11} {prompt:>12} {decode:>12}")
  print_model_terms(long)
  return

 targets=sorted(set(r.requested_fill for r in long))
 print(f'\n  {len(models)} models • {len(targets)} completed workloads')
 for target in targets:
  print('\n'+col(f'Stress comparison — ~{target//1024}K',B)); print()
  h=f"{'MODEL':30} {'TTFT':>11} {'PROMPT':>11} {'DECODE':>11} {'INTEGRITY':>12}"
  print(h); print('─'*len(h))
  for model in models:
   pre=next((r for r in long if r.base_model==model and r.requested_fill==target and r.phase=='Long Prefill'),None)
   dec=next((r for r in long if r.base_model==model and r.requested_fill==target and r.phase=='Long Decode'),None)
   if not pre and not dec: continue
   integrity=f'{pre.checkpoint_hits}/{pre.checkpoint_total} PASS' if pre and pre.checkpoint_pass else ('FAIL' if pre else '-')
   tt=human_duration(dec.ttft_s if dec else pre.ttft_s)
   prompt=f'{pre.prompt_tps:.0f} t/s' if pre else '-'; decode=f'{dec.generation_tps:.0f} t/s' if dec else '-'
   print(f'{model[:30]:30} {tt:>11} {prompt:>11} {decode:>11} {integrity:>12}')
 print_model_terms(long)


def benchmark_formats(rs):
 out=set()
 for r in rs:
  if r.error:continue
  try:
   q=_engine().inspect_model(r.base_model).quantization
   if q:out.add(str(q).upper())
  except:pass
 return out


def result_format_label(result):
 name=result.base_model.lower()
 quant=(getattr(result,'quantization',None) or '').upper()
 parts=[]
 if quant: parts.append(quant)
 elif 'mlx' in name: parts.append('NVFP4')
 if 'mlx' in name: parts.append('MLX')
 return ' / '.join(parts) if parts else 'format reported in guide'


def result_identity(result):
 """Compact screenshot-friendly identity for a result set."""
 try:
  hw=hardware()
  machine=f"{hw.get('chip','Unknown')} • {hw.get('memory','Unknown')}"
 except Exception:
  machine='Hardware unavailable'
 fmt=result_format_label(result)
 thinking=getattr(result,'benchmark_thinking',None) or result.thinking
 return [
  f'  {result.base_model}',
  f'  {fmt} • thinking={thinking} • context={result.context//1024}K',
  f'  {machine}',
 ]


def print_model_terms(rs):
 quants=benchmark_formats(rs); mlx=any('mlx' in r.base_model.lower() for r in rs if not r.error)
 if not quants and not mlx:return
 desc={
  'NVFP4':'4-bit floating-point quantization. Ollama supports model-optimized NVFP4 with its MLX engine on Apple Silicon.',
  'Q4_K_M':'4-bit GGUF/llama.cpp K-quant; a commonly used balanced medium variant.',
  'Q8_0':'8-bit quantization; larger than 4-bit formats and generally closer to higher-precision weights.',
  'Q6_K':'6-bit GGUF K-quant aimed at higher fidelity than lower-bit variants.',
  'Q5_K_M':'5-bit GGUF K-quant balancing size and fidelity.',
  'F16':'16-bit floating-point weights; high fidelity with a substantially larger memory footprint.',
  'BF16':'16-bit bfloat weights; commonly used as a high-precision/reference representation.',
  'MXFP4':'4-bit microscaling floating-point format used by models such as gpt-oss to reduce memory footprint.'}
 print();print(col('Model format / quantization guide',B))
 print()
 for q in sorted(quants):print(f'  {q:<8} {desc.get(q,"Quantization/weight format reported by Ollama for this benchmarked model.")}')
 if mlx:
  print('  MLX      Apple machine-learning framework/runtime optimized for Apple Silicon and unified memory;')
  print('           included because at least one benchmarked model is an MLX variant.')
 print('  Note     Model architecture/family and quantization are separate concepts.')
 print()


def markdown_model_terms(rs):
 quants=benchmark_formats(rs); mlx=any('mlx' in r.base_model.lower() for r in rs if not r.error)
 if not quants and not mlx:return []
 desc={'NVFP4':'4-bit floating-point quantization supported by Ollama’s MLX engine on Apple Silicon.','Q4_K_M':'4-bit GGUF/llama.cpp K-quant; a commonly used balanced medium variant.','Q8_0':'8-bit quantization; larger than 4-bit formats and generally closer to higher-precision weights.','Q6_K':'6-bit GGUF K-quant aimed at higher fidelity than lower-bit variants.','Q5_K_M':'5-bit GGUF K-quant balancing size and fidelity.','F16':'16-bit floating-point weights; high fidelity with a substantially larger memory footprint.','BF16':'16-bit bfloat weights; commonly used as a high-precision/reference representation.','MXFP4':'4-bit microscaling floating-point format used by models such as gpt-oss.'}
 out=['','## Model format / quantization guide','']
 for q in sorted(quants):out.append(f'- **{q}** — {desc.get(q,"quantization/weight format reported by Ollama for this benchmarked model.")}')
 if mlx:out.append('- **MLX** — Apple machine-learning framework/runtime optimized for Apple Silicon and unified memory; included because an MLX model was benchmarked.')
 out.append('- **Architecture vs quantization** — model family/architecture and weight representation are separate concepts.')
 return out


def table(rs):
 banner()
 print(col('Devbits Bench — Custom Results',B))
 print_model_terms(rs)
 print(col('Column guide',B))
 print('  CTX     Configured maximum context window.')
 print('  Run     Execution type; Cold unloads first, Warm reuses the loaded model, Custom is one isolated configured workload.')
 print('  Fill    Approximate synthetic occupied-context target.')
 print('  Prompt  Prompt/prefill processing speed in tokens per second.')
 print('  Decode  Generated-output speed in tokens per second.')
 print('  Load    Model load/setup time reported by Ollama.')
 print('  Total   End-to-end request duration.')
 print()
 h=f"{'MODEL':30} {'CTX':>7} {'RUN':>10} {'THINK':>7} {'FILL':>7} {'PROMPT':>11} {'DECODE':>11} {'LOAD':>8} {'TOTAL':>8}"
 print(h)
 print('-'*len(h))
 for r in rs:
  if r.error:continue
  print(f"{r.base_model[:30]:30} {r.context//1024:>6}K {(r.mode+' ('+str(r.run)+')'):>10} {r.thinking:>7} {r.requested_fill//1024:>6}K {r.prompt_tps:>9.1f}t/s {r.generation_tps:>9.1f}t/s {r.load_s:>7.2f}s {r.total_s:>7.2f}s")
 print()
 print(col('Memory summary',B))
 print('  Sampled immediately before and after each benchmark request.')
 print('  On macOS, free memory alone is contextual; swap and compressed memory help show pressure.')
 print()
 mh=f"{'MODEL':30} {'CTX':>7} {'RUN':>10} {'FREE':>13} {'SWAP':>21} {'COMPRESSED':>21}"
 print(mh); print('-'*len(mh))
 for r in rs:
  if r.error:continue
  mb=r.memory_before; ma=r.memory_after
  bf=f"{mb.get('free_percent')}%" if mb.get('free_percent') is not None else 'n/a'
  af=f"{ma.get('free_percent')}%" if ma.get('free_percent') is not None else 'n/a'
  bsw=f"{mb.get('swap_used_mb'):.0f} MB" if mb.get('swap_used_mb') is not None else 'n/a'
  asw=f"{ma.get('swap_used_mb'):.0f} MB" if ma.get('swap_used_mb') is not None else 'n/a'
  bcp=f"{mb.get('compressed_mb'):.0f} MB" if mb.get('compressed_mb') is not None else 'n/a'
  acp=f"{ma.get('compressed_mb'):.0f} MB" if ma.get('compressed_mb') is not None else 'n/a'
  print(f"{r.base_model[:30]:30} {r.context//1024:>6}K {(r.mode+' ('+str(r.run)+')'):>10} {(bf+' → '+af):>13} {(bsw+' → '+asw):>21} {(bcp+' → '+acp):>21}")


def quick_table(rs):
 banner(); print(col('Devbits Bench — Quick Results',B))
 valid=[r for r in rs if r.phase=='Measured' and r.measured and not r.error]
 if not valid:print('\n  No accepted measured trials.'); return
 first=valid[0]; d=summarize_trials(valid,'generation_tps'); t=summarize_trials(valid,'ttft_s'); p=summarize_trials(valid,'prompt_tps')
 max_cache=max((r.cached_prompt_tokens/r.prompt_tokens*100 if r.prompt_tokens else 0) for r in valid)
 print()
 for line in result_identity(first): print(line)
 print('\n'+col('Performance',B)); print()
 print(f"  Decode       {d['median']:.2f} tok/s")
 if t:print(f"  TTFT         {t['median']:.2f} s")
 print(f"  Prompt       {p['median']:.2f} tok/s")
 print('\n'+col('Consistency',B)); print()
 print(f"  Decode       {d['min']:.2f} – {d['max']:.2f} tok/s")
 if t:print(f"  TTFT         {t['min']:.2f} – {t['max']:.2f} s")
 print(f"  Trials       {len(valid)} measured\n  Cache        ≤{max_cache:.1f}% ✓")
 print_model_terms(valid)


def quick_report(hw,ver,rs,engine_name='Ollama'):
 accepted=[r for r in rs if r.phase=='Measured' and r.measured and not r.error]
 a=['# Devbits Bench — Quick Results','',
    '**devbits.xyz** · **X: @devbits** · **Reddit: /r/devbits/**','',
    f'**Protocol:** `{PROTOCOL}`  ',
    f'**Corpus:** `{CORPUS}`  ',
    f'**Corpus seed:** `{CORPUS_SEED}`','',
    '## System','',
    f"- **Machine:** {hw['machine']}",f"- **Chip:** {hw['chip']}",f"- **Memory:** {hw['memory']}",
    f"- **GPU:** {hw['gpu']}",f"- **OS:** {hw['os']}",f"- **{engine_name}:** {ver}"]
 a+=markdown_model_terms(rs)
 if accepted:
  d=summarize_trials(accepted,'generation_tps'); t=summarize_trials(accepted,'ttft_s')
  at=summarize_trials(accepted,'answer_ttft_s'); pp=summarize_trials(accepted,'prompt_tps')
  max_cache=max((r.cached_prompt_tokens/r.prompt_tokens*100 if r.prompt_tokens else 0) for r in accepted)
  a+=['','## Quick summary','',
      f"- **Accepted measured trials:** {len(accepted)}",
      f"- **Decode median:** {d['median']:.2f} tok/s (min {d['min']:.2f}, max {d['max']:.2f}, stddev {d['stddev']:.2f})",
      f"- **TTFT median:** {t['median']:.2f} s (min {t['min']:.2f}, max {t['max']:.2f}, stddev {t['stddev']:.2f})" if t else '- **TTFT median:** n/a',
      f"- **TTA median:** {at['median']:.2f} s" if at else '- **TTA median:** n/a (no visible answer within output budget)',
      f"- **Prompt median:** {pp['median']:.2f} tok/s",
      f"- **Maximum accepted prompt cache:** {max_cache:.2f}% (limit {CACHE_ACCEPT_RATIO*100:.0f}%)"]
 a+=['','## Raw trials','',
     '| Model | CTX | Input | Run | TTFT | TTA | Prompt tok/s | Decode tok/s | Cache | Status |',
     '|---|---:|---:|---|---:|---:|---:|---:|---:|---|']
 for r in rs:
  tt=f'{r.ttft_s:.2f}s' if r.ttft_s is not None else 'n/a'
  ta=f'{r.answer_ttft_s:.2f}s' if r.answer_ttft_s is not None else 'n/a'
  cr=(r.cached_prompt_tokens/r.prompt_tokens*100) if r.prompt_tokens else 0
  status='Rejected' if r.error else ('Discarded warmup' if not r.measured else 'Accepted')
  a.append(f"| `{r.base_model}` | {r.context:,} | {r.prompt_tokens:,} | {r.phase} ({r.run}) | {tt} | {ta} | {r.prompt_tps:.2f} | **{r.generation_tps:.2f}** | {r.cached_prompt_tokens} ({cr:.2f}%) | {status} |")
 a+=['','## Memory summary','',
     '| Model | Run | Free | Swap | Compressed |',
     '|---|---|---:|---:|---:|']
 for r in rs:
  mb=r.memory_before; ma=r.memory_after
  def pct(x): return f'{x}%' if x is not None else 'n/a'
  def memv(x): return f'{x:.0f} MB' if x is not None else 'n/a'
  a.append(f"| `{r.base_model}` | {r.phase} ({r.run}) | {pct(mb.get('free_percent'))} → {pct(ma.get('free_percent'))} | {memv(mb.get('swap_used_mb'))} → {memv(ma.get('swap_used_mb'))} | {memv(mb.get('compressed_mb'))} → {memv(ma.get('compressed_mb'))} |")
 a+=['','## Methodology','',
     '- Baseline input is deterministic, meaningful structured text generated from a versioned corpus and trial-specific seed.',
     '- Each trial changes its deterministic input to prevent substantial cross-request prompt-cache reuse.',
     f'- Measured trials are accepted when cached prompt tokens are ≤ {CACHE_ACCEPT_RATIO*100:.0f}% of actual input tokens.',
     '- Prompt throughput is calculated from uncached prompt tokens and Ollama prompt evaluation duration.',
     '- TTFT is client-observed wall time from request start to the first streamed generated token, including reasoning tokens.',
     '- TTA is client-observed wall time to the first visible answer token and may be unavailable for reasoning models.',
     '- Decode throughput uses Ollama output-token count and evaluation duration.',
     '- One warmup is discarded; headline Quick statistics use three accepted measured warm trials.',
     '- Memory is sampled immediately before and after every request.',
     '- Results are local measurements, not universal performance claims.']
 return '\n'.join(a)


def report(hw,ver,rs,engine_name='Ollama'):
 a=['# Devbits Bench — Custom Results','','**devbits.xyz** · **X: @devbits** · **Reddit: /r/devbits/**','','## System','',f"- **Machine:** {hw['machine']}",f"- **Chip:** {hw['chip']}",f"- **Memory:** {hw['memory']}",f"- **GPU:** {hw['gpu']}",f"- **OS:** {hw['os']}",f"- **{engine_name}:** {ver}"]
 a+=markdown_model_terms(rs)
 a+=['','## Results','','| Model | Context | Run | Thinking | Approx. fill | Actual prompt | Prompt tok/s | Decode tok/s | Load | Total |','|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
 for r in rs:
  if not r.error:a.append(f"| `{r.base_model}` | {r.context:,} | {r.mode} ({r.run}) | {r.thinking} | {r.requested_fill:,} | {r.prompt_tokens:,} | {r.prompt_tps:.2f} | **{r.generation_tps:.2f}** | {r.load_s:.2f}s | {r.total_s:.2f}s |")
 a+=['','## Memory summary','','Memory is sampled immediately before and after each benchmark request. On macOS, free memory alone is contextual, so swap and compressed memory are included as before → after values.','','| Model | Context | Run | Free | Swap | Compressed |','|---|---:|---|---:|---:|---:|']
 for r in rs:
  if r.error:continue
  mb=r.memory_before; ma=r.memory_after
  bf=f"{mb.get('free_percent')}%" if mb.get('free_percent') is not None else 'n/a'
  af=f"{ma.get('free_percent')}%" if ma.get('free_percent') is not None else 'n/a'
  bsw=f"{mb.get('swap_used_mb'):.0f} MB" if mb.get('swap_used_mb') is not None else 'n/a'
  asw=f"{ma.get('swap_used_mb'):.0f} MB" if ma.get('swap_used_mb') is not None else 'n/a'
  bcp=f"{mb.get('compressed_mb'):.0f} MB" if mb.get('compressed_mb') is not None else 'n/a'
  acp=f"{ma.get('compressed_mb'):.0f} MB" if ma.get('compressed_mb') is not None else 'n/a'
  a.append(f"| `{r.base_model}` | {r.context:,} | {r.mode} ({r.run}) | {bf} → {af} | {bsw} → {asw} | {bcp} → {acp} |")
 a+=['','### Methodology','- Context is the configured maximum context window for the generated benchmark variant.','- Approx. fill is synthetic input; Actual prompt is Ollama’s reported token count.','- Decode tok/s = output token count / eval duration.','- Run identifies the execution type: Cold unloads running Ollama models first; Warm reuses the loaded model; Custom is one isolated configured workload.','- Memory is sampled immediately before and after each request; macOS swap/compression are included when available.','- Temperature 0, seed 42, output capped at 256 tokens.','- Results are local measurements, not universal performance claims.']
 return '\n'.join(a)


def context_advisory(ctx, model_size, hw):
    if ctx < 131072:
        return True
    mem_text=hw.get("memory","")
    m=re.search(r"([0-9.]+)\\s*(GB|GiB)", mem_text, re.I)
    ram=float(m.group(1)) if m else None
    print(col("\n⚠ Large-context advisory", Y))
    print(f"  Requested context: {ctx:,} tokens ({ctx//1024}K)")
    if ram:
        print(f"  Detected memory:   {ram:g} GB")
    if model_size:
        print(f"  Model on disk:     {model_size/1024**3:.1f} GiB")
    print("  A large configured context can increase memory pressure, and actually filling")
    print("  that context can substantially increase prefill latency and memory use.")
    print("  Devbits Bench will not silently lower your requested context.")
    return input("  Continue with this context? [y/N] ").strip().lower() in ("y","yes")


def demo_memory(before, after, swap_before=529, swap_after=529, compressed_before=911, compressed_after=911):
 return (
  {'platform':'darwin','available_percent':before,'swap_used_mb':swap_before,'compressed_mb':compressed_before},
  {'platform':'darwin','available_percent':after,'swap_used_mb':swap_after,'compressed_mb':compressed_after},
 )


def demo_result(model, context, phase, run, input_tokens, ttft, prompt_tps, decode_tps,
                target=0, hits=0, total=0, passed=None, before=89, after=40,
                swap_before=529, swap_after=529, compressed_before=911, compressed_after=911):
 mb,ma=demo_memory(before,after,swap_before,swap_after,compressed_before,compressed_after)
 return Result(
  base_model=model,model=model,context=context,mode=phase,run=run,thinking='medium',
  requested_fill=target,prompt_tokens=input_tokens,cached_prompt_tokens=0,
  output_tokens=256,load_s=0.01,prompt_s=(input_tokens/prompt_tps if prompt_tps else 0),
  total_s=ttft+5,prompt_tps=prompt_tps,generation_tps=decode_tps,
  memory_before=mb,memory_after=ma,ttft_s=ttft,answer_ttft_s=None,
  client_total_s=ttft+5,phase=phase,measured=True,
  checkpoint_pass=passed,checkpoint_hits=hits,checkpoint_total=total,
  benchmark_thinking='medium',
 )


def demo_pause(seconds=.45, label=None):
 """Control demo playback without affecting real benchmark timing."""
 if DEMO_SPEED=='step':
  prompt=f'  Demo paused{(" — "+label) if label else ""}. Press Enter to continue…'
  input(prompt)
  return
 factor=3.0 if DEMO_SPEED=='slow' else 1.0
 time.sleep(seconds*factor)


def demo_step_wait():
 """Show a quiet, removable control line for deterministic screenshot mode."""
 input(col('    [Enter] next screenshot state',D))
 clear_terminal_lines(1)


def demo_step_state(message):
 """Render one deterministic screenshot state and wait without animation."""
 print(f'  ◐ {message}')
 demo_step_wait()
 clear_terminal_lines(1)


def demo_checkpoint(label):
 """Pause only at screenshot-worthy states in step mode."""
 if DEMO_SPEED=='step':
  input(f'  Demo paused — {label}. Press Enter to continue…')


def demo_quick(model='qwen3.8-27b-mlx-128k:latest', render_results=True):
 print('\n'+col('Quick benchmark',B)); print()
 print(f'  {model} • thinking=medium')
 print('  1 cold baseline • 1 warmup • 3 measured trials')
 print()
 rows=[('Cold baseline',7.5,39),('Warmup',3.4,39),('Measured 1/3',3.4,41),('Measured 2/3',3.6,43),('Measured 3/3',3.8,38)]
 completed=[]
 for i,(label,ttft,decode) in enumerate(rows,1):
  if DEMO_SPEED=='step':
   quick_progress_block(completed,i-1,5)
   demo_step_state(f'{label} • processing input…')
   clear_terminal_lines(2+len(completed))
   quick_progress_block(completed,i-1,5)
   demo_step_state(f'{label} • generating tokens…')
   clear_terminal_lines(2+len(completed))
  else:
   clear_terminal_lines(2+len(completed)) if i>1 else None
   quick_progress_block(completed,i-1,5)
   with Spinner(f'{label} • processing input…') as status:
    demo_pause(.30); status.update(f'{label} • generating tokens…'); demo_pause(.35)
  if i<=2: completed.append(f'  ✓ {label}')
  else: completed.append(f'  ✓ {label}   TTFT {ttft:.1f}s     Decode {decode:>2} t/s')
  if DEMO_SPEED=='step':
   quick_progress_block(completed,i,5)
   demo_step_wait()
   clear_terminal_lines(2+len(completed))
 if DEMO_SPEED!='step':
  clear_terminal_lines(2+len(completed))
 quick_progress_block(completed,5,5)

 rs=[demo_result(model,131072,'Measured',1,648,3.38,185.0,40.7,before=40,after=39),demo_result(model,131072,'Measured',2,647,3.64,179.9,43.4,before=39,after=40),demo_result(model,131072,'Measured',3,649,3.79,172.4,37.8,before=40,after=38)]
 if render_results: print(); quick_table(rs)
 return rs


def demo_practical(model='qwen3.8-27b-mlx-128k:latest', render_results=True):
 print('\n'+col('Practical benchmark',B)); print()
 print('  4K → 8K → 16K → 32K • isolated interactions • thinking=medium')
 vals=[(4096,3836,23.4,199,47,90,37),(8192,7566,46.7,177,42,90,38),(16384,15025,90.2,175,32,90,37),(32768,29875,196.1,156,31,91,32)]
 rs=[]; done=0; completed=[]; rendered=0; print()
 for i,(target,inp,ttft,prompt,decode,bef,aft) in enumerate(vals,1):
  if rendered: clear_terminal_lines(rendered)
  rendered=quick_progress_block(completed,done,4)
  if DEMO_SPEED=='step':
   demo_step_state(f'Processing ~{target//1024}K input…')
   clear_terminal_lines(rendered); rendered=quick_progress_block(completed,done,4)
   demo_step_state('Generating tokens…')
   clear_terminal_lines(rendered); rendered=quick_progress_block(completed,done,4)
  else:
   with Spinner(f'Processing ~{target//1024}K input…') as status:
    demo_pause(.28); status.update('Generating tokens…'); demo_pause(.32)
  r=demo_result(model,131072,'Practical',i,inp,ttft,prompt,decode,target=target,before=bef,after=aft); rs.append(r); done+=1
  completed.append(f'  ✓ {target//1024:>2}K   TTFT {human_duration(ttft):<9} Prompt {prompt:>5.0f} t/s   Decode {decode:>4.0f} t/s')
  if DEMO_SPEED=='step':
   clear_terminal_lines(rendered); rendered=quick_progress_block(completed,done,4); demo_step_wait()
 if rendered: clear_terminal_lines(rendered)
 quick_progress_block(completed,4,4)
 print(); print('  ✓ Practical benchmark complete')
 if render_results: print(); practical_table(rs)
 return rs


def demo_stress(model='qwen3.8-27b-mlx-128k:latest', pressure=True, render_results=True):
 print('\n'+col('Stress benchmark',B)); print()
 print(f'  {model} • context=128K • thinking=medium')
 print('  25% → 50% → 75% → 90% • isolated integrity + decode workloads')
 print('\n  Baseline calibration')
 print('  1 cold baseline • 1 warmup • 3 measured trials')
 if DEMO_SPEED=='step':
  demo_step_state('Baseline calibration • generating tokens…')
 else:
  with Spinner('Baseline calibration • generating tokens…'): demo_pause(.55)
 print('  ✓ Baseline calibrated   TTFT 3.5s   Decode 39 t/s')

 vals=[(32768,208.9,152,29,89,38,529,529),(65536,474.6,134,24,89,27,529,2501),(98304,894.6,107,20,89,24,529,2929),(117964,1260.9,87,14,89,22,529,3200)]
 rs=[]; done=0; completed=[]; rendered=0; print()
 for i,(target,ttft,prompt,decode,bef,aft,sw0,sw1) in enumerate(vals,1):
  if rendered: clear_terminal_lines(rendered)
  rendered=stress_progress_block(completed,done,4)
  print(f'  Stress stage {i}/4 • ~{target//1024}K ({[25,50,75,90][i-1]}% of configured context)'); print()
  stage_extra=2
  if DEMO_SPEED=='step':
   demo_step_state(f'Integrity check • processing ~{target//1024}K input…')
   demo_step_state('Integrity check • generating tokens…')
  else:
   with Spinner(f'Integrity check • processing ~{target//1024}K input…') as status:
    demo_pause(.25); status.update('Integrity check • generating tokens…'); demo_pause(.28)
  pre=demo_result(model,131072,'Long Prefill',i,target-1000,ttft+4,prompt,decode+3,target=target,hits=4,total=4,passed=True,before=bef,after=aft,swap_before=sw0,swap_after=sw1)
  if DEMO_SPEED=='step':
   demo_step_state(f'Decode test • processing ~{target//1024}K input…')
   demo_step_state('Decode test • generating tokens…')
  else:
   with Spinner(f'Decode test • processing ~{target//1024}K input…') as status:
    demo_pause(.23); status.update('Decode test • generating tokens…'); demo_pause(.27)
  dec=demo_result(model,131072,'Long Decode',i,target-1000,ttft,prompt,decode,target=target,before=bef,after=aft,swap_before=sw0,swap_after=sw1)
  rs.extend([pre,dec]); done+=1
  completed.append(stress_stage_summary(target,pre,dec))
  clear_terminal_lines(rendered+stage_extra); rendered=stress_progress_block(completed,done,4)
  if DEMO_SPEED=='step':
   demo_step_wait()
  if pressure and i in (2,3):
   if rendered:
    clear_terminal_lines(rendered)
    for line in completed: print(line)
    rendered=0
   print(); print(col('  ⚠ Memory pressure detected',Y)); print()
   print(f'    Available memory   {bef}% → {aft}%\n    Swap               {sw0} MB → {sw1} MB\n    Compressed         911 MB → {411 if i==2 else 420} MB')
   print(f'\n    Next workload: ~{vals[i][0]//1024}K')
   if DEMO_SPEED=='step':
    print('    Continue? [y/N] y   (demo)')
    demo_step_wait()
   else: print('    Continue? [y/N] y   (demo)')
   print()
 if rendered: clear_terminal_lines(rendered)
 stress_progress_block(completed,done,4)
 print(); print('  ✓ Stress benchmark complete')
 if render_results: print(); full_table(rs)
 return rs


def demo_custom(model='qwen3.8-27b-mlx-128k:latest', render_results=True):
 print('\n'+col('Custom benchmark',B)); print()
 context=131072; target=int(context*.90)
 print(f'  {model} • thinking=medium')
 print(f'  Workloads  ~{target//1024}K @ {context//1024}K')
 print('  One isolated interaction per workload • 10% headroom by default')
 completed=[]; rendered=0; print()
 rendered=quick_progress_block(completed,0,1)
 if DEMO_SPEED=='step':
  demo_step_state(f'Processing ~{target//1024}K input…')
  clear_terminal_lines(rendered); rendered=quick_progress_block(completed,0,1)
  demo_step_state('Generating tokens…')
  clear_terminal_lines(rendered); rendered=quick_progress_block(completed,0,1)
 else:
  with Spinner(f'Processing ~{target//1024}K input…') as status:
   demo_pause(.40); status.update('Generating tokens…'); demo_pause(.45)
 r=demo_result(model,context,'Custom',1,target-1024,735.4,105,19,target=target,before=89,after=25,swap_before=529,swap_after=2870)
 completed.append(f'  ✓ ~{target//1024}K @ {context//1024}K   TTFT {human_duration(r.ttft_s):<9} Prompt {r.prompt_tps:>5.0f} t/s   Decode {r.generation_tps:>4.0f} t/s')
 if rendered: clear_terminal_lines(rendered)
 quick_progress_block(completed,1,1)
 print(); print('  ✓ Custom benchmark complete')
 if render_results: print(); table([r])
 return [r]


def demo_multi():
 models=[
  ('qwen3.8-27b-mlx-128k:latest',1.00),
  ('qwen3.8-27b-128k:latest',2.45),
  ('deepseek-r1:14b',.78),
 ]
 rs=[]
 for model,mult in models:
  for target,base_tt,prompt,decode in [(32768,208.9,152,29),(65536,474.6,134,24)]:
   pre=demo_result(model,131072,'Long Prefill',1,target-1000,base_tt*mult,prompt/max(mult,.6),decode/max(mult,.7),target=target,hits=4,total=4,passed=True)
   dec=demo_result(model,131072,'Long Decode',1,target-1000,base_tt*mult,prompt/max(mult,.6),decode/max(mult,.7),target=target)
   rs.extend([pre,dec])
 full_table(rs)


def run_demo(which):
 print(col('UI demo mode',B))
 print('  No Ollama inference requests will be made.')
 if DEMO_SPEED=='slow': print('  Playback: slow (screenshot-friendly)')
 elif DEMO_SPEED=='step': print('  Playback: step-by-step (press Enter at checkpoints)')
 if which=='menu':
  print()
  print('  1  Quick')
  print('  2  Practical')
  print('  3  Stress')
  print('  4  Stress + memory warning')
  print('  5  Custom')
  print('  6  Multi-model results')
  print('  7  All')
  print()
  choice=input('Select demo [1]: ').strip() or '1'
  print()
  which={'1':'quick','2':'practical','3':'stress','4':'stress-pressure','5':'custom','6':'multi','7':'all'}.get(choice,'quick')
 if which=='quick': demo_quick()
 elif which=='practical': demo_practical()
 elif which=='stress': demo_stress(pressure=False)
 elif which=='stress-pressure': demo_stress(pressure=True)
 elif which=='custom': demo_custom()
 elif which=='multi': demo_multi()
 elif which=='all':
  demo_quick(); demo_practical(); demo_stress(pressure=True); demo_custom(); demo_multi()


def parsevals(s):
 out=[]
 for p in s.split(','):
  m=re.fullmatch(r'\s*(\d+)\s*([km]?)\s*',p.lower())
  if not m:raise SystemExit(f'Invalid size: {p}')
  n=int(m.group(1)); suffix=m.group(2)
  # Interactive context sizes are conventionally expressed in K tokens.
  # Treat a bare human-scale value such as 32/64/128/256 as K, while still
  # accepting explicit raw token counts such as 32768 or 131072.
  if not suffix and n <= 1024:
   n *= 1024
  elif suffix:
   n *= {'k':1024,'m':1048576}[suffix]
  out.append(n)
 return out


def choose(models):
 print(col('Available models',B))
 print()
 for i,model in enumerate(models,1):
  print(f"  {col(str(i).rjust(2),C)}  {model.display_name:<34} {hb(model.size_bytes):<10} {model.parameter_count or '?':<7} {model.quantization or '?'}")
 raw=input("\nSelect model numbers (comma-separated, or 'all'): ").strip().lower()
 print()
 if raw=='all':return [model.id for model in models]
 ids=[]
 for part in raw.split(','):
  try: index=int(part.strip())
  except ValueError: raise SystemExit('Invalid selection')
  if not 1<=index<=len(models):raise SystemExit('Invalid selection')
  ids.append(index-1)
 return [models[index].id for index in dict.fromkeys(ids)]

def main():
 p=argparse.ArgumentParser(description='Devbits Bench — practical local-model benchmark'); p.add_argument('--version', action='version', version=f'%(prog)s {VERSION}'); p.add_argument('--mode',choices=['quick','practical','stress','custom']); p.add_argument('--models');p.add_argument('--contexts');p.add_argument('--thinking');p.add_argument('--warm-runs',type=int,default=1,help=argparse.SUPPRESS);p.add_argument('--fill',default=None,help='Custom workload target tokens (one value or one per context; default: 90%% of context)');p.add_argument('--yes',action='store_true');p.add_argument('--verbose',action='store_true');p.add_argument('--no-ansi',action='store_true');p.add_argument('--demo',nargs='?',const='menu',choices=['menu','quick','practical','stress','stress-pressure','custom','multi','all'],help='preview terminal UI without running Ollama inference'); p.add_argument('--demo-speed',choices=['normal','slow','step'],default='normal',help='demo playback: normal, slow, or pause at screenshot checkpoints'); a=p.parse_args()
 global DEMO_SPEED; ui.VERBOSE=a.verbose; ui.NO_ANSI=a.no_ansi; DEMO_SPEED=a.demo_speed
 if a.demo:
  run_demo(a.demo)
  return
 banner()
 mode=a.mode
 if mode is None:
  print(col('Benchmark mode',B))
  print()
  print('  1  Quick      Fast repeatable baseline')
  print('  2  Practical  Everyday usability at 4K, 8K, 16K and 32K')
  print('  3  Stress     Push context toward the configured limit')
  print('  4  Custom     Choose your own context workloads')
  print()
  choice=input('Select mode [1]: ').strip() or '1'
  print()
  mode=(
   'practical' if choice=='2'
   else 'stress' if choice=='3'
   else 'custom' if choice=='4'
   else 'quick'
  )
 engine=_engine()
 if not engine.available(): raise SystemExit(f'{engine.name} engine is not available')
 hw=hardware(); ver=engine.version(); print(col('System detected',B)); print();
 for k,v in hw.items():print(f'  {k.capitalize():8}: {v}')
 print(f'  {engine.name.capitalize():8}: {ver}\n')
 ms=engine.list_models()
 selected=[x.strip() for x in a.models.split(',')] if a.models else choose(ms)
 if mode in ('quick','practical'):
  ctxs=[]
  fills=[]
 else:
  ctxs=parsevals(a.contexts) if a.contexts else parsevals(input('Context windows [32k,64k,128k]: ').strip() or '32k,64k,128k')
  # Stress derives its own 25/50/75/90% workloads. Custom instead treats the
  # selected contexts as the user's workload envelopes and runs one isolated
  # interaction per context. Leave 10% headroom by default for generation and
  # model/runtime overhead; --fill remains available for an explicit target.
  if mode=='custom':
   if a.fill is None:
    fills=[max(1,int(ctx*0.90)) for ctx in ctxs]
   else:
    fills=parsevals(a.fill)
    fills=fills*len(ctxs) if len(fills)==1 else fills
    if len(fills)!=len(ctxs):raise SystemExit('--fill must have one value or one per context')
  else:
   fills=[]
 if not selected: raise SystemExit('No models selected')
 first=engine.inspect_model(selected[0]); default=first.default_reasoning; vals=list(first.reasoning_modes)
 think=a.thinking if a.thinking is not None else (input(f'Thinking {vals or "not advertised"} [{default}]: ').strip() or str(default).lower())
 if mode=='practical':
  rs=[]
  for base in selected:
   info=engine.inspect_model(base)
   current_ctx=info.configured_context
   if not current_ctx:
    print(col('  ✗ ',X)+f'Could not determine configured context for {base}.')
    continue
   supported=[x for x in PRACTICAL_STAGES if x < current_ctx]
   skipped=[x for x in PRACTICAL_STAGES if x >= current_ctx]
   print('\n'+col('Practical configuration',B))
   print()
   print(f'  Model              {base}')
   print(f'  Configured context {current_ctx//1024}K')
   print(f'  Thinking           {think}')
   print('  Workloads          '+(' · '.join(f'{x//1024}K' for x in supported) if supported else 'none'))
   if skipped:print(col('  i ',C)+'Skipped by capacity: '+' · '.join(f'{x//1024}K' for x in skipped))
   if not supported:
    print(col('  ✗ ',X)+'No Practical V1 workload fits safely inside this configured context.')
    continue
   prepared=engine.prepare_model(info,current_ctx)
   try: rs.extend(practical_bench(engine,prepared,think))
   finally: engine.release(prepared)
  practical_table(rs)
  stamp=time.strftime('%Y%m%d-%H%M%S')
  md=Path(f'devbits-bench-{stamp}.md');js=Path(f'devbits-bench-{stamp}.json')
  md.write_text(quick_report(hw,ver,rs,engine.name.capitalize()))
  write_json_report(js, protocol=PRACTICAL_PROTOCOL, suite='practical', corpus=CORPUS, corpus_seed=CORPUS_SEED, output_budget_tokens=512, context_integrity_check=False, system=hw, engine_name=engine.name, engine_version=ver, results=rs)
  print('\n'+col('Reports written:',B)+f'\n  {md}\n  {js}')
  return

 if mode=='stress':
  rs=[]
  for base in selected:
   info=engine.inspect_model(base); current_ctx=info.configured_context
   ctx=(ctxs[0] if ctxs else current_ctx or 32768)
   if current_ctx!=ctx:
    print(col('  i ',C)+f'Stress requested {ctx//1024}K; selected model is '+(f'{current_ctx//1024}K.' if current_ctx else 'not explicitly configured.'))
    if input('Prepare requested context? [Y/n] ').strip().lower() in ('n','no'):continue
   prepared=engine.prepare_model(info,ctx)
   try: rs.extend(full_bench(engine,prepared,think))
   finally: engine.release(prepared)
  full_table(rs)
  stamp=time.strftime('%Y%m%d-%H%M%S');md=Path(f'devbits-bench-{stamp}.md');js=Path(f'devbits-bench-{stamp}.json')
  md.write_text(quick_report(hw,ver,rs,engine.name.capitalize()))
  write_json_report(js, protocol=PROTOCOL, suite='stress', memory_pressure_policy=pressure_policy_description(sys.platform), corpus=CORPUS, corpus_seed=CORPUS_SEED, system=hw, engine_name=engine.name, engine_version=ver, results=rs)
  print('\n'+col('Reports written:',B)+f'\n  {md}\n  {js}')
  return
 if mode=='quick':
  rs=[]
  for base in selected:
   info=engine.inspect_model(base); current_ctx=info.configured_context
   ctx=(ctxs[0] if ctxs else current_ctx or 32768)
   if current_ctx!=ctx:
    print(col('  i ',C)+f'Quick requested {ctx//1024}K; selected model is '+(f'{current_ctx//1024}K.' if current_ctx else 'not explicitly configured.'))
    if input('Prepare requested context? [Y/n] ').strip().lower() in ('n','no'):continue
   prepared=engine.prepare_model(info,ctx)
   try: rs.extend(quick_bench(engine,prepared,think))
   finally: engine.release(prepared)
  quick_table(rs)
  stamp=time.strftime('%Y%m%d-%H%M%S'); md=Path(f'devbits-bench-{stamp}.md'); js=Path(f'devbits-bench-{stamp}.json')
  md.write_text(quick_report(hw,ver,rs,engine.name.capitalize())); write_json_report(js, protocol=PROTOCOL, memory_pressure_policy=pressure_policy_description(sys.platform), corpus=CORPUS, corpus_seed=CORPUS_SEED, system=hw, engine_name=engine.name, engine_version=ver, results=rs)
  print('\n'+col('Reports written:',B)+f'\n  {md}\n  {js}')
  return
 rs=[]; handles=[]
 try:
  for base in selected:
   print('\n'+col('Custom configuration',B)); print()
   info=engine.inspect_model(base)
   native=info.advertised_context
   current_ctx=info.configured_context
   size=next((m.size_bytes for m in ms if m.id==base),None)
   print(f'  Model              {base}')
   if native: print(f'  Advertised context {native//1024}K')
   if current_ctx: print(f'  Configured context {current_ctx//1024}K')
   print(f'  Thinking           {think}')
   print('  Workloads          '+' · '.join(f'~{fill//1024}K @ {ctx//1024}K' for ctx,fill in zip(ctxs,fills)))
   print('  Execution          one isolated interaction per workload')
   prepared=[]
   for ctx,fill in zip(ctxs,fills):
    if fill >= ctx:
     print(col('  ✗ ',X)+f'Workload ~{fill//1024}K must be smaller than its {ctx//1024}K context window.')
     continue
    if native and ctx>native: print(col(f'  ⚠ Requested {ctx:,} exceeds advertised context {native:,}.',Y))
    if not a.yes and not context_advisory(ctx,size,hw):
     print(col('  ↷ Skipped by user.',D)); continue
    if current_ctx == ctx:
     print(col('  ✓ ',G)+f'Reusing selected model directly — already configured for {ctx//1024}K.')
    else:
     msg=(f'Selected model is configured for {current_ctx//1024}K; ' if current_ctx else 'Selected model has no explicit context window; ')
     print(col('  i ',C)+msg+f'{ctx//1024}K requires context preparation.')
     allow=True
     if not a.yes:
      print(col('    The engine will prepare the requested context and release preparation resources after testing.',D))
      allow=input('    Prepare this benchmark context? [Y/n] ').strip().lower() not in ('n','no')
     if not allow:
      print(col('  ↷ Skipped this context; original model left unchanged.',D)); continue
    handle=engine.prepare_model(info,ctx); handles.append(handle)
    prepared.append((handle,fill))
   if prepared:
    rs.extend(custom_bench(engine,base,prepared,think))
 finally:
  if handles: print('\n'+col('Releasing benchmark preparation resources',B))
  for handle in reversed(handles): engine.release(handle)
 table(rs); stamp=time.strftime('%Y%m%d-%H%M%S'); md=Path(f'devbits-bench-{stamp}.md'); js=Path(f'devbits-bench-{stamp}.json'); md.write_text(report(hw,ver,rs,engine.name.capitalize())); write_json_report(js, protocol=PROTOCOL, suite='custom', memory_pressure_policy=pressure_policy_description(sys.platform), corpus=CORPUS, corpus_seed=CORPUS_SEED, system=hw, engine_name=engine.name, engine_version=ver, results=rs); print('\n'+col('Reports written:',B)+f'\n  {md}\n  {js}')


