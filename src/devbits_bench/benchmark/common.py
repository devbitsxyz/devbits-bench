"""Shared measurements and the unchanged v0.1 memory-pressure policy."""
import sys
import time

from ..engines.base import Engine, GenerationResult, PreparedModel
from ..system.metrics import memory_snapshot
from ..ui.terminal import C, D, col, ui_verbose


def attach_generation_evidence(result, generation: GenerationResult):
    """Attach report-only evidence without changing the historical Result dataclass.

    Keeping this as dynamic sidecar data preserves the frozen runner/result fixtures
    and Ollama compatibility schema while allowing multi-engine reports to expose
    native and Devbits measurement provenance.
    """
    result._measurement_provenance = dict(generation.metrics.provenance)
    result._native_metrics = dict(generation.native_metrics)
    result._requested_reasoning = generation.requested_reasoning
    result._effective_reasoning = generation.effective_reasoning
    return result


def result_fields(generation: GenerationResult):
    """Map normalized measurements to the historical Result schema.

    Unsupported measurements must be qualified before enabling another engine;
    do not silently invent zero-valued observations for a future adapter.
    """
    required = ("prompt_tokens", "cached_prompt_tokens", "output_tokens", "load_s",
                "prompt_s", "total_s", "prompt_tps", "generation_tps")
    fields = {name: getattr(generation.metrics, name) for name in required}
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise ValueError("Engine did not supply required protocol metrics: " + ", ".join(missing))
    fields.update({name: getattr(generation.metrics, name)
                   for name in ("ttft_s", "answer_ttft_s", "client_total_s")})
    return fields

def stabilize_for_long_test(engine: Engine, prepared: PreparedModel, seconds=3.0):
 # Long-context interactions must be prompt/KV isolated. Some engines (currently
 # MLX-LM) guarantee that isolation per generate() while retaining model weights;
 # others keep the historical reset-before-interaction behavior.
 keeps_residency = getattr(engine, 'isolated_generation_keeps_residency', None)
 if not (callable(keeps_residency) and keeps_residency()):
  engine.reset(prepared)
 time.sleep(seconds)
 return memory_snapshot()


def memory_delta(before,after):
 def delta(k):
  a=before.get(k); b=after.get(k)
  return None if a is None or b is None else b-a
 return {'free_percent':delta('free_percent'),'swap_used_mb':delta('swap_used_mb'),'compressed_mb':delta('compressed_mb')}


def pressure_warning(before,after):
 reasons=[]
 platform=after.get('platform') or before.get('platform') or sys.platform
 ba=before.get('available_percent'); aa=after.get('available_percent')
 bs=before.get('swap_used_mb'); ass=after.get('swap_used_mb')
 if platform=='darwin':
  bc=before.get('compressed_mb'); ac=after.get('compressed_mb')
  if aa is not None and aa<=8:reasons.append(f'macOS available memory fell to {aa:.0f}%')
  if bs is not None and ass is not None and ass-bs>=512:reasons.append(f'macOS swap increased by {ass-bs:.0f} MB')
  if bc is not None and ac is not None and ac-bc>=1536:reasons.append(f'macOS compressed memory increased by {ac-bc:.0f} MB')
 elif platform.startswith('linux'):
  some=after.get('psi_memory_some_avg10'); full=after.get('psi_memory_full_avg10')
  if aa is not None and aa<=8:reasons.append(f'Linux memory available fell to {aa:.1f}%')
  if bs is not None and ass is not None and ass-bs>=512:reasons.append(f'Linux swap increased by {ass-bs:.0f} MB')
  if full is not None and full>=1.0:reasons.append(f'Linux severe memory stalls reached {full:.2f}% over the recent 10-second window')
  if some is not None and some>=10.0:reasons.append(f'Linux memory stalls reached {some:.2f}% over the recent 10-second window')
 else:
  if aa is not None and aa<=8:reasons.append(f'available memory fell to {aa:.1f}%')
  if bs is not None and ass is not None and ass-bs>=512:reasons.append(f'swap increased by {ass-bs:.0f} MB')
 return reasons


def pressure_policy_description(platform):
 if platform=='darwin':return 'macOS: available memory + swap/compression deltas'
 if platform.startswith('linux'):return 'Linux: available memory + swap + memory stalls'
 return 'Portable fallback: available memory + swap delta'


