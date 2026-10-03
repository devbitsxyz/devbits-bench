"""Dependency-free terminal presentation; no engine or methodology ownership."""
import shutil
import sys
import threading
import time

R='\033[0m'; B='\033[1m'; D='\033[2m'; C='\033[36m'; G='\033[32m'; Y='\033[33m'; X='\033[31m'; M='\033[35m'
VERBOSE=False
NO_ANSI=False

def ui_verbose(message):
 if VERBOSE: print(message)


def ui_clear():
 if sys.stdout.isatty() and not NO_ANSI:
  sys.stdout.write('\r\033[2K'); sys.stdout.flush()


def active_progress(done,total,message):
 """Prefix a one-line live status with completed-work progress."""
 return f'[{progress_bar(done,total)}] {done}/{total} • {message}'


class Spinner:
 """Dependency-free live status for long interactive operations."""
 def __init__(self,message):
  self.message=message
  self.stop=threading.Event()
  self.thread=None
  self.started=None
  self.lock=threading.Lock()

 def update(self,message):
  with self.lock:
   self.message=message

 def __enter__(self):
  self.started=time.monotonic()
  if not sys.stdout.isatty() or NO_ANSI:
   print(f'  … {self.message}')
   return self

  def run():
   frames=('◐','◓','◑','◒')
   i=0
   while not self.stop.wait(.12):
    elapsed=max(0,int(time.monotonic()-self.started))
    minutes,seconds=divmod(elapsed,60)
    clock=f'{minutes:02d}:{seconds:02d}'
    width=shutil.get_terminal_size((100,24)).columns
    with self.lock:
     message=self.message
    text=f'  {frames[i%4]} {message} • {clock} elapsed'
    sys.stdout.write('\r\033[2K'+text[:max(1,width-1)])
    sys.stdout.flush()
    i+=1

  self.thread=threading.Thread(target=run,daemon=True)
  self.thread.start()
  return self

 def __exit__(self,*_):
  self.stop.set()
  if self.thread:
   self.thread.join(.5)
  ui_clear()


def progress_bar(done,total,width=24):
 filled=round(width*done/total) if total else 0
 return '█'*filled+'░'*(width-filled)


class LiveRegion:
 """Own a redrawable terminal region without callers counting cursor lines."""
 def __init__(self):
  self.lines=0

 def render(self,lines):
  self.clear()
  for line in lines:
   print(line)
  self.lines=len(lines)

 def clear(self):
  if self.lines:
   clear_terminal_lines(self.lines)
   self.lines=0

 def finish(self,lines):
  self.render(lines)
  self.lines=0


def progress_lines(completed,done,total):
 return [
  f'  {progress_bar(done,total)}  {done}/{total}',
  '',
  *completed,
 ]


def clear_terminal_lines(count):
 """Erase the previous ``count`` printed terminal lines.

 ``print()`` leaves the cursor on the line *after* the rendered region.  The old
 implementation counted that empty cursor line as one of the lines to erase,
 which left the oldest progress-bar row behind on every redraw.  Repeated
 redraws therefore accumulated the visible staircase.
 """
 if count <= 0 or not sys.stdout.isatty() or NO_ANSI:
  return
 # Discard anything transient on the current cursor line, then erase exactly
 # ``count`` previously printed lines.  Finish at the top of the erased region
 # so the replacement block is written in-place.
 sys.stdout.write('\r\033[2K')
 for _ in range(count):
  sys.stdout.write('\033[1A\r\033[2K')
 sys.stdout.flush()


def quick_progress_block(completed, done, total):
 """Render Quick's single progress bar plus already-completed steps."""
 print(f'  {progress_bar(done,total)}  {done}/{total}')
 print()
 for line in completed:
  print(line)
 # Spinner/status owns the next terminal line.
 return 2 + len(completed)


def stress_progress_block(completed, done, total):
 """Render Stress as one persistent progress block."""
 print(f'  {progress_bar(done,total)}  {done}/{total}')
 print()
 for line in completed:
  print(line)
 return 2 + len(completed)


def stress_stage_summary(target, pre, dec):
 integrity=f'{pre.checkpoint_hits}/{pre.checkpoint_total}' if pre and pre.checkpoint_pass else 'FAIL'
 ttft=human_duration(dec.ttft_s if dec else pre.ttft_s)
 prompt=f'{pre.prompt_tps:.0f} t/s' if pre and pre.prompt_tps is not None else '-'
 decode=f'{dec.generation_tps:.0f} t/s' if dec else '-'
 return f'  ✓ ~{target//1024:<3}K   Integrity {integrity:<4}   TTFT {ttft:<10} Prompt {prompt:<9} Decode {decode}'


def col(s,x): return f'{x}{s}{R}' if sys.stdout.isatty() and not NO_ANSI else s


def banner():
 art = r"""
 ____             _     _ _          ___  _ _                       ____                  _
|  _ \  _____   _| |__ (_) |_ ___  / _ \| | | __ _ _ __ ___   __ _| __ )  ___ _ __   ___| |__
| | | |/ _ \ \ / / '_ \| | __/ __|| | | | | |/ _` | '_ ` _ \ / _` |  _ \ / _ \ '_ \ / __| '_ \ 
| |_| |  __/\ V /| |_) | | |_\__ \| |_| | | | (_| | | | | | | (_| | |_) |  __/ | | | (__| | | |
|____/ \___| \_/ |_.__/|_|\__|___/ \___/|_|_|\__,_|_| |_| |_|\__,_|____/ \___|_| |_|\___|_| |_|
"""
 print(col(art,C))
 print(col('Local model benchmark • context • thinking • cold/warm • reproducible reports',D))
 print()
 print('devbits.xyz  •  X: @devbits  •  Reddit: /r/devbits/\n')


def hb(n):
 if n is None:return '?'
 x=float(n)
 for u in ('B','KiB','MiB','GiB','TiB'):
  if x<1024 or u=='TiB': return f'{x:.1f} {u}'
  x/=1024


def human_duration(seconds):
 """Format latency compactly for humans while preserving raw seconds in JSON."""
 if seconds is None:
  return 'n/a'
 if seconds < 60:
  return f'{seconds:.1f}s'
 minutes = int(seconds // 60)
 remainder = seconds - minutes * 60
 return f'{minutes}m {remainder:04.1f}s'


def custom_summary(result):
 return (f'  ✓ ~{result.requested_fill//1024:<3}K  '
         f'TTFT {human_duration(result.ttft_s):<10} '
         f'Prompt {result.prompt_tps:.0f} t/s   Decode {result.generation_tps:.0f} t/s')


def print_pressure_warning(reasons):
 print(col('  ⚠ Memory-pressure advisory',Y))
 for reason in reasons:
  print(f'    {reason}')

