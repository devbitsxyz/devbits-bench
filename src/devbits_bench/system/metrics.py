from __future__ import annotations
import json, platform, re, subprocess, sys
from pathlib import Path

def _run(cmd):
    return subprocess.run(cmd, text=True, capture_output=True, check=True)

def _human_bytes(n):
    if n is None:return '?'
    x=float(n)
    for u in ('B','KiB','MiB','GiB','TiB'):
        if x<1024 or u=='TiB': return f'{x:.1f} {u}'
        x/=1024

def hardware():
    if sys.platform=='darwin':
        def sp(t):
            try:return json.loads(_run(['system_profiler',t,'-json']).stdout)
            except Exception:return {}
        h=sp('SPHardwareDataType').get('SPHardwareDataType',[{}])[0]; ds=sp('SPDisplaysDataType').get('SPDisplaysDataType',[])
        gpu=', '.join(x.get('sppci_model','') for x in ds if x.get('sppci_model')) or 'Apple integrated GPU'
        return {'os':f"macOS {platform.mac_ver()[0]}",'machine':h.get('machine_name') or h.get('machine_model') or platform.machine(),'chip':h.get('chip_type','Unknown'),'memory':h.get('physical_memory','Unknown'),'gpu':gpu}
    mem='Unknown'
    try:
        kb=int(re.search(r'MemTotal:\s+(\d+)',Path('/proc/meminfo').read_text()).group(1)); mem=_human_bytes(kb*1024)
    except Exception:pass
    gpu='Unknown'
    try:gpu=_run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader']).stdout.strip() or gpu
    except Exception:pass
    return {'os':platform.platform(),'machine':platform.machine(),'chip':platform.processor() or 'Unknown','memory':mem,'gpu':gpu}

def memory_snapshot():
    snap={'platform':sys.platform,'available_percent':None,'swap_used_mb':None,'compressed_mb':None,
          'psi_memory_some_avg10':None,'psi_memory_full_avg10':None}
    if sys.platform=='darwin':
        try:
            o=subprocess.run(['memory_pressure'],text=True,capture_output=True).stdout
            m=re.search(r'System-wide memory free percentage:\s*(\d+)%',o)
            if m:snap['available_percent']=int(m.group(1))
        except Exception:pass
        try:
            o=subprocess.run(['sysctl','-n','vm.swapusage'],text=True,capture_output=True).stdout
            m=re.search(r'used\s*=\s*([0-9.]+)([MG])',o)
            if m:snap['swap_used_mb']=float(m.group(1))*(1024 if m.group(2)=='G' else 1)
        except Exception:pass
        try:
            o=subprocess.run(['vm_stat'],text=True,capture_output=True).stdout
            pm=re.search(r'page size of (\d+) bytes',o); page=int(pm.group(1)) if pm else 16384
            m=re.search(r'Pages occupied by compressor:\s*(\d+)\.',o)
            if m:snap['compressed_mb']=int(m.group(1))*page/1048576
        except Exception:pass
    elif sys.platform.startswith('linux'):
        try:
            vals={}
            for line in Path('/proc/meminfo').read_text().splitlines():
                if ':' in line:
                    k,v=line.split(':',1); m=re.search(r'(\d+)',v)
                    if m:vals[k]=int(m.group(1))
            total=vals.get('MemTotal'); avail=vals.get('MemAvailable')
            if total and avail is not None:snap['available_percent']=avail/total*100
            st=vals.get('SwapTotal'); sf=vals.get('SwapFree')
            if st is not None and sf is not None:snap['swap_used_mb']=(st-sf)/1024
        except Exception:pass
        try:
            for line in Path('/proc/pressure/memory').read_text().splitlines():
                parts=line.split()
                if not parts:continue
                d={k:float(v) for k,v in (x.split('=',1) for x in parts[1:] if '=' in x)}
                if parts[0]=='some':snap['psi_memory_some_avg10']=d.get('avg10')
                if parts[0]=='full':snap['psi_memory_full_avg10']=d.get('avg10')
        except Exception:pass
    snap['free_percent']=snap['available_percent']
    return snap
