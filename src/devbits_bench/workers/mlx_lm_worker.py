"""Optional native MLX-LM worker: discovery, residency and tiny generation."""
from __future__ import annotations
import gc, importlib.metadata, importlib.util, json, os, platform, sys
from pathlib import Path
from typing import Any

PROTOCOL_VERSION=1; WORKER_VERSION="0.3"
_MODEL=None; _TOKENIZER=None; _MODEL_ID=None; _MODEL_PATH=None

def _version(d):
    try:return importlib.metadata.version(d)
    except importlib.metadata.PackageNotFoundError:return None

def _emit(m):sys.stdout.write(json.dumps(m,separators=(",",":"),ensure_ascii=False)+"\n");sys.stdout.flush()
def _error(i,m,kind="worker_error"):_emit({"v":PROTOCOL_VERSION,"id":i,"event":"error","kind":kind,"message":m})

def _hf_hub_root()->Path:
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):return Path(os.environ["HF_HOME"]).expanduser()/"hub"
    return Path.home()/".cache"/"huggingface"/"hub"

def _snapshot(repo_dir:Path)->Path|None:
    ref=repo_dir/"refs"/"main"
    if ref.is_file():
        p=repo_dir/"snapshots"/ref.read_text().strip()
        if p.is_dir():return p
    snaps=repo_dir/"snapshots"
    choices=sorted((p for p in snaps.iterdir() if p.is_dir()),key=lambda p:p.stat().st_mtime,reverse=True) if snaps.is_dir() else []
    return choices[0] if choices else None

def _json(path:Path)->dict:
    try:return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError,json.JSONDecodeError):return {}

def _reasoning(tok:dict)->tuple[list[str],str]:
    # chat_template may be a string, mapping or list depending on tokenizer
    # serialization. JSON preserves nested template text more reliably than
    # assuming one concrete representation.
    raw=tok.get("chat_template")
    try: template=json.dumps(raw,ensure_ascii=False) if not isinstance(raw,str) else raw
    except (TypeError,ValueError): template=str(raw or "")
    if "enable_thinking" not in template and "reasoning_effort" not in template:return [],"false"
    if "reasoning_effort" in template:return ["false","low","medium","xhigh"],"xhigh"
    return ["false","true"],"true"

def _runtime_reasoning(tokenizer, fallback_modes:list[str], fallback_default:str)->tuple[list[str],str]:
    if not bool(getattr(tokenizer,"has_thinking",False)):
        return fallback_modes,fallback_default
    candidates=[getattr(tokenizer,"chat_template",None)]
    inner=getattr(tokenizer,"_tokenizer",None) or getattr(tokenizer,"tokenizer",None)
    if inner is not None:candidates.append(getattr(inner,"chat_template",None))
    for template in candidates:
        modes,default=_reasoning({"chat_template":template})
        if modes:return modes,default
    # A thinking-capable tokenizer without an inspectable reasoning_effort
    # template can truthfully advertise the binary control only.
    return (["false","true"],"true") if not fallback_modes else (fallback_modes,fallback_default)

def _quant(config:dict)->str|None:
    q=config.get("quantization") or config.get("quantization_config") or {}
    if isinstance(q,str): return q.upper()
    if isinstance(q,dict):
        for k in ("quant_method","mode","quantization_method"):
            if q.get(k): return str(q[k]).upper()
        bits=q.get("bits")
        if bits:return f"{bits}-bit"
    return None

def _parameter_count(config:dict)->str|None:
    raw=config.get("num_parameters") or config.get("parameter_count")
    if isinstance(raw,str) and raw.strip():return raw.strip()
    if isinstance(raw,(int,float)) and raw>0:
        n=float(raw)
        if n>=1e9:return f"{n/1e9:.1f}B".replace(".0B","B")
        if n>=1e6:return f"{n/1e6:.1f}M".replace(".0M","M")
        return str(int(n))
    return None

def _is_text_generation_config(config:dict)->bool:
    arches=[str(x) for x in (config.get("architectures") or [])]
    suffixes=("ForCausalLM","ForConditionalGeneration","ForSeq2SeqLM","LMHeadModel")
    return bool(arches) and any(a.endswith(suffixes) for a in arches)

def _model_record(repo_id:str,snap:Path)->dict|None:
    config=_json(snap/"config.json");tok=_json(snap/"tokenizer_config.json")
    if not config or not _is_text_generation_config(config) or not any(snap.glob("*.safetensors")):return None
    arch=(config.get("architectures") or [None])[0] or config.get("model_type")
    ctx=config.get("max_position_embeddings") or tok.get("model_max_length")
    if not isinstance(ctx,(int,float)) or ctx>10_000_000:ctx=None
    modes,default=_reasoning(tok);size=sum(p.stat().st_size for p in snap.glob("*.safetensors") if p.is_file())
    return {"id":repo_id,"display_name":repo_id,"local_path":str(snap),"architecture":arch,
        "quantization":_quant(config),"parameter_count":_parameter_count(config),
        "advertised_context":int(ctx) if ctx else None,"size_bytes":size or None,
        "reasoning_modes":modes,"default_reasoning":default}

def _local_models()->list[dict]:
    root=_hf_hub_root()
    if not root.is_dir():return []
    out=[]
    for d in sorted(root.glob("models--*")):
        if not d.is_dir():continue
        repo_id=d.name[len("models--"):].replace("--","/");snap=_snapshot(d)
        if snap:
            record=_model_record(repo_id,snap)
            if record:out.append(record)
    return out

def _find(repo_id:str)->dict:
    for m in _local_models():
        if m["id"]==repo_id:return m
    raise ValueError(f"local MLX-LM model not found: {repo_id}")

def _inspect(repo_id:str)->dict:
    m=_find(repo_id).copy(); import mlx_lm
    model,tokenizer=mlx_lm.load(m["local_path"],lazy=True)
    m["runtime_model_class"]=f"{type(model).__module__}.{type(model).__name__}"
    m["runtime_tokenizer_class"]=f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
    m["has_thinking"]=bool(getattr(tokenizer,"has_thinking",False))
    modes,default=_runtime_reasoning(tokenizer,list(m.get("reasoning_modes") or []),str(m.get("default_reasoning") or "false"))
    m["reasoning_modes"]=modes;m["default_reasoning"]=default
    return m

def _load_model(repo_id:str)->dict:
    global _MODEL,_TOKENIZER,_MODEL_ID,_MODEL_PATH
    m=_find(repo_id).copy()
    if _MODEL_ID!=repo_id:
        import mlx_lm
        _MODEL,_TOKENIZER=mlx_lm.load(m["local_path"])
        _MODEL_ID=repo_id;_MODEL_PATH=m["local_path"]
    m["runtime_model_class"]=f"{type(_MODEL).__module__}.{type(_MODEL).__name__}"
    m["runtime_tokenizer_class"]=f"{type(_TOKENIZER).__module__}.{type(_TOKENIZER).__name__}"
    m["has_thinking"]=bool(getattr(_TOKENIZER,"has_thinking",False));return m

def _release_model():
    global _MODEL,_TOKENIZER,_MODEL_ID,_MODEL_PATH
    _MODEL=_TOKENIZER=_MODEL_ID=_MODEL_PATH=None;gc.collect()
    try:
        import mlx.core as mx;mx.clear_cache()
    except (ImportError,AttributeError):pass

def _template_kwargs(reasoning:str)->dict:
    if reasoning=="false":return {"enable_thinking":False}
    if reasoning in {"low","medium","xhigh"}:return {"enable_thinking":True,"reasoning_effort":reasoning}
    if reasoning=="true":return {"enable_thinking":True}
    raise ValueError(f"unsupported reasoning mode: {reasoning}")

def _initial_state(tokenizer,prompt)->str:
    if not bool(getattr(tokenizer,"has_thinking",False)):return "normal"
    try:return "reasoning" if tokenizer.rfind_think_start(prompt)>tokenizer.rfind_think_end(prompt) else "normal"
    except (AttributeError,TypeError):return "normal"

def _generate(i:str,m:dict):
    if _MODEL is None or _TOKENIZER is None:raise RuntimeError("no MLX-LM model is loaded")
    if m.get("fresh_cache") is not True:raise ValueError("Pass 5B-3 requires fresh_cache=true")
    prompt=str(m.get("prompt") or "");max_tokens=int(m.get("max_output_tokens") or 0)
    if max_tokens<=0:raise ValueError("max_output_tokens must be positive")
    temperature=float(m.get("temperature",0));seed=int(m.get("seed",42));reasoning=str(m.get("reasoning") or "false")
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.generate import make_text_state_machine
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    template_kwargs=_template_kwargs(reasoning)
    messages=[{"role":"user","content":prompt}]
    rendered=_TOKENIZER.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,**template_kwargs)
    mx.random.seed(seed)
    sampler=make_sampler(temperature)
    cache=make_prompt_cache(_MODEL) # deliberately fresh for every interaction
    text_sm=make_text_state_machine(_TOKENIZER,())
    state_name=_initial_state(_TOKENIZER,rendered);state=text_sm.make_state(initial=state_name)
    last=None;thinking=[];answer=[]
    for gen in stream_generate(_MODEL,_TOKENIZER,rendered,max_tokens=max_tokens,sampler=sampler,prompt_cache=cache):
        last=gen;before=state[0];state,visible,after=text_sm.step(state,gen.text)
        if visible:
            # Native state machine strips structural markers. In the normal
            # Qwen stream a marker transition occupies its own buffered text;
            # visible text is therefore attributed to the state that consumed it.
            channel="reasoning_delta" if before=="reasoning" else "answer_delta"
            (thinking if channel=="reasoning_delta" else answer).append(visible)
            _emit({"v":1,"id":i,"event":channel,"text":visible})
    # Flush any buffered non-marker text using an empty terminal step.
    if state is not None:
        state,visible,_=text_sm.step(state,"")
        if visible:
            channel="reasoning_delta" if state_name=="reasoning" and not answer else "answer_delta"
            (thinking if channel=="reasoning_delta" else answer).append(visible);_emit({"v":1,"id":i,"event":channel,"text":visible})
    if last is None:raise RuntimeError("MLX-LM generation produced no response")
    metrics={"prompt_tokens":last.prompt_tokens,"prompt_tps":last.prompt_tps,
        "generation_tokens":last.generation_tokens,"generation_tps":last.generation_tps,
        "peak_memory_gb":last.peak_memory,"finish_reason":last.finish_reason}
    _emit({"v":1,"id":i,"event":"complete","metrics":metrics,"effective_reasoning":reasoning,
        "fresh_prompt_cache":True})

def _handle(m:Any)->bool:
    if not isinstance(m,dict):_error(None,"request must be a JSON object","protocol_error");return True
    i=m.get("id")
    if m.get("v")!=PROTOCOL_VERSION:_error(i,"unsupported protocol version","protocol_error");return True
    op=m.get("op")
    if op=="hello":
        ml=importlib.util.find_spec("mlx_lm") is not None;mx=importlib.util.find_spec("mlx") is not None
        _emit({"v":1,"id":i,"event":"hello","worker_version":WORKER_VERSION,"python":sys.executable,
            "python_version":platform.python_version(),"platform":platform.platform(),
            "dependencies":{"mlx":_version("mlx") if mx else None,"mlx_lm":_version("mlx-lm") if ml else None},
            "capabilities":["handshake","shutdown","model_discovery","model_inspection","model_residency","generation"]});return True
    if op=="list_models":
        try:_emit({"v":1,"id":i,"event":"models","models":_local_models()})
        except Exception as e:_error(i,f"model discovery failed: {e}","model_discovery_error")
        return True
    if op=="inspect_model":
        try:_emit({"v":1,"id":i,"event":"model","model":_inspect(str(m.get("model_id") or ""))})
        except Exception as e:_error(i,f"model inspection failed: {e}","model_inspection_error")
        return True
    if op=="load_model":
        try:_emit({"v":1,"id":i,"event":"model_loaded","model":_load_model(str(m.get("model_id") or ""))})
        except Exception as e:_error(i,f"model load failed: {e}","model_load_error")
        return True
    if op=="release_model":
        try:_release_model();_emit({"v":1,"id":i,"event":"model_released"})
        except Exception as e:_error(i,f"model release failed: {e}","model_release_error")
        return True
    if op=="generate":
        try:_generate(str(i),m)
        except Exception as e:_error(i,f"generation failed: {e}","generation_error")
        return True
    if op=="shutdown":_release_model();_emit({"v":1,"id":i,"event":"shutdown"});return False
    _error(i,f"unsupported operation: {op!r}","protocol_error");return True

def main():
    for raw in sys.stdin:
        raw=raw.strip()
        if not raw:continue
        try:m=json.loads(raw)
        except json.JSONDecodeError:_error(None,"invalid JSON request","protocol_error");continue
        if not _handle(m):return 0
    return 0
if __name__=="__main__":raise SystemExit(main())
