"""Discovery of an existing MLX-LM runtime without installing or mutating it."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Iterable


class MLXRuntimeError(RuntimeError):
    """MLX runtime discovery/validation failed."""


@dataclass(frozen=True)
class MLXRuntime:
    python: Path
    root: Path | None
    managed: bool = False  # compatibility field; Devbits never creates runtimes
    mlx_version: str | None = None
    mlx_lm_version: str | None = None
    backend: str | None = None
    system: str | None = None
    machine: str | None = None


def _probe_python(executable: Path, require_mlx: bool = False) -> dict | None:
    code = r'''
import importlib.metadata, json, platform, sys

def dist(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None

system = platform.system()
machine = platform.machine()
mlx = dist("mlx") or dist("mlx-cpu") or dist("mlx-cuda") or dist("mlx-cuda-12") or dist("mlx-cuda-13")
mlx_lm = dist("mlx-lm")
backend = None
if system == "Darwin" and machine == "arm64" and mlx:
    backend = "Metal"
elif system == "Linux" and mlx:
    if dist("mlx-cuda-13") or dist("mlx-cuda-12") or dist("mlx-cuda"):
        backend = "CUDA"
    elif dist("mlx-cpu"):
        backend = "CPU"
    else:
        # Source/editable installs may not expose the backend distribution name.
        backend = "Linux"
out = {"version": list(sys.version_info[:3]), "machine": machine, "system": system,
       "mlx": mlx, "mlx_lm": mlx_lm, "backend": backend}
print(json.dumps(out))
'''
    try:
        cp = subprocess.run([str(executable), "-c", code], capture_output=True, text=True, timeout=10)
        if cp.returncode != 0:
            return None
        value = json.loads(cp.stdout.strip())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    if value.get("system") not in ("Darwin", "Linux"):
        return None
    if value.get("system") == "Darwin" and value.get("machine") != "arm64":
        return None
    version = tuple(value.get("version") or ())
    if len(version) != 3 or version < (3, 10, 0):
        return None
    if require_mlx and not (value.get("mlx") and value.get("mlx_lm")):
        return None
    return value


def _python_from_script(command: str) -> Path | None:
    raw = shutil.which(command)
    if not raw:
        return None
    # Preserve the command path: resolving a venv-owned script can escape the venv.
    script = Path(raw).expanduser().absolute()
    try:
        first = script.open("r", encoding="utf-8", errors="ignore").readline().strip()
    except (OSError, UnicodeError):
        return None
    if first.startswith("#!"):
        interpreter = first[2:].strip().split()
        if interpreter:
            p = Path(interpreter[0]).expanduser()
            if p.name == "env" and len(interpreter) > 1:
                found = shutil.which(interpreter[1])
                return Path(found).expanduser().absolute() if found else None
            return p.absolute() if p.is_file() else None
    sibling = script.parent / "python"
    return sibling.absolute() if sibling.is_file() else None


def _conda_pythons() -> Iterable[Path]:
    """Enumerate environments registered with Conda, when Conda is available."""
    conda = shutil.which("conda")
    if not conda:
        return ()
    try:
        cp = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True, timeout=5)
        if cp.returncode != 0:
            return ()
        roots = json.loads(cp.stdout).get("envs") or ()
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return ()
    return tuple(Path(root) / ("python.exe" if os.name == "nt" else "bin/python") for root in roots)


def _pyenv_pythons() -> Iterable[Path]:
    """Enumerate Python installs known to pyenv without scanning arbitrary user directories."""
    pyenv = shutil.which("pyenv")
    if not pyenv:
        return ()
    try:
        cp = subprocess.run([pyenv, "root"], capture_output=True, text=True, timeout=5)
        if cp.returncode != 0 or not cp.stdout.strip():
            return ()
        versions = Path(cp.stdout.strip()) / "versions"
        if not versions.is_dir():
            return ()
        return tuple(entry / "bin/python" for entry in versions.iterdir() if entry.is_dir())
    except (OSError, subprocess.SubprocessError):
        return ()



def _homebrew_pythons() -> Iterable[Path]:
    """Enumerate Python owned by Homebrew's mlx-lm formula on macOS, if installed."""
    if sys.platform != "darwin":
        return ()
    brew = shutil.which("brew")
    if not brew:
        return ()
    try:
        cp = subprocess.run([brew, "--prefix", "mlx-lm"], capture_output=True, text=True, timeout=5)
        if cp.returncode != 0 or not cp.stdout.strip():
            return ()
        prefix = Path(cp.stdout.strip())
        # Homebrew Python applications normally carry their isolated environment under libexec.
        candidates = (prefix / "libexec" / "bin" / "python", prefix / "bin" / "python3", prefix / "bin" / "python")
        return tuple(p for p in candidates if p.is_file())
    except (OSError, subprocess.SubprocessError):
        return ()



def _pipx_pythons() -> Iterable[Path]:
    """Enumerate pipx-managed environments from pipx's own resolved venv root."""
    pipx = shutil.which("pipx")
    if not pipx:
        return ()
    try:
        cp = subprocess.run(
            [pipx, "environment", "--value", "PIPX_LOCAL_VENVS"],
            capture_output=True, text=True, timeout=5,
        )
        if cp.returncode != 0 or not cp.stdout.strip():
            return ()
        root = Path(cp.stdout.strip())
        if not root.is_dir():
            return ()
        result = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            for rel in (Path("bin/python"), Path("Scripts/python.exe")):
                candidate = entry / rel
                if candidate.is_file():
                    result.append(candidate)
        return tuple(result)
    except (OSError, subprocess.SubprocessError):
        return ()


def _conventional_venv_pythons() -> Iterable[Path]:
    """Enumerate bounded conventional user venv roots; never crawl the home tree."""
    home = Path.home()
    roots = (
        home / ".venvs",
        home / ".virtualenvs",
        home / "venvs",
        home / "envs",
    )
    result = []
    for root in roots:
        if not root.is_dir():
            continue
        try:
            entries = tuple(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            for rel in (Path("bin/python"), Path("Scripts/python.exe")):
                candidate = entry / rel
                if candidate.is_file():
                    result.append(candidate)
    return tuple(result)


def _uv_tool_pythons() -> Iterable[Path]:
    """Enumerate uv-managed tool environments without creating or modifying tools."""
    uv = shutil.which("uv")
    if not uv:
        return ()
    try:
        cp = subprocess.run([uv, "tool", "dir"], capture_output=True, text=True, timeout=5)
        if cp.returncode != 0 or not cp.stdout.strip():
            return ()
        root = Path(cp.stdout.strip())
        if not root.is_dir():
            return ()
        result=[]
        for entry in root.iterdir():
            if not entry.is_dir(): continue
            for rel in (Path("bin/python"), Path("Scripts/python.exe")):
                candidate=entry/rel
                if candidate.is_file(): result.append(candidate)
        return tuple(result)
    except (OSError, subprocess.SubprocessError):
        return ()


def _python_candidates() -> Iterable[tuple[str, Path]]:
    """Yield bounded, installation-agnostic Python candidates with provenance."""
    seen: set[str] = set()
    raw_candidates: list[tuple[str, str | Path | None]] = [
        ("active virtualenv", os.environ.get("VIRTUAL_ENV") and Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python"),
        ("active conda environment", os.environ.get("CONDA_PREFIX") and Path(os.environ["CONDA_PREFIX"]) / "bin" / "python"),
        ("mlx_lm on PATH", _python_from_script("mlx_lm")),
        ("mlx_lm.server on PATH", _python_from_script("mlx_lm.server")),
        ("mlx_lm.generate on PATH", _python_from_script("mlx_lm.generate")),
        ("mlx_lm.chat on PATH", _python_from_script("mlx_lm.chat")),
    ]
    raw_candidates.extend(("Homebrew mlx-lm", p) for p in _homebrew_pythons())
    raw_candidates.extend(("uv tool environment", p) for p in _uv_tool_pythons())
    raw_candidates.extend(("pipx environment", p) for p in _pipx_pythons())
    raw_candidates.extend(("conda environment", p) for p in _conda_pythons())
    raw_candidates.extend(("pyenv environment", p) for p in _pyenv_pythons())
    raw_candidates.extend(("conventional venv", p) for p in _conventional_venv_pythons())
    raw_candidates.extend([
        ("python on PATH", shutil.which("python")),
        ("python3 on PATH", shutil.which("python3")),
        ("Devbits Python", sys.executable),
    ])
    for source, raw in raw_candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        # Do not realpath Python executables here. A venv's bin/python is often a
        # symlink to its base interpreter; executing the resolved target loses the
        # venv's sys.prefix and installed packages. Use the absolute spelling as
        # candidate identity and preserve it for probing/worker launch.
        path = path.absolute()
        key = os.path.normcase(os.path.normpath(str(path)))
        if key not in seen and path.is_file():
            seen.add(key)
            yield source, path

def _runtime(python: Path, info: dict) -> MLXRuntime:
    python = python.expanduser().absolute()
    return MLXRuntime(python, python.parent.parent, False, info.get("mlx"), info.get("mlx_lm"),
                      info.get("backend"), info.get("system"), info.get("machine"))


def discover_runtime(diagnostic: Callable[[str], None] | None = None) -> MLXRuntime | None:
    """Find an existing MLX-LM environment on macOS/Linux without mutation."""
    log = diagnostic or (lambda _message: None)
    explicit = os.environ.get("DEVBITS_MLX_PYTHON")
    if explicit:
        python = Path(explicit).expanduser()
        log(f"MLX runtime: checking DEVBITS_MLX_PYTHON={python}")
        info = _probe_python(python, require_mlx=True) if python.is_file() else None
        if info is None:
            raise MLXRuntimeError(
                "DEVBITS_MLX_PYTHON does not point to a usable macOS/Linux MLX-LM Python environment"
            )
        runtime = _runtime(python, info)
        log(f"MLX runtime: accepted {runtime.python} (mlx-lm {runtime.mlx_lm_version}, mlx {runtime.mlx_version}, {runtime.backend})")
        return runtime

    candidates = list(_python_candidates())
    if not candidates:
        log("MLX runtime: no Python candidates found")
    for source, python in candidates:
        log(f"MLX runtime: probing {python} [{source}]")
        info = _probe_python(python, require_mlx=True)
        if info is not None:
            runtime = _runtime(python, info)
            log(f"MLX runtime: accepted {runtime.python} (mlx-lm {runtime.mlx_lm_version}, mlx {runtime.mlx_version}, {runtime.backend})")
            return runtime
        log(f"MLX runtime: rejected {python} (not a usable MLX-LM environment)")
    log("MLX runtime: no usable existing MLX-LM environment found")
    log("MLX runtime: activate its venv, expose MLX-LM on PATH, install it with a supported environment manager, or set DEVBITS_MLX_PYTHON=/path/to/venv/bin/python")
    return None
