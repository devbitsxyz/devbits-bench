import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from devbits_bench.runtimes.mlx import MLXRuntimeError, discover_runtime

INFO_MAC={"mlx":"0.32.3","mlx_lm":"0.32.0","backend":"Metal","system":"Darwin","machine":"arm64"}
INFO_LINUX={"mlx":"0.33.0","mlx_lm":"0.33.1","backend":"CUDA","system":"Linux","machine":"x86_64"}

class MLXRuntimeTests(unittest.TestCase):
    @patch("devbits_bench.runtimes.mlx._probe_python")
    def test_explicit_override_is_advanced_escape_hatch(self, probe):
        probe.return_value=INFO_MAC
        with tempfile.NamedTemporaryFile() as f, patch.dict(os.environ,{"DEVBITS_MLX_PYTHON":f.name}):
            runtime=discover_runtime()
        self.assertEqual(runtime.mlx_lm_version,"0.32.0"); self.assertEqual(runtime.backend,"Metal")

    @patch("devbits_bench.runtimes.mlx._python_candidates")
    @patch("devbits_bench.runtimes.mlx._probe_python")
    def test_existing_runtime_is_auto_discovered(self, probe, candidates):
        candidates.return_value=iter([("test", Path("/existing/venv/bin/python"))]); probe.return_value=INFO_MAC
        with patch.dict(os.environ,{},clear=True): runtime=discover_runtime()
        self.assertEqual(runtime.python,Path("/existing/venv/bin/python")); self.assertFalse(runtime.managed)

    @patch("devbits_bench.runtimes.mlx._python_candidates")
    @patch("devbits_bench.runtimes.mlx._probe_python")
    def test_linux_cuda_runtime_is_supported(self, probe, candidates):
        candidates.return_value=iter([("test", Path("/linux/venv/bin/python"))]); probe.return_value=INFO_LINUX
        with patch.dict(os.environ,{},clear=True): runtime=discover_runtime()
        self.assertEqual(runtime.backend,"CUDA"); self.assertEqual(runtime.system,"Linux")

    def test_invalid_explicit_override_is_actionable(self):
        with patch.dict(os.environ,{"DEVBITS_MLX_PYTHON":"/missing/python"}):
            with self.assertRaisesRegex(MLXRuntimeError,"DEVBITS_MLX_PYTHON"): discover_runtime()

    @patch("devbits_bench.runtimes.mlx._probe_python")
    def test_active_virtualenv_wins_even_when_devbits_uses_another_python(self, probe):
        probe.return_value=INFO_MAC
        with tempfile.TemporaryDirectory() as td:
            python=Path(td)/"bin"/"python"; python.parent.mkdir(); python.touch()
            with patch.dict(os.environ,{"VIRTUAL_ENV":td},clear=True): runtime=discover_runtime()
        self.assertEqual(runtime.python,python.absolute()); probe.assert_called_once_with(python.absolute(),require_mlx=True)

    @patch("devbits_bench.runtimes.mlx._probe_python")
    def test_active_venv_python_symlink_is_not_resolved_to_base_interpreter(self, probe):
        probe.return_value=INFO_MAC
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            base=root/"base"/"bin"/"python3.14"
            base.parent.mkdir(parents=True); base.touch()
            venv=root/"venv"; (venv/"bin").mkdir(parents=True)
            python=venv/"bin"/"python"; python.symlink_to(base)
            with patch.dict(os.environ,{"VIRTUAL_ENV":str(venv)},clear=True):
                runtime=discover_runtime()
        self.assertEqual(runtime.python,python.absolute())
        self.assertNotEqual(runtime.python,base.absolute())
        probe.assert_called_once_with(python.absolute(),require_mlx=True)

    @patch("devbits_bench.runtimes.mlx._probe_python")
    def test_installed_versions_are_reported_not_pinned(self, probe):
        probe.return_value=INFO_LINUX
        with tempfile.NamedTemporaryFile() as f, patch.dict(os.environ,{"DEVBITS_MLX_PYTHON":f.name}): runtime=discover_runtime()
        self.assertEqual(runtime.mlx_version,"0.33.0"); self.assertEqual(runtime.mlx_lm_version,"0.33.1")
    @patch("devbits_bench.runtimes.mlx.subprocess.run")
    @patch("devbits_bench.runtimes.mlx.shutil.which")
    def test_conda_environments_are_enumerated_from_manager_registry(self, which, run):
        which.side_effect=lambda name: "/usr/bin/conda" if name == "conda" else None
        run.return_value=type("CP",(),{"returncode":0,"stdout":"{\"envs\":[\"/opt/conda/envs/mlx\"]}"})()
        from devbits_bench.runtimes.mlx import _conda_pythons
        self.assertEqual(tuple(_conda_pythons()), (Path("/opt/conda/envs/mlx/bin/python"),))

    @patch("devbits_bench.runtimes.mlx.subprocess.run")
    @patch("devbits_bench.runtimes.mlx.shutil.which")
    def test_pyenv_enumeration_uses_manager_root_not_home_scan(self, which, run):
        which.side_effect=lambda name: "/usr/bin/pyenv" if name == "pyenv" else None
        with tempfile.TemporaryDirectory() as td:
            versions=Path(td)/"versions"; (versions/"mlx-env"/"bin").mkdir(parents=True)
            run.return_value=type("CP",(),{"returncode":0,"stdout":td+"\n"})()
            from devbits_bench.runtimes.mlx import _pyenv_pythons
            self.assertEqual(tuple(_pyenv_pythons()), (versions/"mlx-env"/"bin/python",))

    @patch("devbits_bench.runtimes.mlx._python_candidates")
    @patch("devbits_bench.runtimes.mlx._probe_python")
    def test_runtime_diagnostics_explain_rejected_and_accepted_candidates(self, probe, candidates):
        bad=Path("/tmp/plain-python"); good=Path("/tmp/mlx-python")
        candidates.return_value=iter([("PATH", bad),("conda environment", good)]); probe.side_effect=[None,INFO_MAC]
        messages=[]
        with patch.dict(os.environ,{},clear=True): runtime=discover_runtime(diagnostic=messages.append)
        self.assertEqual(runtime.python,good.absolute())
        joined="\n".join(messages)
        self.assertIn(f"probing {bad} [PATH]",joined); self.assertIn(f"rejected {bad}",joined)
        self.assertIn(f"accepted {good.absolute()}",joined)


class MLXPortableProviderTests(unittest.TestCase):
    @patch("devbits_bench.runtimes.mlx.subprocess.run")
    @patch("devbits_bench.runtimes.mlx.shutil.which")
    def test_homebrew_mlx_lm_is_bounded_formula_discovery(self, which, run):
        from devbits_bench.runtimes import mlx
        which.side_effect=lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None
        run.return_value=type("CP",(),{"returncode":0,"stdout":"/opt/homebrew/opt/mlx-lm\n"})()
        with patch.object(mlx.sys,"platform","darwin"), patch.object(Path,"is_file",return_value=True):
            values=tuple(mlx._homebrew_pythons())
        self.assertIn(Path("/opt/homebrew/opt/mlx-lm/libexec/bin/python"),values)
        run.assert_called_once_with(["/opt/homebrew/bin/brew","--prefix","mlx-lm"],capture_output=True,text=True,timeout=5)

    @patch("devbits_bench.runtimes.mlx.subprocess.run")
    @patch("devbits_bench.runtimes.mlx.shutil.which")
    def test_uv_tools_are_enumerated_only_from_uv_managed_root(self, which, run):
        from devbits_bench.runtimes.mlx import _uv_tool_pythons
        which.side_effect=lambda name: "/usr/local/bin/uv" if name == "uv" else None
        with tempfile.TemporaryDirectory() as td:
            python=Path(td)/"mlx-lm"/"bin"/"python"; python.parent.mkdir(parents=True); python.touch()
            run.return_value=type("CP",(),{"returncode":0,"stdout":td+"\n"})()
            self.assertEqual(tuple(_uv_tool_pythons()),(python,))

    @patch("devbits_bench.runtimes.mlx.subprocess.run")
    @patch("devbits_bench.runtimes.mlx.shutil.which")
    def test_pipx_environments_use_pipx_resolved_venv_root(self, which, run):
        from devbits_bench.runtimes.mlx import _pipx_pythons
        which.side_effect=lambda name: "/usr/local/bin/pipx" if name == "pipx" else None
        with tempfile.TemporaryDirectory() as td:
            python=Path(td)/"mlx-lm"/"bin"/"python"
            python.parent.mkdir(parents=True); python.touch()
            run.return_value=type("CP",(),{"returncode":0,"stdout":td+"\n"})()
            self.assertEqual(tuple(_pipx_pythons()),(python,))
        run.assert_called_once_with(
            ["/usr/local/bin/pipx","environment","--value","PIPX_LOCAL_VENVS"],
            capture_output=True,text=True,timeout=5,
        )

    def test_conventional_venv_discovery_is_bounded_to_known_roots(self):
        from devbits_bench.runtimes import mlx
        with tempfile.TemporaryDirectory() as td:
            home=Path(td)
            expected=home/".venvs"/"mlx-lm"/"bin"/"python"
            expected.parent.mkdir(parents=True); expected.touch()
            unrelated=home/"projects"/"secret"/".venv"/"bin"/"python"
            unrelated.parent.mkdir(parents=True); unrelated.touch()
            with patch.object(mlx.Path,"home",return_value=home):
                values=tuple(mlx._conventional_venv_pythons())
        self.assertIn(expected,values)
        self.assertNotIn(unrelated,values)

    @patch("devbits_bench.runtimes.mlx._python_candidates")
    def test_missing_runtime_diagnostic_is_actionable(self, candidates):
        candidates.return_value=iter(())
        messages=[]
        with patch.dict(os.environ,{},clear=True):
            self.assertIsNone(discover_runtime(diagnostic=messages.append))
        self.assertIn("DEVBITS_MLX_PYTHON",messages[-1])
        self.assertIn("activate its venv",messages[-1])

