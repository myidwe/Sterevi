"""Compiler evidence remains distinct from physical GPU execution evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/release/verify_gpu_targets.py"
spec = importlib.util.spec_from_file_location("verify_gpu_targets", SCRIPT)
targets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(targets)


class Compiler:
    def __init__(self, profile="cu126"):
        self.torch_version = f"2.7.1+{profile}"
        self.cuda_version = "12.6" if profile == "cu126" else "12.8"
        self.version = (12, 6) if profile == "cu126" else (12, 8)
        self.targets = (75, 86, 89) if profile == "cu126" else (75, 86, 89, 120)
        self.calls = []

    def compile(self, source, name, target):
        self.calls.append((source, name, target))
        return f".version 8.7\n.target sm_{target}\n.visible .entry kernel() {{}}\n\0".encode()


@pytest.mark.parametrize("profile,count", [("cu126", 18), ("cu128", 24)])
def test_entire_actual_source_inventory_is_bound_to_each_target_with_honest_scope(profile, count):
    compiler = Compiler(profile)
    report = targets.verify_sources(compiler)
    assert len(compiler.calls) == len(report["compilations"]) == count
    assert {(name, target) for _, name, target in compiler.calls} == {
        (name, target) for name in targets.SOURCE_NAMES for target in compiler.targets}
    for call, entry in zip(compiler.calls, report["compilations"]):
        source, name, target = call
        assert source == (targets.SOURCE_DIRECTORY / name).read_bytes()
        assert entry["source_sha256"] == hashlib.sha256(source).hexdigest()
        assert entry["target"] == f"compute_{target}"
        assert entry["executed"] is False
        assert entry["options"] == [f"--gpu-architecture=compute_{target}", "--fmad=false", "--std=c++11"]
    assert report["hardware_validation"] is report["driver_jit_validation"] is False
    assert report["gpu_functions_launched"] == 0


def test_repeated_wrong_architecture_ptx_is_rejected_instead_of_counted_as_coverage():
    compiler = Compiler()
    compiler.compile = lambda *args: b".version 8.5\n.target sm_75\n.visible .entry kernel() {}\n"
    with pytest.raises(RuntimeError, match="different target"):
        targets.verify_sources(compiler)


@pytest.mark.parametrize("ptx", [b".target sm_75\n.entry x() {}", b".version 8.5\n.target sm_75\n"])
def test_invalid_compiler_output_is_not_a_successful_source_check(ptx):
    compiler = Compiler()
    compiler.compile = lambda *args: ptx
    with pytest.raises(RuntimeError, match="invalid PTX"):
        targets.verify_sources(compiler)


def test_missing_source_fails_before_any_compilation(tmp_path):
    compiler = Compiler()
    with pytest.raises(FileNotFoundError):
        targets.verify_sources(compiler, tmp_path)
    assert compiler.calls == []


@pytest.mark.parametrize("version,cuda", [("2.8.0+cu128", "12.8"), ("2.7.1+cu126", "12.8"), ("2.7.1+cpu", None)])
def test_unpinned_or_inconsistent_compiler_runtime_is_refused(version, cuda):
    with pytest.raises(RuntimeError):
        targets.runtime_spec(version, cuda)


def test_stale_nvrtc_or_reduced_target_inventory_is_refused():
    for change in ({"version": (12, 5)}, {"targets": (75,)}, {"targets": (75, 86, 89, 120)}):
        compiler = Compiler()
        compiler.__dict__.update(change)
        with pytest.raises(RuntimeError, match="inventory"):
            targets.verify_sources(compiler)


def test_compiler_profiles_remain_consistent_with_install_and_runtime_policy():
    from quest3d.gpu_runtime import SUPPORTED_TORCH_VERSIONS

    installed = json.loads((ROOT / "config/gpu-runtimes.json").read_text(encoding="utf-8"))
    assert set(targets.PINNED_RUNTIMES) == set(SUPPORTED_TORCH_VERSIONS)
    for torch_version, spec in targets.PINNED_RUNTIMES.items():
        profile, cuda_version, nvrtc_version, architectures = spec
        entry = installed["profiles"][profile]
        assert entry["torch"] == torch_version
        assert entry["cuda_version"] == cuda_version
        assert nvrtc_version == tuple(map(int, cuda_version.split(".")))
        assert {major * 10 + minor for major, minor in entry["capabilities"]} <= set(architectures)


def test_worker_has_a_bounded_deadline_and_does_not_accept_execution_claims(monkeypatch):
    calls = []

    def timeout(*args, **kwargs):
        calls.append((args, kwargs))
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(targets.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="60 seconds"):
        targets.run_worker()
    assert calls[0][1]["timeout"] == 60
    assert calls[0][0][0][-1] == "--worker"
    forged = {"kind": "cuda-source-target-compilation", "all_sources_compiled": True,
              "hardware_validation": True, "gpu_functions_launched": 0}
    monkeypatch.setattr(targets.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(forged), stderr=""))
    with pytest.raises(RuntimeError, match="worker result"):
        targets.run_worker()


def test_import_has_no_torch_or_native_loader_side_effects():
    code = ("import ctypes,importlib.util,sys; "
            "ctypes.WinDLL=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('DLL import')); "
            f"s=importlib.util.spec_from_file_location('gpu_targets',{str(SCRIPT)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print('torch' in sys.modules)")
    checked = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert checked.stdout.strip() == "False"


@pytest.mark.gpu
@pytest.mark.skipif(sys.platform != "win32" or importlib.util.find_spec("torch") is None,
                    reason="Requires pinned Windows Torch/NVRTC package; does not require target GPUs")
def test_actual_nvrtc_compiles_every_bundled_shader_for_all_profile_targets():
    report = targets.run_worker()
    profile = report["runtime_profile"]
    expected = ["compute_75", "compute_86", "compute_89"] + (["compute_120"] if profile == "cu128" else [])
    assert report["targets"] == expected
    assert len(report["compilations"]) == len(expected) * len(targets.SOURCE_NAMES)
    assert all(entry["compiled"] and not entry["executed"] for entry in report["compilations"])
    assert report["hardware_validation"] is False
