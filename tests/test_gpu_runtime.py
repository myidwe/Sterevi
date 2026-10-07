"""Architecture/profile validation without treating mocks as GPU evidence."""
from contextlib import nullcontext
import ctypes as C
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from quest3d import gpu_runtime
from quest3d.gpu_runtime import (
    profile_for_capability, require_cuda_runtime, require_driver_version,
    require_nvrtc_version, select_runtime_policy,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHES_126 = ("sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90")
ARCHES_128 = ("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120")


def policy(capability=(7, 5), profile="cu126", architectures=None):
    return select_runtime_policy(torch_version=f"2.7.1+{profile}",
                                 cuda_version="12.6" if profile == "cu126" else "12.8",
                                 capability=capability,
                                 torch_arches=architectures or (ARCHES_126 if profile == "cu126" else ARCHES_128))


@pytest.mark.parametrize("capability,profile,target,binary", [
    ((7, 5), "cu126", "compute_75", "sm_75"),
    ((8, 6), "cu126", "compute_86", "sm_86"),
    ((8, 9), "cu126", "compute_89", "sm_86"),
    ((12, 0), "cu128", "compute_120", "sm_120"),
    ((7, 5), "cu128", "compute_75", "sm_75"),
    ((8, 6), "cu128", "compute_86", "sm_86"),
    ((8, 9), "cu128", "compute_89", "sm_86"),
])
def test_supported_profile_selects_actual_compile_target_and_torch_coverage(capability, profile, target, binary):
    selected = policy(capability, profile)
    assert selected.profile == profile
    assert selected.architecture == target
    assert selected.architecture_option == f"--gpu-architecture={target}".encode()
    assert selected.torch_binary_architecture == binary
    assert selected.minimum_driver_api == (12060 if profile == "cu126" else 12080)
    assert selected.metadata()["device_capability"] == list(capability)


def test_blackwell_does_not_accept_old_runtime_even_with_fake_matching_arch():
    with pytest.raises(RuntimeError, match="RTX 50.*CUDA 12.8"):
        policy((12, 0), "cu126", ("sm_120",))


@pytest.mark.parametrize("capability", [(5, 2), (6, 1), (8, 0), (9, 0), (10, 0), (12, 1), (13, 0)])
def test_unreviewed_architecture_is_not_enabled_by_generic_forward_compatibility(capability):
    with pytest.raises(RuntimeError, match="Unsupported NVIDIA capability"):
        policy(capability, "cu128", ("compute_75",))


@pytest.mark.parametrize("capability", [None, [7, 5], (True, 5), (7, 5.0), (7,)])
def test_invalid_capability_is_not_silently_coerced(capability):
    with pytest.raises(ValueError, match="capability"):
        profile_for_capability(capability)


@pytest.mark.parametrize("torch_version,cuda_version,platform,expected", [
    ("2.8.0+cu128", "12.8", "win32", "Unsupported Torch"),
    ("2.7.1+cpu", None, "win32", "Unsupported Torch"),
    ("2.7.1+cu128", "12.6", "win32", "mismatch"),
    ("2.7.1+cu126", "12.8", "win32", "mismatch"),
    ("2.7.1+cu126", "12.6", "linux", "Windows"),
])
def test_pins_platform_and_runtime_consistency_are_still_enforced(torch_version, cuda_version, platform, expected):
    with pytest.raises(RuntimeError, match=expected):
        select_runtime_policy(torch_version=torch_version, cuda_version=cuda_version,
                              capability=(7, 5), torch_arches=ARCHES_126, platform=platform)


@pytest.mark.parametrize("architectures", [(), ("sm_75",), ("sm_90",), ("sm_90a",), ("compute_90",)])
def test_torch_wheel_must_cover_device_not_just_report_cuda_available(architectures):
    with pytest.raises(RuntimeError, match="no compatible CUDA binary/PTX"):
        select_runtime_policy(torch_version="2.7.1+cu126", cuda_version="12.6",
                              capability=(8, 9), torch_arches=architectures)


def test_generic_ptx_forward_compatibility_is_explicit_and_suffixes_are_excluded():
    assert policy((8, 9), architectures=("sm_90a", "compute_80")).torch_binary_architecture == "compute_80"
    with pytest.raises(RuntimeError, match="no compatible"):
        policy((8, 9), architectures=("compute_90a", "sm_90a"))


def test_install_profile_table_matches_runtime_policy():
    config = json.loads((ROOT / "config/gpu-runtimes.json").read_text(encoding="utf-8"))
    assert config["schema"] == 1
    seen = set()
    for name, entry in config["profiles"].items():
        for value in entry["capabilities"]:
            capability = tuple(value)
            selected = policy(capability, name)
            seen.add(capability)
            assert profile_for_capability(capability) == name
            assert selected.torch_version == entry["torch"]
            assert f"{selected.cuda_version[0]}.{selected.cuda_version[1]}" == entry["cuda_version"]
            assert selected.minimum_driver_api == entry["minimum_driver_api"]
    assert seen == set(gpu_runtime.SUPPORTED_CAPABILITIES)


def test_module_import_does_not_import_torch_or_load_cuda():
    code = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import quest3d.gpu_runtime; print('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def fake_torch(*, available=True, capability=(7, 5), device_count=1):
    return SimpleNamespace(__version__="2.7.1+cu126", version=SimpleNamespace(cuda="12.6"),
                           cuda=SimpleNamespace(is_available=lambda: available,
                                                device_count=lambda: device_count,
                                                get_device_capability=lambda device: capability,
                                                get_arch_list=lambda: ARCHES_126))


def test_device_validation_uses_requested_index_and_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(gpu_runtime, "sys", SimpleNamespace(platform="win32"))
    observed = []
    fake = fake_torch(capability=(8, 9), device_count=2)
    fake.cuda.get_device_capability = lambda index: observed.append(index) or (8, 9)
    assert require_cuda_runtime(1, torch_module=fake).architecture == "compute_89"
    assert observed == [1]
    with pytest.raises(RuntimeError, match="available NVIDIA"):
        require_cuda_runtime(0, torch_module=fake_torch(available=False))
    for bad in (True, -1, 2, 0.0):
        with pytest.raises(ValueError, match="Invalid CUDA device"):
            require_cuda_runtime(bad, torch_module=fake)


def test_eager_depth_model_rejects_wrong_blackwell_runtime_before_source_or_weights(monkeypatch):
    pytest.importorskip("torch")
    from quest3d import depth
    monkeypatch.setattr(gpu_runtime, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(depth.torch, "__version__", "2.7.1+cu126")
    monkeypatch.setattr(depth.torch.version, "cuda", "12.6")
    monkeypatch.setattr(depth.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(depth.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(depth.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(depth.torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(depth.torch.cuda, "get_arch_list", lambda: ("sm_120",))
    monkeypatch.setattr(depth, "verified_model_source", lambda **kwargs: pytest.fail("Incompatible runtime reached model loading"))
    with pytest.raises(RuntimeError, match="RTX 50.*CUDA 12.8"):
        depth.DepthEngine(execution_mode="eager", fused_resize=False)


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def fake_driver(version=12080, code=0):
    def get_version(value):
        value._obj.value = version
        return code
    return SimpleNamespace(cuDriverGetVersion=Function(get_version))


def test_old_driver_is_not_mistaken_for_valid_cuda_minor_compatibility():
    with pytest.raises(RuntimeError, match="too old.*cu128"):
        require_driver_version(fake_driver(12070), policy((12, 0), "cu128"))
    assert require_driver_version(fake_driver(12080), policy((12, 0), "cu128")) == 12080
    assert require_driver_version(fake_driver(13000), policy()) == 13000
    with pytest.raises(RuntimeError, match="code 999"):
        require_driver_version(fake_driver(13000, 999), policy())


def test_nvrtc_must_match_selected_runtime_exactly():
    selected = policy((12, 0), "cu128")
    require_nvrtc_version(selected, (12, 8))
    for bad in ((12, 6), (12, 9), (13, 0)):
        with pytest.raises(RuntimeError, match="Expected pinned CUDA 12.8"):
            require_nvrtc_version(selected, bad)


KERNELS = (
    ("tonemap_cuda", "CudaToneMapper"), ("stereo_pack_cuda", "CudaStereoPacker"),
    ("resample_cuda", "CudaReferenceCubicResize"), ("colour_fit_cuda", "CudaFloatColourFit"),
    ("depth_edges_cuda", "CudaDepthEdges"), ("forward_warp_cuda", "CudaForwardWarper"),
)


@pytest.mark.parametrize("module_name,class_name", KERNELS)
@pytest.mark.parametrize("capability,profile", [((7, 5), "cu126"), ((8, 6), "cu126"),
    ((8, 9), "cu126"), ((12, 0), "cu128"), ((7, 5), "cu128")])
def test_every_kernel_uses_shared_profile_and_target_before_launch(monkeypatch, module_name, class_name, capability, profile):
    """Compile API calls are simulated; this cannot prove numerical GPU parity."""
    pytest.importorskip("torch")
    module = importlib.import_module(f"quest3d.{module_name}")
    torch = module.torch
    selected = policy(capability, profile)
    compiled = []
    loaded = []
    def set_pointer(pointer, value):
        pointer._obj.value = value
        return 0
    def version(major, minor):
        major._obj.value, minor._obj.value = selected.cuda_version
        return 0
    def compile_program(program, count, options):
        compiled.append(tuple(options[index] for index in range(count)))
        return 0
    nvrtc = SimpleNamespace(
        nvrtcVersion=Function(version),
        nvrtcCreateProgram=Function(lambda program, *args: set_pointer(program, 123)),
        nvrtcCompileProgram=Function(compile_program),
        nvrtcGetProgramLogSize=Function(lambda *args: 0), nvrtcGetProgramLog=Function(lambda *args: 0),
        nvrtcGetPTXSize=Function(lambda program, size: set_pointer(size, 9)),
        nvrtcGetPTX=Function(lambda program, buffer: C.memmove(buffer, b"fake-ptx\0", 9) and 0),
        nvrtcDestroyProgram=Function(lambda program: set_pointer(program, 0)),
    )
    driver = fake_driver(13000)
    driver.cuModuleLoadData = Function(lambda pointer, ptx: loaded.append(True) or set_pointer(pointer, 456))
    driver.cuModuleGetFunction = Function(lambda pointer, module, name: set_pointer(pointer, 789))
    driver.cuLaunchKernel = Function(lambda *args: pytest.fail("Constructor launched a kernel"))
    driver.cuModuleUnload = Function(lambda *args: 0)
    monkeypatch.setattr(gpu_runtime, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(torch, "__version__", selected.torch_version)
    monkeypatch.setattr(torch.version, "cuda", f"{selected.cuda_version[0]}.{selected.cuda_version[1]}")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ARCHES_126 if profile == "cu126" else ARCHES_128)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: None)
    monkeypatch.setattr(C, "WinDLL", lambda path: driver if str(path).endswith("nvcuda.dll") else nvrtc, raising=False)
    with getattr(module, class_name)() as owner:
        assert owner.metadata["architecture"] == selected.architecture
        assert owner.metadata["runtime_profile"] == profile
        assert owner.metadata["driver_cuda_version"] == 13000
    assert compiled == [(selected.architecture_option, b"--fmad=false", b"--std=c++11")]
    assert loaded == [True]
