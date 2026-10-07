"""Pinned NVIDIA runtime policy shared by capture, graphs and NVRTC kernels.

Importing this module does not import Torch, initialize CUDA, or load a DLL.
An accepted policy describes software compatibility; it does not assert that
the user's hardware has passed the separate capture/model/encoder smoke test.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import sys


SUPPORTED_CAPABILITIES = ((7, 5), (8, 6), (8, 9), (12, 0))
SUPPORTED_TORCH_VERSIONS = ("2.7.1+cu126", "2.7.1+cu128")


@dataclass(frozen=True)
class CudaRuntimePolicy:
    profile: str
    torch_version: str
    cuda_version: tuple[int, int]
    capability: tuple[int, int]
    torch_binary_architecture: str

    @property
    def architecture(self) -> str:
        return f"compute_{self.capability[0]}{self.capability[1]}"

    @property
    def minimum_driver_api(self) -> int:
        # CUDA minor-version compatibility alone does not guarantee that the
        # driver can JIT the newer PTX emitted by our pinned NVRTC compiler.
        major, minor = self.cuda_version
        return major * 1000 + minor * 10

    @property
    def architecture_option(self) -> bytes:
        return f"--gpu-architecture={self.architecture}".encode("ascii")

    def metadata(self) -> dict:
        return {"runtime_profile": self.profile, "torch": self.torch_version,
                "device_capability": list(self.capability),
                "torch_binary_architecture": self.torch_binary_architecture}


def profile_for_capability(capability: tuple[int, int]) -> str:
    """Choose the install profile before importing/downloading Torch."""
    if (not isinstance(capability, tuple) or len(capability) != 2
            or any(type(part) is not int for part in capability)):
        raise ValueError("CUDA capability must be an integer (major, minor) tuple")
    if capability not in SUPPORTED_CAPABILITIES:
        raise RuntimeError(f"Unsupported NVIDIA capability {capability}; supported RTX 20/30/40/50 architectures are {SUPPORTED_CAPABILITIES}")
    return "cu128" if capability == (12, 0) else "cu126"


def _torch_binary_architecture(capability, architectures) -> str:
    """Accept a same-major cubin or forward-compatible generic PTX.

    In particular, a generic sm_86 cubin covers sm_89. Architecture-specific
    suffixes such as sm_90a are deliberately excluded from this generic rule.
    """
    native, ptx = [], []
    for architecture in architectures:
        if not isinstance(architecture, str):
            continue
        matched = re.fullmatch(r"(sm|compute)_(\d+)", architecture)
        if matched is None:
            continue
        number = int(matched[2])
        target = (number // 10, number % 10)
        if matched[1] == "sm" and target[0] == capability[0] and target[1] <= capability[1]:
            native.append((target, architecture))
        elif matched[1] == "compute" and target <= capability:
            ptx.append((target, architecture))
    eligible = native or ptx
    if not eligible:
        raise RuntimeError(f"The installed Torch wheel has no compatible CUDA binary/PTX for sm_{capability[0]}{capability[1]}; architectures={tuple(architectures)}")
    return max(eligible)[1]


def select_runtime_policy(*, torch_version: str, cuda_version: str | None,
                          capability: tuple[int, int], torch_arches,
                          platform: str = "win32") -> CudaRuntimePolicy:
    """Pure validation; tests can cover each architecture without a GPU."""
    if platform != "win32":
        raise RuntimeError("Sterevi's pinned NVIDIA runtime requires Windows x64")
    preferred = profile_for_capability(capability)
    if torch_version not in SUPPORTED_TORCH_VERSIONS:
        raise RuntimeError(f"Unsupported Torch runtime {torch_version}; use pinned Torch 2.7.1+cu126 or 2.7.1+cu128")
    profile = torch_version.rsplit("+", 1)[1]
    version = (12, 6) if profile == "cu126" else (12, 8)
    if cuda_version != f"{version[0]}.{version[1]}":
        raise RuntimeError(f"Torch build/runtime mismatch: {torch_version} reports CUDA {cuda_version}")
    if preferred == "cu128" and profile != preferred:
        raise RuntimeError("RTX 50 / sm_120 requires the pinned CUDA 12.8 runtime (Torch 2.7.1+cu128); reinstall with the Blackwell profile")
    architectures = tuple(torch_arches)
    binary = _torch_binary_architecture(capability, architectures)
    return CudaRuntimePolicy(profile, torch_version, version, capability, binary)


def require_cuda_runtime(device=0, *, torch_module=None) -> CudaRuntimePolicy:
    """Validate the requested device; CUDA access happens only when called."""
    if type(device) is not int or device < 0:
        raise ValueError("Invalid CUDA device")
    if sys.platform != "win32":
        raise RuntimeError("Sterevi's pinned NVIDIA runtime requires Windows x64")
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available():
        raise RuntimeError("An available NVIDIA CUDA GPU and driver are required")
    if device >= torch_module.cuda.device_count():
        raise ValueError("Invalid CUDA device")
    return select_runtime_policy(torch_version=str(torch_module.__version__),
                                 cuda_version=torch_module.version.cuda,
                                 capability=torch_module.cuda.get_device_capability(device),
                                 torch_arches=torch_module.cuda.get_arch_list(),
                                 platform=sys.platform)


def require_nvrtc_version(policy: CudaRuntimePolicy, version: tuple[int, int]):
    if version != policy.cuda_version:
        raise RuntimeError(f"Expected pinned CUDA {policy.cuda_version[0]}.{policy.cuda_version[1]} NVRTC for {policy.profile}; got {version}")


def require_driver_version(driver, policy: CudaRuntimePolicy) -> int:
    """Check the driver PTX JIT before compiling/loading any CUDA module."""
    import ctypes as C
    query = driver.cuDriverGetVersion
    query.argtypes, query.restype = [C.POINTER(C.c_int)], C.c_int
    version = C.c_int()
    code = query(C.byref(version))
    if code:
        raise RuntimeError(f"cuDriverGetVersion failed with CUDA code {code}")
    if version.value < policy.minimum_driver_api:
        raise RuntimeError(f"NVIDIA driver CUDA API {version.value} is too old for {policy.profile} PTX; requires >= {policy.minimum_driver_api}. Update the NVIDIA driver")
    return version.value
