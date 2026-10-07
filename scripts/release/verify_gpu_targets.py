"""Compile all bundled CUDA sources for the pinned NVIDIA target architectures.

This is a source/compiler check, not a GPU execution, model, capture, NVENC or
Quest test. NVRTC can compile a target absent from this PC. No generated PTX is
loaded into a driver context or launched. Importing this module does not import
Torch or load a DLL; compiler work runs in a subprocess with a 60-second limit.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRECTORY = ROOT / "src/quest3d/shaders"
SOURCE_NAMES = (
    "tone_map.cu", "depth_edges.cu", "forward_warp.cu", "reference_cubic.cu",
    "stereo_pack.cu", "colour_fit.cu",
)
PINNED_RUNTIMES = {
    "2.7.1+cu126": ("cu126", "12.6", (12, 6), (75, 86, 89)),
    "2.7.1+cu128": ("cu128", "12.8", (12, 8), (75, 86, 89, 120)),
}
MAX_LOG_BYTES = 1_048_576
MAX_PTX_BYTES = 16_777_216
WORKER_DEADLINE_SECONDS = 60


def runtime_spec(torch_version, cuda_version):
    """Accept the two exact compiler profiles without accessing CUDA devices."""
    if torch_version not in PINNED_RUNTIMES:
        raise RuntimeError("Target verification requires pinned Torch 2.7.1+cu126 or 2.7.1+cu128")
    spec = PINNED_RUNTIMES[torch_version]
    if cuda_version != spec[1]:
        raise RuntimeError("Torch build and CUDA runtime version differ")
    return spec


def compile_options(target):
    if type(target) is not int or target not in (75, 86, 89, 120):
        raise ValueError("Unreviewed CUDA compiler target")
    return (f"--gpu-architecture=compute_{target}".encode("ascii"),
            b"--fmad=false", b"--std=c++11")


def _check(code, operation):
    if code:
        raise RuntimeError(f"{operation} failed with NVRTC code {code}")


class NVRTCCompiler:
    """Owned NVRTC loader with no CUDA driver/module/launch API."""

    def __init__(self, torch_module):
        if sys.platform != "win32" or C.sizeof(C.c_void_p) != 8:
            raise RuntimeError("Pinned NVRTC target verification requires Windows x64")
        self.torch_version = str(torch_module.__version__)
        self.profile, self.cuda_version, expected_nvrtc, self.targets = runtime_spec(
            self.torch_version, torch_module.version.cuda)
        library = Path(torch_module.__file__).resolve().parent / "lib/nvrtc64_120_0.dll"
        self.nvrtc = C.WinDLL(str(library))
        declarations = {
            "nvrtcVersion": [C.POINTER(C.c_int), C.POINTER(C.c_int)],
            "nvrtcCreateProgram": [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p,
                                   C.c_int, C.c_void_p, C.c_void_p],
            "nvrtcCompileProgram": [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)],
            "nvrtcGetProgramLogSize": [C.c_void_p, C.POINTER(C.c_size_t)],
            "nvrtcGetProgramLog": [C.c_void_p, C.c_void_p],
            "nvrtcGetPTXSize": [C.c_void_p, C.POINTER(C.c_size_t)],
            "nvrtcGetPTX": [C.c_void_p, C.c_void_p],
            "nvrtcDestroyProgram": [C.POINTER(C.c_void_p)],
        }
        for name, arguments in declarations.items():
            getattr(self.nvrtc, name).argtypes = arguments
            getattr(self.nvrtc, name).restype = C.c_int
        major, minor = C.c_int(), C.c_int()
        _check(self.nvrtc.nvrtcVersion(C.byref(major), C.byref(minor)), "nvrtcVersion")
        self.version = (major.value, minor.value)
        if self.version != expected_nvrtc:
            raise RuntimeError(f"Expected pinned NVRTC {expected_nvrtc}, found {self.version}")

    def compile(self, source, name, target):
        options = compile_options(target)
        arguments = (C.c_char_p * len(options))(*options)
        program = C.c_void_p()
        _check(self.nvrtc.nvrtcCreateProgram(C.byref(program), source, name.encode("ascii"),
                                           0, None, None), "nvrtcCreateProgram")
        try:
            code = self.nvrtc.nvrtcCompileProgram(program, len(options), arguments)
            if code:
                size = C.c_size_t()
                _check(self.nvrtc.nvrtcGetProgramLogSize(program, C.byref(size)), "nvrtcGetProgramLogSize")
                if not 0 < size.value < MAX_LOG_BYTES:
                    raise RuntimeError("Unexpected NVRTC compiler log size")
                log = C.create_string_buffer(size.value)
                _check(self.nvrtc.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
                raise RuntimeError(f"{name} compute_{target} compile failed ({code}): "
                                   + log.value.decode("utf-8", "replace"))
            size = C.c_size_t()
            _check(self.nvrtc.nvrtcGetPTXSize(program, C.byref(size)), "nvrtcGetPTXSize")
            if not 0 < size.value < MAX_PTX_BYTES:
                raise RuntimeError("Unexpected compiled PTX size")
            ptx = C.create_string_buffer(size.value)
            _check(self.nvrtc.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
            return ptx.raw
        finally:
            _check(self.nvrtc.nvrtcDestroyProgram(C.byref(program)), "nvrtcDestroyProgram")


def verify_sources(compiler, source_directory=SOURCE_DIRECTORY):
    """Bind every actual source and PTX result to its requested target."""
    profile, cuda_version, expected_nvrtc, targets = runtime_spec(
        compiler.torch_version, compiler.cuda_version)
    if tuple(compiler.version) != expected_nvrtc or tuple(compiler.targets) != targets:
        raise RuntimeError("Compiler version or target inventory differs from the pinned profile")
    sources = {name: (Path(source_directory) / name).read_bytes() for name in SOURCE_NAMES}
    if any(not source or b"\0" in source for source in sources.values()):
        raise RuntimeError("A bundled CUDA source is empty or contains a NUL byte")
    results = []
    for target in targets:
        for name, source in sources.items():
            ptx = compiler.compile(source, name, target)
            observed = re.search(rb"(?m)^\s*\.target\s+(sm_[0-9]+)(?:\s|,|$)", ptx)
            if observed is None or observed[1] != f"sm_{target}".encode("ascii"):
                raise RuntimeError(f"{name} compute_{target} returned PTX for a different target")
            if not re.search(rb"(?m)^\s*\.version\s+[0-9]+\.[0-9]+", ptx) or b".entry " not in ptx:
                raise RuntimeError(f"{name} compute_{target} returned invalid PTX")
            results.append({"source": f"src/quest3d/shaders/{name}",
                            "target": f"compute_{target}", "ptx_target": observed[1].decode("ascii"),
                            "source_sha256": hashlib.sha256(source).hexdigest(),
                            "ptx_sha256": hashlib.sha256(ptx).hexdigest(), "ptx_bytes": len(ptx),
                            "options": [option.decode("ascii") for option in compile_options(target)],
                            "compiled": True, "executed": False})
    return {"schema": 1, "kind": "cuda-source-target-compilation",
            "torch": compiler.torch_version, "runtime_profile": profile,
            "cuda_version": cuda_version, "nvrtc": list(compiler.version),
            "sources": len(SOURCE_NAMES), "targets": [f"compute_{target}" for target in targets],
            "compilations": results, "all_sources_compiled": True,
            "gpu_functions_launched": 0, "hardware_validation": False,
            "driver_jit_validation": False,
            "scope": "NVRTC source compilation only; no GPU execution, capture, model, NVENC, Quest or performance validation"}


def verify_targets():
    import torch
    return verify_sources(NVRTCCompiler(torch))


def run_worker():
    try:
        checked = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=WORKER_DEADLINE_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("CUDA source compilation exceeded 60 seconds") from error
    if checked.returncode:
        raise RuntimeError("CUDA source compilation failed: " + checked.stderr[-12000:])
    result = json.loads(checked.stdout)
    if (result.get("kind") != "cuda-source-target-compilation"
            or result.get("all_sources_compiled") is not True
            or result.get("hardware_validation") is not False
            or result.get("gpu_functions_launched") != 0):
        raise RuntimeError("Unexpected CUDA source verification worker result")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(verify_targets(), ensure_ascii=True))
        return
    output = json.dumps(run_worker(), indent=2, ensure_ascii=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
