"""Run the fixed depth model once on CUDA, offline, before completing installation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def verify_depth() -> dict:
    import torch
    from quest3d.depth import DepthEngine
    from quest3d.paths import ROOT

    if ROOT.resolve() != Path(__file__).resolve().parents[2]:
        raise RuntimeError("Depth checker imported a different Sterevi installation")
    engine = DepthEngine(input_size=280, fp16=True, execution_mode="cuda-graph",
                         reuse_constants=True, fused_resize=True,
                         model_id="depth_anything_v2_small")
    try:
        engine.prepare_execution(640, 360)
        # The product supplies GPU-resident BGRA. A CPU input has a different
        # preprocessing signature and cannot prove the prepared graph runs.
        image = torch.zeros((360, 640, 4), dtype=torch.uint8, device="cuda")
        image[:, :, 0] = torch.linspace(0, 255, 640, device="cuda").to(torch.uint8)
        image[:, :, 1] = torch.linspace(0, 255, 360, device="cuda").to(torch.uint8)[:, None]
        image[:, :, 2], image[:, :, 3] = 140, 255
        result = engine.infer(image, frame_id=1, generation=1)
        torch.cuda.synchronize()
        if result.frame_id != 1 or result.generation != 1 or tuple(result.tensor.shape) != (1, *result.input_shape):
            raise RuntimeError("Depth inference returned an invalid frame identity or shape")
        if not bool(torch.isfinite(result.tensor).all()):
            raise RuntimeError("Depth inference returned non-finite values")
        if result.execution_mode != "cuda-graph" or result.execution_reason is not None:
            raise RuntimeError(f"Prepared CUDA graph did not run: {result.execution_mode}, {result.execution_reason}")
        if engine.model_metadata["strict_load"] != {"missing": [], "unexpected": []}:
            raise RuntimeError("The fixed depth checkpoint did not load strictly")
        return {"verified": True, "torch": torch.__version__,
                "runtime_profile": engine.runtime_policy.profile,
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "model": engine.model_metadata["name"],
                "checkpoint_sha256": engine.model_metadata["sha256"],
                "strict_checkpoint_load": True, "real_cuda_inference": True,
                "execution_mode": result.execution_mode,
                "execution_reason": result.execution_reason,
                "output_shape": list(result.tensor.shape), "finite_output": True,
                "network_downloads": False,
                "scope": "One generated GPU-resident input; not live capture, image quality, FPS or Quest verification"}
    finally:
        engine.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(verify_depth(), ensure_ascii=True))
        return
    try:
        checked = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Depth model validation exceeded 120 seconds") from error
    if checked.returncode:
        raise RuntimeError("Depth model validation failed: " + checked.stderr[-12000:])
    result = json.loads(checked.stdout)
    output = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, "utf-8")
    print(output)


if __name__ == "__main__":
    main()
