"""Windows installer selects a pinned GPU runtime before downloads or updates."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
HELPER = ROOT / "scripts/release/gpu-discovery.ps1"
POLICY = ROOT / "config/gpu-runtimes.json"
pytestmark = pytest.mark.skipif(sys.platform != "win32" or not SHELL.is_file(), reason="Windows PowerShell 5.1 required")


def q(path):
    return "'" + str(path).replace("'", "''") + "'"


def run(tmp_path, body, *, success=True):
    runner = tmp_path / "probe.ps1"
    runner.write_text("$ErrorActionPreference='Stop'\n" + body, "utf-8-sig")
    environment = {k: v for k, v in os.environ.items() if k.casefold() != "psmodulepath"}
    checked = subprocess.run([str(SHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(runner)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                             cwd=ROOT, env=environment)
    assert (checked.returncode == 0) is success, checked.stdout + checked.stderr
    return checked


@pytest.mark.parametrize("capability,driver,profile", [
    ((7, 5), 12060, "cu126"), ((8, 6), 12060, "cu126"),
    ((8, 9), 12080, "cu126"), ((12, 0), 12080, "cu128"),
])
def test_driver_only_preflight_selects_runtime_without_marketing_name_matching(tmp_path, capability, driver, profile):
    major, minor = capability
    result = run(tmp_path, f"""
. {q(HELPER)}
$policy=Get-Content -LiteralPath {q(POLICY)} -Raw | ConvertFrom-Json
$detection=[pscustomobject]@{{driver_api={driver};devices=@([pscustomobject]@{{index=0;name='GPU fixture';capability=@({major},{minor})}})}}
Select-Quest3DNvidiaRuntime -Policy $policy -Detection $detection | ConvertTo-Json -Depth 8
""")
    plan = json.loads(result.stdout)
    assert plan["profile"] == profile and plan["torch"] == "2.7.1+" + profile
    assert plan["device"]["capability"] == list(capability)
    assert plan["validation"].startswith("driver preflight only")


@pytest.mark.parametrize("capability,driver,reason", [
    ((6, 1), 12080, "Unsupported NVIDIA"), ((12, 1), 12080, "Unsupported NVIDIA"),
    ((7, 5), 12050, "Update the driver"), ((12, 0), 12070, "Update the driver"),
])
def test_unsupported_architecture_or_older_ptx_jit_fails_before_install(tmp_path, capability, driver, reason):
    major, minor = capability
    result = run(tmp_path, f"""
. {q(HELPER)}
$policy=Get-Content -LiteralPath {q(POLICY)} -Raw | ConvertFrom-Json
$detection=[pscustomobject]@{{driver_api={driver};devices=@([pscustomobject]@{{index=0;name='GPU fixture';capability=@({major},{minor})}})}}
Select-Quest3DNvidiaRuntime -Policy $policy -Detection $detection
""", success=False)
    assert reason in result.stdout + result.stderr


def test_preflight_does_not_silently_select_another_gpu(tmp_path):
    result = run(tmp_path, f"""
. {q(HELPER)}
$policy=Get-Content -LiteralPath {q(POLICY)} -Raw | ConvertFrom-Json
$detection=[pscustomobject]@{{driver_api=12080;devices=@(
    [pscustomobject]@{{index=0;name='Unsupported first GPU';capability=@(6,1)}},
    [pscustomobject]@{{index=1;name='Supported second GPU';capability=@(12,0)}})}}
Select-Quest3DNvidiaRuntime -Policy $policy -Detection $detection
""", success=False)
    assert "Unsupported NVIDIA" in result.stdout + result.stderr


def installer_fixture(tmp_path, detection):
    folder = tmp_path / "package"
    names = ("scripts/release/install.ps1", "scripts/release/installation-lifecycle.ps1",
             "scripts/release/gpu-discovery.ps1", "config/gpu-runtimes.json")
    entries = {}
    for name in names:
        content = (ROOT / name).read_bytes()
        if name.endswith("gpu-discovery.ps1"):
            content += ("\nfunction Get-Quest3DCudaDevices {\n" + detection + "\n}\n").encode()
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (folder / "distribution-manifest.json").write_text(json.dumps({"schema": 1, "release": "fixture", "files": entries}), "utf-8")
    return folder


def test_actual_installer_rejects_gpu_before_destination_or_download_changes(tmp_path):
    folder = installer_fixture(tmp_path, "return [pscustomobject]@{driver_api=12080;devices=@([pscustomobject]@{index=0;name='fixture';capability=@(6,1)})}")
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "user-data.txt"
    sentinel.write_bytes(b"must survive")
    result = run(tmp_path, f"& {q(folder / 'scripts/release/install.ps1')} -Destination {q(destination)} -NoShortcuts", success=False)
    assert "Unsupported NVIDIA" in result.stdout + result.stderr
    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"must survive"
    assert not (folder / ".cache").exists()


def test_verify_only_stays_read_only_and_does_not_require_a_gpu(tmp_path):
    folder = installer_fixture(tmp_path, "throw 'GPU preflight should not run in VerifyOnly'")
    destination = tmp_path / "uncreated"
    result = run(tmp_path, f"& {q(folder / 'scripts/release/install.ps1')} -Destination {q(destination)} -VerifyOnly")
    assert "Package integrity verified" in result.stdout
    assert not destination.exists()
