<# Driver-only preflight: no Torch/Python, downloads, driver update or device setting changes. #>
Set-StrictMode -Version Latest

function Get-Quest3DCudaDevices {
    # Use the system NVIDIA driver, not a DLL next to a downloaded installer.
    if ($null -eq ('Sterevi.InstallerCudaDriver' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
namespace Sterevi {
    public static class InstallerCudaDriver {
        [DllImport("nvcuda.dll"), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        public static extern int cuInit(uint flags);
        [DllImport("nvcuda.dll"), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        public static extern int cuDriverGetVersion(out int version);
        [DllImport("nvcuda.dll"), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        public static extern int cuDeviceGetCount(out int count);
        [DllImport("nvcuda.dll"), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        public static extern int cuDeviceGet(out int device, int ordinal);
        [DllImport("nvcuda.dll"), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        public static extern int cuDeviceGetAttribute(out int value, int attribute, int device);
        [DllImport("nvcuda.dll", CharSet = CharSet.Ansi), DefaultDllImportSearchPaths(DllImportSearchPath.System32)]
        public static extern int cuDeviceGetName(StringBuilder name, int size, int device);
    }
}
'@
    }
    try {
        $status = [Sterevi.InstallerCudaDriver]::cuInit(0)
        if ($status -ne 0) { throw "NVIDIA CUDA driver initialization failed (code $status)." }
        [int]$version = 0
        [int]$count = 0
        if ([Sterevi.InstallerCudaDriver]::cuDriverGetVersion([ref]$version) -ne 0 -or
            [Sterevi.InstallerCudaDriver]::cuDeviceGetCount([ref]$count) -ne 0 -or $count -lt 1) {
            throw 'An available NVIDIA CUDA device and driver are required.'
        }
        $devices = @()
        for ($index = 0; $index -lt $count; $index++) {
            [int]$device = 0
            [int]$major = 0
            [int]$minor = 0
            $name = New-Object Text.StringBuilder 256
            if ([Sterevi.InstallerCudaDriver]::cuDeviceGet([ref]$device, $index) -ne 0 -or
                [Sterevi.InstallerCudaDriver]::cuDeviceGetAttribute([ref]$major, 75, $device) -ne 0 -or
                [Sterevi.InstallerCudaDriver]::cuDeviceGetAttribute([ref]$minor, 76, $device) -ne 0 -or
                [Sterevi.InstallerCudaDriver]::cuDeviceGetName($name, $name.Capacity, $device) -ne 0) {
                throw "NVIDIA device $index could not be identified reliably."
            }
            $devices += [pscustomobject]@{index=$index;name=$name.ToString();capability=@($major,$minor)}
        }
        return [pscustomobject]@{driver_api=$version;devices=$devices}
    } catch {
        throw ('NVIDIA GPU preflight failed. Install a current NVIDIA driver and retry. No GPU libraries were downloaded. ' + $_.Exception.Message)
    }
}

function Select-Quest3DNvidiaRuntime {
    param([Parameter(Mandatory=$true)][object]$Policy,
          [Parameter(Mandatory=$true)][object]$Detection)
    if ($Policy.schema -ne 1 -or $Policy.device_index -ne 0) { throw 'Unsupported NVIDIA runtime policy.' }
    $devices = @($Detection.devices)
    $selected = @($devices | Where-Object { $_.index -eq $Policy.device_index })
    if ($selected.Count -ne 1 -or @($selected[0].capability).Count -ne 2) { throw 'CUDA device 0 could not be identified reliably.' }
    $device = $selected[0]
    $key = @($device.capability) -join '.'
    $profiles = @($Policy.profiles.PSObject.Properties | Where-Object {
        $candidates = @($_.Value.capabilities | ForEach-Object { @($_) -join '.' })
        $key -cin $candidates
    })
    if ($profiles.Count -ne 1) {
        throw "Unsupported NVIDIA compute capability $key. This build supports GeForce RTX 20/30/40/50 series; AMD, Intel and older NVIDIA GPUs are not supported. No GPU libraries were downloaded."
    }
    $profile = $profiles[0].Value
    if ($Detection.driver_api -lt $profile.minimum_driver_api) {
        throw "The selected GPU needs an NVIDIA driver exposing CUDA $($profile.cuda_version) or later (driver API $($profile.minimum_driver_api)); found $($Detection.driver_api). Update the driver and retry. No GPU libraries were downloaded."
    }
    return [pscustomobject]@{schema=1;profile=$profiles[0].Name;extra=$profile.extra;
        torch=$profile.torch;torchvision=$profile.torchvision;cuda_version=$profile.cuda_version;device_index=0;device=$device;
        driver_api=$Detection.driver_api;minimum_driver_api=$profile.minimum_driver_api;
        validation='driver preflight only; actual Torch and CUDA kernels are checked after installation'}
}

function Get-Quest3DNvidiaRuntimePlan([string]$PackageRoot) {
    $policy = Get-Content -LiteralPath (Join-Path $PackageRoot 'config/gpu-runtimes.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    return Select-Quest3DNvidiaRuntime -Policy $policy -Detection (Get-Quest3DCudaDevices)
}
