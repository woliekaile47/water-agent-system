[CmdletBinding()]
param(
    [string]$RunId = ("simulation_e2e_" + (Get-Date -Format "yyyyMMdd_HHmmss")),
    [string]$VmHostName = "192.168.218.135",
    [string]$VmUser = "wlkl",
    [string]$VmKeyPath = "D:\yujian_exchange\ssh\codex_vm_ed25519",
    [string]$VmRepo = "/home/wlkl/water_agent_ws/water_agent_system",
    [string]$WslHostName = "127.0.0.1",
    [int]$WslPort = 2222,
    [string]$WslUser = "wlkl",
    [string]$WslKeyPath = "D:\yujian_exchange\ssh\codex_wsl_ed25519",
    [string]$WslVenv = "/home/wlkl/venvs/sam2-gpu",
    [string]$WslCheckpoint = "/home/wlkl/ai_models/sam2/checkpoints/sam2.1_hiera_tiny.pt"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$") {
    throw "RunId may contain only letters, digits, underscore and hyphen."
}
foreach ($key in @($VmKeyPath, $WslKeyPath)) {
    if (-not (Test-Path -LiteralPath $key -PathType Leaf)) {
        throw "SSH private key does not exist: $key"
    }
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Program,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

function ConvertTo-RemoteBash {
    param([Parameter(Mandatory = $true)][string]$Script)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Script)
    $base64 = [Convert]::ToBase64String($bytes)
    return "echo $base64 | base64 -d | bash"
}

function Invoke-VmBash {
    param([Parameter(Mandatory = $true)][string]$Script)
    Invoke-NativeChecked "ssh" @(
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-i", $VmKeyPath,
        "$VmUser@$VmHostName",
        (ConvertTo-RemoteBash $Script)
    )
}

function Invoke-WslBash {
    param([Parameter(Mandatory = $true)][string]$Script)
    Invoke-NativeChecked "ssh" @(
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-i", $WslKeyPath,
        "-p", "$WslPort",
        "$WslUser@$WslHostName",
        (ConvertTo-RemoteBash $Script)
    )
}

$tempParent = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$tempRoot = Join-Path $tempParent ("water_agent_c13_" + $RunId)
if (Test-Path -LiteralPath $tempRoot) {
    throw "Temporary run directory already exists: $tempRoot"
}
New-Item -ItemType Directory -Path $tempRoot | Out-Null

$vmRunDir = "$VmRepo/outputs/phase2d_c13_one_click_runs/$RunId"
$vmInputArchive = "$vmRunDir/exchange_to_wsl.tar.gz"
$vmResultArchive = "$vmRunDir/incoming_sam2.tar.gz"
$localInputArchive = Join-Path $tempRoot "exchange_to_wsl.tar.gz"
$localResultArchive = Join-Path $tempRoot "sam2_result.tar.gz"
$wslInputArchive = "/tmp/water_agent_c13_${RunId}_input.tar.gz"
$wslResultArchive = "/tmp/water_agent_c13_${RunId}_result.tar.gz"

try {
    Write-Host "[1/6] VM: building Ground DEM from the dry simulated LiDAR bag"
    Write-Host "      and generating the automatic temporal SAM2 prompt..."
    Invoke-VmBash @"
set -eo pipefail
cd '$VmRepo'
source /opt/ros/humble/setup.bash
set -u
python3 scripts/run_simulation_agent_e2e_vm.py \
  --config configs/phase2d_c13_one_click_20cm.yaml \
  prepare --run-id '$RunId'
"@

    Write-Host "[2/6] Copying the frozen RGB window and automatic prompt to Windows..."
    Invoke-NativeChecked "scp" @(
        "-q", "-o", "BatchMode=yes", "-i", $VmKeyPath,
        "${VmUser}@${VmHostName}:$vmInputArchive",
        $localInputArchive
    )
    Invoke-NativeChecked "scp" @(
        "-q", "-o", "BatchMode=yes", "-i", $WslKeyPath, "-P", "$WslPort",
        $localInputArchive,
        "${WslUser}@${WslHostName}:$wslInputArchive"
    )

    Write-Host "[3/6] WSL/GPU: running SAM2 video propagation once..."
    Invoke-WslBash @"
set -eo pipefail
work=`$(mktemp -d '/tmp/water_agent_c13_${RunId}_XXXXXX')
cleanup() {
  rm -rf "`$work"
  rm -f '$wslInputArchive'
}
trap cleanup EXIT
tar -xzf '$wslInputArchive' -C "`$work"
source '$WslVenv/bin/activate'
set -u
window_start=`$(python3 -c "import json; print(json.load(open('`$work/payload/exchange_manifest.json'))['window_start'])")
window_end=`$(python3 -c "import json; print(json.load(open('`$work/payload/exchange_manifest.json'))['window_end'])")
anchor_frame=`$(python3 -c "import json; print(json.load(open('`$work/payload/exchange_manifest.json'))['anchor_frame_index'])")
python "`$work/payload/run_sam2_video_propagation.py" \
  --frames-dir "`$work/payload/frames" \
  --prompt-config "`$work/payload/automatic_prompt.json" \
  --window-start "`$window_start" \
  --window-end "`$window_end" \
  --anchor-frame-index "`$anchor_frame" \
  --model-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint '$WslCheckpoint' \
  --output-dir "`$work/result" \
  --device cuda
tar -czf '$wslResultArchive' -C "`$work" result
"@

    Write-Host "[4/6] Returning the frozen SAM2 masks to the VM..."
    Invoke-NativeChecked "scp" @(
        "-q", "-o", "BatchMode=yes", "-i", $WslKeyPath, "-P", "$WslPort",
        "${WslUser}@${WslHostName}:$wslResultArchive",
        $localResultArchive
    )
    Invoke-NativeChecked "scp" @(
        "-q", "-o", "BatchMode=yes", "-i", $VmKeyPath,
        $localResultArchive,
        "${VmUser}@${VmHostName}:$vmResultArchive"
    )
    Invoke-WslBash "rm -f '$wslResultArchive'"

    Write-Host "[5/6] VM: running ray-DEM inversion, quality gate, S5-S8 and Agent..."
    Invoke-VmBash @"
set -eo pipefail
cd '$VmRepo'
source /opt/ros/humble/setup.bash
set -u
python3 scripts/run_simulation_agent_e2e_vm.py \
  --config configs/phase2d_c13_one_click_20cm.yaml \
  finalize --run-id '$RunId' --sam2-archive '$vmResultArchive'
"@

    Write-Host "[6/6] Completed."
    Write-Host "VM result: $vmRunDir/completion_summary.json"
    Write-Host "This run is simulation-only and cannot trigger a real warning or device action."
}
finally {
    $resolvedTemp = [System.IO.Path]::GetFullPath($tempRoot)
    if ($resolvedTemp.StartsWith($tempParent, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedTemp)) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
    }
}
