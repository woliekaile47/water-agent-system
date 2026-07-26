[CmdletBinding()]
param(
    [string]$MatrixId = ("simulation_matrix_" + (Get-Date -Format "yyyyMMdd_HHmmss")),
    [string]$VmHostName = "192.168.218.135",
    [string]$VmUser = "wlkl",
    [string]$VmKeyPath = "D:\yujian_exchange\ssh\codex_vm_ed25519",
    [string]$VmRepo = "/home/wlkl/water_agent_ws/water_agent_system",
    [string]$WslHostName = "127.0.0.1",
    [int]$WslPort = 2222,
    [string]$WslUser = "wlkl",
    [string]$WslKeyPath = "D:\yujian_exchange\ssh\codex_wsl_ed25519"
)

$ErrorActionPreference = "Stop"
if ($MatrixId -notmatch "^[A-Za-z0-9][A-Za-z0-9_-]{0,60}$") {
    throw "MatrixId may contain only letters, digits, underscore and hyphen."
}
$runner = Join-Path $PSScriptRoot "run_simulation_agent_e2e_windows.ps1"
$scenarios = @(
    @{ Label = "5cm"; Config = "configs/phase2d_c15_one_click_5cm.yaml" },
    @{ Label = "10cm"; Config = "configs/phase2d_c15_one_click_10cm.yaml" },
    @{ Label = "20cm"; Config = "configs/phase2d_c13_one_click_20cm.yaml" },
    @{ Label = "40cm"; Config = "configs/phase2d_c15_one_click_40cm.yaml" }
)

foreach ($scenario in $scenarios) {
    Write-Host "=== Phase 2D-C-15: $($scenario.Label) ==="
    & $runner `
        -RunId "${MatrixId}_$($scenario.Label)" `
        -ConfigPath $scenario.Config `
        -VmHostName $VmHostName -VmUser $VmUser -VmKeyPath $VmKeyPath -VmRepo $VmRepo `
        -WslHostName $WslHostName -WslPort $WslPort -WslUser $WslUser `
        -WslKeyPath $WslKeyPath
    if ($LASTEXITCODE -ne 0) {
        throw "Scenario $($scenario.Label) failed."
    }
}

$remote = @"
set -eo pipefail
cd '$VmRepo'
source /opt/ros/humble/setup.bash
python3 scripts/summarize_phase2d_c15_matrix.py --matrix-id '$MatrixId'
"@
$encoded = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($remote))
& ssh -o BatchMode=yes -i $VmKeyPath "$VmUser@$VmHostName" "echo $encoded | base64 -d | bash"
if ($LASTEXITCODE -ne 0) {
    throw "Matrix summary failed."
}
Write-Host "Matrix report: $VmRepo/outputs/phase2d_c15_multiscenario_acceptance/$MatrixId"
