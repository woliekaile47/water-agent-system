from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "scripts" / "run_simulation_agent_e2e_wsl.sh"
MATRIX = ROOT / "scripts" / "run_simulation_multiscenario_acceptance_wsl.sh"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_wsl_local_runners_have_valid_bash_syntax() -> None:
    for script in (SINGLE, MATRIX):
        completed = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_single_runner_reuses_frozen_prediction_components() -> None:
    text = _text(SINGLE)
    assert "run_simulation_agent_e2e_vm.py" in text
    assert "run_sam2_video_propagation.py" in text
    assert "configs/sam2.1/sam2.1_hiera_t.yaml" in text
    assert "finalize --run-id" in text
    assert "source /opt/ros/humble/setup.bash" in text


def test_matrix_uses_all_four_frozen_scenarios() -> None:
    text = _text(MATRIX)
    for label in ("5cm", "10cm", "20cm", "40cm"):
        assert label in text
    for config in (
        "configs/phase2d_c15_one_click_5cm.yaml",
        "configs/phase2d_c15_one_click_10cm.yaml",
        "configs/phase2d_c13_one_click_20cm.yaml",
        "configs/phase2d_c15_one_click_40cm.yaml",
    ):
        assert config in text


def test_local_runners_do_not_depend_on_vm_or_network_transfer() -> None:
    combined = (_text(SINGLE) + _text(MATRIX)).lower()
    for forbidden in (
        "192.168.218.135",
        "scp ",
        "ssh ",
        "vmhostname",
        "ground_truth/",
        "real_device_action_allowed=true",
        "external_notification_allowed=true",
    ):
        assert forbidden not in combined


def test_help_is_non_mutating_and_documents_simulation_safety() -> None:
    for script in (SINGLE, MATRIX):
        completed = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        help_text = completed.stdout.lower().replace("-", " ")
        assert "simulation" in help_text
        assert "ground truth" in help_text
