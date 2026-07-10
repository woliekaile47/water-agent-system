from pathlib import Path
import os
import signal
import subprocess
import sys
import time

from waterlogging_simulation.generator import generate_case


SIMULATION_ROOT = Path(__file__).resolve().parents[1]


def test_ground_truth_node_sigint_exits_cleanly(tmp_path):
    generate_case(
        case_id="sim_dry_baseline_001",
        project_root=SIMULATION_ROOT.parent,
        config_dir=SIMULATION_ROOT / "config",
        world_template=SIMULATION_ROOT / "worlds" / "low_lying_road.sdf",
        output_root=tmp_path,
    )
    manifest_path = tmp_path / "sim_dry_baseline_001" / "manifest.json"
    code = "from waterlogging_simulation.ground_truth_node import main; main()"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SIMULATION_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["ROS_DOMAIN_ID"] = "197"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            "--ros-args",
            "-p",
            "use_sim_time:=false",
            "-p",
            f"project_root:={SIMULATION_ROOT.parent}",
            "-p",
            f"manifest_path:={manifest_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    time.sleep(2.0)
    if process.poll() is not None:
        output, _ = process.communicate(timeout=2)
        raise AssertionError(f"Ground Truth node exited before SIGINT: {output}")
    process.send_signal(signal.SIGINT)
    output, _ = process.communicate(timeout=10)
    assert process.returncode == 0, output
    assert "rcl_shutdown already called" not in output
