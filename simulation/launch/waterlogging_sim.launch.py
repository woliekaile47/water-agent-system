"""Launch one configured Phase 1 Gazebo Fortress scenario."""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from waterlogging_simulation.generator import generate_case


def _launch_setup(context):
    package_share = Path(get_package_share_directory("waterlogging_simulation"))
    ros_gz_share = Path(get_package_share_directory("ros_gz_sim"))
    scenario = LaunchConfiguration("scenario").perform(context)
    project_root = Path(LaunchConfiguration("project_root").perform(context)).expanduser().resolve()
    output_root_value = LaunchConfiguration("output_root").perform(context)
    output_root = Path(output_root_value).expanduser().resolve() if output_root_value else None
    gui = LaunchConfiguration("gui").perform(context).lower() in {"1", "true", "yes"}

    manifest = generate_case(
        case_id=scenario,
        project_root=project_root,
        config_dir=package_share / "config",
        world_template=package_share / "worlds" / "low_lying_road.sdf",
        output_root=output_root,
    )
    world_path = Path(manifest["resolved_world_path"])
    if not world_path.is_absolute():
        world_path = project_root / world_path
    manifest_path = (
        (output_root / scenario / "manifest.json")
        if output_root is not None
        else (project_root / "data" / "simulation" / scenario / "manifest.json")
    )

    gz_args = f"-r -v 3 {world_path}"
    if not gui:
        gz_args = "-s " + gz_args
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(ros_gz_share / "launch" / "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items(),
    )

    topics = manifest["topic_names"]
    bridge_arguments = [
        f"{topics['clock']}@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        f"{topics['camera_image']}@sensor_msgs/msg/Image[gz.msgs.Image",
        f"{topics['camera_info']}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
    ]
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=bridge_arguments,
        output="screen",
    )
    ground_truth = Node(
        package="waterlogging_simulation",
        executable="ground_truth_publisher",
        name="ground_truth_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "project_root": str(project_root),
                "manifest_path": str(manifest_path),
            }
        ],
    )
    return [gazebo, bridge, ground_truth]


def generate_launch_description() -> LaunchDescription:
    default_root = os.environ.get("WATER_AGENT_SYSTEM_ROOT", str(Path.cwd()))
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario",
                default_value="sim_dry_baseline_001",
                description="Scenario ID from simulation/config/scenarios.yaml",
            ),
            DeclareLaunchArgument(
                "project_root",
                default_value=default_root,
                description="water_agent_system project root",
            ),
            DeclareLaunchArgument(
                "output_root",
                default_value="",
                description="Optional Ground Truth output root",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Start Gazebo GUI; camera rendering requires a valid DISPLAY",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
