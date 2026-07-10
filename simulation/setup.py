from glob import glob
from pathlib import Path
from setuptools import find_packages, setup


PACKAGE_NAME = "waterlogging_simulation"


def files_only(pattern):
    return [path for path in glob(pattern) if Path(path).is_file()]


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("tests",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        ("share/" + PACKAGE_NAME, ["package.xml", "README.md"]),
        ("share/" + PACKAGE_NAME + "/launch", glob("launch/*.launch.py")),
        ("share/" + PACKAGE_NAME + "/config", glob("config/*.yaml")),
        ("share/" + PACKAGE_NAME + "/worlds", glob("worlds/*.sdf")),
        ("share/" + PACKAGE_NAME + "/scripts", files_only("scripts/*")),
        ("share/" + PACKAGE_NAME + "/models/road_basin", glob("models/road_basin/*")),
        ("share/" + PACKAGE_NAME + "/models/water_surface", glob("models/water_surface/*")),
        ("share/" + PACKAGE_NAME + "/models/roadside_sensor_rig", glob("models/roadside_sensor_rig/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="water-agent-system team",
    maintainer_email="wlkl@example.com",
    description="Gazebo Fortress minimum loop for two-stage waterlogging perception",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ground_truth_publisher = waterlogging_simulation.ground_truth_node:main",
            "set_scenario = waterlogging_simulation.cli:set_scenario_main",
            "export_ground_truth = waterlogging_simulation.cli:export_ground_truth_main",
        ],
    },
)
