"""Headless MuJoCo plant and MAVLink simulator link; start PX4 separately."""
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def setup(context):
    share = get_package_share_directory("acp_mujoco_simulator")
    quad_name = LaunchConfiguration("quad_name").perform(context)
    doc = xacro.process_file(
        os.path.join(share, "model", "scene_px4.xml.xacro"),
        mappings={"quad_name": quad_name, "init_z": "0.03"},
    )
    # Keep generated MJCF in the run directory so a failed run is inspectable.
    run_dir = LaunchConfiguration("run_dir").perform(context)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "scene_px4.xml")
    with open(model_path, "w") as output:
        output.write(doc.toxml())
    return [Node(
        package="acp_mujoco_simulator", executable="mujoco_px4", output="screen",
        parameters=[{
            "model_path": model_path,
            "quad_name": quad_name,
            "mujoco_plugin_dir": LaunchConfiguration("mujoco_plugin_dir").perform(context),
            "simulator_port": int(LaunchConfiguration("simulator_port").perform(context)),
            "duration_s": float(LaunchConfiguration("duration_s").perform(context)),
            "csv_path": os.path.join(run_dir, "plant.csv"),
        }],
    )]


def generate_launch_description():
    ws = os.environ.get("COLCON_PAYLOAD_WS_DIR", os.getcwd())
    return LaunchDescription([
        DeclareLaunchArgument("quad_name", default_value="quadrotor"),
        DeclareLaunchArgument("mujoco_plugin_dir", default_value=os.path.join(
            ws, "mujoco-3.10.0", "bin", "mujoco_plugin")),
        DeclareLaunchArgument("simulator_port", default_value="4560"),
        DeclareLaunchArgument("duration_s", default_value="0.0"),
        DeclareLaunchArgument("run_dir", default_value=os.path.join(
            tempfile.gettempdir(), "acp_px4_mujoco")),
        OpaqueFunction(function=setup),
    ])
