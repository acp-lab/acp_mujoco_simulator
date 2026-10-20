# ACP MuJoCo Payload Transportation Simulator Setup

![ACP MuJoCo Payload Transportation Simulator](video/simulator_mujoco.gif)

At this point, you should have successfully installed all required **ACP Lab core packages**.

## PX4 SITL through the MuJoCo MAVLink plugin

`MujocoRosUtils::MavlinkInterface` connects PX4 SITL directly to the scene.
It works in MuJoCo's standard `simulate` viewer and the generic
`mujoco_headless` runner through the existing `acp_autonomy` launch files.
PX4 is pinned to v1.16.0, commit
`6ea3539157ca358c70a515878b77077af7d4611d`, in `.px4_runtime/PX4-Autopilot`.

### How the connection fits together

```text
acp_autonomy launch → scene Xacro → MavlinkInterface plugin
MuJoCo flight sensors → MAVLink TCP → PX4 estimator/controllers
MuJoCo RotorMotor actuators ← MAVLink TCP ← PX4 motor outputs
```

The implementation is split across three packages:

| Package | Files to read | Responsibility |
| --- | --- | --- |
| `MujocoRosUtils` | `plugin/MavlinkInterface.cpp` and `.h` | Read native sensors, exchange MAVLink messages, apply motor commands. `Plugin.cpp` registers it; `plugin/CMakeLists.txt` enables its build. |
| `acp_mujoco_simulator` | `model/quadrotor_acro_macro.xml.xacro` and the `scene_acro*.xml.xacro` files used by the launches below | Attach the plugin and select PX4 instead of AcroMode for the same four rotor actuators. |
| `acp_mujoco_simulator` | `scripts/run_px4_sitl.py` and `config/px4_mujoco.json` | Start PX4 with the simulator connection and quadrotor parameters. `scripts/setup_px4_sitl.sh` and `config/requirements-px4.txt` prepare the PX4 build. |
| `acp_autonomy` | The seven simulator launch files listed below | Pass `controller_backend`, timestep and port settings into the scenes. |

PX4's existing controllers implement takeoff, hover and landing. No PX4
controller source changes are required. The existing ROS sensor publishers
continue independently; LiDAR and cameras are not sent to PX4.

### Build

From the workspace root (these binaries are already built in this workspace):

```bash
export COLCON_PAYLOAD_WS_DIR="$PWD"
bash src/acp_mujoco_simulator/scripts/setup_px4_sitl.sh "$PWD"
source install/setup.bash
colcon build --packages-select mujoco_ros_utils --symlink-install \
  --cmake-args -DMUJOCO_ROS_UTILS_MAVLINK=ON \
  -DMUJOCO_ROOT_DIR="$PWD/mujoco-3.10.0" \
  -DMAVLINK_INCLUDE_DIR="$PWD/.px4_runtime/PX4-Autopilot/build/px4_sitl_default/mavlink"
colcon build --packages-select acp_mujoco_simulator acp_autonomy --symlink-install
source install/setup.bash
```

The setup script needs Git, CMake, Ninja, a C/C++ compiler, and `uv` or Python
`venv` support. It downloads PX4 and Python dependencies into `.px4_runtime`.
The connection needs no Gazebo, MAVROS, `px4_msgs`, or DDS agent.

### Run with your existing visual launch files

Terminal 1, from the workspace root:

```bash
export COLCON_PAYLOAD_WS_DIR="$PWD"
source install/setup.bash
ROS_DOMAIN_ID=147 ros2 launch acp_autonomy \
  single_quadrotor_nmpc_acro_simulation_mujoco.launch.py controller_backend:=px4
```

The MuJoCo window opens as usual. The plugin prints
`Waiting for PX4 on TCP 127.0.0.1:4560` once simulation steps begin.
If the viewer is paused, press Space to run it.

Terminal 2, from the same workspace:

```bash
ROS_DOMAIN_ID=147 python3 src/acp_mujoco_simulator/scripts/run_px4_sitl.py \
  --px4-dir .px4_runtime/PX4-Autopilot --run-dir .px4_runtime/mujoco
```

Wait for `lockstep active` in the simulator and `Ready for takeoff!` in PX4.
For a first flight with the single quadrotor, enter at the `pxh>` prompt:

```text
param set MIS_TAKEOFF_ALT 1.5
commander takeoff
```

`commander takeoff` selects PX4's automatic takeoff mode and arms the vehicle.
PX4 climbs to 1.5 m above its starting altitude, then holds position. After
observing the hover, enter:

```text
commander land
```

PX4 descends, detects ground contact, and automatically disarms. This uses
PX4's position, attitude and body-rate controllers, estimator, and motor
allocation; MuJoCo provides the dynamics and simulated flight sensors.
No NMPC controller, LiDAR or camera data is needed for
this initial test. Automatic takeoff/landing use PX4's autonomous flight
modes; manual Acro mode controls body rates and does not hold altitude.

At the `pxh>` prompt, `simulator_mavlink status`, `listener sensor_combined`,
and `listener actuator_outputs_sim` inspect the connection. Once landed,
use `shutdown` to stop PX4, then close the MuJoCo window (or Ctrl+C for a
headless launch).

All seven launch files accept `controller_backend:=px4`:

- `single_quadrotor_nmpc_acro_simulation_mujoco.launch.py`
- `single_quadrotor_nmpc_acro_gate_simulation_mujoco.launch.py`
- `single_quadrotor_nmpc_acro_lidar_simulation_mujoco.launch.py`
- `single_quadrotor_payload_nmpc_acro_simulation_mujoco.launch.py`
- `single_quadrotor_payload_nmpc_acro_lidar_simulation_mujoco.launch.py`
- `multiple_quadrotor_nmpc_acro_simulation_mujoco.launch.py`
- `multiple_quadrotor_nmpc_acro_lidar_simulation_mujoco.launch.py`

Their default is still `controller_backend:=acro`. PX4 selects a default
1 ms physics timestep and 250 Hz sensor/actuator exchange; the original acro
mode keeps its 2.5 ms default. Existing environment, payload, LiDAR and ROS
publisher configuration is preserved. `headless:=true` selects the generic
runner. Single-vehicle launches accept `px4_simulator_port:=4561` to pair with
PX4 `--instance 1`.

### Multiple vehicles

Each vehicle needs a separate PX4 process and parameter directory. Plugin
ports are `4560 + instance`; PX4's offboard UDP ports are `14540 + instance`.
Instances follow the `quadrotors_mujoco` YAML order, starting at
`px4_instance_start` (default 0, supported indices 0–9). The launch prints the
mapping. With the current `dq_multiple.yaml`:

| Vehicle | PX4 instance | Simulator TCP | Offboard UDP |
| --- | --- | --- | --- |
| eagle8 | 0 | 4560 | 14540 |
| eagle10 | 1 | 4561 | 14541 |
| eagle11 | 2 | 4562 | 14542 |

Start the multi-vehicle scene in terminal 1:

```bash
ROS_DOMAIN_ID=147 ros2 launch acp_autonomy \
  multiple_quadrotor_nmpc_acro_lidar_simulation_mujoco.launch.py controller_backend:=px4
```

In three more terminals, run the PX4 command below with `0`, `1`, and `2`,
respectively, changing both the instance and its run directory:

```bash
ROS_DOMAIN_ID=147 python3 src/acp_mujoco_simulator/scripts/run_px4_sitl.py \
  --px4-dir .px4_runtime/PX4-Autopilot \
  --instance 0 --run-dir .px4_runtime/mujoco_0
```

The vehicles share one MuJoCo physics clock. A connected PX4 that stalls can
hold a step for up to two seconds; its plugin then disables that vehicle's
motor commands. A vehicle without a connected PX4 has zero motor commands.
This verifies the simulator connections, not coordinated payload flight.

### Flight sensors and motor commands

| Data | ROS 2 | PX4 simulator connection |
| --- | --- | --- |
| Accelerometer/gyro | Existing ImuPublisher | HIL_SENSOR, 250 Hz |
| Magnetometer/barometer | Independent of ROS publishers | HIL_SENSOR fields, 50 Hz |
| Ground-truth pose/velocity | Existing OdometryPublisher | Position/velocity derive HIL_GPS, 10 Hz |
| LiDAR/cameras | Existing publisher configuration | No point-cloud/image forwarding |

In `MavlinkInterface.cpp`, `sample_sensors()` reads MuJoCo measurements,
`send_sensors()` sends `HIL_SENSOR` and `HIL_GPS`, and `apply_controls()` maps
received `HIL_ACTUATOR_CONTROLS` to the four motors. `Bridge::advance()` manages
the connection and 250 Hz exchange. Sensor `compute()` only caches data:
rendering, repeated sensor calculations and viewer copies do not send packets.
Commands apply after the sampled step, adding one physics step of latency.

The plugin converts body FLU to PX4 FRD and world ENU to NED. Its IMU uses
native measurements, independently of ROS publisher noise. Magnetometer and
barometer readings are synthesized with small noise; GPS derives from the
simulated position and velocity using a fixed geographic origin.

PX4 controls the four existing RotorMotor actuators; AcroMode is omitted for
that vehicle. The bridge expects the shared macro's motor names, 3000 rad/s
speed limit and `kf=2.063751543951938e-6`. `THR_MDL_FAC=1` in
`config/px4_mujoco.json` performs PX4's square-root thrust linearization, so
normalized outputs map to `omega=3000*u`. Existing 18 ms actuator delay and
25 ms motor time constant remain. The supplied gains/hover thrust are an
initial baseline for the 1.24 kg quadrotor. Other masses and payload flight
need appropriate PX4 parameters and controller validation.

The plugin faults on invalid commands, disconnects or a two-second actuator
timeout. Motor commands are zeroed and the viewer continues running; rotor
speed decays through the motor model. Reset the scene and restart PX4 to
establish a new simulation clock after a fault, reset, rewind or reload.
Pause/resume without resetting does not advance simulation timestamps.

The single-quadrotor launch has passed `commander takeoff`, five seconds of
position hold near 1.5 m, and `commander land` with automatic disarming.
All three PX4 instances also reached readiness in the multi-vehicle LiDAR
scene while ROS sensors continued publishing. Full NMPC/payload missions
have not been validated with PX4.

To use the **ACP MuJoCo Simulator** for payload transportation, we recommend creating a **dedicated ROS 2 workspace**:

```text
payload_transportation_ws
```

---

## Optional environment contacts in acro scenes

The acro launch files in `acp_autonomy` expose
`enable_environment_contacts:=true|false` (default `true`). Setting it to `false`
disables collisions with static environment geometry: floor, walls, ceiling,
boxes, pillars, and gates. Geometry remains visible; bodies can pass through it.
Contacts between moving bodies and tendon limits used for cables are unchanged.

```bash
ros2 launch acp_autonomy single_quadrotor_payload_nmpc_acro_simulation_mujoco.launch.py \
  quad_name:=eagle11 init_z:=1.5 payload_x:=0.0 payload_z:=0.5 \
  enable_environment_contacts:=false
```

The option is passed to the scene Xacro, which sets both `contype` and
`conaffinity` to zero for the `environment` geometry class. New static obstacles
should use `class="environment"` to follow this setting. The older dedicated
gate scene has explicit contact masks, so it also sets its gate mask to zero
when the option is false. When true, its separate `gates_collide` option keeps
its existing meaning. No global contact or constraint disable flag is used.

This applies to the single quadrotor acro, payload, gate, and LiDAR scenes and
the multiple quadrotor point-mass scenes. It works with both the viewer and
headless executable because both load the generated MJCF model. No C++ parameter
or headless executable change is needed.

## Optional payload gate

The payload launches, with or without LiDAR, have `enable_gate:=false` by default. Enable it with:

```bash
ros2 launch acp_autonomy single_quadrotor_payload_nmpc_acro_simulation_mujoco.launch.py \
  quad_name:=eagle11 \
  init_x:=0.0 init_y:=0.0 init_z:=1.5 \
  payload_x:=0.0 payload_y:=0.0 payload_z:=0.5 \
  enable_gate:=true
```

The simulator gate is centered at (2.0, 0.0, 2.0) m, matching the planner's
gate position in world coordinates.
Its clear opening radius is 0.40 m, also matching the planner.

| Launch argument | Default | Meaning |
| --- | --- | --- |
| `gate_x`, `gate_y`, `gate_z` | `2.0`, `0.0`, `2.0` | Opening center in world coordinates, metres |
| `gate_radius` | `0.4` | Clear opening radius, metres (0.8 m diameter) |
| `gate_thickness` | `0.1` | Full gate depth along the world x axis, metres |
| `enable_second_gate` | `false` | Add a second gate independently of `enable_gate` |
| `second_gate_x`, `second_gate_y`, `second_gate_z` | `3.5`, `0.0`, `2.0` | Second opening center in world coordinates, metres |
| `second_gate_radius` | `0.4` | Second clear opening radius, metres |
| `second_gate_thickness` | `0.1` | Second full gate depth along world x, metres |

These are launch defaults, not values read automatically from the planner.
If you change the planner's gate, pass the corresponding values here too.

For the `multiple_gates_x3p5.csv` trajectory, enable both gates. They are centered at
(2.0, 0.0, 2.0) m and (3.5, 0.0, 2.0) m:

```bash
ros2 launch acp_autonomy single_quadrotor_payload_nmpc_acro_simulation_mujoco.launch.py \
  quad_name:=eagle11 \
  init_x:=0.0 init_y:=0.0 init_z:=1.5 \
  payload_x:=0.0 payload_y:=0.0 payload_z:=0.5 \
  enable_gate:=true enable_second_gate:=true
```

The orange ring is static, with its opening in the world yz plane. It uses
64 capsule segments with a chord correction to preserve the clear radius.
It represents a finite, rounded ring; the planner uses a wall slab with a
circular opening. The planner's body radii and clearance margins constrain
body centers and do not shrink the simulator's opening.

The ring follows `enable_environment_contacts`: contacts are enabled by
default; setting it to `false` keeps the gate visible without collisions.
Each gate's enable flag removes its ring when false. After adding this macro to an
existing workspace, install it with:

```bash
colcon build --packages-select acp_mujoco_simulator acp_autonomy --symlink-install
source install/setup.bash
```

## Payload simulation with LiDAR

The payload LiDAR launch accepts the same arguments and defaults as the payload launch
above, including initial poses, masses, inertia, cable length, both gates,
sensor rates, and headless duration:

```bash
source install/setup.bash
ros2 launch acp_autonomy single_quadrotor_payload_nmpc_acro_lidar_simulation_mujoco.launch.py \
  quad_name:=eagle11 headless:=false \
  init_x:=0.0 init_y:=0.0 init_z:=1.5 \
  payload_x:=0.0 payload_y:=0.0 payload_z:=0.5 \
  enable_gate:=true enable_second_gate:=true second_gate_x:=3.5
```

This launch runs a scene with one payload attached to the quadrotor.
It retains the LiDAR environment and publishes Livox measurements on
`/eagle11/livox/lidar` (`lidar_hz:=10` by default). The simulated quadrotor
odometry remains ground truth; a separate LiDAR odometry estimator must process
the LiDAR and IMU measurements to supply estimated odometry to TI-MPC/TI-MHE.
This launch does not change controller or estimator parameters or subscriptions.

The LiDAR scene retains its walls, landmarks, and ceiling. The ceiling underside
is at z = 2.96 m; these obstacles also participate in collisions when
`enable_environment_contacts:=true` (the default).

Both gates are disabled by default. `duration_s` only stops headless runs;
use `headless:=true duration_s:=10.0` for a ten-second simulation.

## 1. Create and Configure the Workspace

Create the workspace directory (the location is up to you) and export it as an environment variable.

### Edit your `~/.bashrc`

```bash
vim ~/.bashrc
```

Add the following line (adjust the path if needed):

```bash
export COLCON_WS_DIR="$HOME/payload_transportation_ws"
```

Reload your environment:

```bash
source ~/.bashrc
```

> **Note**
> `~/payload_transportation_ws` is only an example. Ensure this path matches the actual location of your workspace.

---

## 2. Create the Setup Script

Navigate to your workspace:

```bash
cd $COLCON_WS_DIR
```

Create a setup script:

```bash
vim setup_acp_payload_transportation_simulator.sh
```

Paste the following content:

```bash
#!/bin/bash
set -e

mkdir -p src
cd src

# Start SSH agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# ACP autonomy stack
if [ ! -d "acp-autonomy-stack" ]; then
  git clone git@github.com:acp-lab/acp-autonomy-stack.git
  pushd acp-autonomy-stack
  git checkout payload_transportation_simulator
  popd
fi

# Quadrotor control
if [ ! -d "acp-quadrotor-control" ]; then
  git clone git@github.com:acp-lab/acp-quadrotor-control.git
  pushd acp-quadrotor-control
  git checkout payload_transportation_simulator
  popd
fi

# DQ-NMPC
if [ ! -d "dq_nmpc" ]; then
  git clone git@github.com:acp-lab/dq_nmpc.git
  pushd dq_nmpc
  git checkout payload_transportation_simulator
  popd
fi

# Raspberry Pi ROS 2 interface
if [ ! -d "pi_ros2_interface" ]; then
  git clone git@github.com:acp-lab/pi_ros2_interface.git
  pushd pi_ros2_interface
  git checkout main
  popd
fi

# ACP MuJoCo simulator
if [ ! -d "acp_mujoco_simulator" ]; then
  git clone git@github.com:acp-lab/acp_mujoco_simulator.git
  pushd acp_mujoco_simulator
  git checkout payload_transportation_simulator
  popd
fi

cd ..

# Build workspace
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

---

## 3. Make the Script Executable

```bash
chmod +x setup_acp_payload_transportation_simulator.sh
```

---

## 4. Run the Setup Script

From inside your workspace:

```bash
cd $COLCON_WS_DIR
source setup_acp_payload_transportation_simulator.sh
```

This will clone all required repositories and build the workspace.

---

## 5. Build DQ-NMPC (Base Controller)

To build and use **DQ-NMPC** as the base controller for payload transportation:

```bash
cd $COLCON_WS_DIR/src

# Clone dual-quaternion C++ library
git clone git@github.com:acp-lab/dq_cpp.git
cd dq_cpp
git checkout payload_transportation_simulator

# Build DQ-NMPC for onboard usage
cd $COLCON_WS_DIR/src/dq_nmpc
chmod +x build_dq_nmpc_onboard.sh
source build_dq_nmpc_onboard.sh
```

When prompted for the platform type, enter:

```text
eagle
```

> **Note**
> Ensure all dependencies are installed and that the `dq_nmpc` repository was cloned in the previous steps.

---

## 6. Install MuJoCo

We use the **latest stable MuJoCo version** available at the time of this project.

From inside your workspace:

```bash
cd $COLCON_WS_DIR
```

Download and extract MuJoCo:

```bash
curl -L -o mujoco-3.4.0-linux-x86_64.tar.gz \
https://github.com/google-deepmind/mujoco/releases/download/3.4.0/mujoco-3.4.0-linux-x86_64.tar.gz

tar -xzf mujoco-3.4.0-linux-x86_64.tar.gz
rm mujoco-3.4.0-linux-x86_64.tar.gz
```

Clone MuJoCo ROS utilities:

```bash
cd $COLCON_WS_DIR/src
git clone git@github.com:acp-lab/MujocoRosUtils.git
cd MujocoRosUtils
git checkout payload_transportation_simulator
```

Build the MuJoCo ROS utilities:

```bash
cd $COLCON_WS_DIR
colcon build \
  --packages-select mujoco_ros_utils \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMUJOCO_ROOT_DIR=$COLCON_WS_DIR/mujoco-3.4.0

source install/setup.bash
```

---

## 7. Run the Simulator

### Start the MuJoCo simulator

```bash
cd $COLCON_WS_DIR
source install/setup.bash
ros2 launch acp_autonomy single_quadrotor_mujoco_sim.launch.py
```

### Start MAV manager

```bash
ros2 run mav_manager mav_manager_service_exec
```

### Run trajectory manager tests

```bash
ros2 run mav_manager_test main_test
```

### Payload initial pose, physical parameters, and headless duration

The payload simulation launch accepts the quadrotor's initial position and
orientation, the payload's initial position, physical parameters, and an optional
headless duration:

```bash
ros2 launch acp_autonomy single_quadrotor_payload_nmpc_acro_simulation_mujoco.launch.py \
  quad_name:=eagle11 headless:=true duration_s:=20.0 \
  init_roll:=0.0 init_pitch:=0.0 \
  payload_x:=-0.7 payload_y:=0.0 payload_z:=0.02
```

| Argument | Default | Meaning |
|---|---|---|
| `init_x`, `init_y`, `init_z` | `0.0`, `0.0`, `0.03` | Quadrotor body-origin position in world coordinates, metres |
| `init_roll`, `init_pitch`, `init_yaw` | `0.0`, `0.0`, `0.0` | Initial quadrotor Euler angles, radians |
| `payload_x`, `payload_y`, `payload_z` | `-0.7`, `0.0`, `0.02` | Initial payload position in world coordinates, metres |
| `quadrotor_mass` | `1.24` | Quadrotor mass, kg |
| `quadrotor_ixx` | `0.0027338143976` | Quadrotor inertia about the body x-axis through its CoM, kg m² |
| `quadrotor_iyy` | `0.0027336327812` | Quadrotor inertia about the body y-axis through its CoM, kg m² |
| `quadrotor_izz` | `0.0052991944907` | Quadrotor inertia about the body z-axis through its CoM, kg m² |
| `payload_mass` | `0.2` | Mass of the cable-connected payload, kg |
| `cable_length` | `0.95` | Maximum tendon length between attachment sites, metres |
| `duration_s` | `0.0` | Headless simulation duration in simulated seconds; zero runs until stopped |

Pose and physical arguments apply to both graphical and headless runs. `duration_s` applies
only with `headless:=true` and must be finite and nonnegative. The simulator
stops at the first completed physics step reaching the requested duration, so
the endpoint has physics-timestep resolution. Existing launches that omit
`duration_s` continue running until stopped.

For example, to change the plant parameters:

```bash
ros2 launch acp_autonomy single_quadrotor_payload_nmpc_acro_simulation_mujoco.launch.py \
  quad_name:=eagle11 headless:=true duration_s:=20.0 \
  quadrotor_mass:=1.5 \
  quadrotor_ixx:=0.0035 quadrotor_iyy:=0.0035 quadrotor_izz:=0.006 \
  payload_mass:=0.3 cable_length:=1.2
```

The inertia remains diagonal: `diag(Ixx, Iyy, Izz)`. Mass and inertia are
independent inputs; changing `quadrotor_mass` does not rescale the inertia.
These arguments configure the MuJoCo plant. Controller model parameters remain
configured separately, so use matching values there for nominal-model experiments.
The single-payload scenes, with or without LiDAR, attach the cable 5 cm below the quadrotor
body origin (`payload_site_pos="0.0 0.0 -0.05"`). With a level quadrotor,
`init_z:=1.5`, `payload_z:=0.5`, and equal x/y coordinates, the initial
attachment-to-payload distance is `1.5 - 0.05 - 0.5 = 0.95 m`, matching the
default cable length.

Changing `cable_length` sets the tendon limit; initial positions remain the values
supplied through the pose arguments.
