# [TWINS-Software](https://github.com/isri-aist/TWINS-Software)

Software for **TWINS (Tactile Wearable Isomorphic Arm Networked System)**:
wearable leader recording, Dynamixel follower control, eSkin sensing, policy
training and rollout, and RViz visualization. The Python package is
`twin_arms_controller` and uses [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines).

## Requirements and installation

- Ubuntu 22.04, Python 3.10, and ROS 2 Humble.
- A CUDA-capable GPU for policy training.
- TWINS leader/follower hardware for physical operation. Model visualization
  and playback of existing RMB recordings do not require a robot.
- eSkin sensing is optional. Its hardware driver is provided to eSkin
  purchasers and is not included in this repository.

Install ROS 2 Humble first, then install the additional system packages:

```bash
sudo apt install python3-venv ffmpeg libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf \
  ros-humble-rviz2 ros-humble-robot-state-publisher \
  ros-humble-joint-state-publisher-gui
```

Clone both repositories into the same workspace. All later commands run from
the `TWINS-Software` root unless explicitly stated otherwise.

```bash
git clone https://github.com/isri-aist/TWINS-Software.git
cd TWINS-Software
git clone --recursive https://github.com/isri-aist/RoboManipBaselines.git ../RoboManipBaselines
git -C ../RoboManipBaselines checkout TWINS
git -C ../RoboManipBaselines submodule update --init --recursive
```

Create a dedicated environment. The ROS packages installed above remain
available through the system Python packages.

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '../RoboManipBaselines[real-ur5e,diffusion-policy]'
python -m pip install -e ../RoboManipBaselines/third_party/gello_software
python -m pip install -e ../RoboManipBaselines/third_party/diffusion_policy
python -m pip install -e .
```

In each new terminal, enter this repository and activate ROS and the environment:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
```

## Hardware and configuration

Connect one device at a time and obtain its USB serial number:

```bash
udevadm info --query=property --name=/dev/ttyUSB0 | grep '^ID_SERIAL_SHORT='
```

Replace the six `*_SERIAL` placeholders in
[udev/99-twin-arms.rules](udev/99-twin-arms.rules). They correspond to the left
and right leader encoders and the X/P-series follower interfaces on each side.

```bash
sudo install -m 644 udev/99-twin-arms.rules /etc/udev/rules.d/99-twin-arms.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
```

Reconnect the USB cables and check the device names:

```bash
ls -l /dev/leader_arm_* /dev/follower_arm_*
```

Edit the YAML files in [twin_arms_controller/envs/configs](twin_arms_controller/envs/configs)
directly to match your hardware.

**Before enabling follower torque**, check motor IDs, joint directions, offsets,
limits and current settings. The motor mapping is in
[RealTwinArmsFollowerEnv.py](twin_arms_controller/envs/follower/RealTwinArmsFollowerEnv.py)
and the drive settings are in
[RealTwinArmsFollower.yaml](twin_arms_controller/envs/configs/RealTwinArmsFollower.yaml).
Supplied offsets are reference-hardware values, not a calibration for every robot.

To recalibrate the leader offsets, record a short episode at the URDF zero
posture and run:

```bash
python -m twin_arms_controller.misc.PrintLeaderEncoderOffsets /path/to/zero_pose.rmb
```

Apply the printed arrays to `encoder_offset_left` and `encoder_offset_right` in
[RealTwinArmsLeaderEnv.py](twin_arms_controller/envs/leader/RealTwinArmsLeaderEnv.py).

## Start eSkin (optional)

eSkin requires a purchaser-supplied driver and a compatible ZeroMQ data
publisher, neither of which is included here. Configure `eskin_endpoint` and
`eskin_patch_names` for your setup and start the stream before recording or
rollout. The expected data format is defined in
[TwinArmsEskinReceiver.py](twin_arms_controller/envs/common/TwinArmsEskinReceiver.py).

To skip eSkin, set `eskin_endpoint: null` and `publish_eskin_markers: false`
in the configurations. Policies trained with tactile input require that input.

## Record demonstrations

Record the wearable leader:

```bash
python -m twin_arms_controller.bin.Teleop RealTwinArmsLeader \
  --config twin_arms_controller/envs/configs/RealTwinArmsLeader.yaml
```

Alternatively, record by moving the unpowered follower by hand:

```bash
python -m twin_arms_controller.bin.Teleop RealTwinArmsFollower \
  --config twin_arms_controller/envs/configs/RealTwinArmsFollowerReadOnly.yaml
```

Follow the terminal prompts: `n` advances through synchronization, recording
and saving. Press keys with the operation window focused. `Esc` exits. The
saved `.rmb` directory is a recording, not a single file. Leader recording
reads the wearable encoders; it does not simultaneously drive the follower.

To replay recorded commands on the powered follower:

```bash
python -m twin_arms_controller.bin.Teleop RealTwinArmsFollower \
  --config twin_arms_controller/envs/configs/RealTwinArmsFollower.yaml --replay_log /path/to/episode.rmb
```

## Train and run a policy

Place demonstration `.rmb` directories in a dataset directory. This example
trains a camera-free Diffusion Policy with joint positions and tactile input:

```bash
python -m twin_arms_controller.bin.Train DiffusionPolicy \
  --dataset_dir /path/to/dataset --checkpoint_dir checkpoints/dp \
  --state_keys measured_joint_pos measured_eskin --action_keys command_joint_pos \
  --n_obs_steps 2 --n_action_steps 8 --horizon 16 --camera_names
```

For joint-only training, omit `measured_eskin`. To restrict tactile inputs,
use `--eskin_patch_names`, `--no_eskin_prox` or `--no_eskin_force`.
Run `python -m twin_arms_controller.bin.Train DiffusionPolicy --help` for other
training options. Datasets and pretrained policies are not bundled here.

Keep `model_meta_info.pkl` beside the checkpoint. After checking the follower
configuration and initial posture, run:

```bash
python -m twin_arms_controller.bin.Rollout DiffusionPolicy RealTwinArmsFollower \
  --config twin_arms_controller/envs/configs/RealTwinArmsFollower.yaml \
  --checkpoint checkpoints/dp/policy_last.ckpt --wait_before_start
```

`n` starts/ends policy execution and then resets; `Esc` exits and disables
torque. Input keys and normalization are loaded from the checkpoint metadata.

## Visualize live data or recordings

While a follower recording or rollout is running, show measured and commanded
poses in another terminal:

```bash
ros2 launch ./twin_arms_controller/visualization/launch/RealTwinArmsFollower.launch.py
```

For leader recording, use `RealTwinArmsLeader.launch.py` instead. To show eSkin
markers published by either operation:

```bash
ros2 launch ./twin_arms_controller/visualization/launch/RealTwinArmsEskin.launch.py
```

For offline playback, start the corresponding RViz launch above, then run:

```bash
python -m twin_arms_controller.visualization.PlayRmbDataTwinArms /path/to/episode.rmb --loop
```

Add `--no_eskin` for joint-only data. For follower measured/commanded playback,
add `--urdf twin_arms_controller/urdf/TwinArmsFollower.urdf --publish_command_joint_states`.
The bundled sensor layout supports playback without the live driver; for a
different installation, create a layout while its stream is running:

```bash
mkdir -p local
python -m twin_arms_controller.visualization.DumpEskinLayout \
  --config twin_arms_controller/envs/configs/RealTwinArmsFollower.yaml --output local/eskin_layout.json
```

Pass that file with `--eskin_layout local/eskin_layout.json` during playback.

## License

Code and robot models are distributed under the [BSD 2-Clause License](LICENSE).
External libraries and the purchaser-supplied eSkin driver have their own licenses.
