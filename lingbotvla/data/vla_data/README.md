# Customized Downstream Task Dataset Construction

This guide explains how to prepare downstream VLA data for post-training.
The default data path is a **single LeRobot dataset**. If you want to train on multiple datasets, use `data.data_name: multi` and pass a text list through `data.train_path`.

## 1. Prepare LeRobot Dataset

LingBot-VLA 2.0 loads downstream data through the [LeRobot](https://github.com/huggingface/lerobot) library (`LeRobotDataset`). Each dataset entry can be a HuggingFace repo id or a local LeRobot dataset directory.

Both **LeRobot v2.1** and **LeRobot v3.0** layouts are supported directly. You do not need to merge datasets or convert v2.1 data to v3.0 before training.

## 2. Prepare Dataset Input

### Single Dataset, Default

For one LeRobot dataset, set `data.data_name` to the robot config name and set `data.train_path` to the dataset repo id or local directory.

For RoboTwin, use `robotwin` so the loader resolves:

```text
configs/robot_configs/robotwin.yaml
```

The corresponding VLA config should use:

```yaml
data:
  datasets_type: vla
  data_name: robotwin  # Single dataset: use the robot config name.
  train_path: /path/to/lerobot_dataset
  robot_config_root: ./configs/robot_configs
  joints:
    - arm.position: 14          # must >= total dim of concatenated arm position slices (6+6=12); 14 recommended for padding headroom
    - end.position: 14
    - effector.position: 2      # must >= total dim of concatenated effector position slices (1+1=2); 2 recommended for padding headroom
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
```

### Multiple Datasets

For multiple LeRobot datasets, set `data.data_name: multi` and put all datasets into a text file. `data.train_path` should point to this file.

Each non-empty line has two columns separated by a space:

```text
<robot_config_name> <lerobot_repo_or_local_path>
```

Example `assets/training_data/robotwin.txt`:

```text
robotwin /path/to/lerobot_task_a
robotwin /path/to/lerobot_task_b
robotwin /path/to/lerobot_task_c
```

The corresponding VLA config should use:

```yaml
data:
  datasets_type: vla
  data_name: multi  # Multi-dataset mode: keep this value as "multi".
  train_path: assets/training_data/robotwin.txt
  robot_config_root: ./configs/robot_configs
  joints:
    - arm.position: 14          # must >= total dim of concatenated arm position slices (6+6=12); 14 recommended for padding headroom
    - end.position: 14
    - effector.position: 2      # must >= total dim of concatenated effector position slices (1+1=2); 2 recommended for padding headroom
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
```

`MultiVLADataset` will instantiate one `VLADataset` per line and concatenate them at runtime. This replaces the old "merge datasets first" workflow.

## 3. Prepare Robot Config

The robot config maps raw LeRobot feature names to the unified feature space used by LingBot-VLA 2.0. For the current RoboTwin setup, use:

```text
configs/robot_configs/robotwin.yaml
```

The file name, without `.yaml`, is used as `data.data_name` for a single dataset or as the first column in the multi-dataset list:

```text
robotwin /path/to/lerobot_dataset
```

### States

`states` maps raw observation keys to unified state features. Multiple slices from the same raw tensor are concatenated in order.

Current RoboTwin state mapping:

```yaml
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state:       # left arm joints [0:6)
            start: 0
            end: 6
        - observation.state:       # right arm joints [7:13)
            start: 7
            end: 13

  - observation.state.effector.position:
      origin_keys:
        - observation.state:       # left gripper [6:7)
            start: 6
            end: 7
        - observation.state:       # right gripper [13:14)
            start: 13
            end: 14
```

This means:

| Unified State | Raw Slices | Total Dim |
|---|---|---|
| `observation.state.arm.position` | `[0:6)` + `[7:13)` | 12 |
| `observation.state.effector.position` | `[6:7)` + `[13:14)` | 2 |

### Actions

`actions` has the same structure as `states`, but maps raw action tensors to unified action features. `subtract_state: False` means the model learns absolute actions instead of state-relative deltas.

Current RoboTwin action mapping:

```yaml
actions:
  - action.arm.position:
      origin_keys:
        - action:
            start: 0
            end: 6
        - action:
            start: 7
            end: 13
      subtract_state: False

  - action.effector.position:
      origin_keys:
        - action:
            start: 6
            end: 7
        - action:
            start: 13
            end: 14
      subtract_state: False
```

> <p><span style="color:red; font-size:1.em; font-weight:bold;">Note</span>: We recommend setting <code>subtract_state</code> of <code>action.arm.position</code> to <code>True</code> and for<code>action.effector.position</code> to <code>False</code> when training the model with real-world data. See <a href="../../../configs/robot_configs/agilex_cobot_magic.yaml"><code>configs/robot_configs/agilex_cobot_magic.yaml</code></a> for a complete example.</p>

### Images

`images` maps raw camera keys to the unified camera names declared in the VLA training config.

Current RoboTwin camera mapping:

```yaml
images:
  - observation.images.camera_top:
      origin_keys: observation.images.cam_high
  - observation.images.camera_wrist_left:
      origin_keys: observation.images.cam_left_wrist
  - observation.images.camera_wrist_right:
      origin_keys: observation.images.cam_right_wrist
```

If a raw key already matches the target key, use the short form:

```yaml
images:
  - observation.images.camera_top
```

### Normalization Stats

The robot config also points to the normalization file:

```yaml
norm_stats: assets/norm_stats/robotwin.json
```

Training reads this path from the robot config. After recomputing norm stats, make sure this field points to the generated JSON file.

### Consistency With VLA Training Config

The joint types and camera names used in the robot config must be declared in the VLA training config.

Example:

```yaml
# configs/vla/robotwin/robotwin.yaml
data:
  datasets_type: vla
  data_name: robotwin  # Single dataset: use the robot config name; use "multi" only with a dataset list.
  train_path: /path/to/lerobot_dataset
  robot_config_root: ./configs/robot_configs
  joints:
    - arm.position: 14          # must >= total dim of concatenated arm position slices (6+6=12); 14 recommended for padding headroom
    - end.position: 14
    - effector.position: 2      # must >= total dim of concatenated effector position slices (1+1=2); 2 recommended for padding headroom
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
```

Rules:

- `observation.state.<joint_type>` and `action.<joint_type>` in the robot config must have matching entries in `data.joints`.
- `observation.images.<camera_name>` in the robot config must be listed in `data.cameras`.
- The configured joint dimension should be greater than or equal to the concatenated raw slice dimension. For RoboTwin, `arm.position` uses 12 dims and `effector.position` uses 2 dims.
- Extra joint entries may exist in the training config, but they are only used when the robot config maps data to them. In the RoboTwin example, `end.position: 14` is kept to align with the pretraining config and model action/state head dimensions.

> See `configs/robot_configs/robotwin.yaml` for a complete example.

> **Important:**
> - The `<joint_type>` used in states (`observation.state.<joint_type>`) and actions (`action.<joint_type>`) must be defined in `data.joints` of your VLA training config.
> - Camera names in the images section (`observation.images.<camera_name>`) must be listed in `data.cameras`.

For example, if `configs/vla/robotwin/robotwin.yaml` declares `joints: [{arm.position: 14}, {end.position: 14}, {effector.position: 2}]` and `cameras: [camera_top, camera_wrist_left, camera_wrist_right]`, then only these joint types and camera names are valid in the robot config. Using an undefined joint type or camera name will raise a `ValueError` at runtime.

> **Note:** You can define additional joint types beyond `arm.position` and `effector.position` by adding new entries to `data.joints`. If you add end-effector (EEF) dimensions, we recommend learning **absolute action** (`subtract_state: False`), as relative rotation computation is not currently supported.

## 4. Compute Normalization Statistics

Use the robot training config directly. No merge or format-conversion step is required.

For a single LeRobot dataset directory, set `data.data_name` to the robot config name and pass the dataset directory as `data.train_path`:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name robotwin \
  --data.train_path /path/to/lerobot_dataset \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/robotwin.json \
  --data.data_ratio_for_norm_compute 1
```

For example:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name robotwin \
  --data.train_path /path/to/beat_block_hammer-aloha-agilex_randomized_500-1000/ \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path debug.json \
  --data.data_ratio_for_norm_compute 1
```

For a multi-dataset list, keep `data.data_name: multi` and pass the list file as `data.train_path`:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name multi \
  --data.train_path assets/training_data/robotwin.txt \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_path assets/norm_stats/robotwin.json \
  --data.data_ratio_for_norm_compute 1
```

Optional: compute stats for only selected robot config names in the list:

```bash
--data.robot_name robotwin
```

The output JSON should match the `norm_stats` path in `configs/robot_configs/robotwin.yaml`.

## 5. Training

After the LeRobot dataset, robot config, and norm stats are ready, start post-training. The default usage is a single dataset with `data.data_name` set to the robot config name.

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.norm_stats_file assets/norm_stats/robotwin.json
```

For a single dataset, override paths from the command line like this:

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name robotwin \
  --data.train_path /path/to/lerobot_dataset \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_stats_file assets/norm_stats/robotwin.json \
  --train.output_dir output/
```

For multiple datasets, set `--data.data_name multi` and pass a text list through `--data.train_path`:

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name multi \
  --data.train_path assets/training_data/robotwin.txt \
  --data.robot_config_root ./configs/robot_configs \
  --data.norm_stats_file assets/norm_stats/robotwin.json \
  --train.output_dir output/
```

## Quick Checklist

- Dataset entries are LeRobot v2.1 or v3.0 repos/local directories.
- Single dataset: `data.data_name` is the robot config name, for example `robotwin`, and `data.train_path` points to one LeRobot dataset directory.
- Multiple datasets: `data.data_name` is `multi`, and `data.train_path` points to a text file whose first column is the robot config name, for example `robotwin`.
- `configs/robot_configs/robotwin.yaml` exists.
- `norm_stats` in the robot config points to an existing JSON file.
- During training, pass `--data.norm_stats_file path/to/norm_stats.json` if you want to override or explicitly set the norm stats JSON.
- `data.joints` and `data.cameras` contain every unified joint and camera used by the robot config.
