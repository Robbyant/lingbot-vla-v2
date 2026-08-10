# Training Configuration Guide

This document explains all configuration parameters used in LingBot-VLA 2.0 post-training (both **Real-World** and RoboTwin 2.0 simulation scenarios).

## Real-World / Real Deploy Example

The current real deploy template is `configs/vla/real_robot/real_robot.yaml`.

```yaml
model:
  model_path: /path/to/pretain_ckpt/hf_ckpt
  tokenizer_path: /path/to/Qwen3-VL-4B-Instruct
  post_training: true
  adanorm_time: true
  config_key: LingbotVLAV2Config
  moe_implementation: fused

data:
  datasets_type: vla
  data_name: multi
  train_path: path/to/training_data.txt
  robot_config_root: configs/robot_configs
  joints:
    - arm.position: 14
    - end.position: 14
    - effector.position: 2
    - waist.position: 4
    - head.position: 2
    - base.position: 3
    - hand.position: 12
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
  norm_type:
    - arm.position: meanstd
    - end.position: meanstd
    - effector.position: meanstd
    - waist.position: meanstd
    - head.position: meanstd
    - base.position: meanstd
    - hand.position: meanstd
  num_workers: 8
  use_future_image: true

train:
  output_dir: /path/to/save_ckpt
  moe_monitor_interval: 1000
  enable_gradient_checkpointing: true  # Saves GPU memory. Set to false when VRAM is sufficient for faster training.
  precompute_grid_thw: true
  vlm_causal: true
  vlm_fsdp: true
  attention_implementation: flex_cached
  use_moe: true
  token_moe_layers: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
  token_num_experts: 32
  token_top_k: 4
  token_moe_intermediate_size: 512
  token_shared_intermediate_size: 704
  bias_update_speed: 0.00025
  router_activation: "sigmoid"
  routed_scaling_factor: 4.0
  use_shared_expert_gate: false
  use_moe_expert_lr: true
  data_parallel_mode: fsdp2 # Use FSDP2 for model
  enable_full_shard: false
  module_fsdp_enable: true
  use_compile: true # Apply torch.compile() to model
  use_wandb: false
  rmpad: false
  rmpad_with_pos_ids: false
  ulysses_parallel_size: 1
  freeze_vision_encoder: false
  tokenizer_max_length: 72
  action_dim: 55
  max_action_dim: 55
  max_state_dim: 55
  lr: 5.0e-5
  lr_decay_style: constant
  num_train_epochs: 29000
  micro_batch_size: 1
  global_batch_size: 3
  max_steps: 40000
  ckpt_manager: dcp
  save_steps: 40000
  save_epochs: 29000
  enable_fp32: true # Use float32 precision for the action expert
  enable_resume: true
  # ---- Depth Injection (native LingBot-VLA 2.0) ----
  align_params:
    mode: 'query'
    num_task_tokens: 8
    depth_loss_weight: 0.004
    future_depth_loss_weight: 0.004
    use_future_video: true
    llm:
      dim_out: 2560
      image_token_size: 8
      image_input_size: 224
    depth:
      model_type: MoRGBD
      moge_path: /path/to/depth/moge2-vitb-normal.pt
      morgbd_path: /path/to/depth/morgbd_v2_mixdata.pt
      num_layers: 1
      num_heads: 4
      dim_head: 32
      ff_mult: 1
      num_backbone_tokens: 256
      token_size: 16
      dim_out: 1024
      input_size: 224
    video:
      ckpt_path: /path/to/video-dino/teacher_step_10000.pth
      config_path: /path/to/video-dino/config.yaml
      attention_mode: flex_block_causal
      input_size: 256
      block_suffix_to_future_video: true
      block_warmup_steps: 0
      block_warmup_gradual: false
      # Enables grouped future depth/video query seed with resampler heads.
      share_future_depth_query: true
      use_shared_future_task_proj: true
      use_current_shared_task_proj: true
      shared_query_head_type: resampler
      num_future_frames: 1
      # Keeps DINO teacher input as [warmup current, current, future].
      # Current-DINO target uses current_index=1 in code.
      use_warmup_frame: true
      effective_fps: 1.0
      n_blocks: 1
      cls_pool: last
      head_type: resampler
      detach_image_feats: true
      num_layers: 1
      num_heads: 4
      dim_head: 32
      ff_mult: 1
      num_backbone_tokens: 256
      dim_out: 1024
      target_type: absolute
      future_video_loss_weight: 0.004
      use_smooth_l1_loss: false
      use_mse_loss: true
      mse_loss_weight: 1.0
      # Enables current-DINO patch alignment and current depth/DINO shared query projection.
      use_patch_loss: true
      use_current_patch_loss: true
      use_cosine_loss: false
      cosine_loss_weight: 0.2
      use_cls_loss: false
      cls_loss_type: mse
      cls_loss_weight: 0.2
      log_max_samples: 32
      log_scale: 16
    visual_steps: 5000
```

## RoboTwin 2.0 Example (50 tasks, clean + randomized)

```yaml
model:
  model_path: /path/to/pretain_ckpt/hf_ckpt
  tokenizer_path: /path/to/Qwen3-VL-4B-Instruct
  post_training: true
  adanorm_time: true
  config_key: LingbotVLAV2Config
  moe_implementation: fused

data:
  datasets_type: vla
  data_name: multi
  train_path: assets/training_data/robotwin.txt  # 50 RoboTwin 2.0 tasks with clean and randomized data.
  robot_config_root: ./configs/robot_configs
  joints:
    - arm.position: 14
    - end.position: 14
    - effector.position: 2
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
  prompt_type: global
  norm_type:
    - arm.position: bounds_99_woclip
    - end.position: bounds_99_woclip
    - effector.position: bounds_99_woclip
  num_workers: 8
  use_future_image: true
  norm_stats_file: assets/norm_stats/robotwin.json

train:
  output_dir: /path/to/save_ckpt
  moe_monitor_interval: 1000
  enable_gradient_checkpointing: true  # Saves GPU memory. Set to false when VRAM is sufficient for faster training.
  precompute_grid_thw: true
  vlm_causal: true
  vlm_fsdp: true
  attention_implementation: flex_cached
  use_moe: true
  token_moe_layers: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
  token_num_experts: 32
  token_top_k: 4
  token_moe_intermediate_size: 512
  token_shared_intermediate_size: 704
  bias_update_speed: 0.00025
  router_activation: "sigmoid"
  routed_scaling_factor: 4.0
  use_shared_expert_gate: false
  use_moe_expert_lr: true
  loss_type: L1_fm
  data_parallel_mode: fsdp2 # Use FSDP2 for model
  enable_full_shard: false
  module_fsdp_enable: true
  use_compile: true # Apply torch.compile() to model
  use_wandb: false
  rmpad: false
  rmpad_with_pos_ids: false
  ulysses_parallel_size: 1
  freeze_vision_encoder: false
  tokenizer_max_length: 72
  action_dim: 55
  max_action_dim: 55
  max_state_dim: 55
  lr: 1.0e-4
  lr_decay_style: constant
  num_train_epochs: 29000
  micro_batch_size: 1
  global_batch_size: 3
  max_steps: 30000
  ckpt_manager: dcp
  save_steps: 30000
  save_epochs: 29000
  enable_fp32: true # Use float32 precision for the action expert
  enable_resume: true
  # ---- Depth Injection (native LingBot-VLA 2.0) ----
  align_params:
    mode: 'query'
    num_task_tokens: 8
    depth_loss_weight: 0.004
    future_depth_loss_weight: 0.004
    use_future_video: true
    llm:
      dim_out: 2560
      image_token_size: 8
      image_input_size: 224
    depth:
      model_type: MoRGBD
      moge_path: /path/to/depth/moge2-vitb-normal.pt
      morgbd_path: /path/to/depth/morgbd_v2_mixdata.pt
      num_layers: 1
      num_heads: 4
      dim_head: 32
      ff_mult: 1
      num_backbone_tokens: 256
      token_size: 16
      dim_out: 1024
      input_size: 224
      use_future_depth: true
      block_future_depth_to_action: true
      future_depth_head_type: resampler
      block_warmup_steps: 0
      block_warmup_gradual: false
      detach_future_image_feats: true
    video:
      ckpt_path: /path/to/video-dino/teacher_step_10000.pth
      config_path: /path/to/video-dino/config.yaml
      attention_mode: flex_block_causal
      input_size: 256
      block_suffix_to_future_video: true
      block_warmup_steps: 0
      block_warmup_gradual: false
      # Enables grouped future depth/video query seed with resampler heads.
      share_future_depth_query: true
      use_shared_future_task_proj: true
      use_current_shared_task_proj: true
      shared_query_head_type: resampler
      num_future_frames: 1
      # Keeps DINO teacher input as [warmup current, current, future].
      # Current-DINO target uses current_index=1 in code.
      use_warmup_frame: true
      effective_fps: 1.0
      n_blocks: 1
      cls_pool: last
      head_type: resampler
      detach_image_feats: true
      num_layers: 1
      num_heads: 4
      dim_head: 32
      ff_mult: 1
      num_backbone_tokens: 256
      dim_out: 1024
      target_type: absolute
      future_video_loss_weight: 0.004
      use_smooth_l1_loss: false
      use_mse_loss: true
      mse_loss_weight: 1.0
      # Enables current-DINO patch alignment and current depth/DINO shared query projection.
      use_patch_loss: true
      use_current_patch_loss: true
      use_cosine_loss: false
      cosine_loss_weight: 0.2
      use_cls_loss: false
      cls_loss_type: mse
      cls_loss_weight: 0.2
      log_max_samples: 32
      log_scale: 16
    visual_steps: 5000
```

---

## Parameter Reference

### Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | — | Path to pre-trained VLA model weights. |
| `tokenizer_path` | str | - | Path to VLM. |
| `post_training` | bool | `false` | Enable post-training mode. |
| `adanorm_time` | bool | `false` | Add the time embedding to AdaNorm in the action expert. |
| `config_key` | str | `"LingbotVLAConfig"` | Model config registry key. LingBot-VLA 2.0 uses `"LingbotVLAV2Config"`. |
| `moe_implementation` | str | `None` | MoE implementation. LingBot-VLA 2.0 uses `"fused"` in the provided configs. |

### Data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_name` | str | — | Robot config name for a single dataset, or `"multi"` when `train_path` points to a dataset list. Must be consistent with normalization statistics. |
| `train_path` | str | — | Path to a LeRobot v2.1/v3.0 dataset directory, or a text list when `data_name: multi`. |
| `robot_config_root` | str | — | Directory containing robot config YAML files, for example `configs/robot_configs`. |
| `joints` | List[Dict] | — | Max dim of each named joints in data. |
| `cameras` | List[str] | — | Camera names in data. |
| `prompt_type` | str | `"global"` | Prompt type. The released data pipeline currently supports only `"global"`; `"subtask"` and `"both"` are rejected instead of silently falling back to the global task prompt. |
| `norm_type` | List[Dict[str, str]] | — | Per-joint normalization type used by `train_lingbotvla.py`. Each joint type in `data.joints` that appears in states/actions should have one entry, for example `[{arm.position: bounds_99_woclip}, {effector.position: bounds_99_woclip}]`. Options include `"meanstd"`, `"bounds_98"`, `"bounds_99"`, `"bounds_98_woclip"`, `"bounds_99_woclip"`, `"std"`, `"minmax"`, `"minmax_woclip"`, `"sincos"`, and `"identity"`. |
| `norm_stats_file` | str | — | Path to pre-computed normalization statistics JSON file. Must be the same when computing normalization statistics! |
| `use_future_image` | bool | `false` | Load future image frames for native-depth/future-video training. |

### Train — Batch Size & Gradient Accumulation

If you have limited GPU memory, we support enabling **gradient accumulation** through setting `gradient_accumulation_steps` > 1 to achieve a larger global batch size.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `micro_batch_size` | int | - | Number of samples per forward pass per GPU. |
| `global_batch_size` | int | `None` | Total batch size across all GPUs and accumulation steps. If `None`, auto-computed as `micro_batch_size × data_parallel_size(num_gpus) × gradient_accumulation_steps`. If set, must equal that value or an error is raised. |
| `gradient_accumulation_steps` | int | `1` | Number of gradient accumulation steps. `global_batch_size` is always derived from this value. |

**How gradient accumulation works:**

`global_batch_size` is always computed as `micro_batch_size × data_parallel_size(num_gpus) × gradient_accumulation_steps`. You only need to set `gradient_accumulation_steps`:

```yaml
micro_batch_size: 32
gradient_accumulation_steps: 2
# global_batch_size is auto-computed: 32 × num_gpus × 2
```

If you also set `global_batch_size` explicitly, it must be consistent with the computed value, otherwise an error is raised.

### Train — Training Duration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_train_epochs` | int | `None` | Number of training epochs. If `None`, trains indefinitely until `max_steps`. |
| `max_steps` | int | `None` | Global maximum number of update steps. If `None`, trains until all epochs complete. |

**How training duration is controlled:**

`num_train_epochs` and `max_steps` jointly control when training stops. At least one must be specified.

- **Only `max_steps`**: set `num_train_epochs` to `None`. Training runs across epochs indefinitely and stops at `max_steps`.
  ```yaml
  max_steps: 20000
  # num_train_epochs: not set → runs until 20000 steps
  ```

- **Only `num_train_epochs`**: set `max_steps` to `None`. Training runs for the specified number of epochs.
  ```yaml
  num_train_epochs: 69
  # max_steps: not set → runs all 69 epochs
  ```

- **Both specified**: training stops at whichever limit is reached first.
  ```yaml
  num_train_epochs: 69
  max_steps: 20000
  # stops at 20000 steps even if 69 epochs are not finished
  ```
> **Note:** When training stops at `max_steps`, a checkpoint is always saved automatically.

### Train — Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | str | `"adamw"` | Optimizer type. Supported values are `"adamw"`, `"anyprecision_adamw"`, and `"muon"`. Set `optimizer: muon` or pass `--train.optimizer muon` to enable the Muon optimizer. |
| `loss_type` | str | `"fm"` | Loss function. `"fm"` for MSE flow-matching, `"L1_fm"` for L1 flow-matching. |
| `data_parallel_mode` | str | `"ddp"` | Distributed data parallel strategy. Options: `"ddp"`, `"fsdp1"`, `"fsdp2"`. |
| `enable_gradient_checkpointing` | bool | `false` | Enable gradient checkpointing to reduce GPU memory usage. Keep it `true` on memory-constrained GPUs; set it to `false` when VRAM is sufficient for faster training. |
| `precompute_grid_thw` | bool | `false` | Precompute and cache fixed-resolution `grid_thw` derived tensors. |
| `vlm_causal` | bool | `false` | Use causal attention for image and language tokens in the VLM. |
| `vlm_fsdp` | bool | `false` | Apply FSDP2 to the VLM. |
| `attention_implementation` | str | `"flex"` | VLA attention implementation. Supported values include `"flex"`, `"flex_cached"`, and `"eager"`. |
| `use_compile` | bool | `false` | Enable `torch.compile` for training acceleration. |
| `ckpt_manager` | str | `"dcp"` | Checkpoint backend. Options: `"dcp"` (PyTorch Distributed Checkpoint), `"bytecheckpoint"`. |
| `enable_fp32` | bool | `false` | Use float32 precision for the action expert. |
| `enable_resume` | bool | `false` | Automatically resume training from the latest checkpoint in `output_dir`. |

### Train — MoE Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_moe` | bool | `false` | Enable MoE. |
| `token_moe_layers` | List[int] | `None` | Layer indices for token-level routing MoE. LingBot-VLA 2.0 configs use layers `0..35`. |
| `token_num_experts` | int | `32` | Number of experts per token-level MoE layer. |
| `token_top_k` | int | `1` | Top-k experts selected per token. |
| `token_moe_intermediate_size` | int | `256` | Intermediate size for token-level MoE expert FFN. |
| `token_shared_intermediate_size` | int | `256` | Intermediate size for token-level shared expert FFN. |
| `bias_update_speed` | float | `0.001` | Bias update speed for loss-free MoE load balancing. |
| `router_activation` | str | `"softmax"` | Router activation function, for example `"softmax"` or `"sigmoid"`. |
| `routed_scaling_factor` | float | `1.0` | Scaling factor applied to routing weights after normalization. |
| `use_shared_expert_gate` | bool | `true` | Use sigmoid gate on shared expert output. |
| `use_moe_expert_lr` | bool | `false` | Apply scaled learning rate to MoE routed experts. |

### Train — Native Depth / Future Video Parameters

`align_params` is a nested dictionary used by the native-depth LingBot-VLA 2.0 configs. The most important fields are listed below; see `docs/config/lingbotvla_config_doc.md` for the full argument table.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `align_params.mode` | str | `"query"` | Alignment mode. |
| `align_params.num_task_tokens` | int | `8` | Number of task/query tokens. |
| `align_params.depth_loss_weight` | float | `0.004` | Current-depth loss weight. |
| `align_params.future_depth_loss_weight` | float | `0.004` | Future-depth loss weight. |
| `align_params.use_future_video` | bool | `true` | Enable future-video/DINO supervision. |
| `align_params.visual_steps` | int | `5000` | Visual logging interval. |
| `align_params.depth.moge_path` | str | — | Path to MoGe weights. |
| `align_params.depth.morgbd_path` | str | — | Path to MoRGBD/LingBot-Depth weights. |
| `align_params.depth.use_future_depth` | bool | `true` | Enable future-depth targets. |
| `align_params.depth.block_future_depth_to_action` | bool | `true` | Block gradients from future-depth branch to action branch. |
| `align_params.depth.future_depth_head_type` | str | `"resampler"` | Future-depth head type. |
| `align_params.depth.detach_future_image_feats` | bool | `true` | Detach future image features for the future-depth branch. |
| `align_params.video.ckpt_path` | str | — | Path to video-DINO teacher checkpoint. |
| `align_params.video.config_path` | str | — | Path to video-DINO teacher config. |
| `align_params.video.attention_mode` | str | `"flex_block_causal"` | Video teacher attention mode. |
| `align_params.video.num_future_frames` | int | `1` | Number of future frames. |
| `align_params.video.use_warmup_frame` | bool | `true` | Keep DINO teacher input as `[warmup current, current, future]`. |
| `align_params.video.future_video_loss_weight` | float | `0.004` | Future-video loss weight. |
| `align_params.video.use_patch_loss` | bool | `true` | Enable DINO patch alignment. |
| `align_params.video.use_current_patch_loss` | bool | `true` | Enable current-frame DINO patch alignment. |


---

> **⚠️ Important:** Due to differences between real-world and simulation environments, their training configurations differ in two key aspects:
>
> | | Real-World | RoboTwin 2.0 |
> |---|---|---|
> | `norm_type` | per-joint `meanstd` | per-joint `bounds_99_woclip` |
> | `loss_type` | default (MSE flow-matching) | `L1_fm` (L1 flow-matching) |


---

## Example: Training Setting on A6000

You can fine-tune `LingBot-VLA 2.0` on **4 × A6000 GPUs** platforms:

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/real_robot/real_robot.yaml \
    --data.train_path /path/to/real_robot_dataset_or_list \
    --data.data_name robot_config_name_or_multi \
    --data.norm_stats_file /path/to/norm_stats.json \
    --train.output_dir output/ \
    --train.micro_batch_size 1 \
    --train.gradient_accumulation_steps 1 \
    --train.global_batch_size 4 \
    --train.enable_gradient_checkpointing true

# train.global_batch_size will be auto-computed as:
# micro_batch_size × data_parallel_size(num_gpus) × gradient_accumulation_steps = 1 × 4 × 1 = 4
```
This example needs at least about 49 GB of VRAM per GPU on A6000-class GPUs. Actual VRAM usage depends on the model checkpoint, dataset, image/depth options, and whether gradient checkpointing is enabled.
If GPU memory is sufficient, set `--train.enable_gradient_checkpointing false` to reduce recomputation overhead and improve speed.
