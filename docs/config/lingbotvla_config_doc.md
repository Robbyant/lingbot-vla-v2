## LingBot-VLA 2.0 Config Arguments

This page documents the configuration fields used by the LingBot-VLA 2.0
post-training entrypoint:

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml
```

The source of truth for top-level arguments is:

- `lingbotvla/utils/arguments.py`
- `tasks/vla/train_lingbotvla.py`

For runnable examples, see `configs/vla/Training_Config.md`. For LeRobot data,
robot configs, and normalization statistics, see
`lingbotvla/data/vla_data/README.md`.

### Model Arguments

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `model.config_path` | Optional[str] | `None` | Path to the model config. Defaults to `model.model_path`. |
| `model.model_path` | Optional[str] | `None` | Path to the pre-trained model. If unspecified, use random initialization. |
| `model.tokenizer_path` | Optional[str] | `None` | Path to the tokenizer. Defaults to `model.config_path`. |
| `model.vlm_repo_id` | Optional[str] | `None` | Path or repo id of the VLM. |
| `model.post_training` | bool | `False` | Whether to use post-training mode. |
| `model.vocab_size` | int | `0` | Vocabulary size override. |
| `model.incremental_training` | bool | `False` | Whether to apply incremental training. |
| `model.adanorm_time` | bool | `False` | Whether to apply an extra time embedding to `ada_norm` in the action expert. |
| `model.encoders` | Dict | `{}` | Multimodal encoder config and weights. |
| `model.decoders` | Dict | `{}` | Multimodal decoder config and weights. |
| `model.input_encoder` | `"encoder"` or `"decoder"` | `"encoder"` | Use the encoder or decoder encoder for input images. |
| `model.output_encoder` | `"encoder"` or `"decoder"` | `"decoder"` | Use the encoder or decoder encoder for output images. |
| `model.encode_target` | bool | `False` | Whether to encode target images with the decoder. |
| `model.attn_implementation` | `"eager"`, `"sdpa"`, `"flash_attention_2"`, `"flash_attention_3"` | `"flash_attention_2"` | Attention implementation used when loading supported HF modules. |
| `model.moe_implementation` | `None`, `"eager"`, `"fused"` | `None` | MoE implementation to use. LingBot-VLA 2.0 configs usually use `"fused"`. |
| `model.basic_modules` | List[str] | `[]` | Extra modules beyond `model._no_split_modules` to shard in FSDP. |
| `model.force_use_huggingface` | bool | `False` | Force loading model through HuggingFace. |
| `model.use_lm_head` | bool | `False` | Whether to use the language-model head. |
| `model.final_norm_adanorm` | bool | `False` | Whether to use AdaNorm in the final norm. |
| `model.config_key` | `"LingbotVLAConfig"`, `"GrootConfig"`, `"LingbotVLAV2Config"` | `"LingbotVLAConfig"` | Which model config registry key to use. LingBot-VLA 2.0 uses `"LingbotVLAV2Config"`. |
| `model.vit_attn_implementation` | str | `"flash_attention_2"` | `_attn_implementation` used by the vision tower config. |

### Data Arguments

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `data.train_path` | str | Required | Path of the training data. For VLA, use a LeRobot dataset directory or a text list when `data.data_name: multi`. |
| `data.datasets_type` | `"mapping"`, `"iterable"`, `"vla"` | `"mapping"` | Dataset builder type. LingBot-VLA 2.0 post-training uses `"vla"`. |
| `data.data_name` | Optional[str] | `None` | Dataset name for multimodal/VLA training. For one VLA dataset, use the robot config name. For multiple datasets, use `"multi"`. |
| `data.source_name` | Optional[str] | `None` | Source name of the dataset. |
| `data.robot_config_root` | Optional[str] | `None` | Root directory containing robot config YAML files, for example `configs/robot_configs`. |
| `data.joints` | Optional[List[Dict[str, int]]] | `None` | Ordered joint types and max dims used by the unified VLA feature space. |
| `data.cameras` | Optional[List[str]] | `None` | Ordered camera names used by the VLA model. |
| `data.norm_type` | Optional[List[Dict[str, str]]] | `None` | Per-joint normalization type. Scalar strings are not supported by `train_lingbotvla.py`. Example: `[{arm.position: bounds_99_woclip}, {effector.position: bounds_99_woclip}]`. |
| `data.norm_stats_file` | Optional[str] | `None` | Path to the normalization stats JSON file. |
| `data.prompt_type` | `"global"`, `"subtask"` | `"global"` | Prompt type used by the VLA dataset. Current RoboTwin config uses `"global"`. |
| `data.use_future_image` | bool | `False` | Whether to load future image frames for native-depth/future-video training. |
| `data.img_size` | int | `256` | Image size used by VLA data utilities. |
| `data.state_norm_type` | str | `"none"` | Normalization type for VLA state features. Use `"none"` to reuse `norm_type`; use `"sincos"` to encode raw state angles as cos/sin while actions still use `norm_type`. |
| `data.image_augment` | bool | `False` | Enable training-time image augmentation for VLA datasets. Random color parameters are sampled once per sample and replayed across all camera views. |
| `data.num_workers` | int | `20` | Number of workers used to load data. |
| `data.prefetch_factor` | int | `4` | Number of batches loaded in advance by each worker. |
| `data.drop_last` | bool | `True` | Whether to drop the last incomplete batch. |
| `data.pin_memory` | bool | `True` | Whether to pin memory for dataloader transfer. |
| `data.train_size` | int | `10000000` | Number of tokens used to compute training steps for dynamic-batch dataloaders. |
| `data.data_type` | `"plaintext"`, `"conversation"`, `"diffusion"` | `"conversation"` | Generic training data type. VLA datasets usually leave this unchanged. |
| `data.dataloader_type` | `"native"` | `"native"` | Dataloader implementation. |
| `data.data_root` | Optional[str] | `None` | Root path of datasets. |
| `data.data_tag` | `"default"`, `"mmtag"` | `"default"` | Dataset tag for multimodal training. |
| `data.text_keys` | Optional[str] | `None` | Key used to get text from generic training data. |
| `data.image_keys` | str | `"images"` | Key used to get images from generic training data. |
| `data.chat_template` | str | `"default"` | Chat template name. |
| `data.max_seq_len` | int | `2048` | Maximum sequence length for generic training data. |

### Core Training Arguments

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `train.output_dir` | str | Required | Path to save model checkpoints. |
| `train.lr` | float | `5e-5` | Maximum/default learning rate, or initial LR for warmup. |
| `train.lr_min` | float | `1e-7` | Minimum learning rate. |
| `train.lr_start` | float | `0.0` | Learning rate at the start of warmup. |
| `train.weight_decay` | float | `0` | L2 regularization strength. |
| `train.optimizer` | `"adamw"`, `"anyprecision_adamw"`, `"muon"` | `"adamw"` | Optimizer type. Set to `"muon"` to enable the Muon optimizer. |
| `train.max_grad_norm` | float | `1.0` | Gradient norm clipping value. |
| `train.micro_batch_size` | int | `1` | Number of samples per iteration on each device. |
| `train.gradient_accumulation_steps` | Optional[int] | `1` | Gradient accumulation steps. If set, this value participates in global batch-size validation. |
| `train.global_batch_size` | Optional[int] | `None` | Global batch size. If set, it must equal `micro_batch_size * data_parallel_size * gradient_accumulation_steps`; otherwise training raises an error. |
| `train.num_train_epochs` | Optional[int] | `None` | Epochs to train. If `None`, train until `max_steps` is reached. |
| `train.max_steps` | Optional[int] | `None` | Global max training steps. If `None`, train until all epochs complete. At least one of `num_train_epochs` and `max_steps` must be set. |
| `train.lr_warmup_ratio` | float | `0` | Ratio of learning-rate warmup steps. |
| `train.lr_decay_style` | str | `"constant"` | Learning-rate scheduler name. |
| `train.lr_decay_ratio` | float | `1.0` | Ratio of learning-rate decay steps. |
| `train.loss_type` | str | `"fm"` | Loss type. RoboTwin config uses `"L1_fm"`. |
| `train.enable_fp32` | bool | `False` | Enable fp32 training/action expert precision. |
| `train.enable_mixed_precision` | bool | `True` | Enable mixed-precision training. |
| `train.enable_gradient_checkpointing` | bool | `True` | Enable gradient checkpointing to reduce memory usage. Set to `false` when VRAM is sufficient and speed is preferred. |
| `train.enable_resume` | bool | `False` | Automatically resume training from a checkpoint in `output_dir`. |
| `train.resume_dataloader_state` | bool | `True` | Whether to resume dataloader state. |
| `train.freeze_vit` | bool | `False` | Whether to freeze ViT parameters. |
| `train.vit_lr` | float | `1e-6` | Maximum learning rate for ViT parameters. |
| `train.freeze_vision_encoder` | bool | `False` | Whether to freeze the vision encoder. |
| `train.train_expert_only` | bool | `False` | Whether to train only the action expert. |
| `train.train_state_proj` | bool | `True` | Whether to train the state projection. |
| `train.tokenizer_max_length` | int | `48` | Maximum tokenizer length. Current V2 configs often set `72`. |
| `train.action_dim` | int | `7` | Action dimension. |
| `train.max_action_dim` | int | `32` | Action dimension after padding. |
| `train.max_state_dim` | int | `32` | State dimension after padding. |
| `train.chunk_size` | int | `50` | Action chunk size. |
| `train.vlm_causal` | bool | `False` | Whether to use causal attention for image and language tokens in the VLM. |
| `train.attention_implementation` | str | `"flex"` | VLA attention implementation. Supported values include `"flex"`, `"flex_cached"`, and `"eager"`. |
| `train.my_tokenizer_max_length` | int | `72` | Extra tokenizer length knob used by VLA code paths. |
| `train.decayed_max_grad_norm` | float | `1.0` | Maximum norm for decayed gradients. |
| `train.stable_train_steps` | int | `100000` | Training steps before `decayed_max_grad_norm` is applied. |

### Parallelism, Checkpoint, and Logging Arguments

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `train.data_parallel_mode` | `"ddp"`, `"fsdp1"`, `"fsdp2"`, `"fsdp2-vescale"` | `"ddp"` | Data parallel mode. Current V2 configs use `"fsdp2"`. |
| `train.enable_full_shard` | bool | `True` | Enable full-shard FSDP training. |
| `train.module_fsdp_enable` | bool | `True` | Enable module-level FSDP wrapping. |
| `train.vlm_fsdp` | bool | `False` | Whether to apply FSDP2 to the VLM. |
| `train.use_compile` | bool | `False` | Whether to enable `torch.compile`. |
| `train.enable_reentrant` | bool | `False` | Use reentrant gradient checkpointing. |
| `train.allow_partial_checkpoint` | bool | `False` | Allow partial checkpoint loading and skip missing keys when model structure differs. |
| `train.enable_forward_prefetch` | bool | `True` | Enable forward prefetch for FSDP1. |
| `train.enable_fsdp_offload` | bool | `False` | Enable CPU offload for FSDP1. |
| `train.enable_activation_offload` | bool | `False` | Enable activation offload to CPU. |
| `train.activation_gpu_limit` | float | `0.0` | Amount of activations, in GB, allowed to remain on GPU when activation offload is enabled. |
| `train.enable_manual_eager` | bool | `False` | Enable veScale manual eager mode. |
| `train.init_device` | `"cpu"`, `"cuda"`, `"meta"` | `"cuda"` | Device used to initialize model weights. |
| `train.enable_full_determinism` | bool | `False` | Enable full determinism. |
| `train.empty_cache_steps` | int | `500` | Number of steps between CUDA cache clear operations. |
| `train.data_parallel_replicate_size` | int | `-1` | Data parallel replicate size. |
| `train.data_parallel_shard_size` | int | `-1` | Data parallel shard degree. |
| `train.tensor_parallel_size` | int | `1` | Tensor parallel size. |
| `train.expert_parallel_size` | int | `1` | Expert parallel size. |
| `train.pipeline_parallel_size` | int | `1` | Pipeline parallel size. |
| `train.ulysses_parallel_size` | int | `1` | Ulysses sequence parallel size. |
| `train.context_parallel_size` | int | `1` | Ring-attention context parallel size. |
| `train.rmpad` | bool | `True` | Enable padding-free training with `cu_seqlens`. V2 examples usually set `false`. |
| `train.rmpad_with_pos_ids` | bool | `False` | Enable padding-free training with `position_ids`. Cannot be used together with `rmpad`. |
| `train.dyn_bsz` | bool | `True` | Enable dynamic batch size for padding-free training. |
| `train.dyn_bsz_margin` | int | `0` | Number of pad tokens in dynamic batch. |
| `train.dyn_bsz_buffer_size` | int | `200` | Buffer size for dynamic batch size. |
| `train.bsz_warmup_ratio` | float | `0` | Ratio of batch-size warmup steps. |
| `train.bsz_warmup_init_mbtoken` | int | `200` | Initial number of tokens in a batch during warmup. |
| `train.use_doptim` | bool | `False` | Use veScale ZeRO optimizer. |
| `train.ckpt_manager` | `"bytecheckpoint"`, `"dcp"` | `"bytecheckpoint"` | Checkpoint manager. Current V2 configs use `"dcp"`. |
| `train.load_checkpoint_path` | Optional[str] | `None` | Path to a checkpoint to resume from. |
| `train.save_steps` | int | `0` | Number of steps between checkpoint saves. |
| `train.save_epochs` | int | `1` | Number of epochs between checkpoint saves. |
| `train.save_hf_weights` | bool | `True` | Save HuggingFace-format weights to the latest checkpoint directory. |
| `train.async_save_hf_weights` | bool | `False` | Save HuggingFace-format weights on rank 0 in a best-effort background thread. |
| `train.async_hf_max_pending` | int | `1` | Maximum pending async HF checkpoint conversions on rank 0. |
| `train.seed` | int | `42` | Random seed. |
| `train.use_wandb` | bool | `True` | Use wandb logging. |
| `train.wandb_project` | str | `"lingbotvla"` | Wandb project name. |
| `train.wandb_name` | Optional[str] | `None` | Wandb experiment name. |
| `train.enable_profiling` | bool | `False` | Enable torch profiling. |
| `train.profile_start_step` | int | `1` | Profiling start step. |
| `train.profile_end_step` | int | `2` | Profiling end step. |
| `train.profile_trace_dir` | str | `"./trace"` | Directory used to export profiling results. |
| `train.profile_record_shapes` | bool | `True` | Whether to record input tensor shapes in profiler. |
| `train.profile_profile_memory` | bool | `True` | Whether to profile memory usage. |
| `train.profile_with_stack` | bool | `True` | Whether to record stack traces in profiler. |

### MoE and Action Expert Arguments

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `train.use_moe` | bool | `False` | Whether to use MoE. |
| `train.token_moe_layers` | Optional[List[int]] | `None` | Layer indices for token-level routing MoE, for example `[0,1,...,35]`. |
| `train.token_num_experts` | int | `32` | Number of experts per token-level MoE layer. |
| `train.token_top_k` | int | `1` | Top-k for token-level routing. |
| `train.token_moe_intermediate_size` | int | `256` | Intermediate size for token-level MoE expert FFN. |
| `train.token_shared_intermediate_size` | int | `256` | Intermediate size for token-level shared expert FFN. |
| `train.bias_update_speed` | float | `0.001` | Bias update speed for loss-free MoE load balancing. |
| `train.bias_centering` | bool | `False` | Center loss-free correction bias each step to prevent cumulative drift. |
| `train.bias_update_interval` | int | `1` | Apply the loss-free bias update every N optimizer steps. |
| `train.sequence_wise_loss_coeff` | float | `0.0` | Coefficient for sequence-wise MoE balance loss. `0` disables it. |
| `train.sequence_wise_mode` | str | `"per_sequence"` | Granularity of sequence-wise balance loss: `"per_sequence"` or `"global"`. |
| `train.router_z_loss_coeff` | float | `0.0` | Coefficient for router z-loss on raw router logits. `0` disables it. |
| `train.moe_monitor_interval` | int | `50` | Step interval for writing MoE monitor scalars and expert-selection histograms. |
| `train.router_activation` | str | `"softmax"` | Router activation function, `"softmax"` or `"sigmoid"`. |
| `train.routed_scaling_factor` | float | `1.0` | Scaling factor applied to routing weights after norm. `1.0` disables additional scaling. |
| `train.use_shared_expert_gate` | bool | `True` | Whether to use a sigmoid gate on shared expert output. |
| `train.use_moe_expert_lr` | bool | `False` | Whether to apply scaled learning rate to MoE routed experts. |
| `train.split_fused_experts_from_decoder_fsdp` | bool | `False` | Exclude `Qwen2FusedExperts` params from Qwen decoder FSDP2 units without wrapping experts in FSDP2. |
| `train.expert_hidden_size` | int | `768` | Hidden size for the action expert. |
| `train.expert_intermediate_size` | int | `2752` | FFN intermediate size for the action expert. |
| `train.action_fp32` | bool | `False` | Whether to use fp32 action and state tensors. |
| `train.precompute_grid_thw` | bool | `False` | Precompute and cache `grid_thw`-derived tensors such as `rotary_pos_emb` and `window_index` for fixed-resolution training. |

### Depth, Future Image, and Video Alignment Arguments

`train.align_params` is a nested dictionary. It is not a dataclass, but the
current LingBot-VLA 2.0 configs use the fields below.

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `train.align_params.mode` | str | `"query"` in V2 configs | Alignment mode. |
| `train.align_params.num_task_tokens` | int | `8` in V2 configs | Number of task/query tokens. |
| `train.align_params.depth_loss_weight` | float | `0.004` in V2 configs | Weight for current-depth supervision. |
| `train.align_params.future_depth_loss_weight` | float | `0.004` in V2 configs | Weight for future-depth supervision. |
| `train.align_params.use_future_video` | bool | `true` in V2 configs | Whether to enable future-video/DINO supervision. |
| `train.align_params.visual_steps` | int | `5000` in V2 configs | Interval for visual logging. |
| `train.align_params.llm.dim_out` | int | `2560` in V2 configs | LLM hidden/output dimension used by alignment modules. |
| `train.align_params.llm.image_token_size` | int | `8` in V2 configs | Image token size used by the LLM side. |
| `train.align_params.llm.image_input_size` | int | `224` in V2 configs | Image input size used by the LLM side. |
| `train.align_params.depth.model_type` | str | `"MoRGBD"` in V2 configs | Depth model type. |
| `train.align_params.depth.moge_path` | str | Required for depth | Path to MoGe weights. |
| `train.align_params.depth.morgbd_path` | str | Required for depth | Path to MoRGBD/LingBot-Depth weights. |
| `train.align_params.depth.num_layers` | int | `1` in V2 configs | Depth adapter layer count. |
| `train.align_params.depth.num_heads` | int | `4` in V2 configs | Depth adapter attention heads. |
| `train.align_params.depth.dim_head` | int | `32` in V2 configs | Depth adapter head dimension. |
| `train.align_params.depth.ff_mult` | int | `1` in V2 configs | Depth adapter FFN multiplier. |
| `train.align_params.depth.num_backbone_tokens` | int | `256` in V2 configs | Number of depth backbone tokens. |
| `train.align_params.depth.token_size` | int | `16` in V2 configs | Depth token size. |
| `train.align_params.depth.dim_out` | int | `1024` in V2 configs | Depth adapter output dimension. |
| `train.align_params.depth.input_size` | int | `224` in V2 configs | Depth input image size. |
| `train.align_params.depth.use_future_depth` | bool | `true` in V2 configs | Whether to enable future-depth targets. |
| `train.align_params.depth.block_future_depth_to_action` | bool | `true` in V2 configs | Block gradients from future-depth branch to action branch. |
| `train.align_params.depth.future_depth_head_type` | str | `"resampler"` in V2 configs | Future-depth head type. |
| `train.align_params.depth.block_warmup_steps` | int | `0` in V2 configs | Warmup steps for blocking depth gradients. |
| `train.align_params.depth.block_warmup_gradual` | bool | `false` in V2 configs | Whether depth blocking warmup is gradual. |
| `train.align_params.depth.detach_future_image_feats` | bool | `true` in V2 configs | Detach future-image features for the future-depth branch. |
| `train.align_params.video.ckpt_path` | str | Required for video | Path to video-DINO teacher checkpoint. |
| `train.align_params.video.config_path` | str | Required for video | Path to video-DINO teacher config. |
| `train.align_params.video.attention_mode` | str | `"flex_block_causal"` in V2 configs | Video attention mode. |
| `train.align_params.video.input_size` | int | `256` in V2 configs | Video teacher input size. |
| `train.align_params.video.block_suffix_to_future_video` | bool | `true` in V2 configs | Block suffix tokens to future-video branch. |
| `train.align_params.video.block_warmup_steps` | int | `0` in V2 configs | Warmup steps for future-video blocking. |
| `train.align_params.video.block_warmup_gradual` | bool | `false` in V2 configs | Whether future-video blocking warmup is gradual. |
| `train.align_params.video.share_future_depth_query` | bool | `true` in V2 configs | Share grouped future depth/video query seed. |
| `train.align_params.video.use_shared_future_task_proj` | bool | `true` in V2 configs | Use shared future task projection. |
| `train.align_params.video.use_current_shared_task_proj` | bool | `true` in V2 configs | Use current shared task projection. |
| `train.align_params.video.shared_query_head_type` | str | `"resampler"` in V2 configs | Shared query head type. |
| `train.align_params.video.num_future_frames` | int | `1` in V2 configs | Number of future frames. |
| `train.align_params.video.use_warmup_frame` | bool | `true` in V2 configs | Keep DINO teacher input as `[warmup current, current, future]`. |
| `train.align_params.video.effective_fps` | float | `1.0` in V2 configs | Effective FPS used for future-video target. |
| `train.align_params.video.n_blocks` | int | `1` in V2 configs | Number of video blocks. |
| `train.align_params.video.cls_pool` | str | `"last"` in V2 configs | CLS pooling type. |
| `train.align_params.video.head_type` | str | `"resampler"` in V2 configs | Video head type. |
| `train.align_params.video.detach_image_feats` | bool | `true` in V2 configs | Detach image features for video branch. |
| `train.align_params.video.num_layers` | int | `1` in V2 configs | Video adapter layer count. |
| `train.align_params.video.num_heads` | int | `4` in V2 configs | Video adapter attention heads. |
| `train.align_params.video.dim_head` | int | `32` in V2 configs | Video adapter head dimension. |
| `train.align_params.video.ff_mult` | int | `1` in V2 configs | Video adapter FFN multiplier. |
| `train.align_params.video.num_backbone_tokens` | int | `256` in V2 configs | Number of video backbone tokens. |
| `train.align_params.video.dim_out` | int | `1024` in V2 configs | Video adapter output dimension. |
| `train.align_params.video.target_type` | str | `"absolute"` in V2 configs | Video target type. |
| `train.align_params.video.future_video_loss_weight` | float | `0.004` in V2 configs | Weight for future-video loss. |
| `train.align_params.video.use_smooth_l1_loss` | bool | `false` in V2 configs | Whether to use Smooth L1 for video loss. |
| `train.align_params.video.use_mse_loss` | bool | `true` in V2 configs | Whether to use MSE for video loss. |
| `train.align_params.video.mse_loss_weight` | float | `1.0` in V2 configs | MSE loss weight. |
| `train.align_params.video.use_patch_loss` | bool | `true` in V2 configs | Enable DINO patch alignment. |
| `train.align_params.video.use_current_patch_loss` | bool | `true` in V2 configs | Enable current-frame DINO patch alignment. |
| `train.align_params.video.use_cosine_loss` | bool | `false` in V2 configs | Whether to use cosine loss. |
| `train.align_params.video.cosine_loss_weight` | float | `0.2` in V2 configs | Cosine loss weight. |
| `train.align_params.video.use_cls_loss` | bool | `false` in V2 configs | Whether to use CLS loss. |
| `train.align_params.video.cls_loss_type` | str | `"mse"` in V2 configs | CLS loss type. |
| `train.align_params.video.cls_loss_weight` | float | `0.2` in V2 configs | CLS loss weight. |
| `train.align_params.video.log_max_samples` | int | `32` in V2 configs | Max samples for visual logging. |
| `train.align_params.video.log_scale` | int | `16` in V2 configs | Scale factor used by visual logging. |

### Batch Size Rule

When `train.global_batch_size` is explicitly set, the runtime checks:

```text
global_batch_size == micro_batch_size * data_parallel_size * gradient_accumulation_steps
```

If the values do not match, training stops with a `ValueError`. For example, on
4 GPUs with `micro_batch_size: 1` and `gradient_accumulation_steps: 1`, set
`global_batch_size: 4`.
