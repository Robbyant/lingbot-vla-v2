import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Literal
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
import wandb
from PIL import Image
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from lingbotvla.utils.async_tb_writer import AsyncTBWriter
from lingbotvla.checkpoint import build_checkpointer
from lingbotvla.data import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
    VLADataCollatorWithPacking,
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
    build_vla_dataset,
)
from lingbotvla.distributed.offloading import build_activation_offloading_context
from lingbotvla.distributed.parallel_state import get_parallel_state, init_parallel_state
from lingbotvla.distributed.torch_parallelize import build_parallelize_model
from lingbotvla.models import build_foundation_model, build_processor, save_model_assets, build_tokenizer
from lingbotvla.optim import build_lr_scheduler, build_muon_optimizer, build_optimizer
from lingbotvla.utils import helper
from lingbotvla.utils.async_hf_checkpoint import AsyncHFCheckpointSaver
from lingbotvla.utils.arguments import EvalArguments, DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from lingbotvla.utils.dist_utils import all_reduce
from lingbotvla.models.config_registry import get_config_registry

from lingbotvla.models.vla.vision_models.module_utils import (
    build_depth_model,
    build_video_model,
    get_depth_target,
    get_video_target,
    log_video,
)
from lingbotvla.models.vla.lingbot_vla.moe_load_balance import build_moe_load_balance_hook
import gc
gc.set_threshold(50000, 50, 50)

logger = helper.create_logger(__name__)
# try:
#     from aistudio_tracking import training_tracking as wandb
# except Exception as e:
#     logger.info_rank0(f"Failed to import aistudio_tracking: {repr(e)}.")

def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]


def get_moe_param_groups(model: "torch.nn.Module", args_train) -> Optional[List[Dict]]:
    """
    Build optimizer param groups with token-MoE expert LR scaling.

    Only routed expert weights (.mlp.experts.*) receive the scaled LR.
    Gate / shared-expert / all dense params keep the base LR.

    Scale formula (always auto-computed):
        token-level MoE:  scale = (token_num_experts  / token_top_k)^0.5

    Returns None when MoE is disabled so the caller can fall back to the
    default single-group behaviour.
    """
    if not getattr(args_train, 'use_moe', False):
        return None
    if not getattr(args_train, 'use_moe_expert_lr', False):
        return None

    base_lr = args_train.lr

    token_moe_layers = set(getattr(args_train, 'token_moe_layers', None) or [])

    # Auto-compute token expert scale: (num_experts / top_k)^0.5
    token_scale = (args_train.token_num_experts / args_train.token_top_k) ** 0.5 if token_moe_layers else None

    if token_scale is None:
        return None

    # Match routed-expert params by FQN: ...layers.<idx>.mlp.experts....
    # Gate / shared_expert / shared_expert_gate keep base_lr.
    layer_expert_re = re.compile(r'\.layers\.(\d+)\.mlp\.experts\.')

    lr_to_params: Dict[float, List] = {base_lr: []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        m = layer_expert_re.search(name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx in token_moe_layers and token_scale is not None:
                expert_lr = base_lr * token_scale
            else:
                expert_lr = base_lr
            lr_to_params.setdefault(expert_lr, []).append(param)
        else:
            lr_to_params[base_lr].append(param)

    return [{"params": params, "lr": lr} for lr, params in lr_to_params.items() if params]

@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )
    freeze_vision_encoder: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vision encoder in PI0 model."},
    )
    train_expert_only: bool = field(
        default=False,
        metadata={"help": "Train action expert only or not."},
    )
    train_state_proj: bool = field(
        default=True,
        metadata={"help": "Train state proj only or not."},
    )
    tokenizer_max_length: int = field(
        default=48,
        metadata={"help": "Maximum length of the tokenizer."},
    )
    action_dim: int = field(
        default=7,
        metadata={"help": "Action dimension."},
    )
    max_action_dim: int = field(
        default=32,
        metadata={"help": "Action dimension after padding."},
    )
    max_state_dim: int = field(
        default=32,
        metadata={"help": "State dimension after padding."},
    )
    chunk_size: int = field(
        default=50,
        metadata={"help": "Chunk size of action."},
    )
    vlm_causal: bool = field(
        default=False,
        metadata={"help": "Whether to use causal atten for img anb lang tokens in vlm."},
    )
    loss_type: str = field(
        default='fm',
        metadata={"help": "Which loss to use."},
    )
    align_params: Optional[Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"help": "The config of vaco"},
    )
    attention_implementation: str = field(
        default="flex",
        metadata={"help": "Attention implementation: flex, flex_cached, or eager."},
    )
    my_tokenizer_max_length: int = field(
        default=72,
        metadata={"help": ""},
    )
    decayed_max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum norm for the decayed gradients."},
    )
    stable_train_steps: int = field(
        default=100000,
        metadata={"help": "Training steps for stable training, after this step, the decayed_max_grad_norm will be applied."},
    )
    resume_dataloader_state: bool = field(
        default=True,
        metadata={"help": "Whether to resume dataloader."},
    )
    use_moe: bool = field(
        default=False,
        metadata={"help": "Whether to use MoE."},
    )
    token_moe_layers: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Layer indices for token-level routing MoE, e.g. [1,2,...,34]."},
    )
    token_num_experts: int = field(
        default=32,
        metadata={"help": "Number of experts per token-level MoE layer."},
    )
    token_top_k: int = field(
        default=1,
        metadata={"help": "Top-k for token-level routing."},
    )
    token_moe_intermediate_size: int = field(
        default=256,
        metadata={"help": "Intermediate size for token-level MoE expert FFN."},
    )
    token_shared_intermediate_size: int = field(
        default=256,
        metadata={"help": "Intermediate size for token-level shared expert FFN."},
    )
    bias_update_speed: float = field(
        default=0.001,
        metadata={"help": "Bias update speed for loss-free MoE load balancing."},
    )
    bias_centering: bool = field(
        default=False,
        metadata={"help": "If True, center the loss-free correction bias each step (subtract per-layer mean) to prevent cumulative drift. Routing-invariant; pure numerical hygiene."},
    )
    bias_update_interval: int = field(
        default=1,
        metadata={"help": "Apply the loss-free bias update once every N optimizer steps, accumulating tokens_per_expert in between. >1 averages load over more tokens so sign(load-mean) is reliable when the global batch is small (per-step load too noisy -> bias random-walks). Also makes the load monitor (maxvio/min_load/has_dead) windowed/less noisy."},
    )
    sequence_wise_loss_coeff: float = field(
        default=0.0,
        metadata={"help": "Coefficient for sequence-wise MoE balance loss (DeepSeek-V3 style, complementary to loss-free bias). 0 = disabled."},
    )
    sequence_wise_mode: str = field(
        default="per_sequence",
        metadata={"help": "Granularity of the sequence-wise balance loss: 'per_sequence' (balance experts within each sample's token sequence, DeepSeek-V3 intent) or 'global' (treat the whole batch as one sequence). Ablation knob."},
    )
    router_z_loss_coeff: float = field(
        default=0.0,
        metadata={"help": "Coefficient for router z-loss computed on raw router logits. 0 = disabled."},
    )
    moe_monitor_interval: int = field(
        default=50,
        metadata={"help": "Step interval for writing per-layer MoE monitor scalars (moe_maxvio/minvio/minload/entropy/topksigmoid) and expert-selection histograms to TensorBoard. moe_summary/* is always written every step. Set small (e.g. 1/10) for close debugging."},
    )
    router_activation: str = field(
        default="softmax",
        metadata={"help": "Router activation function: 'softmax' or 'sigmoid'. Default 'softmax' for backward compat."},
    )
    routed_scaling_factor: float = field(
        default=1.0,
        metadata={"help": "Scaling factor applied to routing weights after norm (DeepSeek-V3 style). 1.0 = disabled."},
    )
    use_shared_expert_gate: bool = field(
        default=True,
        metadata={"help": "Whether to use sigmoid gate on shared expert output. True = current behavior, False = direct add (DS-V3 style)."},
    )
    # Action expert architecture params
    expert_hidden_size: int = field(
        default=768,
        metadata={"help": "Hidden size for action expert."},
    )
    expert_intermediate_size: int = field(
        default=2752,
        metadata={"help": "FFN intermediate size for action expert."},
    )
    action_fp32: bool = field(
        default=False,
        metadata={"help": "Whether to use fp32 action and state."},
    )
    precompute_grid_thw: bool = field(
        default=False,
        metadata={"help": "Whether to precompute and cache grid_thw-derived tensors (rotary_pos_emb, window_index, etc.) for fixed-resolution training."},
    )
    use_moe_expert_lr: bool = field(
        default=False,
        metadata={"help": "Whether to apply scaled LR to MoE routed experts."},
    )
    split_fused_experts_from_decoder_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to exclude Qwen2FusedExperts params from Qwen2DecoderLayer FSDP2 units without wrapping the experts in FSDP2."},
    )
    vlm_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to apply FSDP2 for VLM."},
    )

@dataclass
class MyDataArguments(DataArguments):
    source_name: str = field(
        default=None,
        metadata={"help": "Source name of dataset."},
    )
    robot_config_root: str = field(
        default=None,
        metadata={"help": "Path to get all robot configs."},
    )
    joints: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The order of joints and their dim"},
    )
    cameras:Optional[List[str]] = field(
        default=None,
        metadata={"help": "The order of used images"},
    )
    norm_type: Optional[List[str]] = field(default=None, metadata={"help": "Normalization type."})
    img_size: int = field(
        default=256,
        metadata={"help": "Size of the image."},
    )
    norm_stats_file: str = field(
        default=None,
        metadata={"help": "Path to the normalization stats file."},
    )
    prompt_type: Literal["global", "subtask"] = field(
        default="global",
        metadata={"help": "Type of the prompt."},
    )
    use_future_image: bool = field(
        default=False,
        metadata={"help": "Whether to use future image."},
    )


@dataclass
class Arguments:

    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)
    eval: "EvalArguments" = field(default_factory=EvalArguments)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare model")
    config_kwargs = {**vars(args.model), **vars(args.train)}
    config_registry = get_config_registry()

    config_key = args.model.config_key

    if config_key in config_registry.supported_configs:
        config_cls = config_registry.get_config_cls_from_config_key(config_key)
        config_cls = config_cls(**config_kwargs)
        logger.info_rank0(f"Successfully loaded: {config_cls.__class__.__name__}")
    else:
        config_cls = None
    model = build_foundation_model(
        config_path=args.model.config_path,
        config_cls=config_cls,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
        config_kwargs=config_kwargs,
        moe_implementation=getattr(args.model, 'moe_implementation', None),
    )
    use_depth_align = True if args.train.align_params != {} else False
    use_future_depth = args.train.align_params.get('depth', {}).get('use_future_depth', False)
    use_future_video = use_depth_align and args.train.align_params.get('use_future_video', False)
    depth_model_type = None
    video_teacher = None
    future_video_loss_weight = 1.0
    depth_loss_weight = 1.0
    future_depth_loss_weight = 1.0
    if use_future_video:
        if not args.data.use_future_image:
            raise ValueError("align_params.use_future_video=True requires data.use_future_image=True.")
        video_cfg = args.train.align_params.get("video", {})
        future_video_loss_weight = video_cfg.get(
            "future_video_loss_weight",
            args.train.align_params.get(
                "future_video_loss_weight",
                args.train.align_params.get("depth_loss_weight", 1.0),
            ),
        )
    if use_depth_align:
        depth_model_type = args.train.align_params['depth']['model_type']
        if depth_model_type != 'MoRGBD':
            raise ValueError(f"Only MoRGBD depth distillation is supported, got {depth_model_type!r}.")
        depth_loss_weight = args.train.align_params.get("depth_loss_weight", 1.0)
        future_depth_loss_weight = args.train.align_params.get("future_depth_loss_weight", 1.0)
        print('====Loading Depth Model====')
        moge_model, morgbd_model = build_depth_model(args.train.align_params)
        if args.train.use_compile:
            moge_model = torch.compile(moge_model)
            morgbd_model = torch.compile(morgbd_model)
        if 'visual_dir' not in args.train.align_params or not args.train.align_params['visual_dir']:
            args.train.align_params['visual_dir'] = os.path.join(args.train.output_dir, 'images')
        os.makedirs(args.train.align_params['visual_dir'], exist_ok=True)
        if use_future_video:
            print('====Loading Future Video Model====')
            video_teacher = build_video_model(args.train.align_params['video'])
    from lingbotvla.utils.moe_utils import log_model_param_stats
    log_model_param_stats(model)

    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path) # if use build_processor,  tokenizer is processor.tokenizer

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = []
    if args.data.datasets_type == 'vla':
        data_collate_fn.append(VLADataCollatorWithPacking())
    else:
        if args.train.rmpad_with_pos_ids:
            data_collate_fn.append(OmniDataCollatorWithPacking()) # TODO 8.21
        else:
            data_collate_fn.append(OmniDataCollatorWithPadding())
    
    if args.data.dataloader_type == "native":
        if args.data.datasets_type == 'vla':
            args.data.chunk_size = args.train.chunk_size
            train_dataset = build_vla_dataset(dataset_config=args.data, model_config=args.model, config=model.config, processor=processor, use_depth_align=use_depth_align)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))
        
        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=data_collate_fn,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor if args.data.num_workers > 0 else None,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    fsdp_kwargs = {}
    if args.train.freeze_vit:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_fp32=args.train.enable_fp32,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules if args.train.module_fsdp_enable else None,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        fsdp_llm_blocks=False,
        ignore_norm=False,
        use_depth_align=use_depth_align,
        split_fused_experts_from_decoder_fsdp=args.train.split_fused_experts_from_decoder_fsdp,
        vlm_fsdp=args.train.vlm_fsdp,
        use_future_image=args.data.use_future_image,
    )
    logger.info_rank0(model)
    if args.train.use_compile:
        model = torch.compile(model)

    moe_param_groups = get_moe_param_groups(model, args.train)
    if moe_param_groups is not None:
        n_expert = sum(len(g["params"]) for g in moe_param_groups if g["lr"] != args.train.lr)
        group_summary = {f"lr={g['lr']:.2e}": len(g["params"]) for g in moe_param_groups}
        logger.info_rank0(
            f"MoE expert LR scaling enabled: {n_expert} expert param tensors use scaled LR. "
            f"Groups: {group_summary}"
        )
    if args.train.optimizer == "muon":
        optimizer = build_muon_optimizer(
            model,
            args.train,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
        )
        muon_groups, adamw_groups = (
            optimizer.optimizers[0].param_groups,
            optimizer.optimizers[1].param_groups if len(optimizer.optimizers) > 1 else [],
        )
        muon_summary = {f"lr={g['lr']:.2e}": len(g["params"]) for g in muon_groups}
        adamw_summary = {f"lr={g['lr']:.2e}": len(g["params"]) for g in adamw_groups}
        logger.info_rank0(
            f"Muon enabled. Muon groups: {muon_summary}; AdamW (1D/embed) groups: {adamw_summary}"
        )
    else:
        optimizer = build_optimizer(
            model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=False,
            optimizer_type=args.train.optimizer,
            post_training=args.model.post_training,
            param_groups=moe_param_groups,
        )

    # Register loss-free load balancing hook (before optimizer.step).
    # The hook also all-reduces and snapshots
    # last_tokens_per_expert (the global load used for monitoring). Setting
    # bias_update_speed=0 makes the bias update a no-op (bias frozen at 0) while
    # keeping the global load monitoring intact.
    if args.train.use_moe:
        _lb_hook = build_moe_load_balance_hook(
            model, coeff=args.train.bias_update_speed, bias_centering=args.train.bias_centering,
            update_interval=args.train.bias_update_interval,
        )
        optimizer.register_step_pre_hook(_lb_hook)
        logger.info_rank0(
            f"Registered MoE load-balance pre-hook (coeff={args.train.bias_update_speed}, "
            f"bias_centering={args.train.bias_centering}, "
            f"update_interval={args.train.bias_update_interval})"
        )

    total_train_steps = args.train.train_steps * args.train.num_train_epochs
    if args.train.max_steps is not None:
        total_train_steps = min(total_train_steps, args.train.max_steps)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=total_train_steps,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        log_dir=f"{args.train.output_dir}/runs/"
        writer = AsyncTBWriter(log_dir=log_dir)
        if args.train.use_wandb:
            wandb.init(
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        model_assets = [model_config, processor]
        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    current_epoch_for_eval, current_epoch_step_for_eval = 1, 0
    save_checkpoint_path = None
    hf_failure_log_path = (
        os.path.join(args.train.save_checkpoint_path, "async_hf_failures.jsonl")
        if args.train.save_checkpoint_path
        else None
    )
    hf_saver = AsyncHFCheckpointSaver(
        enabled=args.train.async_save_hf_weights,
        max_pending=args.train.async_hf_max_pending,
        logger=logger,
        failure_log_path=hf_failure_log_path,
        eval_args=args.eval,
    )

    def save_hf_checkpoint_best_effort(
        checkpoint_path: str | None,
        checkpoint_state: Dict[str, Any],
        step: int,
        epoch: int | None = None,
        epoch_step: int | None = None,
    ) -> None:
        if args.train.global_rank != 0:
            return
        if not args.train.save_hf_weights or checkpoint_path is None:
            return
        hf_saver.submit(
            global_step=step,
            save_checkpoint_path=checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
            save_ema=checkpoint_state.get("ema") is not None,
            enable_fp32=args.train.enable_fp32,
            model_assets=model_assets,
            epoch=epoch,
            epoch_step=epoch_step,
        )

    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    load_checkpoint_path = None
    candidates = []
    if args.train.load_checkpoint_path or args.train.enable_resume:
        if args.train.load_checkpoint_path:
            load_checkpoint_path = args.train.load_checkpoint_path
            candidates = [load_checkpoint_path]
        elif args.train.enable_resume:
            checkpoint_dir = f'{args.train.output_dir}/checkpoints'
            if os.path.exists(checkpoint_dir):
                pattern = re.compile(r"global_step_(\d+)")
                tmp = []
                for dirname in os.listdir(checkpoint_dir):
                    match = pattern.fullmatch(dirname)
                    if match:
                        step = int(match.group(1))
                        tmp.append((step, os.path.join(checkpoint_dir, dirname)))
                tmp.sort(key=lambda x: x[0], reverse=True)
                candidates = [p for _, p in tmp]
            if candidates:
                load_checkpoint_path = candidates[0]
            else:
                logger.info_rank0(f"No checkpoints in {args.train.output_dir} now!")
    if candidates:
        last_err = None
        loaded = False
        for cp in candidates:
            state = {"model": model, "ema": None, "optimizer": optimizer, "extra_state": {}}  # cannot be None
            try:
                Checkpointer.load(cp, state, allow_partial_load=getattr(args.train, 'allow_partial_checkpoint', False))
                global_step = state["extra_state"]["global_step"]
                start_epoch = global_step // args.train.train_steps
                start_step = global_step % args.train.train_steps
                lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
                if start_step > 0 and args.train.resume_dataloader_state:
                    train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
                environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
                torch.set_rng_state(state["extra_state"]["torch_rng_state"])
                if start_step == 0:  # resume at the end of epoch
                    iter(train_dataloader)  # clear resume state and prefetch data
                dist.barrier()
                logger.info_rank0(f"Load distributed checkpoint from {cp} successfully!")
                loaded = True
                break
            except Exception as e:
                last_err = e
                logger.info_rank0(f"Failed to load checkpoint {cp}: {repr(e)}. Trying older one...")
                continue
        if not loaded:
            logger.info_rank0("Starting training from scratch. No valid checkpoint could be loaded.")
    else:
        logger.info_rank0("Starting training from scratch.")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    # create the path in advance to save loss log
    if args.train.global_rank == 0:
        os.makedirs(args.train.save_checkpoint_path, exist_ok=True)
    reached_max_steps = False
    max_steps_driven = (
        args.train.max_steps is not None
        and args.train.max_steps < args.train.train_steps * args.train.num_train_epochs
    )
    if max_steps_driven:
        data_loader_tqdm = trange(
            args.train.max_steps,
            bar_format="Step: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
            initial=global_step,
            disable=args.train.local_rank != 0,
        )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        current_epoch_for_eval = epoch + 1
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        if not max_steps_driven:
            data_loader_tqdm = trange(
                args.train.train_steps,
                desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
                total=args.train.train_steps,
                initial=start_step,
                disable=args.train.local_rank != 0,
            )
        data_iterator = iter(train_dataloader)
        for epoch_step in range(start_step, args.train.train_steps):
            current_epoch_step_for_eval = epoch_step + 1
            global_step += 1
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss, total_vla_loss, total_depth_loss, total_future_depth_loss, total_future_video_loss, total_seq_wise_loss, total_router_z_loss = 0, 0, 0, 0, 0, 0, 0
            depth_targets, depth_preds = None, None
            future_depth_targets, future_depth_preds = None, None
            future_video_targets, future_video_preds = None, None
            future_video_current_preds = None
            future_video_cls_targets = None
            future_video_current_dino = None
            future_video_current_rgb, future_video_target_rgb = None, None
            ignore_batch_num = 0
            torch.cuda.synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                future_video_targets = None
                future_video_current_preds = None
                future_video_cls_targets = None
                future_video_current_dino = None
                dataset_names = micro_batch.pop('rep_id', None)
                environ_meter.add(micro_batch)
                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()
                }
                future_video_effective_fps = micro_batch.pop('future_video_effective_fps', None)
                depth_forward_time = 0
                if use_depth_align:
                    with torch.no_grad():
                        depth_start_time = time.time()
                        pil_images = micro_batch.pop('pil_images', None)
                        future_pil_images = micro_batch.pop('future_pil_images', None) if (use_future_depth or use_future_video) else None
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            depth_targets, cls_token = get_depth_target(depth_model_type, (moge_model, morgbd_model), pil_images)
                            if use_future_depth:
                                future_depth_targets, future_cls_token = get_depth_target(depth_model_type, (moge_model, morgbd_model), future_pil_images)
                        if use_future_video:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                future_video_target_bundle = get_video_target(
                                    video_teacher,
                                    pil_images,
                                    future_pil_images,
                                    args.train.align_params['video'],
                                    effective_fps=future_video_effective_fps,
                                )
                                if isinstance(future_video_target_bundle, dict):
                                    future_video_targets = future_video_target_bundle["patch"]
                                    future_video_cls_targets = future_video_target_bundle.get("cls")
                                    future_video_current_dino = future_video_target_bundle.get("current_patch")
                                elif isinstance(future_video_target_bundle, tuple):
                                    future_video_targets, future_video_cls_targets = future_video_target_bundle
                                else:
                                    future_video_targets = future_video_target_bundle
                            future_video_current_rgb = pil_images
                            future_video_target_rgb = future_pil_images
                        depth_forward_time = time.time() - depth_start_time

                with model_fwd_context:
                    model_outputs = model(
                        **micro_batch,
                        depth_targets=depth_targets,
                        future_depth_targets=future_depth_targets,
                        future_video_targets=future_video_targets,
                        future_video_cls_targets=future_video_cls_targets,
                        future_video_current_patch=future_video_current_dino,
                    )
                    if len(model_outputs) == 6:
                        loss, vla_loss, depth_loss, seq_wise_loss, loss_log, depth_preds = model_outputs
                        future_depth_loss = 0
                        future_video_loss = 0
                        future_depth_preds = None
                        future_video_preds = None
                        future_video_current_preds = None
                    elif len(model_outputs) == 8:
                        loss, vla_loss, depth_loss, future_depth_loss, seq_wise_loss, loss_log, depth_preds, future_depth_preds = model_outputs
                        future_video_loss = 0
                        future_video_preds = None
                        future_video_current_preds = None
                    elif len(model_outputs) == 10:
                        loss, vla_loss, depth_loss, future_depth_loss, future_video_loss, seq_wise_loss, loss_log, depth_preds, future_depth_preds, future_video_preds = model_outputs
                        future_video_current_preds = None
                    elif len(model_outputs) == 11:
                        loss, vla_loss, depth_loss, future_depth_loss, future_video_loss, seq_wise_loss, loss_log, depth_preds, future_depth_preds, future_video_preds, future_video_current_preds = model_outputs
                    else:
                        raise ValueError(f"Unexpected model output length: {len(model_outputs)}")

                    loss = loss / len(micro_batches)
                    vla_loss = vla_loss / len(micro_batches)
                    depth_loss = depth_loss / len(micro_batches)
                    future_depth_loss = future_depth_loss / len(micro_batches)
                    future_video_loss = future_video_loss / len(micro_batches)
                    seq_wise_loss = seq_wise_loss / len(micro_batches)
                    router_z_loss = loss_log.get("router_z_loss", loss_log.get("moe_zloss/weighted", 0))
                    avg_lang_length = micro_batch['lang_masks'].sum(dim=-1).float().mean()

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                total_vla_loss += vla_loss.item()
                if not (isinstance(depth_loss, int) or isinstance(depth_loss, float)):
                    total_depth_loss += depth_loss.item()
                if not (isinstance(future_depth_loss, int) or isinstance(future_depth_loss, float)):
                    total_future_depth_loss += future_depth_loss.item()
                if not (isinstance(future_video_loss, int) or isinstance(future_video_loss, float)):
                    total_future_video_loss += future_video_loss.item()
                if not (isinstance(seq_wise_loss, int) or isinstance(seq_wise_loss, float)):
                    total_seq_wise_loss += seq_wise_loss.item()
                if not (isinstance(router_z_loss, int) or isinstance(router_z_loss, float)):
                    total_router_z_loss += router_z_loss.item() / len(micro_batches)
                del micro_batch
            # --- TEMP gate-gradient probe (GATE_GRAD_PROBE=1): dump router gate grad
            # magnitude vs routed-expert grad, plus per-expert symmetry, pre-clip. ---
            if os.environ.get("GATE_GRAD_PROBE") and global_step < 12:
                from torch.distributed.tensor import DTensor
                def _full(t):
                    return t.full_tensor() if isinstance(t, DTensor) else t
                _moe_blocks = [m for m in model.modules()
                               if type(m).__name__ == "Qwen2TokenMoeBlock"]
                _lines = []
                for _li, _blk in enumerate(_moe_blocks):
                    _gw = _blk.gate.weight
                    _gg = _gw.grad
                    _gw_f = _full(_gw.detach())
                    _gg_f = _full(_gg.detach()) if _gg is not None else None
                    # routed-expert reference grad (experts[0].gate_proj), collective on all ranks
                    _eg_f = None
                    if hasattr(_blk.experts, "__getitem__"):
                        _ep = _blk.experts[0].gate_proj.weight
                        if _ep.grad is not None:
                            _eg_f = _full(_ep.grad.detach())
                    if args.train.local_rank == 0:
                        if _gg_f is None:
                            _lines.append(f"L{_li}: gate.grad=None rg={_gw.requires_grad}")
                        else:
                            _wn = _gw_f.norm().item(); _gn = _gg_f.norm().item()
                            _row = _gg_f.norm(dim=1)
                            _eref = f" exp|g|={_eg_f.norm().item():.2e}" if _eg_f is not None else ""
                            _lines.append(
                                f"L{_li}: |g|={_gn:.2e} |w|={_wn:.2e} g/w={_gn/(_wn+1e-9):.1e} "
                                f"rowμ={_row.mean():.2e} rowσ={_row.std():.2e} "
                                f"[{_row.min():.1e},{_row.max():.1e}]{_eref}")
                if args.train.local_rank == 0:
                    print(f"[GATE_GRAD_PROBE step={global_step}]\n  " + "\n  ".join(_lines), flush=True)
            if global_step > args.train.stable_train_steps:
                max_grad_norm = args.train.decayed_max_grad_norm
            else:
                max_grad_norm = args.train.max_grad_norm
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(max_grad_norm).item()
            elif hasattr(model, '_ep_param_set'):
                from lingbotvla.distributed.fsdp2.clip_grad_norm import ep_fsdp2_clip_grad_norm
                grad_norm = ep_fsdp2_clip_grad_norm(model, max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, total_vla_loss, total_depth_loss, total_future_depth_loss, total_future_video_loss, total_seq_wise_loss, total_router_z_loss, avg_lang_length, grad_norm = all_reduce((total_loss, total_vla_loss, total_depth_loss, total_future_depth_loss, total_future_video_loss, total_seq_wise_loss, total_router_z_loss, avg_lang_length, grad_norm), group=get_parallel_state().fsdp_group)
            total_depth_loss = total_depth_loss / depth_loss_weight
            total_future_depth_loss = total_future_depth_loss / future_depth_loss_weight
            total_future_video_loss = total_future_video_loss / future_video_loss_weight
            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            all_lrs = lr_scheduler.get_last_lr()
            if args.train.use_moe_expert_lr and len(all_lrs) > 1:
                lr = min(all_lrs)          # base (non-expert) lr
                expert_lr = max(all_lrs)   # routed-expert lr
            else:
                lr = max(all_lrs)
                expert_lr = None
            train_metrics = environ_meter.step(delta_time, global_step=global_step)
            data_loader_tqdm.update()
            expert_lr_str = f"Expert_LR {expert_lr:.2e}, " if expert_lr is not None else ""
            maxvio_val = loss_log.get("moe_summary/maxvio_avg", loss_log.get("token_moe/avg_maxvio", None))
            maxvio_str = f"MaxVio {maxvio_val.item() if torch.is_tensor(maxvio_val) else maxvio_val:.4f}, " if maxvio_val is not None else ""
            sigmoid_val = loss_log.get("moe_summary/topk_sigmoid_avg_rank0", loss_log.get("token_moe/avg_topk_sigmoid", None))
            sigmoid_str = f"AvgSigmoid {sigmoid_val.item() if torch.is_tensor(sigmoid_val) else sigmoid_val:.4f}, " if sigmoid_val is not None else ""
            logger.info_rank0(
                f"Step {global_step}/{args.train.train_steps}, "
                f"Epoch {epoch+1}, "
                f"Loss {total_loss:.4f}, "
                f"VLA_Loss {total_vla_loss:.4f}, "
                f"Depth_Loss {total_depth_loss:.4f}, "
                f"Future_Depth_Loss {total_future_depth_loss:.4f}, "
                f"FutureVideo_Loss {total_future_video_loss:.4f}, "
                f"SeqWise_Loss {total_seq_wise_loss:.4f}, "
                f"RouterZ_Loss {total_router_z_loss:.4f}, "
                f"{maxvio_str}"
                f"{sigmoid_str}"
                f"GradNorm {grad_norm:.4f}, "
                f"LR {lr:.2e}, "
                f"{expert_lr_str}"
                f"StepTime {delta_time:.3f}s, "
                f"Depth_Forward_Time {depth_forward_time: .3f}s, "
                f"Ignore_Batch_Num {ignore_batch_num}"
            )


            if args.train.global_rank == 0:
                writer.add_scalar("training/loss", total_loss, global_step)
                writer.add_scalar("training/vla_loss", total_vla_loss, global_step)
                writer.add_scalar("training/depth_loss", total_depth_loss, global_step)
                writer.add_scalar("training/future_depth_loss", total_future_depth_loss, global_step)
                writer.add_scalar("training/future_video_loss", total_future_video_loss, global_step)
                writer.add_scalar("training/sequence_wise_loss", total_seq_wise_loss, global_step)
                writer.add_scalar("training/router_z_loss", total_router_z_loss, global_step)
                # MoE monitoring metrics.
                #   moe_summary/*            -> every step (cheap cross-layer health glance)
                #   moe_<metric>/layerXX     -> every moe_monitor_interval steps (per-layer, downsampled)
                moe_interval = max(1, args.train.moe_monitor_interval)
                log_moe_perlayer = (global_step % moe_interval == 0)
                moe_perlayer_prefixes = (
                    "moe_maxvio/", "moe_minvio/", "moe_minload/",
                    "moe_entropy_rank0/", "moe_topksigmoid_rank0/", "moe_bias/",
                    "moe_seqwise/layer",   # per-layer seq-wise loss -> downsampled
                    "moe_zloss/layer",     # per-layer raw router z-loss -> downsampled
                )

                def _tb_scalar(value):
                    if torch.is_tensor(value):
                        if value.numel() != 1:
                            return None
                        return value.detach().float().item()
                    return value

                for key, value in loss_log.items():
                    # every step: cross-layer summaries + seq-wise average + legacy V1 keys
                    if (key.startswith("moe_summary/")
                            or key in ("moe_seqwise/avg", "moe_zloss/avg_raw", "moe_zloss/weighted")
                            or key.startswith("token_moe/")):
                        scalar = _tb_scalar(value)
                        if scalar is not None:
                            writer.add_scalar(key, scalar, global_step)
                    elif key.startswith(moe_perlayer_prefixes) and log_moe_perlayer:
                        scalar = _tb_scalar(value)
                        if scalar is not None:
                            writer.add_scalar(key, scalar, global_step)
                    elif key.startswith("align/"):
                        scalar = _tb_scalar(value)
                        if scalar is not None:
                            writer.add_scalar(key, scalar, global_step)
                align_training_aliases = {
                    "align/current_video_loss": "training/current_video_loss",
                    "align/current_video_loss_weighted": "training/current_video_loss_weighted",
                    "align/future_video_loss": "training/future_video_patch_loss",
                    "align/future_video_loss_weighted": "training/future_video_patch_loss_weighted",
                    "align/video_loss": "training/video_loss",
                    "align/video_loss_weighted": "training/video_loss_weighted",
                }
                for src_key, dst_key in align_training_aliases.items():
                    if src_key in loss_log:
                        scalar = _tb_scalar(loss_log[src_key])
                        if scalar is not None:
                            writer.add_scalar(dst_key, scalar, global_step)
                # MoE expert-selection monitoring (per layer, every moe_monitor_interval):
                #   moe_expert_selection/      -> original add_histogram (Histograms tab; bins
                #                                 expert IDs, edges look inflated -- kept as-is).
                #   moe_expert_selection_bar/  -> exact per-expert bar chart (Images tab; one bar =
                #                                 that expert's true token count, no binning artifact).
                #                                 Aligned with VideoPretrain's bar chart.
                # Plus a global (summed-over-layers) bar + load CV (std/mean) imbalance scalar.
                if log_moe_perlayer:
                    if "_token_moe_expert_counts" in loss_log:
                        total_counts = None
                        for layer_id, counts in loss_log["_token_moe_expert_counts"]:
                            c = counts.detach().float().cpu()
                            writer.add_histogram_from_counts(
                                f"moe_expert_selection/layer{layer_id:02d}", c, global_step,
                            )
                            writer.add_expert_bar(
                                f"moe_expert_selection_bar/layer{layer_id:02d}", c, global_step,
                                title=f"Expert load layer{layer_id:02d} (step {global_step})",
                            )
                            total_counts = c.clone() if total_counts is None else total_counts + c
                        if total_counts is not None:
                            writer.add_expert_bar(
                                "moe_expert_load_bar", total_counts, global_step,
                                title=f"Global expert load, all layers (step {global_step})",
                            )
                            mean = total_counts.mean()
                            load_cv = (total_counts.std(unbiased=False) / (mean + 1e-9)).item()
                            writer.add_scalar("moe_summary/load_cv", load_cv, global_step)
                writer.add_scalar("training/grad_norm", grad_norm, global_step)
                writer.add_scalar("training/lr", lr, global_step)
                if expert_lr is not None:
                    writer.add_scalar("training/expert_lr", expert_lr, global_step)
                writer.add_scalar("training/avg_lang_length", avg_lang_length, global_step)
                writer.add_scalar("training/max_norm_batch", ignore_batch_num, global_step)
                writer.add_scalar("steptime", delta_time, global_step)
                # we only log the last mini batch if grad acc is activated
                if dataset_names is not None and 'batch_mean_losses' in loss_log:
                    batch_mean_losses = loss_log['batch_mean_losses']  # shape (B,)
                    if hasattr(batch_mean_losses, "detach"):
                        batch_mean_losses = batch_mean_losses.detach().cpu()

                    group_losses = defaultdict(list)
                    for name, loss_value in zip(dataset_names, batch_mean_losses):
                        group_losses[name].append(loss_value.item() if hasattr(loss_value, "item") else float(loss_value))

                    for name, values in group_losses.items():
                        mean_loss = sum(values) / len(values)
                        writer.add_scalar(f"detailed_loss/{name}", mean_loss, global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()
                        helper.upload_trace(
                            args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir
                        )

                # loss.jsonl disabled to avoid NFS write blocking rank 0
                # loss_record = {
                #     "step": global_step,
                #     "epoch": epoch + 1,
                #     "loss": total_loss,
                #     "grad_norm": grad_norm,
                #     "lr": lr,
                #     "step_time": delta_time
                # }
                # loss_file_path = os.path.join(args.train.save_checkpoint_path, "loss.jsonl")
                # try:
                #     with open(loss_file_path, "a", encoding="utf-8") as f:
                #         f.write(json.dumps(loss_record, ensure_ascii=False) + "\n")
                # except Exception as e:
                #     logger.info_rank0(f"⚠️ Failed to write loss.jsonl: {e}")

                if use_depth_align:
                    if global_step % args.train.align_params['visual_steps'] == 0:
                        with torch.no_grad():
                            if use_future_video:
                                log_video(
                                    future_video_preds,
                                    future_video_targets,
                                    steps=global_step,
                                    config=args.train.align_params,
                                    is_future=True,
                                    current_rgb_images=future_video_current_rgb,
                                    target_rgb_images=future_video_target_rgb,
                                    current_target_feats=future_video_current_dino,
                                    current_pred_feats=future_video_current_preds,
                                )

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                
                # param_to_name = {}
                # for name, param in model.named_parameters():
                #     param_to_name[name] = param
                # for group in optimizer.state.values():
                #     for param in group:
                #         if param in param_to_name:
                #             print(param_to_name[param])
                #         else:
                #             print("⚠️ Unidentified parameter in optimizer state")

                state = {
                    "model": model,
                    "ema": None,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                if args.train.global_rank == 0:
                    writer.flush()
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                save_hf_checkpoint_best_effort(
                    save_checkpoint_path,
                    state,
                    global_step,
                    current_epoch_for_eval,
                    current_epoch_step_for_eval,
                )

            if args.train.max_steps is not None and global_step >= args.train.max_steps:
                logger.info_rank0(f"Reached max_steps={args.train.max_steps}, stopping training.")
                reached_max_steps = True
                break

        if not max_steps_driven:
            data_loader_tqdm.close()
        if args.train.global_rank == 0:
            writer.flush()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if reached_max_steps:
            already_saved = args.train.save_steps and global_step % args.train.save_steps == 0
            if not already_saved:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "ema": None,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                save_hf_checkpoint_best_effort(
                    save_checkpoint_path,
                    state,
                    global_step,
                    current_epoch_for_eval,
                    current_epoch_step_for_eval,
                )
            break
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "ema": None,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
            save_hf_checkpoint_best_effort(
                save_checkpoint_path,
                state,
                global_step,
                current_epoch_for_eval,
                current_epoch_step_for_eval,
            )

    if max_steps_driven:
        data_loader_tqdm.close()
    if args.train.global_rank == 0:
        writer.close()
    torch.cuda.synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # Ensure the last checkpoint has an HF conversion scheduled, then wait for async work.
    if save_checkpoint_path is not None:
        save_hf_checkpoint_best_effort(
            save_checkpoint_path,
            state,
            global_step,
            current_epoch_for_eval,
            current_epoch_step_for_eval,
        )
    hf_saver.wait_all_across_ranks()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
