# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import types
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state_if_fully_sharded_module
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import create_selective_checkpoint_contexts, noop_context_fn

from ..models import load_model_weights
from ..utils import logging
from ..utils.import_utils import is_torch_version_greater_than
from .checkpoint import CheckpointFunction
from .fsdp import (
    clip_grad_norm_,
    init_fsdp_fn,
    parallel_init_fsdp_fn,
    parallel_load_safetensors,
    register_checkpoint_extension,
)
from .parallel_state import get_parallel_state
from .utils import get_module_from_path, set_module_from_path


if is_torch_version_greater_than("2.4"):
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor.parallel import parallelize_module


logger = logging.get_logger(__name__)

def _resolve(root: Any, paths: List[tuple[str, ...]]) -> tuple[Optional[Any], Optional[tuple[str, ...]]]:
    for path in paths:
        current = root
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return current, path
    return None, None

def verbose_fsdp_grouping(model, prefix="", depth=0):
    indent = "    " * depth

    for name, child in model.named_children():
        if isinstance(child, FullyShardedDataParallel):
            module_names = [m_name for m_name, _ in child.named_modules()][1:]  # [1:] excludes self
            strategy = child.sharding_strategy
            logger.debug_rank0(f"{indent}├── [FSDP Group] {prefix}{name}")
            logger.debug_rank0(
                f"{indent}│   ├── Sharding Strategy: {strategy}, Mixed Precision: {child.mixed_precision}"
            )
            logger.debug_rank0(f"{indent}│   └── Contains Modules: {module_names}")

            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth + 1)
        else:
            verbose_fsdp_grouping(child, prefix=f"{prefix}{name}.", depth=depth)


def build_parallelize_model(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    sharding_plan: Optional[Dict[str, Any]] = None,
    enable_full_shard: bool = True,
    enable_mixed_precision: bool = True,
    enable_fp32: bool = False,
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    fsdp_llm_blocks: bool = True,
    ignore_norm: bool = False,
    use_depth_align: bool = False,
    split_fused_experts_from_decoder_fsdp: bool = False,
    vlm_fsdp: bool = False,
    use_compile: bool = False,
    enable_fine_grained_fsdp: bool = False,
    **kwargs,
) -> "nn.Module":
    """
    Applies parallel strategies to the model.
    """
    parallel_state = get_parallel_state()
    fsdp_no_shard_states = None

    if enable_fine_grained_fsdp and (
        not parallel_state.fsdp_enabled or parallel_state.dp_mode != "fsdp2" or not vlm_fsdp
    ):
        raise ValueError("enable_fine_grained_fsdp requires FSDP2 training and vlm_fsdp=True.")

    if not parallel_state.fsdp_enabled:
        if kwargs.get("init_device") != "cuda":
            raise ValueError("Only FSDP training supports `init_device=cpu` or `init_device=meta`.")
        if kwargs.pop("enable_fsdp_offload", False):
            raise ValueError("Only FSDP training supports `enable_fsdp_offload`.")

    if enable_mixed_precision:  # upcast to float32 before feed it to optimizer
        model = model.float()

    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info_rank0("Enable gradient checkpointing.")
        use_reentrant = kwargs.pop("enable_reentrant", False)
        if use_reentrant:
            torch.utils.checkpoint.CheckpointFunction = CheckpointFunction

        ops_to_save = kwargs.pop("ops_to_save", None)
        gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant}
        if ops_to_save is not None:
            context_fn = partial(create_selective_checkpoint_contexts, ops_to_save)
            gradient_checkpointing_kwargs["context_fn"] = context_fn
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    if parallel_state.tp_enabled:
        logger.info_rank0("Apply tensor parallel to the model.")
        model = parallelize_module(
            model,
            device_mesh=parallel_state.tp_mesh,
        )

    if parallel_state.ep_enabled:
        parallel_plan = model.get_parallel_plan()
        ep_param_suffix = parallel_plan.ep_param_suffix

        fqn2spec_info = parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)
        fsdp_no_shard_states_fqn_to_module = parallel_plan.get_fsdp_no_shard_info(model)

        fsdp_no_shard_states = list(fsdp_no_shard_states_fqn_to_module.values())
        fsdp_no_shard_states_fqn = list(fsdp_no_shard_states_fqn_to_module.keys())
        logger.info_rank0(f"Apply expert parallel to the model successfully.\nEP modules: {fsdp_no_shard_states_fqn}.")
    else:
        fqn2spec_info = None
        ep_param_suffix = None
        fsdp_no_shard_states = None
        fsdp_no_shard_states_fqn = None

    if parallel_state.fsdp_enabled:
        logger.info_rank0(f"Apply data parallel to the model: {parallel_state.dp_mode}.")
        if parallel_state.dp_mode == "fsdp2":
            def merge_ignored_params(fsdp_options: Dict[str, Any], params: set) -> None:
                if not params:
                    return
                ignored_params = set(fsdp_options.get("ignored_params", set()))
                ignored_params.update(params)
                fsdp_options["ignored_params"] = ignored_params

            def module_local_fsdp_kwargs(module: "nn.Module", fsdp_options: Dict[str, Any]) -> Dict[str, Any]:
                ignored_params = fsdp_options.get("ignored_params", None)
                if not ignored_params:
                    return fsdp_options

                local_params = set(module.parameters())
                local_ignored_params = ignored_params.intersection(local_params)
                module_kwargs = dict(fsdp_options)
                if local_ignored_params:
                    module_kwargs["ignored_params"] = local_ignored_params
                else:
                    module_kwargs.pop("ignored_params", None)
                return module_kwargs

            def collect_fused_expert_modules(root_module: "nn.Module") -> list:
                if not split_fused_experts_from_decoder_fsdp:
                    return []
                try:
                    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2FusedExperts
                except Exception as exc:
                    logger.warning_rank0(f"Failed to import Qwen2FusedExperts for FSDP2 split: {exc}")
                    return []

                fused_expert_modules = []
                for submodule in root_module.modules():
                    if isinstance(submodule, Qwen2FusedExperts):
                        fused_expert_modules.append(submodule)

                if fused_expert_modules:
                    param_numel = sum(
                        param.numel()
                        for fused_expert in fused_expert_modules
                        for param in fused_expert.parameters()
                    )
                    logger.info_rank0(
                        "Split Qwen2FusedExperts from decoder FSDP2 units and wrap them as separate FSDP2 units: "
                        f"{len(fused_expert_modules)} modules, {param_numel:,} parameters."
                    )
                else:
                    logger.info_rank0(
                        "split_fused_experts_from_decoder_fsdp=True, but no Qwen2FusedExperts modules were found."
                    )
                return fused_expert_modules

            fsdp_kwargs = {
                "mesh": parallel_state.fsdp_mesh,
                "reshard_after_forward": enable_full_shard,
                **kwargs.pop("fsdp_kwargs", {}),
            }
            if enable_mixed_precision and not enable_fp32:
                logger.info_rank0("Enable mixed precision training.")
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                )
                fsdp_kwargs["mp_policy"] = mp_policy
            elif enable_fp32:
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.float32,
                )
                fsdp_kwargs["mp_policy"] = mp_policy
            if ignore_norm:
                ignored_modules = set()
                for layer in model.model.qwenvl_with_expert.qwenvl.language_model.model.layers:
                    ignored_modules.add(layer.input_layernorm.weight)
                    ignored_modules.add(layer.post_attention_layernorm.weight)
                for expert_layers in model.model.qwenvl_with_expert.qwen_expert.model.layers:
                    ignored_modules.add(expert_layers.input_layernorm.weight)
                    ignored_modules.add(expert_layers.post_attention_layernorm.weight)
                merge_ignored_params(fsdp_kwargs, ignored_modules)

            mp_fsdp_kwargs = {
                "mesh": parallel_state.fsdp_mesh,
                "reshard_after_forward": enable_full_shard,
                **kwargs.pop("fsdp_kwargs", {}),
            }

            fused_expert_modules_to_shard = collect_fused_expert_modules(model)
            fused_expert_modules_to_shard_set = set(fused_expert_modules_to_shard)

            mp_fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                )
            ignore_modules_in_mixed_precision = tuple()
            if hasattr(model, "get_ignore_modules_in_mixed_precision"):
                ignore_modules_in_mixed_precision = model.get_ignore_modules_in_mixed_precision()

            # EP: build ep_fsdp_kwargs for expert modules
            ep_fsdp_kwargs = None
            if parallel_state.ep_enabled:
                ep_fsdp_mesh = parallel_state.ep_fsdp_device_mesh["ep_fsdp"]
                ep_fsdp_kwargs = {
                    "mesh": ep_fsdp_mesh,
                    "reshard_after_forward": enable_full_shard,
                }
                if "mp_policy" in fsdp_kwargs:
                    ep_fsdp_kwargs["mp_policy"] = fsdp_kwargs["mp_policy"]

            def apply_fsdp_to_decoder_blocks(module: "nn.Module") -> None:
                if module.__class__.__name__ in basic_modules or module.__class__ in ignore_modules_in_mixed_precision:
                    if split_fused_experts_from_decoder_fsdp:
                        for submodule in module.modules():
                            if submodule in fused_expert_modules_to_shard_set:
                                logger.debug("Apply separate FSDP2 to Qwen2FusedExperts.")
                                fully_shard(submodule, **module_local_fsdp_kwargs(submodule, fsdp_kwargs))

                    # EP: fully_shard expert module before the decoder block
                    if (
                        ep_fsdp_kwargs is not None
                        and not split_fused_experts_from_decoder_fsdp
                        and hasattr(module, 'mlp')
                        and hasattr(module.mlp, 'experts')
                    ):
                        from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2FusedExperts
                        expert_mod = module.mlp.experts
                        if isinstance(expert_mod, Qwen2FusedExperts):
                            fully_shard(expert_mod, **ep_fsdp_kwargs)
                            expert_mod.set_gradient_divide_factor(parallel_state.world_size)

                    logger.debug(f"Apply FSDP2 to {module.__class__.__name__}.")
                    module_fsdp_kwargs = module_local_fsdp_kwargs(module, fsdp_kwargs)
                    if module.__class__ in ignore_modules_in_mixed_precision:
                        fully_shard(module, **{k: v for k, v in module_fsdp_kwargs.items() if k != "mp_policy"})
                    else:
                        fully_shard(module, **module_fsdp_kwargs)

            if basic_modules:
                model.apply(apply_fsdp_to_decoder_blocks)

                # EP: attach fqn2spec_info and build _ep_param_set for gradient clipping
                if parallel_state.ep_enabled and fqn2spec_info is not None:
                    from torch.distributed._tensor import Shard as DTensorShard
                    model._fqn2spec_info = fqn2spec_info
                    ep_fqns = {fqn for fqn, spec in fqn2spec_info.items()
                               if isinstance(spec.placement, DTensorShard)}
                    ep_param_set = set()
                    for fqn, param in model.named_parameters():
                        if fqn in ep_fqns:
                            ep_param_set.add(param)
                    model._ep_param_set = ep_param_set
                    logger.info_rank0(f"EP param set: {len(ep_param_set)} parameters identified.")

                    # EP: set up forward/backward prefetch for expert layers
                    try:
                        expert_layers = list(model.model.qwenvl_with_expert.qwen_expert.model.layers)
                        for i in range(len(expert_layers) - 1):
                            cur = expert_layers[i]
                            nxt = expert_layers[i + 1]
                            if hasattr(cur.mlp, 'experts') and hasattr(nxt.mlp, 'experts'):
                                from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2FusedExperts
                                if isinstance(cur.mlp.experts, Qwen2FusedExperts):
                                    cur.set_modules_to_forward_prefetch([nxt.mlp.experts, nxt])
                        for i in range(len(expert_layers) - 1, 0, -1):
                            cur = expert_layers[i]
                            prev = expert_layers[i - 1]
                            if hasattr(cur.mlp, 'experts') and hasattr(prev.mlp, 'experts'):
                                from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2FusedExperts
                                if isinstance(cur.mlp.experts, Qwen2FusedExperts):
                                    cur.set_modules_to_backward_prefetch([prev.mlp.experts, prev])
                        logger.info_rank0("EP prefetch configured for expert layers.")
                    except Exception as e:
                        logger.warning_rank0(f"Failed to set up EP prefetch: {e}")

            elif fsdp_llm_blocks:
                layers = model.model.qwenvl_with_expert.qwenvl.language_model.model.layers
                expert_layers = model.model.qwenvl_with_expert.qwen_expert.model.layers
                if not hasattr(layers, '__iter__') or not hasattr(expert_layers, '__iter__'):
                    raise TypeError("Expected 'layers' to be a module list or container.")
                logger.info_rank0(f"Applying FSDP to {len(layers)} transformer layers in qwenvl and Gemma decoder.")
                for i, layer in enumerate(layers):
                    logger.debug(f"Sharding layer {i} ({layer.__class__.__name__})")
                    fully_shard(layer, **module_local_fsdp_kwargs(layer, fsdp_kwargs))
                for i, layer in enumerate(expert_layers):
                    logger.debug(f"Sharding layer {i} ({layer.__class__.__name__})")
                    fully_shard(layer, **module_local_fsdp_kwargs(layer, fsdp_kwargs))
            
            if vlm_fsdp:
                llm_layers, llm_path = _resolve(model.model.qwenvl_with_expert.qwenvl, [
                    ("model", "language_model", "layers"),    # Qwen3-VL: language_model on model
                    ("model", "layers"),                       # Qwen2.5-VL: layers directly on model
                    ("language_model", "model", "layers"),     # legacy fallback
                ])

                if llm_layers is None or not hasattr(llm_layers, "__iter__"):
                    raise RuntimeError(
                        "Could not locate Qwen-VL decoder layers under any known path "
                        "(model.language_model.layers / model.layers / language_model.model.layers). "
                        "sharding would silently fall back and produce a 5+ GB root AllGather."
                    )

                FSDP_LAYER_CLASS_NAMES = {
                    "Qwen2_5_VLDecoderLayer",
                    "Qwen2_5_VLVisionBlock",
                    "Qwen3VLTextDecoderLayer",
                    "Qwen3VLVisionBlock",
                }
                logger.info_rank0(f"FSDP2 sharding LLM layers via {llm_path}, n_layers={len(llm_layers)}")
                for layer in llm_layers:
                    if layer.__class__.__name__ in FSDP_LAYER_CLASS_NAMES:
                        logger.info_rank0(f"Apply FSDP2 to {layer.__class__.__name__}.")
                        fully_shard(layer, **mp_fsdp_kwargs)

                logger.info_rank0(
                    "Fine-grained VLM FSDP2 sharding "
                    f"{'enabled' if enable_fine_grained_fsdp else 'disabled'}: "
                    "vision stages, token embeddings, alignment heads and shared task projections."
                )
                if enable_fine_grained_fsdp:
                    # Decoder-only traversal leaves the entire vision transformer and
                    # token table in the root all-gather. Apply FSDP bottom-up to the
                    # vision blocks, their remaining stem/mergers, and token table too.
                    qwenvl = model.model.qwenvl_with_expert.qwenvl
                    vision, vision_path = _resolve(qwenvl, [("model", "visual"), ("visual",)])
                    embedding, embedding_path = _resolve(qwenvl, [
                        ("model", "language_model", "embed_tokens"),
                        ("model", "embed_tokens"),
                        ("language_model", "model", "embed_tokens"),
                    ])
                    if vision is None or not isinstance(getattr(vision, "blocks", None), nn.ModuleList):
                        raise RuntimeError("Could not locate the Qwen-VL vision transformer blocks for FSDP2.")
                    if not isinstance(embedding, nn.Embedding):
                        raise RuntimeError("Could not locate the Qwen-VL token embedding for FSDP2.")
                    logger.info_rank0(
                        f"FSDP2 sharding vision via {vision_path}, n_blocks={len(vision.blocks)}; "
                        f"token embedding via {embedding_path}"
                    )
                    # Rotary tensors are computed inside the outer model in FP32.
                    # New FSDP boundaries must not downcast these auxiliary inputs.
                    policy = mp_fsdp_kwargs["mp_policy"]
                    vision_fsdp_kwargs = dict(mp_fsdp_kwargs, mp_policy=MixedPrecisionPolicy(
                        param_dtype=policy.param_dtype,
                        reduce_dtype=policy.reduce_dtype,
                        output_dtype=policy.output_dtype,
                        cast_forward_inputs=False,
                    ))
                    for block in vision.blocks:
                        fully_shard(block, **vision_fsdp_kwargs)
                        if use_compile:
                            # Compile the math inside FSDP/checkpoint hooks so the
                            # new boundaries retain the original vision fusions.
                            block.forward = torch.compile(block.forward)
                    # Keep positional embeddings in the visual parent: the HF
                    # interpolator inspects their dtype before calling the embedding.
                    # Stems and mergers are real forward modules and can be sharded.
                    vision_stages = []
                    patch_embed = getattr(vision, "patch_embed", None)
                    mergers = list(getattr(vision, "deepstack_merger_list", []))
                    final_merger = getattr(vision, "merger", None)
                    for component in [patch_embed, *mergers, final_merger]:
                        if isinstance(component, nn.Module):
                            fully_shard(component, **vision_fsdp_kwargs)
                            if use_compile:
                                component.forward = torch.compile(component.forward)
                    if isinstance(patch_embed, nn.Module):
                        vision_stages.append(patch_embed)
                    deepstack_indexes = list(getattr(vision, "deepstack_visual_indexes", []))
                    for index, block in enumerate(vision.blocks):
                        vision_stages.append(block)
                        if index in deepstack_indexes:
                            vision_stages.append(mergers[deepstack_indexes.index(index)])
                    if isinstance(final_merger, nn.Module):
                        vision_stages.append(final_merger)
                    fully_shard(vision, **vision_fsdp_kwargs)
                    # Shared parameters must stay owned by a common ancestor. A
                    # multi-module group is unsafe when its output head is unused.
                    lm_head = getattr(qwenvl, "lm_head", None)
                    if lm_head is not None and getattr(lm_head, "weight", None) is embedding.weight:
                        logger.warning_rank0("Keeping tied input/output embeddings in the root FSDP group.")
                    else:
                        fully_shard(embedding, **mp_fsdp_kwargs)
                        vision_stages.append(embedding)
                    if mp_fsdp_kwargs["reshard_after_forward"] is False:
                        # Parameters already stay resident through forward in this
                        # mode. Queue smaller gathers early so the visual pipeline
                        # can compute while the remaining weights arrive.
                        vision.set_modules_to_forward_prefetch(vision_stages)

                    # Alignment projections prepare queries before the joint text /
                    # action backbone, while alignment heads consume its outputs.
                    # Keep these two lifetimes out of the root parameter all-gather.
                    alignment_heads = [
                        module for name in (
                            "depth_align_head", "future_depth_align_head",
                            "current_video_align_head", "future_video_align_head",
                        )
                        if isinstance(module := getattr(model.model, name, None), nn.Module)
                    ]
                    alignment_projections = [
                        module for name in ("current_shared_task_proj", "future_shared_task_proj")
                        if isinstance(module := getattr(model.model, name, None), nn.Module)
                    ]
                    # Separate units tolerate optional heads not executed in a given
                    # configuration without requiring a multi-module forward group.
                    for component in [*alignment_projections, *alignment_heads]:
                        fully_shard(component, **mp_fsdp_kwargs)
                        if use_compile:
                            component.forward = torch.compile(component.forward)
                    if mp_fsdp_kwargs["reshard_after_forward"] is False:
                        # Vision parameters were already queued by the visual parent.
                        # Launch query projections when visual computation starts.
                        if alignment_projections:
                            vision.blocks[0].set_modules_to_forward_prefetch(alignment_projections)
                        if alignment_heads:
                            # Make the immediately needed decoder parameters ready
                            # before queueing the heads. Otherwise the heads' gathers
                            # can simply move the initial wait into the decoder.
                            expert_layers = model.model.qwenvl_with_expert.qwen_expert.model.layers
                            decoder_lookahead = [expert_layers[0]]
                            if len(llm_layers) > 1:
                                decoder_lookahead.extend([llm_layers[1], expert_layers[1]])
                            llm_layers[0].set_modules_to_forward_prefetch(
                                [*decoder_lookahead, *alignment_heads]
                            )

            fully_shard(model, **mp_fsdp_kwargs)

            if kwargs.get("init_device") == "meta":
                if weights_path is None:
                    # shard init empty model with fsdp2
                    model.to_empty(device="cuda")
                    model.init_weights()
                else:
                    from torch.distributed.tensor import distribute_tensor

                    load_model_weights(model, weights_path, "cuda", dtensor_factory=distribute_tensor)

        elif parallel_state.dp_mode == "fsdp1":
            wrap_policy = partial(
                lambda_auto_wrap_policy, lambda_fn=lambda module: module.__class__.__name__ in basic_modules
            )

            # set fsdp/hsdp sharding strategy
            if parallel_state.fsdp_mesh.ndim > 1 and parallel_state.fsdp_mesh.size() > 1:
                strategy = ShardingStrategy.HYBRID_SHARD
            else:
                strategy = ShardingStrategy.FULL_SHARD

            fsdp_kwargs = {
                "auto_wrap_policy": wrap_policy,
                "ignored_states": fsdp_no_shard_states,
                "device_id": torch.cuda.current_device(),
                "sharding_strategy": strategy if enable_full_shard else ShardingStrategy.NO_SHARD,
                "use_orig_params": True,
            }

            fsdp_kwargs["device_mesh"] = parallel_state.fsdp_mesh

            fsdp_kwargs.update(kwargs.pop("fsdp_kwargs", {}))

            if enable_mixed_precision:
                logger.info_rank0("Enable mixed precision training.")
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.float32,
                )
                if hasattr(model, "get_ignore_modules_in_mixed_precision"):
                    mixed_precision._module_classes_to_ignore += model.get_ignore_modules_in_mixed_precision()

                fsdp_kwargs["mixed_precision"] = mixed_precision

            if kwargs.get("init_device") == "cpu":
                logger.info_rank0("Enable rank0-only initialization.")
                fsdp_kwargs["sync_module_states"] = True
                if parallel_state.global_rank != 0:
                    fsdp_kwargs["param_init_fn"] = init_fsdp_fn(model, device="cuda")
            elif kwargs.get("init_device") == "meta":
                # assert weights_path is not None, "`weights_path` must be provided when `init_device=meta` for fsdp1."

                logger.info_rank0("Enable meta initialization.")
                if weights_path is None:
                    logger.info_rank0("weights_path is None during meta initialization.")

                ignore_param_names = (
                    [".".join([fqn, k]) for fqn in fsdp_no_shard_states_fqn for k in ep_param_suffix]
                    if fsdp_no_shard_states_fqn is not None
                    else None
                )
                shard_states = (
                    parallel_load_safetensors(weights_path, ignore_param_name=ignore_param_names)
                    if weights_path
                    else kwargs.get("state_dict", {})
                )
                fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
                    model, shard_states, ignore_param_name=ignore_param_names
                )

            if kwargs.pop("enable_fsdp_offload", False):
                logger.info_rank0("Enable offloading for parameters & gradients & optimizer states.")
                fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

            if kwargs.pop("enable_forward_prefetch", False):
                fsdp_kwargs["forward_prefetch"] = True
            else:
                fsdp_kwargs["forward_prefetch"] = False
                fsdp_kwargs["backward_prefetch"] = None

            # FULLY_SHARD first
            model = FullyShardedDataParallel(model, **fsdp_kwargs)

            if fsdp_no_shard_states is not None:
                # apply NO_SHARD the ignored_states, but wrap into DDP
                if parallel_state.ep_fsdp_mesh["ep_fsdp"].size() == 1:
                    moe_sharding_strategy = ShardingStrategy.NO_SHARD
                    ep_fsdp_device_mesh = parallel_state.fsdp_mesh
                else:
                    moe_sharding_strategy = ShardingStrategy.FULL_SHARD
                    ep_fsdp_device_mesh = parallel_state.ep_fsdp_mesh["ep_fsdp"]

                logger.info_rank0(f"Apply {moe_sharding_strategy} states on '{fsdp_no_shard_states_fqn}'.")
                fsdp_kwargs.pop("ignored_states", None)
                fsdp_kwargs.pop("auto_wrap_policy", None)
                fsdp_kwargs["sharding_strategy"] = moe_sharding_strategy
                fsdp_kwargs["device_mesh"] = ep_fsdp_device_mesh
                logger.info_rank0(f"{ep_fsdp_device_mesh=}")
                for fqn in fsdp_no_shard_states_fqn:
                    no_shard_module = get_module_from_path(model, fqn)
                    if kwargs.get("init_device") == "meta":
                        specific_param_name = [".".join([fqn, k]) for k in ep_param_suffix]
                        shard_states = (
                            parallel_load_safetensors(weights_path, specific_param_name=specific_param_name)
                            if weights_path
                            else {}
                        )
                        if weights_path:
                            for suffix in ep_param_suffix:
                                shard_states[suffix] = shard_states.pop(".".join([fqn, suffix]))
                        fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
                            no_shard_module, shard_states, specific_param_name=ep_param_suffix
                        )
                    fsdp_module = FullyShardedDataParallel(no_shard_module, **fsdp_kwargs)
                    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(fsdp_module)
                    fsdp_state._gradient_postdivide_factor *= parallel_state.ep_size
                    set_module_from_path(model, fqn, fsdp_module)

            _lazy_init(model, model)

            # Apply fsdp extension to FSDP model
            save_hook_mesh = parallel_state.ep_fsdp_device_mesh if parallel_state.ep_enabled else None
            logger.info_rank0("Register Checkpoints Extension hook to the model")
            register_checkpoint_extension(
                fsdp_model=model,
                save_hook_mesh=save_hook_mesh,
                fqn2spec_info=fqn2spec_info,
            )

            if parallel_state.ep_enabled:
                model.clip_grad_norm_ = types.MethodType(clip_grad_norm_, model)

            verbose_fsdp_grouping(model)
        else:
            ddp_kwargs = {"device_ids": [parallel_state.local_rank]}
            if enable_mixed_precision:
                logger.info_rank0("Enable mixed precision training.")
                if enable_fp32:
                    mixed_precision = MixedPrecision(
                        param_dtype=torch.float32,
                        reduce_dtype=torch.float32,
                        buffer_dtype=torch.float32,
                    )
                else:
                    mixed_precision = MixedPrecision(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                        buffer_dtype=torch.bfloat16,
                    )
                ddp_kwargs["mixed_precision"] = mixed_precision

            model = DDP(model, **ddp_kwargs)

    return model
