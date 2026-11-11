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


from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)
from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from .loader import BaseModelLoader, get_loader

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.get_logger(__name__)


def build_tokenizer(tokenizer_path: str) -> "PreTrainedTokenizer":
    """
    Builds the tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right", trust_remote_code=True)


def build_processor(processor_path: str) -> "ProcessorMixin":
    """
    Builds the processor.
    """
    return AutoProcessor.from_pretrained(processor_path, padding_side="right", trust_remote_code=True)


def build_foundation_model(
    config_path: str | None,
    config_cls:  Optional[str] = None,
    weights_path: Optional[str] = None,
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2", "flex"]] = "flash_attention_2",
    moe_implementation: Optional[Literal["eager", "fused"]] = None,
    init_device: Literal["cpu", "cuda", "meta"] = "cuda",
    config_kwargs: Optional[Dict[str, Any]] = None,
    force_use_huggingface: bool = False,
) -> "PreTrainedModel":
    """
    Builds the foundation model.

    If weights_path is provided, it loads the pre-trained weights, otherwise it initializes weights.
    """

    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    if config_kwargs is None:
        config_kwargs = {}
    vlm_repo_id = config_kwargs['vlm_repo_id'] if 'vlm_repo_id' in config_kwargs else None
    tokenizer_path_for_patch = config_kwargs['tokenizer_path'] if 'tokenizer_path' in config_kwargs else None
    vlm_key = f"{vlm_repo_id or ''} {tokenizer_path_for_patch or ''}".lower()
    if "qwen3" in vlm_key and "vl" in vlm_key:
        from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
        apply_lingbot_qwen3_vl_patch(compile_rmsnorm=bool(config_kwargs.get("enable_compiled_rmsnorm", False)))
    else:
        from lingbotvla.models.vla.lingbot_vla.qwenvl_in_vla import apply_lingbot_qwen25_vl_patch
        apply_lingbot_qwen25_vl_patch()
    apply_lingbot_qwen2_patch()
    post_training = config_kwargs['post_training'] if 'post_training' in config_kwargs else False
    adanorm_time = config_kwargs['adanorm_time'] if 'adanorm_time' in config_kwargs else False
    incremental_training = config_kwargs['incremental_training'] if 'incremental_training' in config_kwargs else False
    if config_cls is not None:
        config = config_cls
    else:
        raise ValueError(f'Invalid Model Config based on {config_kwargs}!!!')

    if moe_implementation is not None:
        if moe_implementation not in ["eager", "fused"]:
            raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
        config._moe_implementation = moe_implementation
        logger.info_rank0(f"Moe implementation: {moe_implementation}")

    loader: Optional[BaseModelLoader] = get_loader(config, force_use_huggingface)
    if 'pi0' in config_path:
        init_kwargs = {
            "config": config,
            "torch_dtype": getattr(torch, torch_dtype),
            "attn_implementation": attn_implementation,
            "ckpt_path": weights_path,
            "trust_remote_code": True,
        }
    else:
        init_kwargs = {
            "config": config,
            "torch_dtype": getattr(torch, torch_dtype),
            "attn_implementation": attn_implementation,
            "trust_remote_code": True,
        }

    if (init_device == "cpu" and get_parallel_state().global_rank != 0) or init_device == "meta":
        empty_init = True
    else:
        empty_init = False
    weights_path = vlm_repo_id if vlm_repo_id else weights_path
    model = loader.load_model(
        init_kwargs=init_kwargs,
        weights_path=weights_path,
        empty_init=empty_init,
        init_device=init_device,
        vlm_repo_id=vlm_repo_id,
        post_training=post_training,
        adanorm_time=adanorm_time,
        incremental_training=incremental_training,
    )
    return model
