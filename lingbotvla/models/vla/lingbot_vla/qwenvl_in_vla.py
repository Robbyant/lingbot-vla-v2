import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import Tensor, nn
from typing import List, Optional, Tuple, Union, Callable, Dict, Any
import math
from transformers import (
    PreTrainedModel,
)
from dataclasses import dataclass
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from transformers.cache_utils import Cache, SlidingWindowCache, StaticCache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, flash_attn_supports_top_left_mask, is_flash_attn_available
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.processing_utils import Unpack
import torch.distributed._tensor as dt

import importlib.util as _ilu

# transformers 5.x is_flash_attn_available() may report True without the
# flash_attn package installed; require the actual package.
if is_flash_attn_available() and _ilu.find_spec("flash_attn") is not None:
    from flash_attn.layers.rotary import apply_rotary_emb
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as hf_qwen25vl 
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
except ImportError:  # renamed in transformers 5.x
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRMSNorm as Qwen2RMSNorm
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLMLP,
    Qwen2_5_VLAttention,
    Qwen2MLP,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLRotaryEmbedding,
    apply_rotary_pos_emb_vision,
    eager_attention_forward
)

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLTextModel as _Qwen2_5_VLTextModel,
    Qwen2_5_VLForConditionalGeneration as _Qwen2_5_VLForConditionalGeneration
)
logger = logging.get_logger(__name__)


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False
        # print(f"ViT Attention Type is {self.config._attn_implementation}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            out_fp32_atten = False
            if key_states.dtype == torch.float32:
                out_fp32_atten = True
                query_states, key_states, value_states = query_states.to(torch.bfloat16), key_states.to(torch.bfloat16), value_states.to(torch.bfloat16)
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
            if out_fp32_atten:
                attn_output = attn_output.to(torch.float32)
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "flash_attention_2") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2_5_VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False  # TODO (joao): fix. torch.compile failing probably due to `cache_positions`

    # def _init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, (nn.Linear, nn.Conv3d)):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()


class Qwen2_5_VLDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen2_5_VLAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.norm_qkv:
            self.q_layernorm = Qwen2RMSNorm(self.self_attn.head_dim, eps=config.rms_norm_eps)
            self.k_layernorm = Qwen2RMSNorm(self.self_attn.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: Optional[torch.Tensor] = None,
        start: Optional[int] = 0,
        end: Optional[int] = 0,
        compute_kqv: bool = False,
        norm_qkv: bool = False,
        output_atten: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        if compute_kqv:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

            query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)

            if norm_qkv:
                query_state = self.q_layernorm(query_state)
                key_state = self.k_layernorm(key_state)

            return query_state, key_state, value_state
        
        elif output_atten:
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])

            # first residual
            out_emb += hidden_states
            after_first_residual = out_emb.clone()

            out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)

            # second residual
            out_emb += after_first_residual

            return out_emb

        else:
            raise ValueError(f"Invaild Operation compute_kqv={compute_kqv} and output_atten={output_atten} with Qwen2_5_VLDecoderLayer in LingBot-VLA")


class Qwen2_5_VLTextModel(Qwen2_5_VLPreTrainedModel):
    get_input_embeddings = _Qwen2_5_VLTextModel.get_input_embeddings
    set_input_embeddings = _Qwen2_5_VLTextModel.set_input_embeddings
    forward = _Qwen2_5_VLTextModel.forward
        
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self._init_weights = lambda module: None
        self.post_init()


class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    get_input_embeddings = _Qwen2_5_VLForConditionalGeneration.get_input_embeddings
    set_input_embeddings = _Qwen2_5_VLForConditionalGeneration.set_input_embeddings
    get_output_embeddings = _Qwen2_5_VLForConditionalGeneration.get_output_embeddings
    set_output_embeddings = _Qwen2_5_VLForConditionalGeneration.set_output_embeddings
    get_decoder = _Qwen2_5_VLForConditionalGeneration.get_decoder
    set_decoder = _Qwen2_5_VLForConditionalGeneration.set_decoder
    forward = _Qwen2_5_VLForConditionalGeneration.forward
    prepare_inputs_for_generation = _Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLTextModel._from_config(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()



def preprcess_grid_thw(self, grid_thw: torch.Tensor):
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=grid_thw.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    return rotary_pos_emb, window_index, cu_window_seqlens, cu_seqlens


def forward_without_grid_thw(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor = None,
    rotary_pos_emb = None,
    window_index = None,
    cu_window_seqlens = None,
    cu_seqlens = None,
    **kwargs
) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)

    if rotary_pos_emb is None or window_index is None or cu_window_seqlens is None or cu_seqlens is None:
        rotary_pos_emb, window_index, cu_window_seqlens, cu_seqlens = self.preprcess_grid_thw(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states


def apply_lingbot_qwen25_vl_patch():
    logger.info_rank0("apply patch")
    hf_qwen25vl.Qwen2_5_VLPreTrainedModel = Qwen2_5_VLPreTrainedModel
    hf_qwen25vl.Qwen2_5_VLDecoderLayer = Qwen2_5_VLDecoderLayer
    hf_qwen25vl.Qwen2_5_VLTextModel = Qwen2_5_VLTextModel
    hf_qwen25vl.Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration
    hf_qwen25vl.Qwen2_5_VLVisionAttention = Qwen2_5_VLVisionAttention
    hf_qwen25vl.Qwen2_5_VLVisionBlock = Qwen2_5_VLVisionBlock
    hf_qwen25vl.Qwen2_5_VisionTransformerPretrainedModel.forward = forward_without_grid_thw
    hf_qwen25vl.Qwen2_5_VisionTransformerPretrainedModel.preprcess_grid_thw = preprcess_grid_thw