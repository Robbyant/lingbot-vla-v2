import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple

from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
import transformers.models.qwen3_vl.modeling_qwen3_vl as hf_qwen3vl
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
    Qwen3VLModel as _Qwen3VLModel,
    Qwen3VLTextModel as _Qwen3VLTextModel,
    Qwen3VLPreTrainedModel as _Qwen3VLPreTrainedModel,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    Qwen3VLVisionMLP,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)

# Retain the Transformers implementation even if the LingBot patch is applied
# repeatedly. The optimization is selected per visual-model instance below.
_original_fast_pos_embed_interpolate = getattr(
    Qwen3VLVisionModel, "_lingbot_original_fast_pos_embed_interpolate",
    Qwen3VLVisionModel.fast_pos_embed_interpolate,
)


def _qwen3vl_no_init_weights(self, module):
    return

_Qwen3VLPreTrainedModel._init_weights = _qwen3vl_no_init_weights
Qwen3VLPreTrainedModel = _Qwen3VLPreTrainedModel


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        max_seqlen: Optional[int] = None,
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
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            out_fp32_atten = False
            if key_states.dtype == torch.float32:
                out_fp32_atten = True
                query_states = query_states.to(torch.bfloat16)
                key_states = key_states.to(torch.bfloat16)
                value_states = value_states.to(torch.bfloat16)
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


class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

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


class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: Optional[torch.Tensor] = None,
        start: Optional[int] = 0,
        end: Optional[int] = 0,
        compute_kqv: bool = False,
        output_atten: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        param_dtype = self.self_attn.q_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        if att_output is not None:
            att_output = att_output.to(param_dtype)

        if compute_kqv:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)
            query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
            key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
            return query_state, key_state, value_state

        if output_atten:
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])
            out_emb += hidden_states
            after_first_residual = out_emb.clone()
            out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)
            out_emb += after_first_residual
            return out_emb

        position_embeddings = kwargs.pop("position_embeddings", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if position_embeddings is not None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            return residual + hidden_states

        raise ValueError(
            f"Invalid operation compute_kqv={compute_kqv} and output_atten={output_atten} "
            "with Qwen3VLTextDecoderLayer in LingBot-VLA"
        )


class Qwen3VLTextModel(_Qwen3VLTextModel):
    def __init__(self, config: Qwen3VLTextConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3VLModel(_Qwen3VLModel):
    def __init__(self, config: Qwen3VLConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen3VLForConditionalGeneration(_Qwen3VLForConditionalGeneration, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


@torch.compiler.disable
def get_lingbot_grid_metadata(self, grid_thw):
    """Cache only weight-independent metadata, keyed by the complete visual grid.

    LingBot V2 transfers the complete input grid to CPU once. External CUDA grids
    use one explicit D2H transfer rather than scalar synchronizations in a loop.
    Cache entries are not buffers/state_dict values and never contain a learned
    position embedding or an autograd graph.
    """
    from collections import OrderedDict

    grid = tuple(tuple(int(v) for v in row) for row in grid_thw.detach().cpu().tolist())
    weight = self.pos_embed.weight
    inv_freq = self.rotary_pos_emb.inv_freq
    key = (grid, weight.device, weight.dtype, self.spatial_merge_size,
           self.num_grid_per_side, inv_freq.device, inv_freq.dtype,
           inv_freq.data_ptr(), inv_freq._version, torch.is_inference_mode_enabled())
    cache = getattr(self, "_lingbot_grid_metadata_cache", None)
    if cache is None:
        cache = self._lingbot_grid_metadata_cache = OrderedDict()
    if key in cache:
        cache.move_to_end(key)
        return cache[key]

    merge = self.spatial_merge_size
    indices, weights, coords, permutation = [], [], [], []
    lengths, split_sizes = [], []
    offset = 0
    with torch.no_grad():
        for frames, height, width in grid:
            if frames <= 0 or height <= 0 or width <= 0 or height % merge or width % merge:
                raise ValueError("Visual grid dimensions must be positive and divisible by spatial_merge_size")
            # Match the original CPU linspace and the four-corner summation order.
            hs = torch.linspace(0, self.num_grid_per_side - 1, height)
            ws = torch.linspace(0, self.num_grid_per_side - 1, width)
            hf, wf = hs.int(), ws.int()
            hc = (hf + 1).clamp(max=self.num_grid_per_side - 1)
            wc = (wf + 1).clamp(max=self.num_grid_per_side - 1)
            dh, dw = hs - hf, ws - wf
            indices.append(torch.stack([
                (hf[:, None] * self.num_grid_per_side + wf[None, :]).flatten(),
                (hf[:, None] * self.num_grid_per_side + wc[None, :]).flatten(),
                (hc[:, None] * self.num_grid_per_side + wf[None, :]).flatten(),
                (hc[:, None] * self.num_grid_per_side + wc[None, :]).flatten(),
            ]))
            weights.append(torch.stack([
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]))
            order = torch.arange(height * width).view(height // merge, merge, width // merge, merge)
            order = order.permute(0, 2, 1, 3).reshape(-1)
            coords.append(torch.stack((order // width, order % width), -1).repeat(frames, 1))
            permutation.append((order + offset).repeat(frames))
            offset += height * width
            lengths.extend([height * width] * frames)
            split_sizes.append(frames * height * width // (merge * merge))
        pos_ids = torch.cat(coords).to(inv_freq.device, dtype=torch.long)
        rotary = self.rotary_pos_emb(max(max(h, w) for _, h, w in grid))[pos_ids].flatten(1)
        emb = torch.cat((rotary, rotary), dim=-1)
        cumulative = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        metadata = dict(
            indices=torch.cat(indices, dim=1).to(weight.device, dtype=torch.long),
            weights=torch.cat(weights, dim=1).to(weight.device, dtype=weight.dtype),
            permutation=torch.cat(permutation).to(weight.device, dtype=torch.long),
            position_embeddings=(emb.cos(), emb.sin()),
            cu_seqlens=cumulative.to(weight.device),
            split_sizes=tuple(split_sizes), max_seqlen=max(lengths),
        )
    cache[key] = metadata
    # Bound dynamic-resolution workloads; a different device/dtype cannot reuse a stale entry.
    if len(cache) > 4:
        cache.popitem(last=False)
    return metadata


def fast_pos_embed_interpolate_cached(self, grid_thw, metadata=None):
    if not getattr(self.config, "enable_vision_position_cache", False):
        return _original_fast_pos_embed_interpolate(self, grid_thw)
    if metadata is None:
        metadata = self.get_lingbot_grid_metadata(grid_thw)
    # Re-evaluate the learned table on every call so optimizer updates and gradients
    # remain live. Preserve the original BF16 multiply/add rounding order.
    corners = self.pos_embed(metadata["indices"]) * metadata["weights"][:, :, None]
    positions = corners[0] + corners[1] + corners[2] + corners[3]
    return positions.index_select(0, metadata["permutation"])


@torch.compiler.disable
def preprcess_grid_thw(self, grid_thw: torch.Tensor):
    if getattr(self.config, "enable_vision_position_cache", False):
        metadata = self.get_lingbot_grid_metadata(grid_thw)
        return (None, metadata["position_embeddings"], metadata["cu_seqlens"],
                metadata["split_sizes"], metadata["max_seqlen"])

    # Preserve the pre-optimization preprocessing path and its legacy caller.
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    seq_len = int(torch.prod(grid_thw, dim=1).sum().item())
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return None, position_embeddings, cu_seqlens, split_sizes, max_seqlen


def forward_without_grid_thw(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor = None,
    pos_embeds: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    metadata = kwargs.pop("grid_metadata", None)
    use_position_cache = getattr(self.config, "enable_vision_position_cache", False)
    needs_position_data = (
        pos_embeds is None or position_embeddings is None or cu_seqlens is None or max_seqlen is None
    )
    # A complete precomputed input must also work without grid_thw, as before.
    # Partial inputs retain the original all-or-recompute behavior.
    if needs_position_data and grid_thw is None and (not use_position_cache or metadata is None):
        raise ValueError("grid_thw is required unless all position inputs are provided.")
    if use_position_cache and needs_position_data and metadata is None:
        metadata = self.get_lingbot_grid_metadata(grid_thw)
    hidden_states = self.patch_embed(hidden_states)

    if needs_position_data:
        if use_position_cache:
            position_embeddings = metadata["position_embeddings"]
            cu_seqlens, max_seqlen = metadata["cu_seqlens"], metadata["max_seqlen"]
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw, metadata)
        else:
            pos_embeds, position_embeddings, cu_seqlens, _, max_seqlen = self.preprcess_grid_thw(grid_thw)
    if pos_embeds is None:
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

    hidden_states = hidden_states + pos_embeds
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)
    return hidden_states, deepstack_feature_lists


def apply_lingbot_qwen3_vl_patch():
    logger.info_rank0("apply Qwen3-VL Lingbot patch")
    hf_qwen3vl.Qwen3VLPreTrainedModel = Qwen3VLPreTrainedModel
    hf_qwen3vl.Qwen3VLTextDecoderLayer = Qwen3VLTextDecoderLayer
    hf_qwen3vl.Qwen3VLTextModel = Qwen3VLTextModel
    hf_qwen3vl.Qwen3VLModel = Qwen3VLModel
    hf_qwen3vl.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    hf_qwen3vl.Qwen3VLVisionAttention = Qwen3VLVisionAttention
    hf_qwen3vl.Qwen3VLVisionBlock = Qwen3VLVisionBlock
    hf_qwen3vl.Qwen3VLVisionModel.forward = forward_without_grid_thw
    hf_qwen3vl.Qwen3VLVisionModel.preprcess_grid_thw = preprcess_grid_thw
    hf_qwen3vl.Qwen3VLVisionModel.get_lingbot_grid_metadata = get_lingbot_grid_metadata
    hf_qwen3vl.Qwen3VLVisionModel._lingbot_original_fast_pos_embed_interpolate = _original_fast_pos_embed_interpolate
    hf_qwen3vl.Qwen3VLVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate_cached
