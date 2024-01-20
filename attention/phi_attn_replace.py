# Modified based on https://github.com/lm-sys/FastChat

import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import transformers
from einops import rearrange
from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func
)
from transformers.models.phi.modeling_phi import apply_rotary_pos_emb, rotate_half
from flash_attn.bert_padding import unpad_input, pad_input
import math

group_size_ratio = 1/4
def forward_flashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if not self.training:
        raise ValueError("This function is only for training. For inference, please use forward_flashattn_inference.")

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    if self.qk_layernorm:
        query_states = self.q_layernorm(query_states)
        key_states = self.k_layernorm(key_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # query_states = (
    #     self.q_proj(hidden_states)
    #     .view(bsz, q_len, self.num_heads, self.head_dim)
    #     .transpose(1, 2)
    # )
    # key_states = (
    #     self.k_proj(hidden_states)
    #     .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    #     .transpose(1, 2)
    # )
    # value_states = (
    #     self.v_proj(hidden_states)
    #     .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    #     .transpose(1, 2)
    # )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]


    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    
    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : self.rotary_emb.dim],
        query_states[..., self.rotary_emb.dim :],
    )
    key_rot, key_pass = (
        key_states[..., : self.rotary_emb.dim],
        key_states[..., self.rotary_emb.dim :],
    )
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

    #kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value[0].shape[-2]
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # query_states, key_states = apply_rotary_pos_emb(
    #     query_states, key_states, cos, sin, position_ids
    # )

    # # Past Key value support
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    # if past_key_value is not None:
    #     # reuse k, v, self_attention
    #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # past_key_value = (key_states, value_states) if use_cache else None

    if query_states.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

    # repeat k/v heads if n_kv_heads < n_heads
    #key_states = repeat_kv(key_states, self.num_key_value_groups)
    #value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    key_padding_mask = attention_mask.repeat(2, 1)
    nheads = qkv.shape[-2]
    # shift

    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d." % (q_len, group_size))

    qkv = qkv.reshape(bsz, q_len, 3, 2, self.num_heads // 2, self.head_dim).permute(0, 3, 1, 2, 4, 5).reshape(bsz * 2,
                                                                                                              q_len, 3,
                                                                                                              self.num_heads // 2,
                                                                                                              self.head_dim)
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    cu_q_len_tmp = torch.arange(0, max_s, group_size, device=key_padding_mask.device, dtype=cu_q_lens.dtype)
    cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp + group_size // 2]).repeat(bsz, 1) + cu_q_lens[:-1].unsqueeze(-1)
    cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)], dim=-1).view(-1)

    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads // 2
    )
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz * 2, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads // 2,
    )
    output = output.reshape(bsz, 2, q_len, nheads // 2, self.head_dim).transpose(1, 2).reshape(bsz, q_len, nheads,
                                                                                               self.head_dim)

    return self.dense(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value

def forward_flashattn_full(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    key_padding_mask = attention_mask
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
    )
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


def forward_noflashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * group_size_ratio)

    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size

    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    if self.qk_layernorm:
        query_states = self.q_layernorm(query_states)
        key_states = self.k_layernorm(key_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    
    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : self.rotary_emb.dim],
        query_states[..., self.rotary_emb.dim :],
    )
    key_rot, key_pass = (
        key_states[..., : self.rotary_emb.dim],
        key_states[..., self.rotary_emb.dim :],
    )
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

    
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # shift
    def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
        qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
        qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
        return qkv

    query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
            f" {attn_weights.size()}"
        )

    attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
    if attention_mask is not None:
        if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    # shift back
    attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.dense(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask

def apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k


def forward_flashattn_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        assert (
            flash_attn_version >= "2.1.0"
        ), "past_key_value support requires flash-attn >= 2.1.0"
        # reuse k, v
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

    if attention_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value

def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask

def replace_llama_attn(use_flash_attn=True, use_full=False, inference=False):
    if use_flash_attn:
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
        if inference:
            transformers.models.phi.modeling_phi.PhiModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
            #transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_inference
        else:
            transformers.models.phi.modeling_phi.PhiModel._prepare_decoder_attention_mask = (
                _prepare_decoder_attention_mask
            )
            transformers.models.phi.modeling_phi.PhiFlashAttention2.forward = forward_flashattn_full if use_full else forward_flashattn
    else:
        #raise NotImplementedError
        transformers.models.phi.modeling_phi.PhiAttention.forward = forward_noflashattn
