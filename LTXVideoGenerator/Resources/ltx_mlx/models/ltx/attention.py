"""Attention module for LTX-2."""
import math
from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from ltx_mlx.models.ltx.config import LTXRopeType
from ltx_mlx.models.ltx.rope import apply_rotary_emb


def scaled_dot_product_attention(q: mx.array, k: mx.array, v: mx.array, heads: int, mask: Optional[mx.array] = None) -> mx.array:
    b, q_seq_len, dim = q.shape
    _, kv_seq_len, _ = k.shape
    dim_head = dim // heads
    q = mx.reshape(q, (b, q_seq_len, heads, dim_head))
    k = mx.reshape(k, (b, kv_seq_len, heads, dim_head))
    v = mx.reshape(v, (b, kv_seq_len, heads, dim_head))
    q = mx.swapaxes(q, 1, 2)
    k = mx.swapaxes(k, 1, 2)
    v = mx.swapaxes(v, 1, 2)
    if mask is not None:
        if mask.ndim == 2:
            mask = mx.expand_dims(mask, axis=0)
        if mask.ndim == 3:
            mask = mx.expand_dims(mask, axis=1)
    scale = 1.0 / math.sqrt(dim_head)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    out = mx.swapaxes(out, 1, 2)
    out = mx.reshape(out, (b, q_seq_len, heads * dim_head))
    return out


class Attention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64, norm_eps: float = 1e-6, rope_type: LTXRopeType = LTXRopeType.INTERLEAVED):
        super().__init__()
        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=True)
        self.q_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(inner_dim, eps=norm_eps)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)

    def __call__(self, x: mx.array, context: Optional[mx.array] = None, mask: Optional[mx.array] = None, pe: Optional[Tuple[mx.array, mx.array]] = None, k_pe: Optional[Tuple[mx.array, mx.array]] = None) -> mx.array:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k_pe_to_use = pe if k_pe is None else k_pe
            k = apply_rotary_emb(k, k_pe_to_use, self.rope_type)
        out = scaled_dot_product_attention(q, k, v, self.heads, mask)
        return self.to_out(out)
