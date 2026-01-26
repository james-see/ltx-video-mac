import math
from typing import List, Optional, Tuple
import mlx.core as mx
import numpy as np
from ltx_mlx.models.ltx.config import LTXRopeType


def apply_rotary_emb(input_tensor: mx.array, freqs_cis: Tuple[mx.array, mx.array], rope_type: LTXRopeType = LTXRopeType.INTERLEAVED) -> mx.array:
    if rope_type == LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    elif rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, freqs_cis[0], freqs_cis[1])
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def apply_interleaved_rotary_emb(input_tensor: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.astype(mx.float32)
    cos_freqs = cos_freqs.astype(mx.float32)
    sin_freqs = sin_freqs.astype(mx.float32)
    shape = input_tensor.shape
    input_tensor = mx.reshape(input_tensor, shape[:-1] + (shape[-1] // 2, 2))
    t1 = input_tensor[..., 0]
    t2 = input_tensor[..., 1]
    t_rot = mx.stack([-t2, t1], axis=-1)
    input_tensor = mx.reshape(input_tensor, shape)
    t_rot = mx.reshape(t_rot, shape)
    out = input_tensor * cos_freqs + t_rot * sin_freqs
    return out.astype(input_dtype)


def apply_split_rotary_emb(input_tensor: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    input_dtype = input_tensor.dtype
    needs_reshape = False
    original_shape = input_tensor.shape
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = mx.reshape(input_tensor, (b, t, h, -1))
        input_tensor = mx.swapaxes(input_tensor, 1, 2)
        needs_reshape = True
    input_tensor = input_tensor.astype(mx.float32)
    cos_freqs = cos_freqs.astype(mx.float32)
    sin_freqs = sin_freqs.astype(mx.float32)
    dim = input_tensor.shape[-1]
    split_input = mx.reshape(input_tensor, input_tensor.shape[:-1] + (2, dim // 2))
    first_half = split_input[..., 0, :]
    second_half = split_input[..., 1, :]
    output_first = first_half * cos_freqs - sin_freqs * second_half
    output_second = second_half * cos_freqs + sin_freqs * first_half
    output = mx.stack([output_first, output_second], axis=-2)
    output = mx.reshape(output, input_tensor.shape)
    if needs_reshape:
        b, h, t, d = output.shape
        output = mx.swapaxes(output, 1, 2)
        output = mx.reshape(output, (b, t, h * d))
    return output.astype(input_dtype)


def generate_freq_grid(positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int) -> mx.array:
    theta = positional_embedding_theta
    n_elem = 2 * positional_embedding_max_pos_count
    log_start = math.log(1.0) / math.log(theta)
    log_end = math.log(theta) / math.log(theta)
    num_indices = inner_dim // n_elem
    if num_indices == 0:
        num_indices = 1
    lin_space = mx.linspace(log_start, log_end, num_indices)
    pow_indices = mx.power(theta, lin_space)
    return pow_indices * (math.pi / 2)


def get_fractional_positions(indices_grid: mx.array, max_pos: List[int]) -> mx.array:
    n_pos_dims = indices_grid.shape[1]
    fractional_positions = []
    for i in range(n_pos_dims):
        frac = indices_grid[:, i] / max_pos[i]
        fractional_positions.append(frac)
    return mx.stack(fractional_positions, axis=-1)


def generate_freqs(indices: mx.array, indices_grid: mx.array, max_pos: List[int], use_middle_indices_grid: bool) -> mx.array:
    if use_middle_indices_grid:
        indices_grid_start = indices_grid[..., 0]
        indices_grid_end = indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]
    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    scaled_positions = fractional_positions * 2 - 1
    freqs = mx.expand_dims(scaled_positions, axis=-1) * mx.expand_dims(mx.expand_dims(mx.expand_dims(indices, axis=0), axis=0), axis=0)
    freqs = mx.swapaxes(freqs, -1, -2)
    freqs = mx.reshape(freqs, freqs.shape[:-2] + (-1,))
    return freqs


def split_freqs_cis(freqs: mx.array, pad_size: int, num_attention_heads: int) -> Tuple[mx.array, mx.array]:
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)
    if pad_size != 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)
    b, t = cos_freq.shape[0], cos_freq.shape[1]
    cos_freq = mx.reshape(cos_freq, (b, t, num_attention_heads, -1))
    sin_freq = mx.reshape(sin_freq, (b, t, num_attention_heads, -1))
    cos_freq = mx.swapaxes(cos_freq, 1, 2)
    sin_freq = mx.swapaxes(sin_freq, 1, 2)
    return cos_freq, sin_freq


def interleaved_freqs_cis(freqs: mx.array, pad_size: int) -> Tuple[mx.array, mx.array]:
    cos_freq = mx.cos(freqs)
    sin_freq = mx.sin(freqs)
    cos_freq = mx.repeat(cos_freq, 2, axis=-1)
    sin_freq = mx.repeat(sin_freq, 2, axis=-1)
    if pad_size != 0:
        cos_padding = mx.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = mx.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = mx.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = mx.concatenate([sin_padding, sin_freq], axis=-1)
    return cos_freq, sin_freq


def precompute_freqs_cis(indices_grid: mx.array, dim: int, theta: float = 10000.0, max_pos: Optional[List[int]] = None, use_middle_indices_grid: bool = False, num_attention_heads: int = 32, rope_type: LTXRopeType = LTXRopeType.INTERLEAVED, double_precision: bool = False) -> Tuple[mx.array, mx.array]:
    if max_pos is None:
        max_pos = [20, 2048, 2048]
    if double_precision:
        return _precompute_freqs_cis_double_precision(indices_grid, dim, theta, max_pos, use_middle_indices_grid, num_attention_heads, rope_type)
    indices = generate_freq_grid(theta, indices_grid.shape[1], dim)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)
    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
    return cos_freq, sin_freq


def _precompute_freqs_cis_double_precision(indices_grid: mx.array, dim: int, theta: float, max_pos: List[int], use_middle_indices_grid: bool, num_attention_heads: int, rope_type: LTXRopeType) -> Tuple[mx.array, mx.array]:
    indices_grid_np = np.array(indices_grid.astype(mx.float32)).astype(np.float64)
    n_pos_dims = indices_grid_np.shape[1]
    n_elem = 2 * n_pos_dims
    log_start = math.log(1.0) / math.log(theta)
    log_end = math.log(theta) / math.log(theta)
    num_indices = dim // n_elem
    if num_indices == 0:
        num_indices = 1
    lin_space = np.linspace(log_start, log_end, num_indices)
    indices_np = np.power(theta, lin_space) * (math.pi / 2)
    if use_middle_indices_grid:
        indices_grid_start = indices_grid_np[..., 0]
        indices_grid_end = indices_grid_np[..., 1]
        indices_grid_np = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid_np.shape) == 4:
        indices_grid_np = indices_grid_np[..., 0]
    batch_size = indices_grid_np.shape[0]
    seq_len = indices_grid_np.shape[2]
    fractional_positions = np.zeros((batch_size, seq_len, n_pos_dims), dtype=np.float64)
    for i in range(n_pos_dims):
        fractional_positions[:, :, i] = indices_grid_np[:, i, :] / max_pos[i]
    scaled_positions = fractional_positions * 2 - 1
    freqs = np.expand_dims(scaled_positions, axis=-1) * indices_np.reshape(1, 1, 1, -1)
    freqs = np.swapaxes(freqs, -1, -2)
    freqs = freqs.reshape(freqs.shape[:-2] + (-1,))
    cos_freq = np.cos(freqs)
    sin_freq = np.sin(freqs)
    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = cos_freq.shape[-1]
        pad_size = expected_freqs - current_freqs
        if pad_size > 0:
            cos_padding = np.ones((*cos_freq.shape[:-1], pad_size), dtype=np.float64)
            sin_padding = np.zeros((*sin_freq.shape[:-1], pad_size), dtype=np.float64)
            cos_freq = np.concatenate([cos_padding, cos_freq], axis=-1)
            sin_freq = np.concatenate([sin_padding, sin_freq], axis=-1)
        b, t = cos_freq.shape[0], cos_freq.shape[1]
        cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
        sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)
        cos_freq = np.swapaxes(cos_freq, 1, 2)
        sin_freq = np.swapaxes(sin_freq, 1, 2)
    else:
        cos_freq = np.repeat(cos_freq, 2, axis=-1)
        sin_freq = np.repeat(sin_freq, 2, axis=-1)
        pad_size = dim % n_elem
        if pad_size > 0:
            cos_padding = np.ones((*cos_freq.shape[:-1], pad_size), dtype=np.float64)
            sin_padding = np.zeros((*sin_freq.shape[:-1], pad_size), dtype=np.float64)
            cos_freq = np.concatenate([cos_padding, cos_freq], axis=-1)
            sin_freq = np.concatenate([sin_padding, sin_freq], axis=-1)
    cos_freq = mx.array(cos_freq.astype(np.float32))
    sin_freq = mx.array(sin_freq.astype(np.float32))
    return cos_freq, sin_freq
