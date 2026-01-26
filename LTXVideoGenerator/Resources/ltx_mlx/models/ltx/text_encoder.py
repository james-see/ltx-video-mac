"""LTX-2 Text Encoder - requires mlx-vlm for Gemma 3 support.

This module uses mlx-vlm's Gemma 3 implementation for text encoding.
Install with: pip install mlx-vlm
"""
import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from ltx_mlx.utils import rms_norm, apply_quantization

# Check for mlx-vlm dependency
try:
    from mlx_vlm.models.gemma3.language import Gemma3Model
    from mlx_vlm.models.gemma3.config import TextConfig
    MLX_VLM_AVAILABLE = True
except ImportError:
    MLX_VLM_AVAILABLE = False
    Gemma3Model = None
    TextConfig = None


PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_system_prompt(prompt_name: str) -> str:
    prompt_path = PROMPTS_DIR / prompt_name
    if prompt_path.exists():
        with open(prompt_path, "r") as f:
            return f.read()
    return ""


class GemmaFeaturesExtractor(nn.Module):
    def __init__(self, input_dim: int = 188160, output_dim: int = 3840):
        super().__init__()
        self.aggregate_embed = nn.Linear(input_dim, output_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.aggregate_embed(x)


class ConnectorAttention(nn.Module):
    def __init__(self, dim: int = 3840, num_heads: int = 30, head_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(inner_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(inner_dim, eps=1e-6)

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None, pe: Optional[Tuple[mx.array, mx.array]] = None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = mx.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        k = mx.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        v = mx.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)).transpose(0, 2, 1, 3)
        if pe is not None:
            q = self._apply_split_rope(q, pe[0], pe[1])
            k = self._apply_split_rope(k, pe[0], pe[1])
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.to_out(out)

    def _apply_split_rope(self, x: mx.array, cos_freq: mx.array, sin_freq: mx.array) -> mx.array:
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        cos_freq = cos_freq.astype(mx.float32)
        sin_freq = sin_freq.astype(mx.float32)
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        out1 = x1 * cos_freq - x2 * sin_freq
        out2 = x2 * cos_freq + x1 * sin_freq
        return mx.concatenate([out1, out2], axis=-1).astype(input_dtype)


class ConnectorFeedForward(nn.Module):
    def __init__(self, dim: int = 3840, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        self.proj_in = nn.Linear(dim, inner_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(inner_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.gelu(self.proj_in(x))
        x = self.dropout(x)
        x = self.proj_out(x)
        return x


class ConnectorTransformerBlock(nn.Module):
    def __init__(self, dim: int = 3840, num_heads: int = 30, head_dim: int = 128):
        super().__init__()
        self.attn1 = ConnectorAttention(dim, num_heads, head_dim)
        self.ff = ConnectorFeedForward(dim)

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array] = None, pe: Optional[mx.array] = None) -> mx.array:
        norm_x = rms_norm(x)
        if norm_x.ndim == 4:
            norm_x = mx.squeeze(norm_x, axis=1)
        attn_out = self.attn1(norm_x, attention_mask, pe)
        x = x + attn_out
        if x.ndim == 4:
            x = mx.squeeze(x, axis=1)
        norm_x = rms_norm(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        if x.ndim == 4:
            x = mx.squeeze(x, axis=1)
        return x


class Embeddings1DConnector(nn.Module):
    def __init__(self, dim: int = 3840, num_heads: int = 30, head_dim: int = 128, num_layers: int = 2, num_learnable_registers: int = 128, positional_embedding_theta: float = 10000.0, positional_embedding_max_pos: list = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_learnable_registers = num_learnable_registers
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos or [4096]
        self.transformer_1d_blocks = {i: ConnectorTransformerBlock(dim, num_heads, head_dim) for i in range(num_layers)}
        if num_learnable_registers > 0:
            self.learnable_registers = mx.zeros((num_learnable_registers, dim))

    def _precompute_freqs_cis(self, seq_len: int, dtype: mx.Dtype) -> Tuple[mx.array, mx.array]:
        dim = self.num_heads * self.head_dim
        theta = self.positional_embedding_theta
        max_pos = self.positional_embedding_max_pos
        n_elem = 2 * len(max_pos)
        num_indices = dim // n_elem
        log_start = np.log(1.0) / np.log(theta)
        log_end = np.log(theta) / np.log(theta)
        lin_space = np.linspace(log_start, log_end, num_indices, dtype=np.float64)
        indices = (np.power(theta, lin_space) * (np.pi / 2)).astype(np.float64)
        positions = np.arange(seq_len, dtype=np.float64)
        fractional_positions = positions / max_pos[0]
        scaled_positions = fractional_positions * 2 - 1
        freqs = scaled_positions[:, None] * indices[None, :]
        cos_freq = np.cos(freqs)
        sin_freq = np.sin(freqs)
        cos_freq = cos_freq.reshape(seq_len, self.num_heads, self.head_dim // 2)
        sin_freq = sin_freq.reshape(seq_len, self.num_heads, self.head_dim // 2)
        cos_freq = np.transpose(cos_freq, (1, 0, 2))[np.newaxis, ...]
        sin_freq = np.transpose(sin_freq, (1, 0, 2))[np.newaxis, ...]
        cos_full = mx.array(cos_freq.astype(np.float32))
        sin_full = mx.array(sin_freq.astype(np.float32))
        return cos_full.astype(dtype), sin_full.astype(dtype)

    def _replace_padded_with_registers(self, hidden_states: mx.array, attention_mask: mx.array) -> Tuple[mx.array, mx.array]:
        batch_size, seq_len, dim = hidden_states.shape
        dtype = hidden_states.dtype
        mask_binary = (attention_mask.squeeze(1).squeeze(1) >= -9000.0).astype(mx.int32)
        num_tiles = seq_len // self.num_learnable_registers
        registers = mx.tile(self.learnable_registers, (num_tiles, 1)).astype(dtype)
        result_list = []
        for b in range(batch_size):
            mask_b = mask_binary[b]
            hs_b = hidden_states[b]
            num_valid = int(mx.sum(mask_b))
            valid_tokens = hs_b[seq_len - num_valid:]
            pad_length = seq_len - num_valid
            if pad_length > 0:
                padding = mx.zeros((pad_length, dim), dtype=dtype)
                adjusted = mx.concatenate([valid_tokens, padding], axis=0)
            else:
                adjusted = valid_tokens
            flipped_mask = mx.concatenate([mx.ones((num_valid,), dtype=mx.int32), mx.zeros((pad_length,), dtype=mx.int32)], axis=0)
            flipped_mask_expanded = flipped_mask[:, None].astype(dtype)
            combined = flipped_mask_expanded * adjusted + (1 - flipped_mask_expanded) * registers
            result_list.append(combined)
        hidden_states = mx.stack(result_list, axis=0)
        attention_mask = mx.zeros_like(attention_mask)
        return hidden_states, attention_mask

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        if self.num_learnable_registers > 0 and attention_mask is not None:
            hidden_states, attention_mask = self._replace_padded_with_registers(hidden_states, attention_mask)
        seq_len = hidden_states.shape[1]
        freqs_cis = self._precompute_freqs_cis(seq_len, hidden_states.dtype)
        for i in range(len(self.transformer_1d_blocks)):
            hidden_states = self.transformer_1d_blocks[i](hidden_states, attention_mask, freqs_cis)
        hidden_states = rms_norm(hidden_states)
        return hidden_states, attention_mask


def norm_and_concat_hidden_states(hidden_states: List[mx.array], attention_mask: mx.array, padding_side: str = "left") -> mx.array:
    stacked = mx.stack(hidden_states, axis=-1)
    dtype = stacked.dtype
    b, t, d, num_layers = stacked.shape
    sequence_lengths = mx.sum(attention_mask, axis=-1)
    token_indices = mx.arange(t)[None, :]
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    else:
        start_indices = t - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    mask = mask[:, :, None, None]
    eps = mx.array(1e-6, dtype=dtype)
    masked = mx.where(mask, stacked, mx.zeros_like(stacked))
    denom = (sequence_lengths * d).reshape(b, 1, 1, 1).astype(dtype)
    mean = mx.sum(masked, axis=(1, 2), keepdims=True) / (denom + eps)
    x_for_min = mx.where(mask, stacked, mx.full(stacked.shape, float('inf'), dtype=dtype))
    x_for_max = mx.where(mask, stacked, mx.full(stacked.shape, float('-inf'), dtype=dtype))
    x_min = mx.min(x_for_min, axis=(1, 2), keepdims=True)
    x_max = mx.max(x_for_max, axis=(1, 2), keepdims=True)
    range_val = x_max - x_min
    normed = 8 * (stacked - mean) / (range_val + eps)
    normed = mx.reshape(normed, (b, t, -1))
    mask_flat = mx.broadcast_to(mask[:, :, :, 0], (b, t, d * num_layers))
    normed = mx.where(mask_flat, normed, mx.zeros_like(normed))
    return normed


class LTX2TextEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 3840, audio_dim: int = 2048, num_layers: int = 49):
        super().__init__()
        if not MLX_VLM_AVAILABLE:
            raise ImportError("mlx-vlm is required for LTX-2 text encoding. Install with: pip install mlx-vlm")
        self.hidden_dim = hidden_dim
        self.audio_dim = audio_dim
        self.num_layers = num_layers
        self.language_model = None
        self.feature_extractor = GemmaFeaturesExtractor(input_dim=hidden_dim * num_layers, output_dim=hidden_dim)
        self.video_embeddings_connector = Embeddings1DConnector(dim=hidden_dim, num_heads=30, head_dim=128, num_layers=2, num_learnable_registers=128, positional_embedding_max_pos=[4096])
        self.audio_embeddings_connector = Embeddings1DConnector(dim=hidden_dim, num_heads=30, head_dim=128, num_layers=2, num_learnable_registers=128, positional_embedding_max_pos=[4096])
        self.processor = None

    def load(self, model_path: Optional[str] = None, text_encoder_path: Optional[str] = "google/gemma-3-12b-it"):
        import json
        from pathlib import Path
        model_path = Path(str(model_path))
        if Path(str(text_encoder_path)).joinpath("text_encoder").is_dir():
            text_encoder_path = str(Path(text_encoder_path) / "text_encoder")
        weight_files = sorted(Path(text_encoder_path).glob("*.safetensors"))
        config_file = Path(text_encoder_path) / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            self.language_model = LanguageModel(config=TextConfig.from_dict(config_dict["text_config"]))
        else:
            raise ValueError(f"Config file not found at {text_encoder_path}")
        quantization = config_dict.get("quantization", None)
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(str(wf)))
        if hasattr(self.language_model, "sanitize"):
            weights = self.language_model.sanitize(weights=weights)
        apply_quantization(model=self.language_model, weights=weights, quantization=quantization)
        self.language_model.load_weights(list(weights.items()), strict=False)
        transformer_files = list(model_path.glob("ltx-2-19*.safetensors"))
        if transformer_files:
            transformer_weights = mx.load(str(transformer_files[0]))
            if "text_embedding_projection.aggregate_embed.weight" in transformer_weights:
                self.feature_extractor.aggregate_embed.weight = transformer_weights["text_embedding_projection.aggregate_embed.weight"]
            connector_weights = {}
            for key, value in transformer_weights.items():
                if key.startswith("model.diffusion_model.video_embeddings_connector."):
                    new_key = key.replace("model.diffusion_model.video_embeddings_connector.", "")
                    connector_weights[new_key] = value
            if connector_weights:
                mapped_weights = {}
                for key, value in connector_weights.items():
                    new_key = key
                    new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
                    new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
                    new_key = new_key.replace(".to_out.0.", ".to_out.")
                    mapped_weights[new_key] = value
                self.video_embeddings_connector.load_weights(list(mapped_weights.items()), strict=False)
                if "learnable_registers" in connector_weights:
                    self.video_embeddings_connector.learnable_registers = connector_weights["learnable_registers"]
        from transformers import AutoTokenizer
        tokenizer_path = model_path / "tokenizer"
        if tokenizer_path.exists():
            self.processor = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
        else:
            self.processor = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=True)
        self.processor.padding_side = "left"
        print("Text encoder loaded successfully")

    def encode(self, prompt: str, max_length: int = 1024, return_audio_embeddings: bool = True) -> Tuple[mx.array, mx.array]:
        if self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        inputs = self.processor(prompt, return_tensors="np", max_length=max_length, truncation=True, padding="max_length")
        input_ids = mx.array(inputs["input_ids"])
        attention_mask = mx.array(inputs["attention_mask"])
        _, all_hidden_states = self.language_model(inputs=input_ids, input_embeddings=None, attention_mask=attention_mask, output_hidden_states=True)
        concat_hidden = norm_and_concat_hidden_states(all_hidden_states, attention_mask, padding_side="left")
        features = self.feature_extractor(concat_hidden)
        additive_mask = (attention_mask - 1).astype(features.dtype)
        additive_mask = additive_mask.reshape(attention_mask.shape[0], 1, 1, -1) * 1e9
        video_embeddings, _ = self.video_embeddings_connector(features, additive_mask)
        if return_audio_embeddings:
            audio_embeddings, _ = self.audio_embeddings_connector(features, additive_mask)
            return video_embeddings, audio_embeddings
        else:
            return video_embeddings, attention_mask

    def __call__(self, prompt: str, max_length: int = 1024, return_audio_embeddings: bool = True) -> Tuple[mx.array, mx.array]:
        return self.encode(prompt, max_length, return_audio_embeddings)


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(self.config)

    def _create_causal_mask_with_padding(self, seq_len: int, attention_mask: Optional[mx.array], dtype: mx.Dtype) -> mx.array:
        causal_mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            padding_mask = attention_mask.astype(mx.bool_)
            combined = causal_mask[None, :, :] & padding_mask[:, None, :]
            min_val = mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            mask = mx.where(combined, mx.zeros(combined.shape, dtype=dtype), mx.full(combined.shape, min_val, dtype=dtype))
            return mask[:, None, :, :]
        else:
            min_val = mx.finfo(dtype).min if dtype in (mx.float16, mx.bfloat16) else -1e9
            mask = mx.where(causal_mask, mx.zeros((seq_len, seq_len), dtype=dtype), mx.full((seq_len, seq_len), min_val, dtype=dtype))
            return mask[None, None, :, :]

    def __call__(self, inputs: mx.array, input_embeddings: Optional[mx.array] = None, attention_mask: Optional[mx.array] = None, output_hidden_states: bool = False, cache: Optional[List[mx.array]] = None) -> Tuple[mx.array, List[mx.array]]:
        batch_size, seq_len = inputs.shape
        h = input_embeddings if input_embeddings is not None else self.model.embed_tokens(inputs)
        h *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)
        mx.eval(h)
        all_hidden_states = [h] if output_hidden_states else []
        if cache is None:
            cache = [None] * len(self.model.layers)
        full_causal_mask = self._create_causal_mask_with_padding(seq_len, attention_mask, h.dtype)
        sliding_mask = full_causal_mask
        num_layers = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            is_global = (i % self.config.sliding_window_pattern == self.config.sliding_window_pattern - 1)
            if is_global:
                local_mask = full_causal_mask
            else:
                local_mask = sliding_mask
            h = layer(h, local_mask, cache[i])
            mx.eval(h)
            if output_hidden_states and i < num_layers - 1:
                all_hidden_states.append(h)
        hidden_states = self.model.norm(h)
        mx.eval(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        prefix = "language_model."
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(prefix):
                if hasattr(value, "dtype") and value.dtype == mx.float32:
                    sanitized[key[len(prefix):]] = value.astype(mx.bfloat16)
                else:
                    sanitized[key[len(prefix):]] = value
        return sanitized

    @property
    def layers(self) -> List[nn.Module]:
        return self.model.layers


def load_text_encoder(model_path: str = "/tmp/ltx2") -> LTX2TextEncoder:
    encoder = LTX2TextEncoder()
    encoder.load(model_path=model_path)
    return encoder
