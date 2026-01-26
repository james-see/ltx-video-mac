from typing import Tuple
import mlx.core as mx
import mlx.nn as nn
from ltx_mlx.utils import get_timestep_embedding


class AdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6, use_additional_conditions: bool = False):
        super().__init__()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim=embedding_dim,
            size_emb_dim=0 if not use_additional_conditions else embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
        )
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def __call__(self, timestep: mx.array, added_cond_kwargs: dict | None = None, batch_size: int | None = None, hidden_dtype: mx.Dtype | None = None) -> Tuple[mx.array, mx.array]:
        added_cond_kwargs = added_cond_kwargs or {}
        embedded_timestep = self.emb(timestep, batch_size=batch_size, hidden_dtype=hidden_dtype, **added_cond_kwargs)
        scale_shift_params = self.linear(self.silu(embedded_timestep))
        return scale_shift_params, embedded_timestep


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, size_emb_dim: int = 0, use_additional_conditions: bool = False, timestep_proj_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.size_emb_dim = size_emb_dim
        self.use_additional_conditions = use_additional_conditions
        self.time_proj = Timesteps(timestep_proj_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(timestep_proj_dim, embedding_dim, out_dim=embedding_dim)
        if use_additional_conditions and size_emb_dim > 0:
            self.additional_embedder = ConditionEmbedding(size_emb_dim, embedding_dim)

    def __call__(self, timestep: mx.array, resolution: mx.array | None = None, aspect_ratio: mx.array | None = None, batch_size: int | None = None, hidden_dtype: mx.Dtype | None = None) -> mx.array:
        timesteps_proj = self.time_proj(timestep)
        if hidden_dtype is not None:
            timesteps_proj = timesteps_proj.astype(hidden_dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        if self.use_additional_conditions and self.size_emb_dim > 0:
            if resolution is not None and aspect_ratio is not None:
                additional_embeds = self.additional_embedder(resolution, aspect_ratio, hidden_dtype)
                timesteps_emb = timesteps_emb + additional_embeds
        return timesteps_emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool = False, downscale_freq_shift: float = 1.0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def __call__(self, timesteps: mx.array) -> mx.array:
        return get_timestep_embedding(timesteps, self.num_channels, flip_sin_to_cos=self.flip_sin_to_cos, downscale_freq_shift=self.downscale_freq_shift)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu", out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or time_embed_dim
        self.linear1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        self.linear2 = nn.Linear(time_embed_dim, out_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear1(sample)
        sample = self.act(sample)
        sample = self.linear2(sample)
        return sample


class ConditionEmbedding(nn.Module):
    def __init__(self, size_emb_dim: int, embedding_dim: int):
        super().__init__()
        self.resolution_embedder = TimestepEmbedding(size_emb_dim, embedding_dim)
        self.aspect_ratio_embedder = TimestepEmbedding(size_emb_dim, embedding_dim)

    def __call__(self, resolution: mx.array, aspect_ratio: mx.array, hidden_dtype: mx.Dtype | None = None) -> mx.array:
        resolution_emb = self.resolution_embedder(resolution)
        aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio)
        if hidden_dtype is not None:
            resolution_emb = resolution_emb.astype(hidden_dtype)
            aspect_ratio_emb = aspect_ratio_emb.astype(hidden_dtype)
        return resolution_emb + aspect_ratio_emb
