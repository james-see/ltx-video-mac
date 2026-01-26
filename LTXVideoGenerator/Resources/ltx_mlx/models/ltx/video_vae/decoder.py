"""Video VAE Decoder for LTX-2 with timestep conditioning."""
import math
from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from ltx_mlx.models.ltx.video_vae.convolution import CausalConv3d, PaddingModeType
from ltx_mlx.models.ltx.video_vae.ops import unpatchify
from ltx_mlx.models.ltx.video_vae.sampling import DepthToSpaceUpsample
from ltx_mlx.models.ltx.video_vae.tiling import TilingConfig, decode_with_tiling


def get_timestep_embedding(timesteps: mx.array, embedding_dim: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0, scale: float = 1, max_period: int = 10000) -> mx.array:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = scale * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)
        self.act = nn.SiLU()

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class PixArtAlphaTimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def __call__(self, timestep: mx.array, hidden_dtype: mx.Dtype = mx.float32) -> mx.array:
        timesteps_proj = get_timestep_embedding(timestep, embedding_dim=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        timesteps_emb = self.timestep_embedder(timesteps_proj.astype(hidden_dtype))
        return timesteps_emb


class ResnetBlock3DSimple(nn.Module):
    def __init__(self, channels: int, spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT, timestep_conditioning: bool = False):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning

        class ConvWrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = CausalConv3d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, spatial_padding_mode=spatial_padding_mode)
            def __call__(self_inner, x, causal=False):
                return self_inner.conv(x, causal=causal)
        self.conv1 = ConvWrapper()
        self.conv2 = ConvWrapper()
        self.act = nn.SiLU()
        if timestep_conditioning:
            self.scale_shift_table = mx.zeros((4, channels))

    def pixel_norm(self, x: mx.array, eps: float = 1e-8) -> mx.array:
        return x / mx.sqrt(mx.mean(x ** 2, axis=1, keepdims=True) + eps)

    def __call__(self, x: mx.array, causal: bool = False, timestep_embed: Optional[mx.array] = None) -> mx.array:
        residual = x
        batch_size = x.shape[0]
        x = self.pixel_norm(x)
        if self.timestep_conditioning and timestep_embed is not None:
            ada_values = self.scale_shift_table[None, :, :, None, None, None]
            channels = self.scale_shift_table.shape[1]
            ts_reshaped = timestep_embed.reshape(batch_size, 4, channels, 1, 1, 1)
            ada_values = ada_values + ts_reshaped
            shift1 = ada_values[:, 0]
            scale1 = ada_values[:, 1]
            shift2 = ada_values[:, 2]
            scale2 = ada_values[:, 3]
            x = x * (1 + scale1) + shift1
        x = self.act(x)
        x = self.conv1(x, causal=causal)
        x = self.pixel_norm(x)
        if self.timestep_conditioning and timestep_embed is not None:
            x = x * (1 + scale2) + shift2
        x = self.act(x)
        x = self.conv2(x, causal=causal)
        return x + residual


class ResBlockGroup(nn.Module):
    def __init__(self, channels: int, num_layers: int = 5, spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT, timestep_conditioning: bool = False):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaTimestepEmbedder(embedding_dim=channels * 4)
        self.res_blocks = {i: ResnetBlock3DSimple(channels, spatial_padding_mode, timestep_conditioning=timestep_conditioning) for i in range(num_layers)}

    def __call__(self, x: mx.array, causal: bool = False, timestep: Optional[mx.array] = None) -> mx.array:
        timestep_embed = None
        if self.timestep_conditioning and timestep is not None:
            batch_size = x.shape[0]
            timestep_embed = self.time_embedder(timestep.flatten(), hidden_dtype=x.dtype)
            timestep_embed = timestep_embed.reshape(batch_size, -1, 1, 1, 1)
        for res_block in self.res_blocks.values():
            x = res_block(x, causal=causal, timestep_embed=timestep_embed)
        return x


class LTX2VideoDecoder(nn.Module):
    def __init__(self, in_channels: int = 128, out_channels: int = 3, patch_size: int = 4, num_layers_per_block: int = 5, spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT, timestep_conditioning: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.timestep_conditioning = timestep_conditioning
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05
        self.latents_mean = mx.zeros((in_channels,))
        self.latents_std = mx.ones((in_channels,))

        class ConvInWrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = CausalConv3d(in_channels=in_channels, out_channels=1024, kernel_size=3, stride=1, padding=1, spatial_padding_mode=spatial_padding_mode)
            def __call__(self_inner, x, causal=False):
                return self_inner.conv(x, causal=causal)
        self.conv_in = ConvInWrapper()
        self.up_blocks = {
            0: ResBlockGroup(1024, num_layers_per_block, spatial_padding_mode, timestep_conditioning),
            1: DepthToSpaceUpsample(dims=3, in_channels=1024, stride=(2, 2, 2), residual=True, out_channels_reduction_factor=2, spatial_padding_mode=spatial_padding_mode),
            2: ResBlockGroup(512, num_layers_per_block, spatial_padding_mode, timestep_conditioning),
            3: DepthToSpaceUpsample(dims=3, in_channels=512, stride=(2, 2, 2), residual=True, out_channels_reduction_factor=2, spatial_padding_mode=spatial_padding_mode),
            4: ResBlockGroup(256, num_layers_per_block, spatial_padding_mode, timestep_conditioning),
            5: DepthToSpaceUpsample(dims=3, in_channels=256, stride=(2, 2, 2), residual=True, out_channels_reduction_factor=2, spatial_padding_mode=spatial_padding_mode),
            6: ResBlockGroup(128, num_layers_per_block, spatial_padding_mode, timestep_conditioning),
        }
        final_out_channels = out_channels * patch_size * patch_size

        class ConvOutWrapper(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = CausalConv3d(in_channels=128, out_channels=final_out_channels, kernel_size=3, stride=1, padding=1, spatial_padding_mode=spatial_padding_mode)
            def __call__(self_inner, x, causal=False):
                return self_inner.conv(x, causal=causal)
        self.conv_out = ConvOutWrapper()
        self.act = nn.SiLU()
        if timestep_conditioning:
            self.timestep_scale_multiplier = mx.array(1000.0)
            self.last_time_embedder = PixArtAlphaTimestepEmbedder(embedding_dim=128 * 2)
            self.last_scale_shift_table = mx.zeros((2, 128))

    def denormalize(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        mean = self.latents_mean.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        std = self.latents_std.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        return (x * std + mean).astype(dtype)

    def pixel_norm(self, x: mx.array, eps: float = 1e-8) -> mx.array:
        return x / mx.sqrt(mx.mean(x ** 2, axis=1, keepdims=True) + eps)

    def __call__(self, sample: mx.array, causal: bool = False, timestep: Optional[mx.array] = None, debug: bool = False, chunked_conv: bool = False) -> mx.array:
        batch_size = sample.shape[0]
        if self.timestep_conditioning:
            noise = mx.random.normal(sample.shape) * self.decode_noise_scale
            sample = noise + (1.0 - self.decode_noise_scale) * sample
        sample = self.denormalize(sample)
        if timestep is None and self.timestep_conditioning:
            timestep = mx.full((batch_size,), self.decode_timestep)
        scaled_timestep = None
        if self.timestep_conditioning and timestep is not None:
            scaled_timestep = timestep * self.timestep_scale_multiplier
        x = self.conv_in(sample, causal=causal)
        for i, block in self.up_blocks.items():
            if isinstance(block, ResBlockGroup):
                x = block(x, causal=causal, timestep=scaled_timestep)
            elif isinstance(block, DepthToSpaceUpsample):
                x = block(x, causal=causal, chunked_conv=chunked_conv)
            else:
                x = block(x, causal=causal)
        x = self.pixel_norm(x)
        if self.timestep_conditioning and scaled_timestep is not None:
            embedded_timestep = self.last_time_embedder(scaled_timestep.flatten(), hidden_dtype=x.dtype)
            embedded_timestep = embedded_timestep.reshape(batch_size, -1, 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, :, :, None, None, None]
            ts_reshaped = embedded_timestep.reshape(batch_size, 2, 128, 1, 1, 1)
            ada_values = ada_values + ts_reshaped
            shift = ada_values[:, 0]
            scale = ada_values[:, 1]
            x = x * (1 + scale) + shift
        x = self.act(x)
        x = self.conv_out(x, causal=causal)
        x = unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)
        return x

    def decode_tiled(self, sample: mx.array, tiling_config: Optional[TilingConfig] = None, tiling_mode: str = "auto", causal: bool = False, timestep: Optional[mx.array] = None, debug: bool = False, on_frames_ready: Optional[callable] = None) -> mx.array:
        if tiling_config is None:
            tiling_config = TilingConfig.default()
        _, _, f, h, w = sample.shape
        needs_spatial_tiling = False
        needs_temporal_tiling = False
        spatial_scale = 32
        temporal_scale = 8
        if tiling_config.spatial_config is not None:
            s_cfg = tiling_config.spatial_config
            tile_size_latent = s_cfg.tile_size_in_pixels // spatial_scale
            if h > tile_size_latent or w > tile_size_latent:
                needs_spatial_tiling = True
        if tiling_config.temporal_config is not None:
            t_cfg = tiling_config.temporal_config
            tile_size_latent = t_cfg.tile_size_in_frames // temporal_scale
            if f > tile_size_latent:
                needs_temporal_tiling = True
        use_chunked_conv = tiling_mode in ("conservative", "none", "auto", "default", "spatial")
        if not needs_spatial_tiling and not needs_temporal_tiling:
            return self(sample, causal=causal, timestep=timestep, debug=debug, chunked_conv=use_chunked_conv)
        return decode_with_tiling(decoder_fn=self, latents=sample, tiling_config=tiling_config, spatial_scale=32, temporal_scale=8, causal=causal, timestep=timestep, chunked_conv=use_chunked_conv, on_frames_ready=on_frames_ready)


def load_vae_decoder(model_path: str, timestep_conditioning: Optional[bool] = None) -> LTX2VideoDecoder:
    from pathlib import Path
    import json
    from safetensors import safe_open
    model_path = Path(model_path)
    if model_path.is_file() and model_path.suffix == ".safetensors":
        weights_path = model_path
    elif (model_path / "ltx-2-19b-distilled.safetensors").exists():
        weights_path = model_path / "ltx-2-19b-distilled.safetensors"
    elif (model_path / "vae" / "diffusion_pytorch_model.safetensors").exists():
        weights_path = model_path / "vae" / "diffusion_pytorch_model.safetensors"
    else:
        raise FileNotFoundError(f"VAE weights not found at {model_path}")
    print(f"Loading VAE decoder from {weights_path}...")
    if timestep_conditioning is None:
        try:
            with safe_open(str(weights_path), framework="numpy") as f:
                metadata = f.metadata()
                if metadata and "config" in metadata:
                    configs = json.loads(metadata["config"])
                    vae_config = configs.get("vae", {})
                    timestep_conditioning = vae_config.get("timestep_conditioning", False)
                else:
                    timestep_conditioning = False
        except Exception:
            timestep_conditioning = False
    decoder = LTX2VideoDecoder(timestep_conditioning=timestep_conditioning)
    weights = mx.load(str(weights_path))
    has_vae_prefix = any(k.startswith("vae.") for k in weights.keys())
    if has_vae_prefix:
        prefix = "vae.decoder."
        stats_prefix = "vae.per_channel_statistics."
    else:
        prefix = "decoder."
        stats_prefix = ""
    mean_key = f"{stats_prefix}mean-of-means" if stats_prefix else "latents_mean"
    std_key = f"{stats_prefix}std-of-means" if stats_prefix else "latents_std"
    if mean_key in weights:
        decoder.latents_mean = weights[mean_key]
    if std_key in weights:
        decoder.latents_std = weights[std_key]
    decoder_weights = {}
    for key, value in weights.items():
        if not key.startswith(prefix):
            continue
        new_key = key[len(prefix):]
        if ".conv.weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        if ".conv.weight" in new_key or ".conv.bias" in new_key:
            if ".conv.conv.weight" not in new_key and ".conv.conv.bias" not in new_key:
                new_key = new_key.replace(".conv.weight", ".conv.conv.weight")
                new_key = new_key.replace(".conv.bias", ".conv.conv.bias")
        decoder_weights[new_key] = value
    decoder.load_weights(list(decoder_weights.items()), strict=False)
    print("VAE decoder loaded successfully")
    return decoder
