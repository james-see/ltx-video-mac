"""Video VAE Encoder and Decoder for LTX-2."""
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from ltx_mlx.models.ltx.video_vae.convolution import CausalConv3d, PaddingModeType
from ltx_mlx.models.ltx.video_vae.ops import PerChannelStatistics, patchify, unpatchify
from ltx_mlx.models.ltx.video_vae.resnet import NormLayerType, ResnetBlock3D, UNetMidBlock3D, get_norm_layer
from ltx_mlx.models.ltx.video_vae.sampling import DepthToSpaceUpsample, SpaceToDepthDownsample
from ltx_mlx.utils import PixelNorm


class LogVarianceType(Enum):
    PER_CHANNEL = "per_channel"
    UNIFORM = "uniform"
    CONSTANT = "constant"
    NONE = "none"


def _make_encoder_block(block_name: str, block_config: Dict[str, Any], in_channels: int, convolution_dimensions: int, norm_layer: NormLayerType, norm_num_groups: int, spatial_padding_mode: PaddingModeType) -> Tuple[nn.Module, int]:
    out_channels = in_channels
    if block_name == "res_x":
        block = UNetMidBlock3D(dims=convolution_dimensions, in_channels=in_channels, num_layers=block_config["num_layers"], resnet_eps=1e-6, resnet_groups=norm_num_groups, norm_layer=norm_layer, spatial_padding_mode=spatial_padding_mode)
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(dims=convolution_dimensions, in_channels=in_channels, out_channels=out_channels, eps=1e-6, groups=norm_num_groups, norm_layer=norm_layer, spatial_padding_mode=spatial_padding_mode)
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(dims=convolution_dimensions, in_channels=in_channels, out_channels=out_channels, stride=(2, 2, 2), spatial_padding_mode=spatial_padding_mode)
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(dims=convolution_dimensions, in_channels=in_channels, out_channels=out_channels, stride=(1, 2, 2), spatial_padding_mode=spatial_padding_mode)
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(dims=convolution_dimensions, in_channels=in_channels, out_channels=out_channels, stride=(2, 1, 1), spatial_padding_mode=spatial_padding_mode)
    else:
        raise ValueError(f"Unknown encoder block: {block_name}")
    return block, out_channels


class VideoEncoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(self, convolution_dimensions: int = 3, in_channels: int = 3, out_channels: int = 128, encoder_blocks: List[Tuple[str, Any]] = None, patch_size: int = 4, norm_layer: NormLayerType = NormLayerType.PIXEL_NORM, latent_log_var: LogVarianceType = LogVarianceType.UNIFORM, encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS):
        super().__init__()
        if encoder_blocks is None:
            encoder_blocks = []
        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)
        in_channels = in_channels * patch_size ** 2
        feature_channels = out_channels
        self.conv_in = CausalConv3d(in_channels=in_channels, out_channels=feature_channels, kernel_size=3, stride=1, padding=1, causal=True, spatial_padding_mode=encoder_spatial_padding_mode)
        self.down_blocks = {}
        for i, (block_name, block_params) in enumerate(encoder_blocks):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params
            block, feature_channels = _make_encoder_block(block_name=block_name, block_config=block_config, in_channels=feature_channels, convolution_dimensions=convolution_dimensions, norm_layer=norm_layer, norm_num_groups=self._norm_num_groups, spatial_padding_mode=encoder_spatial_padding_mode)
            self.down_blocks[i] = block
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_groups=self._norm_num_groups, dims=feature_channels, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()
        self.conv_act = nn.SiLU()
        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1
        self.conv_out = CausalConv3d(in_channels=feature_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1, causal=True, spatial_padding_mode=encoder_spatial_padding_mode)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample, causal=True)
        for down_block in self.down_blocks.values():
            if isinstance(down_block, (UNetMidBlock3D, ResnetBlock3D)):
                sample = down_block(sample, causal=True)
            else:
                sample = down_block(sample, causal=True)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=True)
        if self.latent_log_var == LogVarianceType.UNIFORM:
            means = sample[:, :-1, ...]
            logvar = sample[:, -1:, ...]
            num_channels = means.shape[1]
            repeated_logvar = mx.tile(logvar, (1, num_channels, 1, 1, 1))
            sample = mx.concatenate([means, repeated_logvar], axis=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30
            sample = mx.concatenate([sample, mx.full_like(sample, approx_ln_0)], axis=1)
        means = sample[:, :self.latent_channels, ...]
        return self.per_channel_statistics.normalize(means)


class VideoDecoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32

    def __init__(self, convolution_dimensions: int = 3, in_channels: int = 128, out_channels: int = 3, decoder_blocks: List[Tuple[str, Any]] = None, patch_size: int = 4, norm_layer: NormLayerType = NormLayerType.PIXEL_NORM, causal: bool = False, timestep_conditioning: bool = False, decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT):
        super().__init__()
        if decoder_blocks is None:
            decoder_blocks = []
        self.patch_size = patch_size
        out_channels = out_channels * patch_size ** 2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05
        feature_channels = in_channels
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = block_params if isinstance(block_params, dict) else {}
            if block_name == "res_x_y":
                feature_channels = feature_channels * block_config.get("multiplier", 2)
            if block_name == "compress_all":
                feature_channels = feature_channels * block_config.get("multiplier", 1)
        self.conv_in = CausalConv3d(in_channels=in_channels, out_channels=feature_channels, kernel_size=3, stride=1, padding=1, causal=True, spatial_padding_mode=decoder_spatial_padding_mode)
        self.up_blocks = []
        for block_name, block_params in list(reversed(decoder_blocks)):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params
            if block_name == "res_x":
                block = UNetMidBlock3D(dims=convolution_dimensions, in_channels=feature_channels, num_layers=block_config["num_layers"], resnet_eps=1e-6, resnet_groups=self._norm_num_groups, norm_layer=norm_layer, inject_noise=block_config.get("inject_noise", False), timestep_conditioning=timestep_conditioning, spatial_padding_mode=decoder_spatial_padding_mode)
            elif block_name == "compress_all":
                out_channels_block = feature_channels // block_config.get("multiplier", 1)
                block = DepthToSpaceUpsample(dims=convolution_dimensions, in_channels=feature_channels, stride=(2, 2, 2), residual=block_config.get("residual", False), out_channels_reduction_factor=block_config.get("multiplier", 1), spatial_padding_mode=decoder_spatial_padding_mode)
                feature_channels = out_channels_block
            else:
                continue
            self.up_blocks.append(block)
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_groups=self._norm_num_groups, dims=feature_channels, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(in_channels=feature_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, causal=True, spatial_padding_mode=decoder_spatial_padding_mode)

    def __call__(self, sample: mx.array, timestep: Optional[mx.array] = None) -> mx.array:
        batch_size = sample.shape[0]
        if self.timestep_conditioning:
            noise = mx.random.normal(sample.shape) * self.decode_noise_scale
            sample = noise + (1.0 - self.decode_noise_scale) * sample
        sample = self.per_channel_statistics.un_normalize(sample)
        if timestep is None and self.timestep_conditioning:
            timestep = mx.full((batch_size,), self.decode_timestep)
        sample = self.conv_in(sample, causal=self.causal)
        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                sample = up_block(sample, causal=self.causal)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal)
            else:
                sample = up_block(sample, causal=self.causal)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        return sample
