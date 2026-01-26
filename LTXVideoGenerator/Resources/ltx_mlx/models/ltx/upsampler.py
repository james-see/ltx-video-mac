"""Latent upsampler for LTX-2."""
from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn


class Conv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]] = 3, stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[int, Tuple[int, int, int]] = 0, dilation: Union[int, Tuple[int, int, int]] = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        scale = 1.0 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]) ** 0.5
        self.weight = mx.random.uniform(low=-scale, high=scale, shape=(out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv3d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            y = y + self.bias
        return y


class GroupNorm3d(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        n, d, h, w, c = x.shape
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        x = mx.reshape(x, (n, d * h * w, self.num_groups, c // self.num_groups))
        mean = mx.mean(x, axis=(1, 3), keepdims=True)
        var = mx.var(x, axis=(1, 3), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = mx.reshape(x, (n, d, h, w, c))
        weight = self.weight.astype(mx.float32)
        bias = self.bias.astype(mx.float32)
        x = x * weight + bias
        x = x.astype(input_dtype)
        return x


class PixelShuffle2D(nn.Module):
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.upscale_factor = upscale_factor

    def __call__(self, x: mx.array) -> mx.array:
        n, h, w, c = x.shape
        r = self.upscale_factor
        out_c = c // (r * r)
        x = mx.reshape(x, (n, h, w, out_c, r, r))
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
        x = mx.reshape(x, (n, h * r, w * r, out_c))
        return x


class SpatialRationalResampler(nn.Module):
    def __init__(self, mid_channels: int = 1024, scale: float = 2.0):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = PixelShuffle2D(2)

    def __call__(self, x: mx.array) -> mx.array:
        n, d, h, w, c = x.shape
        x = mx.reshape(x, (n * d, h, w, c))
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = mx.reshape(x, (n, d, h * 2, w * 2, c))
        return x


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = GroupNorm3d(32, channels)
        self.conv2 = Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm3d(32, channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nn.silu(x + residual)
        return x


class LatentUpsampler(nn.Module):
    def __init__(self, in_channels: int = 128, mid_channels: int = 1024, num_blocks_per_stage: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.initial_conv = Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = GroupNorm3d(32, mid_channels)
        self.res_blocks = {i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)}
        self.upsampler = SpatialRationalResampler(mid_channels=mid_channels, scale=2.0)
        self.post_upsample_res_blocks = {i: ResBlock3D(mid_channels) for i in range(num_blocks_per_stage)}
        self.final_conv = Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def __call__(self, latent: mx.array, debug: bool = False) -> mx.array:
        x = mx.transpose(latent, (0, 2, 3, 4, 1))
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = nn.silu(x)
        for i in sorted(self.res_blocks.keys()):
            x = self.res_blocks[i](x)
        x = self.upsampler(x)
        for i in sorted(self.post_upsample_res_blocks.keys()):
            x = self.post_upsample_res_blocks[i](x)
        x = self.final_conv(x)
        x = mx.transpose(x, (0, 4, 1, 2, 3))
        return x


def upsample_latents(latent: mx.array, upsampler: LatentUpsampler, latent_mean: mx.array, latent_std: mx.array, debug: bool = False) -> mx.array:
    latent_mean = latent_mean.reshape(1, -1, 1, 1, 1)
    latent_std = latent_std.reshape(1, -1, 1, 1, 1)
    latent = latent * latent_std + latent_mean
    latent = upsampler(latent, debug=debug)
    latent = (latent - latent_mean) / latent_std
    return latent


def load_upsampler(weights_path: str) -> LatentUpsampler:
    print(f"Loading spatial upsampler from {weights_path}...")
    raw_weights = mx.load(weights_path)
    sample_key = "res_blocks.0.conv1.weight"
    if sample_key in raw_weights:
        mid_channels = raw_weights[sample_key].shape[0]
    else:
        mid_channels = 1024
    upsampler = LatentUpsampler(in_channels=128, mid_channels=mid_channels, num_blocks_per_stage=4)
    sanitized = {}
    for key, value in raw_weights.items():
        new_key = key
        if "conv" in key and "weight" in key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        if "conv" in key and "weight" in key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))
        sanitized[new_key] = value
    upsampler.load_weights(list(sanitized.items()), strict=False)
    return upsampler
