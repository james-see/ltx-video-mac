"""Operations for Video VAE."""
import mlx.core as mx
import mlx.nn as nn


def patchify(x: mx.array, patch_size_hw: int = 4, patch_size_t: int = 1) -> mx.array:
    b, c, f, h, w = x.shape
    new_h = h // patch_size_hw
    new_w = w // patch_size_hw
    new_f = f // patch_size_t
    new_c = c * patch_size_hw * patch_size_hw * patch_size_t
    x = mx.reshape(x, (b, c, new_f, patch_size_t, new_h, patch_size_hw, new_w, patch_size_hw))
    x = mx.transpose(x, (0, 1, 3, 7, 5, 2, 4, 6))
    x = mx.reshape(x, (b, new_c, new_f, new_h, new_w))
    return x


def unpatchify(x: mx.array, patch_size_hw: int = 4, patch_size_t: int = 1) -> mx.array:
    b, c_packed, f, h, w = x.shape
    c = c_packed // (patch_size_hw * patch_size_hw * patch_size_t)
    x = mx.reshape(x, (b, c, patch_size_t, patch_size_hw, patch_size_hw, f, h, w))
    x = mx.transpose(x, (0, 1, 5, 2, 6, 4, 7, 3))
    x = mx.reshape(x, (b, c, f * patch_size_t, h * patch_size_hw, w * patch_size_hw))
    return x


class PerChannelStatistics(nn.Module):
    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.latent_channels = latent_channels
        self.mean = mx.zeros((latent_channels,))
        self.std = mx.ones((latent_channels,))

    def normalize(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        mean = self.mean.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        std = self.std.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        return ((x - mean) / std).astype(dtype)

    def un_normalize(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        mean = self.mean.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        std = self.std.astype(mx.float32).reshape(1, -1, 1, 1, 1)
        return (x * std + mean).astype(dtype)
