"""Sampling operations for Video VAE."""
from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from ltx_mlx.models.ltx.video_vae.convolution import CausalConv3d, PaddingModeType


class SpaceToDepthDownsample(nn.Module):
    def __init__(self, dims: int, in_channels: int, out_channels: int, stride: Union[int, Tuple[int, int, int]], spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride
        self.dims = dims
        self.out_channels = out_channels
        multiplier = stride[0] * stride[1] * stride[2]
        self.group_size = in_channels * multiplier // out_channels
        conv_out_channels = out_channels // multiplier
        self.conv = CausalConv3d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1, spatial_padding_mode=spatial_padding_mode)

    def _space_to_depth(self, x: mx.array) -> mx.array:
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride
        x = mx.reshape(x, (b, c, d // st, st, h // sh, sh, w // sw, sw))
        x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
        new_c = c * st * sh * sw
        new_d = d // st
        new_h = h // sh
        new_w = w // sw
        x = mx.reshape(x, (b, new_c, new_d, new_h, new_w))
        return x

    def __call__(self, x: mx.array, causal: bool = True) -> mx.array:
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride
        if st == 2:
            x = mx.concatenate([x[:, :, :1, :, :], x], axis=2)
            d = d + 1
        pad_d = (st - d % st) % st
        pad_h = (sh - h % sh) % sh
        pad_w = (sw - w % sw) % sw
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)])
        x_in = self._space_to_depth(x)
        b2, c2, d2, h2, w2 = x_in.shape
        x_in = mx.reshape(x_in, (b2, self.out_channels, self.group_size, d2, h2, w2))
        x_in = mx.mean(x_in, axis=2)
        x_conv = self.conv(x, causal=causal)
        x_conv = self._space_to_depth(x_conv)
        return x_conv + x_in


class DepthToSpaceUpsample(nn.Module):
    def __init__(self, dims: int, in_channels: int, stride: Union[int, Tuple[int, int, int]], residual: bool = False, out_channels_reduction_factor: int = 1, spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride
        self.dims = dims
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor
        multiplier = stride[0] * stride[1] * stride[2]
        out_channels = in_channels // out_channels_reduction_factor
        self.out_channels = out_channels
        self.conv = CausalConv3d(in_channels=in_channels, out_channels=out_channels * multiplier, kernel_size=3, stride=1, padding=1, spatial_padding_mode=spatial_padding_mode)

    def _depth_to_space(self, x: mx.array) -> mx.array:
        b, c_packed, d, h, w = x.shape
        st, sh, sw = self.stride
        c = c_packed // (st * sh * sw)
        x = mx.reshape(x, (b, c, st, sh, sw, d, h, w))
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = mx.reshape(x, (b, c, d * st, h * sh, w * sw))
        return x

    def __call__(self, x: mx.array, causal: bool = True, chunked_conv: bool = False) -> mx.array:
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride
        x_residual = None
        if self.residual:
            x_residual = self._depth_to_space(x)
            num_repeat = (st * sh * sw) // self.out_channels_reduction_factor
            x_residual = mx.tile(x_residual, (1, num_repeat, 1, 1, 1))
            if st > 1:
                x_residual = x_residual[:, :, 1:, :, :]
        if chunked_conv and d > 4:
            x = self._chunked_conv_depth_to_space(x, causal)
        else:
            x = self.conv(x, causal=causal)
            x = self._depth_to_space(x)
        if st > 1:
            x = x[:, :, 1:, :, :]
        if self.residual and x_residual is not None:
            x = x + x_residual
        return x

    def _chunked_conv_depth_to_space(self, x: mx.array, causal: bool = True) -> mx.array:
        b, c, d, h, w = x.shape
        st, sh, sw = self.stride
        chunk_size = 4
        kernel_t = 3
        if causal:
            pad_start = kernel_t - 1
            pad_end = 0
        else:
            pad_start = (kernel_t - 1) // 2
            pad_end = (kernel_t - 1) // 2
        outputs = []
        t_pos = 0
        while t_pos < d:
            t_end = min(t_pos + chunk_size, d)
            in_start = max(0, t_pos - pad_start)
            in_end = min(d, t_end + pad_end)
            chunk = x[:, :, in_start:in_end, :, :]
            chunk_conv = self.conv(chunk, causal=causal)
            chunk_out = self._depth_to_space(chunk_conv)
            out_start = (t_pos - in_start) * st
            out_end = out_start + (t_end - t_pos) * st
            chunk_out = chunk_out[:, :, out_start:out_end, :, :]
            outputs.append(chunk_out)
            mx.eval(outputs[-1])
            t_pos = t_end
        if len(outputs) == 1:
            return outputs[0]
        return mx.concatenate(outputs, axis=2)
