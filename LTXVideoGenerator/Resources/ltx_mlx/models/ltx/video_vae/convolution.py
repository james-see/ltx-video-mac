"""Convolution operations for Video VAE."""
from enum import Enum
from typing import Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn


class PaddingModeType(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"


def reflect_pad_2d(x: mx.array, pad_h: int, pad_w: int) -> mx.array:
    if pad_h == 0 and pad_w == 0:
        return x
    if pad_h > 0:
        top_pad = x[:, :, 1:pad_h+1, :, :][:, :, ::-1, :, :]
        bottom_pad = x[:, :, -pad_h-1:-1, :, :][:, :, ::-1, :, :]
        x = mx.concatenate([top_pad, x, bottom_pad], axis=2)
    if pad_w > 0:
        left_pad = x[:, :, :, 1:pad_w+1, :][:, :, :, ::-1, :]
        right_pad = x[:, :, :, -pad_w-1:-1, :][:, :, :, ::-1, :]
        x = mx.concatenate([left_pad, x, right_pad], axis=3)
    return x


class CausalConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[int, Tuple[int, int, int], str] = 0, causal: bool = False, spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS):
        super().__init__()
        self.causal = causal
        self.spatial_padding_mode = spatial_padding_mode
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.time_kernel_size = kernel_size[0]
        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        self.spatial_padding = (height_pad, width_pad)
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=True)

    def __call__(self, x: mx.array, causal: Optional[bool] = None) -> mx.array:
        use_causal = causal if causal is not None else self.causal
        if self.time_kernel_size > 1:
            if use_causal:
                first_frame_pad = mx.repeat(x[:, :, :1, :, :], self.time_kernel_size - 1, axis=2)
                x = mx.concatenate([first_frame_pad, x], axis=2)
            else:
                pad_size = (self.time_kernel_size - 1) // 2
                if pad_size > 0:
                    first_frame_pad = mx.repeat(x[:, :, :1, :, :], pad_size, axis=2)
                    last_frame_pad = mx.repeat(x[:, :, -1:, :, :], pad_size, axis=2)
                    x = mx.concatenate([first_frame_pad, x, last_frame_pad], axis=2)
        x = mx.transpose(x, (0, 2, 3, 4, 1))
        pad_h, pad_w = self.spatial_padding
        if pad_h > 0 or pad_w > 0:
            if self.spatial_padding_mode == PaddingModeType.REFLECT:
                x = reflect_pad_2d(x, pad_h, pad_w)
            else:
                pad_width = [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)]
                x = mx.pad(x, pad_width)
        x = self._chunked_conv3d(x)
        x = mx.transpose(x, (0, 4, 1, 2, 3))
        return x

    def _chunked_conv3d(self, x: mx.array) -> mx.array:
        b, d, h, w, c = x.shape
        total_elements = d * h * w * c
        max_safe_elements = 30 * 192 * 192 * 128
        if total_elements <= max_safe_elements:
            return self.conv(x)
        elements_per_frame = h * w * c
        max_frames_per_chunk = max(1, max_safe_elements // elements_per_frame)
        chunk_size = min(max_frames_per_chunk, 24)
        kernel_t = self.time_kernel_size
        overlap = kernel_t - 1
        expected_output_frames = d - overlap
        outputs = []
        out_idx = 0
        in_start = 0
        while out_idx < expected_output_frames:
            remaining = expected_output_frames - out_idx
            out_frames_this_chunk = min(chunk_size, remaining)
            in_frames_needed = out_frames_this_chunk + overlap
            in_end = min(in_start + in_frames_needed, d)
            chunk = x[:, in_start:in_end, :, :, :]
            chunk_out = self.conv(chunk)
            mx.eval(chunk_out)
            outputs.append(chunk_out)
            out_idx += chunk_out.shape[1]
            in_start += chunk_out.shape[1]
        if len(outputs) == 1:
            return outputs[0]
        return mx.concatenate(outputs, axis=1)
