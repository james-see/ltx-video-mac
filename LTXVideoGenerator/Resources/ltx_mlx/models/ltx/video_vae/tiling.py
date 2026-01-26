"""VAE Tiling Configuration for decoding large videos."""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import mlx.core as mx
import gc


def compute_trapezoidal_mask_1d(length: int, ramp_left: int, ramp_right: int, left_starts_from_0: bool = False) -> mx.array:
    if length <= 0:
        raise ValueError("Mask length must be positive.")
    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))
    mask = [1.0] * length
    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in_full = [i / (interval_length - 1) for i in range(interval_length)]
        fade_in = fade_in_full[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        for i in range(min(ramp_left, len(fade_in))):
            mask[i] *= fade_in[i]
    if ramp_right > 0:
        fade_out = [(ramp_right + 1 - i) / (ramp_right + 1) for i in range(1, ramp_right + 1)]
        for i in range(ramp_right):
            mask[length - ramp_right + i] *= fade_out[i]
    return mx.clip(mx.array(mask), 0, 1)


@dataclass(frozen=True)
class SpatialTilingConfig:
    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0


@dataclass(frozen=True)
class TemporalTilingConfig:
    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0


@dataclass(frozen=True)
class TilingConfig:
    spatial_config: Optional[SpatialTilingConfig] = None
    temporal_config: Optional[TemporalTilingConfig] = None

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls(spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64), temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24))

    @classmethod
    def spatial_only(cls, tile_size: int = 512, overlap: int = 64) -> "TilingConfig":
        return cls(spatial_config=SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=overlap), temporal_config=None)

    @classmethod
    def temporal_only(cls, tile_size: int = 64, overlap: int = 24) -> "TilingConfig":
        return cls(spatial_config=None, temporal_config=TemporalTilingConfig(tile_size_in_frames=tile_size, tile_overlap_in_frames=overlap))

    @classmethod
    def aggressive(cls) -> "TilingConfig":
        return cls(spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64), temporal_config=TemporalTilingConfig(tile_size_in_frames=32, tile_overlap_in_frames=8))

    @classmethod
    def conservative(cls) -> "TilingConfig":
        return cls(spatial_config=SpatialTilingConfig(tile_size_in_pixels=768, tile_overlap_in_pixels=64), temporal_config=TemporalTilingConfig(tile_size_in_frames=96, tile_overlap_in_frames=24))

    @classmethod
    def auto(cls, height: int, width: int, num_frames: int, spatial_threshold: int = 512, temporal_threshold: int = 65) -> Optional["TilingConfig"]:
        needs_spatial = height > spatial_threshold or width > spatial_threshold
        needs_temporal = num_frames > temporal_threshold
        if not needs_spatial and not needs_temporal:
            return None
        estimated_output_gb = (3 * num_frames * height * width * 4) / (1024**3)
        if estimated_output_gb > 2.0 or (height * width > 768 * 1024 and num_frames > 100):
            return cls.aggressive()
        spatial_config = None
        temporal_config = None
        if needs_spatial:
            max_dim = max(height, width)
            if max_dim > 1024:
                tile_size = 384
            elif max_dim > 768:
                tile_size = 512
            else:
                tile_size = 384
            spatial_config = SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=64)
        if needs_temporal:
            if num_frames > 200:
                tile_size, overlap = 32, 8
            elif num_frames > 100:
                tile_size, overlap = 48, 16
            else:
                tile_size, overlap = 64, 24
            temporal_config = TemporalTilingConfig(tile_size_in_frames=tile_size, tile_overlap_in_frames=overlap)
        return cls(spatial_config=spatial_config, temporal_config=temporal_config)


@dataclass
class DimensionIntervals:
    starts: List[int]
    ends: List[int]
    left_ramps: List[int]
    right_ramps: List[int]


def split_in_spatial(size: int, overlap: int, dimension_size: int) -> DimensionIntervals:
    if dimension_size <= size:
        return DimensionIntervals(starts=[0], ends=[dimension_size], left_ramps=[0], right_ramps=[0])
    amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
    starts = [i * (size - overlap) for i in range(amount)]
    ends = [start + size for start in starts]
    ends[-1] = dimension_size
    left_ramps = [0] + [overlap] * (amount - 1)
    right_ramps = [overlap] * (amount - 1) + [0]
    return DimensionIntervals(starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps)


def split_in_temporal(size: int, overlap: int, dimension_size: int) -> DimensionIntervals:
    if dimension_size <= size:
        return DimensionIntervals(starts=[0], ends=[dimension_size], left_ramps=[0], right_ramps=[0])
    intervals = split_in_spatial(size, overlap, dimension_size)
    starts = intervals.starts.copy()
    left_ramps = intervals.left_ramps.copy()
    for i in range(1, len(starts)):
        starts[i] = starts[i] - 1
        left_ramps[i] = left_ramps[i] + 1
    return DimensionIntervals(starts=starts, ends=intervals.ends, left_ramps=left_ramps, right_ramps=intervals.right_ramps)


def map_temporal_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, mx.array]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp_scaled = 1 + (left_ramp - 1) * scale if left_ramp > 0 else 0
    right_ramp_scaled = right_ramp * scale
    mask = compute_trapezoidal_mask_1d(stop - start, left_ramp_scaled, right_ramp_scaled, True)
    return slice(start, stop), mask


def map_spatial_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, mx.array]:
    start = begin * scale
    stop = end * scale
    left_ramp_scaled = left_ramp * scale
    right_ramp_scaled = right_ramp * scale
    mask = compute_trapezoidal_mask_1d(stop - start, left_ramp_scaled, right_ramp_scaled, False)
    return slice(start, stop), mask


def decode_with_tiling(decoder_fn, latents: mx.array, tiling_config: TilingConfig, spatial_scale: int = 32, temporal_scale: int = 8, causal: bool = False, timestep: Optional[mx.array] = None, chunked_conv: bool = False, on_frames_ready: Optional[Callable[[mx.array, int], None]] = None) -> mx.array:
    b, c, f_latent, h_latent, w_latent = latents.shape
    out_f = 1 + (f_latent - 1) * temporal_scale
    out_h = h_latent * spatial_scale
    out_w = w_latent * spatial_scale
    if tiling_config.spatial_config is not None:
        s_cfg = tiling_config.spatial_config
        spatial_tile_size = s_cfg.tile_size_in_pixels // spatial_scale
        spatial_overlap = s_cfg.tile_overlap_in_pixels // spatial_scale
    else:
        spatial_tile_size = max(h_latent, w_latent)
        spatial_overlap = 0
    if tiling_config.temporal_config is not None:
        t_cfg = tiling_config.temporal_config
        temporal_tile_size = t_cfg.tile_size_in_frames // temporal_scale
        temporal_overlap = t_cfg.tile_overlap_in_frames // temporal_scale
    else:
        temporal_tile_size = f_latent
        temporal_overlap = 0
    temporal_intervals = split_in_temporal(temporal_tile_size, temporal_overlap, f_latent)
    height_intervals = split_in_spatial(spatial_tile_size, spatial_overlap, h_latent)
    width_intervals = split_in_spatial(spatial_tile_size, spatial_overlap, w_latent)
    output = mx.zeros((b, 3, out_f, out_h, out_w), dtype=mx.float32)
    weights = mx.zeros((b, 1, out_f, out_h, out_w), dtype=mx.float32)
    mx.eval(output, weights)
    for t_idx in range(len(temporal_intervals.starts)):
        t_start = temporal_intervals.starts[t_idx]
        t_end = temporal_intervals.ends[t_idx]
        t_left = temporal_intervals.left_ramps[t_idx]
        t_right = temporal_intervals.right_ramps[t_idx]
        out_t_slice, t_mask = map_temporal_slice(t_start, t_end, t_left, t_right, temporal_scale)
        for h_idx in range(len(height_intervals.starts)):
            h_start = height_intervals.starts[h_idx]
            h_end = height_intervals.ends[h_idx]
            h_left = height_intervals.left_ramps[h_idx]
            h_right = height_intervals.right_ramps[h_idx]
            out_h_slice, h_mask = map_spatial_slice(h_start, h_end, h_left, h_right, spatial_scale)
            for w_idx in range(len(width_intervals.starts)):
                w_start = width_intervals.starts[w_idx]
                w_end = width_intervals.ends[w_idx]
                w_left = width_intervals.left_ramps[w_idx]
                w_right = width_intervals.right_ramps[w_idx]
                out_w_slice, w_mask = map_spatial_slice(w_start, w_end, w_left, w_right, spatial_scale)
                tile_latents = latents[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                tile_output = decoder_fn(tile_latents, causal=causal, timestep=timestep, debug=False, chunked_conv=chunked_conv)
                mx.eval(tile_output)
                del tile_latents
                _, _, decoded_t, decoded_h, decoded_w = tile_output.shape
                expected_t = out_t_slice.stop - out_t_slice.start
                expected_h = out_h_slice.stop - out_h_slice.start
                expected_w = out_w_slice.stop - out_w_slice.start
                actual_t = min(decoded_t, expected_t)
                actual_h = min(decoded_h, expected_h)
                actual_w = min(decoded_w, expected_w)
                t_mask_slice = t_mask[:actual_t] if len(t_mask) > actual_t else t_mask
                h_mask_slice = h_mask[:actual_h] if len(h_mask) > actual_h else h_mask
                w_mask_slice = w_mask[:actual_w] if len(w_mask) > actual_w else w_mask
                blend_mask = (t_mask_slice.reshape(1, 1, -1, 1, 1) * h_mask_slice.reshape(1, 1, 1, -1, 1) * w_mask_slice.reshape(1, 1, 1, 1, -1))
                tile_output_slice = tile_output[:, :, :actual_t, :actual_h, :actual_w].astype(mx.float32)
                del tile_output
                t_out_start = out_t_slice.start
                t_out_end = t_out_start + actual_t
                h_out_start = out_h_slice.start
                h_out_end = h_out_start + actual_h
                w_out_start = out_w_slice.start
                w_out_end = w_out_start + actual_w
                weighted_tile = tile_output_slice * blend_mask
                output[:, :, t_out_start:t_out_end, h_out_start:h_out_end, w_out_start:w_out_end] = (output[:, :, t_out_start:t_out_end, h_out_start:h_out_end, w_out_start:w_out_end] + weighted_tile)
                weights[:, :, t_out_start:t_out_end, h_out_start:h_out_end, w_out_start:w_out_end] = (weights[:, :, t_out_start:t_out_end, h_out_start:h_out_end, w_out_start:w_out_end] + blend_mask)
                mx.eval(output, weights)
                del tile_output_slice, weighted_tile, blend_mask
                gc.collect()
    weights = mx.maximum(weights, 1e-8)
    output = output / weights
    mx.eval(output)
    del weights
    gc.collect()
    return output.astype(latents.dtype)
