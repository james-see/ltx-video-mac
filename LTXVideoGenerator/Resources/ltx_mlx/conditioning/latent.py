"""Latent-based conditioning for I2V (Image-to-Video) generation."""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import mlx.core as mx


@dataclass
class VideoConditionByLatentIndex:
    """Condition video generation by injecting latents at a specific frame index."""
    latent: mx.array
    frame_idx: int = 0
    strength: float = 1.0

    def get_num_latent_frames(self) -> int:
        return self.latent.shape[2]


@dataclass
class LatentState:
    """State for latent diffusion with conditioning support."""
    latent: mx.array
    clean_latent: mx.array
    denoise_mask: mx.array

    def clone(self) -> "LatentState":
        return LatentState(
            latent=self.latent,
            clean_latent=self.clean_latent,
            denoise_mask=self.denoise_mask,
        )


def create_initial_state(
    shape: Tuple[int, ...],
    seed: Optional[int] = None,
    noise_scale: float = 1.0,
) -> LatentState:
    if seed is not None:
        mx.random.seed(seed)
    noise = mx.random.normal(shape)
    return LatentState(
        latent=noise * noise_scale,
        clean_latent=mx.zeros(shape),
        denoise_mask=mx.ones((shape[0], 1, shape[2], 1, 1)),
    )


def apply_conditioning(
    state: LatentState,
    conditionings: List[VideoConditionByLatentIndex],
) -> LatentState:
    state = state.clone()
    dtype = state.latent.dtype
    b, c, f, h, w = state.latent.shape

    for cond in conditionings:
        cond_latent = cond.latent
        frame_idx = cond.frame_idx
        strength = cond.strength
        _, cond_c, cond_f, cond_h, cond_w = cond_latent.shape
        
        if (cond_c, cond_h, cond_w) != (c, h, w):
            raise ValueError(f"Shape mismatch: ({cond_c}, {cond_h}, {cond_w}) vs ({c}, {h}, {w})")
        if frame_idx >= f:
            raise ValueError(f"Frame index {frame_idx} out of bounds for {f} frames")

        num_cond_frames = cond_f
        end_idx = min(frame_idx + num_cond_frames, f)

        latent_list, clean_list, mask_list = [], [], []
        for i in range(f):
            if frame_idx <= i < end_idx:
                cond_idx = i - frame_idx
                latent_list.append(cond_latent[:, :, cond_idx:cond_idx+1])
                clean_list.append(cond_latent[:, :, cond_idx:cond_idx+1])
                mask_list.append(mx.full((b, 1, 1, 1, 1), 1.0 - strength, dtype=dtype))
            else:
                latent_list.append(state.latent[:, :, i:i+1])
                clean_list.append(state.clean_latent[:, :, i:i+1])
                mask_list.append(state.denoise_mask[:, :, i:i+1])

        state.latent = mx.concatenate(latent_list, axis=2)
        state.clean_latent = mx.concatenate(clean_list, axis=2)
        state.denoise_mask = mx.concatenate(mask_list, axis=2)

    return state


def apply_denoise_mask(denoised: mx.array, clean: mx.array, denoise_mask: mx.array) -> mx.array:
    one = mx.array(1.0, dtype=denoised.dtype)
    return denoised * denoise_mask + clean * (one - denoise_mask)


def add_noise_with_state(state: LatentState, noise_scale: float) -> LatentState:
    state = state.clone()
    noise = mx.random.normal(state.latent.shape)
    effective_scale = noise_scale * state.denoise_mask
    one = mx.array(1.0, dtype=state.latent.dtype)
    state.latent = noise * effective_scale + state.latent * (one - effective_scale)
    return state
