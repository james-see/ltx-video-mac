import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from functools import partial
from pathlib import Path
from huggingface_hub import snapshot_download
from PIL import Image
import sys


def get_model_path(model_repo: str):
    """Get or download LTX-2 model path."""
    try:
        return Path(snapshot_download(repo_id=model_repo, local_files_only=True))
    except Exception:
        print("DOWNLOAD:START:" + model_repo, file=sys.stderr)
        print("Downloading LTX-2 model weights...", file=sys.stderr)
        path = Path(snapshot_download(
            repo_id=model_repo,
            local_files_only=False,
            resume_download=True,
            allow_patterns=["*.safetensors", "*.json"],
        ))
        print("DOWNLOAD:COMPLETE:" + model_repo, file=sys.stderr)
        return path


def apply_quantization(model: nn.Module, weights: mx.array, quantization: dict):
    if quantization is not None:
        def get_class_predicate(p, m):
            if p in quantization:
                return quantization[p]
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.shape[0] % 64 != 0:
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )


@partial(mx.compile, shapeless=True)
def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, mx.ones((x.shape[-1],), dtype=x.dtype), eps)


@partial(mx.compile, shapeless=True)
def to_denoised(
    noisy: mx.array,
    velocity: mx.array,
    sigma: mx.array | float
) -> mx.array:
    """Convert velocity prediction to denoised output."""
    if isinstance(sigma, (int, float)):
        sigma_arr = mx.array(sigma, dtype=velocity.dtype)
        return noisy - sigma_arr * velocity
    else:
        sigma = sigma.astype(velocity.dtype)
        while sigma.ndim < velocity.ndim:
            sigma = mx.expand_dims(sigma, axis=-1)
        return noisy - sigma * velocity


def repeat_interleave(x: mx.array, repeats: int, axis: int = -1) -> mx.array:
    """Repeat elements of tensor along an axis."""
    if axis < 0:
        axis = x.ndim + axis
    shape = list(x.shape)
    x = mx.expand_dims(x, axis=axis + 1)
    tile_pattern = [1] * x.ndim
    tile_pattern[axis + 1] = repeats
    x = mx.tile(x, tile_pattern)
    new_shape = shape.copy()
    new_shape[axis] *= repeats
    return mx.reshape(x, new_shape)


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x / mx.sqrt(mx.mean(x * x, axis=1, keepdims=True) + self.eps)


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """Create sinusoidal timestep embeddings."""
    assert timesteps.ndim == 1, "Timesteps should be 1D"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = (timesteps[:, None].astype(mx.float32) * scale) * emb[None, :]
    if flip_sin_to_cos:
        emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    else:
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


def load_image(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Load and preprocess an image for I2V conditioning."""
    image = Image.open(image_path).convert("RGB")
    if height is not None and width is not None:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    elif height is not None or width is not None:
        orig_w, orig_h = image.size
        if height is not None:
            scale = height / orig_h
            new_w = int(orig_w * scale)
            new_w = (new_w // 32) * 32
            image = image.resize((new_w, height), Image.Resampling.LANCZOS)
        else:
            scale = width / orig_w
            new_h = int(orig_h * scale)
            new_h = (new_h // 32) * 32
            image = image.resize((width, new_h), Image.Resampling.LANCZOS)
    else:
        orig_w, orig_h = image.size
        new_w = (orig_w // 32) * 32
        new_h = (orig_h // 32) * 32
        if new_w != orig_w or new_h != orig_h:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    image_np = np.array(image).astype(np.float32) / 255.0
    return mx.array(image_np, dtype=dtype)


def prepare_image_for_encoding(
    image: mx.array,
    target_height: int,
    target_width: int,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Prepare image for VAE encoding."""
    h, w = image.shape[:2]
    if h != target_height or w != target_width:
        image_np = np.array(image)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        image = mx.array(np.array(pil_image).astype(np.float32) / 255.0)
    image = image * 2.0 - 1.0
    image = mx.transpose(image, (2, 0, 1))
    image = mx.expand_dims(image, axis=0)
    image = mx.expand_dims(image, axis=2)
    return image.astype(dtype)
