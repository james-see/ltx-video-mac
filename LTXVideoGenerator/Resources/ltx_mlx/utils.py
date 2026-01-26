import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from functools import partial
from pathlib import Path
from huggingface_hub import snapshot_download, try_to_load_from_cache
from PIL import Image
import sys
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Custom tqdm that outputs progress for Swift GUI parsing."""

    def __init__(self, *args, **kwargs):
        # Filter out kwargs that tqdm doesn't understand
        # huggingface_hub passes 'name' which tqdm rejects
        kwargs.pop("name", None)
        kwargs["disable"] = False
        kwargs["file"] = sys.stderr
        self._last_reported = -1
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        # Output progress in parseable format
        total = getattr(self, "total", None)
        current = getattr(self, "n", 0)
        if total and total > 0:
            pct = int(100 * current / total)
            if pct != self._last_reported:
                self._last_reported = pct
                print(f"DOWNLOAD:PROGRESS:{current}:{total}:{pct}%", file=sys.stderr)
                sys.stderr.flush()

    def close(self):
        total = getattr(self, "total", None)
        current = getattr(self, "n", 0)
        if total and current >= total:
            print(f"DOWNLOAD:PROGRESS:{total}:{total}:100%", file=sys.stderr)
            sys.stderr.flush()
        super().close()


def get_model_path(model_repo: str):
    """Get or download model from HuggingFace Hub."""
    from huggingface_hub.constants import HF_HUB_CACHE
    import os

    # Debug: Show cache info
    cache_dir = Path(HF_HUB_CACHE)
    print(f"DEBUG:CACHE_DIR:{cache_dir}", file=sys.stderr)
    print(f"DEBUG:LOOKING_FOR:{model_repo}", file=sys.stderr)

    # List what's in the cache
    if cache_dir.exists():
        cached_repos = []
        for item in cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                repo_name = item.name.replace("models--", "").replace("--", "/")
                cached_repos.append(repo_name)
        print(f"DEBUG:CACHED_REPOS:{cached_repos}", file=sys.stderr)
    else:
        print(f"DEBUG:CACHE_DIR_NOT_FOUND", file=sys.stderr)

    # Required files for LTX-2 distilled model
    required_files = [
        "ltx-2-19b-distilled.safetensors",
        "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    ]

    # Check if model is already cached by looking for required files
    cache_hit = True
    for req_file in required_files:
        cached = try_to_load_from_cache(repo_id=model_repo, filename=req_file)
        status = "FOUND" if cached else "MISSING"
        print(f"DEBUG:FILE:{req_file}:{status}", file=sys.stderr)
        if cached is None:
            cache_hit = False

    sys.stderr.flush()

    if cache_hit:
        # All required files are cached, get the path
        path = Path(snapshot_download(repo_id=model_repo, local_files_only=True))
        print(f"MODEL:CACHED:{model_repo}", file=sys.stderr)
        print(f"DEBUG:MODEL_PATH:{path}", file=sys.stderr)
        sys.stderr.flush()
        return path

    # Need to download - show actual repo being downloaded
    print(f"DOWNLOAD:START:{model_repo}", file=sys.stderr)
    print(f"Downloading {model_repo}...", file=sys.stderr)
    sys.stderr.flush()

    path = Path(
        snapshot_download(
            repo_id=model_repo,
            local_files_only=False,
            allow_patterns=["*.safetensors", "*.json"],
            tqdm_class=DownloadProgressBar,
        )
    )

    # Verify download completed successfully
    missing_files = [f for f in required_files if not (path / f).exists()]
    if missing_files:
        msg = f"ERROR: Download incomplete. Missing: {missing_files}"
        print(msg, file=sys.stderr)
        sys.stderr.flush()
        raise FileNotFoundError(f"Download failed - missing: {missing_files}")

    print(f"DOWNLOAD:COMPLETE:{model_repo}", file=sys.stderr)
    sys.stderr.flush()
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
    noisy: mx.array, velocity: mx.array, sigma: mx.array | float
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
        pil_image = pil_image.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
        image = mx.array(np.array(pil_image).astype(np.float32) / 255.0)
    image = image * 2.0 - 1.0
    image = mx.transpose(image, (2, 0, 1))
    image = mx.expand_dims(image, axis=0)
    image = mx.expand_dims(image, axis=2)
    return image.astype(dtype)
