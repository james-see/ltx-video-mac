"""Weight conversion utilities for LTX-2 models."""
import json
from pathlib import Path
from typing import Any, Dict, Optional
import mlx.core as mx
from ltx_mlx.utils import get_model_path


def sanitize_transformer_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize transformer weight names from PyTorch LTX-2 format to MLX format."""
    sanitized = {}
    for key, value in weights.items():
        new_key = key
        if not key.startswith("model.diffusion_model."):
            continue
        new_key = key.replace("model.diffusion_model.", "")
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.proj_in.")
        new_key = new_key.replace(".ff.net.2.", ".ff.proj_out.")
        new_key = new_key.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
        new_key = new_key.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
        new_key = new_key.replace(".linear_1.", ".linear1.")
        new_key = new_key.replace(".linear_2.", ".linear2.")
        sanitized[new_key] = value
    return sanitized


def sanitize_vae_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE weight names from PyTorch format to MLX format."""
    sanitized = {}
    for key, value in weights.items():
        new_key = key
        if "position_ids" in key:
            continue
        if not key.startswith("vae."):
            continue
        if "vae.per_channel_statistics" in key:
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics.mean"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics.std"
            else:
                continue
        elif key.startswith("vae.decoder."):
            new_key = key.replace("vae.decoder.", "")
        else:
            continue
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))
        sanitized[new_key] = value
    return sanitized


def sanitize_vae_encoder_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Sanitize VAE encoder weight names from PyTorch format to MLX format."""
    sanitized = {}
    for key, value in weights.items():
        new_key = key
        if "position_ids" in key:
            continue
        if not key.startswith("vae."):
            continue
        if "vae.per_channel_statistics" in key:
            if key == "vae.per_channel_statistics.mean-of-means":
                new_key = "per_channel_statistics._mean_of_means"
            elif key == "vae.per_channel_statistics.std-of-means":
                new_key = "per_channel_statistics._std_of_means"
            else:
                continue
        elif key.startswith("vae.encoder."):
            new_key = key.replace("vae.encoder.", "")
        else:
            continue
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))
        if "conv" in new_key.lower() and "weight" in new_key and value.ndim == 4:
            value = mx.transpose(value, (0, 2, 3, 1))
        sanitized[new_key] = value
    return sanitized


def load_config(model_path: Path) -> Dict[str, Any]:
    """Load model configuration."""
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}
