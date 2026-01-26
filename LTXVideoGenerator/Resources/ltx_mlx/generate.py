"""LTX-2 Video Generation with MLX - with progress output for GUI integration."""
import sys
import time
from pathlib import Path
from typing import Optional
import mlx.core as mx
import numpy as np
from tqdm import tqdm

from ltx_mlx.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType
from ltx_mlx.models.ltx.ltx import LTXModel
from ltx_mlx.models.ltx.transformer import Modality
from ltx_mlx.convert import sanitize_transformer_weights
from ltx_mlx.utils import to_denoised, load_image, prepare_image_for_encoding, get_model_path
from ltx_mlx.models.ltx.video_vae.decoder import load_vae_decoder
from ltx_mlx.models.ltx.video_vae.encoder import load_vae_encoder
from ltx_mlx.models.ltx.video_vae.tiling import TilingConfig
from ltx_mlx.models.ltx.upsampler import load_upsampler, upsample_latents
from ltx_mlx.conditioning import VideoConditionByLatentIndex, apply_conditioning
from ltx_mlx.conditioning.latent import LatentState


# Distilled sigma schedules
STAGE_1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


def progress_output(stage: int, step: int, total_steps: int, message: str = ""):
    """Output progress for Swift to parse."""
    print(f"STAGE:{stage}:STEP:{step}:{total_steps}:{message}", file=sys.stderr)
    sys.stderr.flush()


def status_output(message: str):
    """Output status message for Swift to parse."""
    print(f"STATUS:{message}", file=sys.stderr)
    sys.stderr.flush()


def create_position_grid(batch_size: int, num_frames: int, height: int, width: int, temporal_scale: int = 8, spatial_scale: int = 32, fps: float = 24.0, causal_fix: bool = True) -> mx.array:
    """Create position grid for RoPE in pixel space."""
    patch_size_t, patch_size_h, patch_size_w = 1, 1, 1
    t_coords = np.arange(0, num_frames, patch_size_t)
    h_coords = np.arange(0, height, patch_size_h)
    w_coords = np.arange(0, width, patch_size_w)
    t_grid, h_grid, w_grid = np.meshgrid(t_coords, h_coords, w_coords, indexing='ij')
    patch_starts = np.stack([t_grid, h_grid, w_grid], axis=0)
    patch_size_delta = np.array([patch_size_t, patch_size_h, patch_size_w]).reshape(3, 1, 1, 1)
    patch_ends = patch_starts + patch_size_delta
    latent_coords = np.stack([patch_starts, patch_ends], axis=-1)
    num_patches = num_frames * height * width
    latent_coords = latent_coords.reshape(3, num_patches, 2)
    latent_coords = np.tile(latent_coords[np.newaxis, ...], (batch_size, 1, 1, 1))
    scale_factors = np.array([temporal_scale, spatial_scale, spatial_scale]).reshape(1, 3, 1, 1)
    pixel_coords = (latent_coords * scale_factors).astype(np.float32)
    if causal_fix:
        pixel_coords[:, 0, :, :] = np.clip(pixel_coords[:, 0, :, :] + 1 - temporal_scale, a_min=0, a_max=None)
    pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps
    return mx.array(pixel_coords, dtype=mx.float32)


def denoise(latents: mx.array, positions: mx.array, text_embeddings: mx.array, transformer: LTXModel, sigmas: list, stage: int = 1, verbose: bool = True, state: Optional[LatentState] = None) -> mx.array:
    """Run denoising loop with progress output."""
    dtype = latents.dtype
    if state is not None:
        latents = state.latent
    total_steps = len(sigmas) - 1
    
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        # Output progress
        progress_output(stage, i + 1, total_steps, f"Denoising step {i+1}/{total_steps}")
        
        b, c, f, h, w = latents.shape
        num_tokens = f * h * w
        latents_flat = mx.transpose(mx.reshape(latents, (b, c, -1)), (0, 2, 1))
        
        if state is not None:
            denoise_mask_flat = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
            denoise_mask_flat = mx.broadcast_to(denoise_mask_flat, (b, 1, f, h, w))
            denoise_mask_flat = mx.reshape(denoise_mask_flat, (b, num_tokens))
            timesteps = mx.array(sigma, dtype=dtype) * denoise_mask_flat
        else:
            timesteps = mx.full((b, num_tokens), sigma, dtype=dtype)
        
        video_modality = Modality(latent=latents_flat, timesteps=timesteps, positions=positions, context=text_embeddings, context_mask=None, enabled=True)
        velocity, _ = transformer(video=video_modality, audio=None)
        mx.eval(velocity)
        velocity = mx.reshape(mx.transpose(velocity, (0, 2, 1)), (b, c, f, h, w))
        denoised = to_denoised(latents, velocity, sigma)
        
        if state is not None:
            from ltx_mlx.conditioning.latent import apply_denoise_mask
            denoised = apply_denoise_mask(denoised, state.clean_latent, state.denoise_mask)
        mx.eval(denoised)
        
        if sigma_next > 0:
            sigma_next_arr = mx.array(sigma_next, dtype=dtype)
            sigma_arr = mx.array(sigma, dtype=dtype)
            latents = denoised + sigma_next_arr * (latents - denoised) / sigma_arr
        else:
            latents = denoised
        mx.eval(latents)
    
    return latents


def generate_video(
    model_repo: str = "Lightricks/LTX-2",
    text_encoder_repo: str = None,
    prompt: str = "",
    height: int = 512,
    width: int = 512,
    num_frames: int = 33,
    seed: int = 42,
    fps: int = 24,
    output_path: str = "output.mp4",
    verbose: bool = True,
    image: Optional[str] = None,
    image_strength: float = 1.0,
    image_frame_idx: int = 0,
    tiling: str = "auto",
):
    """Generate video from text prompt."""
    start_time = time.time()
    
    # Validate dimensions
    assert height % 64 == 0, f"Height must be divisible by 64, got {height}"
    assert width % 64 == 0, f"Width must be divisible by 64, got {width}"
    
    if num_frames % 8 != 1:
        adjusted_num_frames = round((num_frames - 1) / 8) * 8 + 1
        num_frames = adjusted_num_frames
    
    is_i2v = image is not None
    mode_str = "I2V" if is_i2v else "T2V"
    status_output(f"Starting {mode_str} generation: {width}x{height}, {num_frames} frames")
    
    # Get model path
    model_path = get_model_path(model_repo)
    text_encoder_path = model_path if text_encoder_repo is None else get_model_path(text_encoder_repo)
    
    # Calculate latent dimensions
    stage1_h, stage1_w = height // 2 // 32, width // 2 // 32
    stage2_h, stage2_w = height // 32, width // 32
    latent_frames = 1 + (num_frames - 1) // 8
    
    mx.random.seed(seed)
    
    # Load text encoder
    status_output("Loading text encoder...")
    from ltx_mlx.models.ltx.text_encoder import LTX2TextEncoder
    text_encoder = LTX2TextEncoder()
    text_encoder.load(model_path=model_path, text_encoder_path=text_encoder_path)
    mx.eval(text_encoder.parameters())
    
    status_output("Encoding prompt...")
    text_embeddings, _ = text_encoder(prompt, return_audio_embeddings=False)
    model_dtype = text_embeddings.dtype
    mx.eval(text_embeddings)
    
    del text_encoder
    mx.clear_cache()
    
    # Load transformer
    status_output("Loading transformer model...")
    raw_weights = mx.load(str(model_path / 'ltx-2-19b-distilled.safetensors'))
    sanitized = sanitize_transformer_weights(raw_weights)
    sanitized = {k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v for k, v in sanitized.items()}
    
    config = LTXModelConfig(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=48,
        cross_attention_dim=4096,
        caption_channels=3840,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1000,
    )
    
    transformer = LTXModel(config)
    transformer.load_weights(list(sanitized.items()), strict=False)
    mx.eval(transformer.parameters())
    
    # Load VAE encoder if I2V
    stage1_image_latent = None
    stage2_image_latent = None
    if is_i2v:
        status_output("Loading VAE encoder...")
        vae_encoder = load_vae_encoder(str(model_path / 'ltx-2-19b-distilled.safetensors'))
        mx.eval(vae_encoder.parameters())
        
        input_image = load_image(image, height=height // 2, width=width // 2, dtype=model_dtype)
        stage1_image_tensor = prepare_image_for_encoding(input_image, height // 2, width // 2, dtype=model_dtype)
        stage1_image_latent = vae_encoder(stage1_image_tensor)
        mx.eval(stage1_image_latent)
        
        input_image = load_image(image, height=height, width=width, dtype=model_dtype)
        stage2_image_tensor = prepare_image_for_encoding(input_image, height, width, dtype=model_dtype)
        stage2_image_latent = vae_encoder(stage2_image_tensor)
        mx.eval(stage2_image_latent)
        
        del vae_encoder
        mx.clear_cache()
    
    # Stage 1: Generate at half resolution
    status_output(f"Stage 1: Generating at {width//2}x{height//2}...")
    mx.random.seed(seed)
    
    positions = create_position_grid(1, latent_frames, stage1_h, stage1_w)
    mx.eval(positions)
    
    state1 = None
    if is_i2v and stage1_image_latent is not None:
        latent_shape = (1, 128, latent_frames, stage1_h, stage1_w)
        state1 = LatentState(
            latent=mx.zeros(latent_shape, dtype=model_dtype),
            clean_latent=mx.zeros(latent_shape, dtype=model_dtype),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
        )
        conditioning = VideoConditionByLatentIndex(latent=stage1_image_latent, frame_idx=image_frame_idx, strength=image_strength)
        state1 = apply_conditioning(state1, [conditioning])
        noise = mx.random.normal(latent_shape, dtype=model_dtype)
        noise_scale = mx.array(STAGE_1_SIGMAS[0], dtype=model_dtype)
        scaled_mask = state1.denoise_mask * noise_scale
        state1 = LatentState(
            latent=noise * scaled_mask + state1.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
            clean_latent=state1.clean_latent,
            denoise_mask=state1.denoise_mask,
        )
        latents = state1.latent
        mx.eval(latents)
    else:
        latents = mx.random.normal((1, 128, latent_frames, stage1_h, stage1_w), dtype=model_dtype)
        mx.eval(latents)
    
    latents = denoise(latents, positions, text_embeddings, transformer, STAGE_1_SIGMAS, stage=1, verbose=verbose, state=state1)
    
    # Upsample latents
    status_output("Upsampling latents 2x...")
    upsampler = load_upsampler(str(model_path / 'ltx-2-spatial-upscaler-x2-1.0.safetensors'))
    mx.eval(upsampler.parameters())
    
    vae_decoder = load_vae_decoder(str(model_path / 'ltx-2-19b-distilled.safetensors'), timestep_conditioning=None)
    
    latents = upsample_latents(latents, upsampler, vae_decoder.latents_mean, vae_decoder.latents_std)
    mx.eval(latents)
    
    del upsampler
    mx.clear_cache()
    
    # Stage 2: Refine at full resolution
    status_output(f"Stage 2: Refining at {width}x{height}...")
    positions = create_position_grid(1, latent_frames, stage2_h, stage2_w)
    mx.eval(positions)
    
    state2 = None
    if is_i2v and stage2_image_latent is not None:
        state2 = LatentState(
            latent=latents,
            clean_latent=mx.zeros_like(latents),
            denoise_mask=mx.ones((1, 1, latent_frames, 1, 1), dtype=model_dtype),
        )
        conditioning = VideoConditionByLatentIndex(latent=stage2_image_latent, frame_idx=image_frame_idx, strength=image_strength)
        state2 = apply_conditioning(state2, [conditioning])
        noise = mx.random.normal(latents.shape).astype(model_dtype)
        noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
        scaled_mask = state2.denoise_mask * noise_scale
        state2 = LatentState(
            latent=noise * scaled_mask + state2.latent * (mx.array(1.0, dtype=model_dtype) - scaled_mask),
            clean_latent=state2.clean_latent,
            denoise_mask=state2.denoise_mask,
        )
        latents = state2.latent
        mx.eval(latents)
    else:
        noise_scale = mx.array(STAGE_2_SIGMAS[0], dtype=model_dtype)
        one_minus_scale = mx.array(1.0 - STAGE_2_SIGMAS[0], dtype=model_dtype)
        noise = mx.random.normal(latents.shape).astype(model_dtype)
        latents = noise * noise_scale + latents * one_minus_scale
        mx.eval(latents)
    
    latents = denoise(latents, positions, text_embeddings, transformer, STAGE_2_SIGMAS, stage=2, verbose=verbose, state=state2)
    
    del transformer
    mx.clear_cache()
    
    # Decode to video
    status_output("Decoding video...")
    
    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    elif tiling == "spatial":
        tiling_config = TilingConfig.spatial_only()
    elif tiling == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        tiling_config = TilingConfig.auto(height, width, num_frames)
    
    if tiling_config is not None:
        video = vae_decoder.decode_tiled(latents, tiling_config=tiling_config, tiling_mode=tiling)
    else:
        video = vae_decoder(latents)
    mx.eval(video)
    mx.clear_cache()
    
    # Convert to uint8 frames
    video = mx.squeeze(video, axis=0)
    video = mx.transpose(video, (1, 2, 3, 0))
    video = mx.clip((video + 1.0) / 2.0, 0.0, 1.0)
    video = (video * 255).astype(mx.uint8)
    video_np = np.array(video)
    
    # Save video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    status_output("Saving video...")
    try:
        import cv2
        h, w = video_np.shape[1], video_np.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for frame in video_np:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        status_output(f"Saved video to {output_path}")
    except Exception as e:
        status_output(f"Error saving video: {e}")
    
    elapsed = time.time() - start_time
    status_output(f"Done! Generated in {elapsed:.1f}s")
    
    return video_np


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate videos with MLX LTX-2")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt")
    parser.add_argument("--height", "-H", type=int, default=512, help="Output video height")
    parser.add_argument("--width", "-W", type=int, default=512, help="Output video width")
    parser.add_argument("--num-frames", "-n", type=int, default=33, help="Number of frames")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=24, help="FPS")
    parser.add_argument("--output-path", type=str, default="output.mp4", help="Output path")
    parser.add_argument("--model-repo", type=str, default="Lightricks/LTX-2", help="Model repo")
    parser.add_argument("--image", "-i", type=str, default=None, help="Conditioning image")
    parser.add_argument("--image-strength", type=float, default=1.0, help="Image conditioning strength")
    parser.add_argument("--tiling", type=str, default="auto", choices=["auto", "none", "default", "aggressive", "conservative", "spatial", "temporal"], help="Tiling mode")
    args = parser.parse_args()
    
    generate_video(**vars(args))
