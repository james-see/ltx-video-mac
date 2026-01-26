#!/usr/bin/env python3
"""
LTX Video Generator - Python Helper Script for Apple Silicon

This script provides the core video generation functionality using the LTX-2 model
on Apple Silicon Macs (M1/M2/M3/M4) via MPS (Metal Performance Shaders).
It can be used standalone for testing or invoked by the Swift app.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video


# Available model variants
MODEL_VARIANTS = {
    "full": {
        "subfolder": "ltx-2-19b-dev",
        "description": "Full 19B model - best quality",
        "recommended_steps": 40,
        "recommended_guidance": 4.0,
    },
    "distilled": {
        "subfolder": "ltx-2-19b-distilled",
        "description": "Distilled model - 8 steps, much faster",
        "recommended_steps": 8,
        "recommended_guidance": 1.0,  # CFG must be 1.0 for distilled
    },
    "fp8": {
        "subfolder": "ltx-2-19b-dev-fp8",
        "description": "FP8 quantized - lower memory usage",
        "recommended_steps": 40,
        "recommended_guidance": 4.0,
    },
}


class LTXGenerator:
    """LTX-2 Video Generator wrapper for Apple Silicon."""

    def __init__(self, dtype: str = "float16", model_variant: str = "full"):
        self.device = "mps"
        self.dtype = dtype
        self.model_variant = model_variant
        self.pipe = None

    def load_model(self) -> None:
        """Load the LTX-2 model on Apple Silicon (MPS)."""
        variant_info = MODEL_VARIANTS.get(self.model_variant, MODEL_VARIANTS["full"])
        subfolder = variant_info["subfolder"]

        print(
            f"Loading LTX-2 model ({self.model_variant}) on Apple Silicon...",
            file=sys.stderr,
        )
        print(f"  Variant: {variant_info['description']}", file=sys.stderr)

        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

        # MPS FIX: Force float32 as default - MPS doesn't support float64
        torch.set_default_dtype(torch.float32)

        # device_map=None prevents automatic CPU offloading - we want pure MPS
        self.pipe = LTX2Pipeline.from_pretrained(
            "Lightricks/LTX-2",
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            device_map=None,
        )

        # MPS FIX: Disable double_precision on RoPE modules - MPS doesn't support float64
        if hasattr(self.pipe, "transformer") and hasattr(
            self.pipe.transformer, "double_precision"
        ):
            self.pipe.transformer.double_precision = False
        if hasattr(self.pipe, "connectors"):
            for name, module in self.pipe.connectors.named_modules():
                if hasattr(module, "double_precision"):
                    module.double_precision = False

        self.pipe.to("mps")

        # Verify pipeline is on MPS
        print(f"Pipeline device: {self.pipe.device}", file=sys.stderr)
        print(f"Transformer device: {self.pipe.transformer.device}", file=sys.stderr)
        print(f"Model loaded with {self.dtype} on MPS!", file=sys.stderr)

    def generate(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = "",
        source_image_path: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        width: int = 768,
        height: int = 512,
        num_frames: int = 121,
        fps: int = 24,
        seed: int | None = None,
    ) -> dict:
        """
        Generate a video from a text prompt, optionally starting from a source image.

        Args:
            prompt: Text description of the video to generate
            output_path: Where to save the output video
            negative_prompt: What to avoid in the generation
            source_image_path: Optional path to source image for image-to-video
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            width: Video width in pixels
            height: Video height in pixels
            num_frames: Number of frames to generate
            fps: Frames per second
            seed: Random seed for reproducibility

        Returns:
            dict with 'video_path', 'seed', and 'mode' keys
        """
        if self.pipe is None:
            self.load_model()

        # Load source image if provided (image-to-video mode)
        source_image = None
        if source_image_path:
            from PIL import Image

            source_image = Image.open(source_image_path).convert("RGB")
            print(
                f"  Source image: {source_image_path} ({source_image.size[0]}x{source_image.size[1]})",
                file=sys.stderr,
            )

        # Get variant-specific defaults
        variant_info = MODEL_VARIANTS.get(self.model_variant, MODEL_VARIANTS["full"])
        is_distilled = self.model_variant == "distilled"

        # Use variant defaults if not specified
        if num_inference_steps is None:
            num_inference_steps = variant_info["recommended_steps"]
        if guidance_scale is None:
            guidance_scale = variant_info["recommended_guidance"]

        # Distilled model requires CFG=1 and no negative prompt
        if is_distilled:
            guidance_scale = 1.0
            negative_prompt = ""

        # Set up generator with seed
        if seed is None:
            seed = torch.randint(0, 2**31, (1,)).item()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        mode = "image-to-video" if source_image else "text-to-video"
        print(f"Generating video with seed {seed}...", file=sys.stderr)
        print(f"  Mode: {mode}", file=sys.stderr)
        print(f"  Model: {self.model_variant}", file=sys.stderr)
        print(f"  Prompt: {prompt[:100]}...", file=sys.stderr)
        print(f"  Size: {width}x{height}, {num_frames} frames", file=sys.stderr)
        print(
            f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}",
            file=sys.stderr,
        )

        # Build pipeline arguments - only include image for image-to-video mode
        pipe_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": float(fps),
            "generator": generator,
            "output_type": "np",
            "return_dict": False,
        }

        # Add optional parameters only when needed
        if negative_prompt and not is_distilled:
            pipe_kwargs["negative_prompt"] = negative_prompt
        if source_image is not None:
            pipe_kwargs["image"] = source_image

        # Generate video and audio with LTX-2
        video, audio = self.pipe(**pipe_kwargs)

        # Convert video to tensor format for encode_video
        video = (video * 255).round().astype("uint8")
        video_tensor = torch.from_numpy(video)

        # Export video with audio using LTX-2's encode_video utility
        encode_video(
            video_tensor[0],
            fps=fps,
            audio=audio[0].float().cpu(),
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
            output_path=output_path,
        )

        print(f"Video saved to: {output_path}", file=sys.stderr)

        return {
            "video_path": output_path,
            "seed": seed,
            "mode": mode,
        }

    def unload_model(self) -> None:
        """Unload model and free memory."""
        self.pipe = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2 Video Generator for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Variants:
  full      - Full 19B model, best quality (default)
  distilled - Distilled model, 8 steps, much faster
  fp8       - FP8 quantized, lower memory usage

Examples:
  Text-to-video:
    %(prog)s "a cat playing piano" -o cat.mp4
    %(prog)s "ocean waves" --model distilled --steps 8
  
  Image-to-video:
    %(prog)s "person turns head and smiles" --image photo.jpg -o animated.mp4
    %(prog)s "waves crashing" --image beach.png --frames 161
""",
    )
    parser.add_argument("prompt", help="Text prompt for video generation")
    parser.add_argument(
        "-o", "--output", default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "-i", "--image", default=None, help="Source image for image-to-video mode"
    )
    parser.add_argument("-n", "--negative-prompt", default="", help="Negative prompt")
    parser.add_argument(
        "--model",
        type=str,
        default="full",
        choices=["full", "distilled", "fp8"],
        help="Model variant: full, distilled, fp8",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps (default: varies by model)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Guidance scale (default: varies by model)",
    )
    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--dtype", type=str, default="float16", help="Data type: float16, float32"
    )
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    generator = LTXGenerator(dtype=args.dtype, model_variant=args.model)

    result = generator.generate(
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        source_image_path=args.image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        fps=args.fps,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(result))
    else:
        print(f"\nGenerated: {result['video_path']}")
        print(f"Mode: {result['mode']}")
        print(f"Seed: {result['seed']}")


if __name__ == "__main__":
    main()
