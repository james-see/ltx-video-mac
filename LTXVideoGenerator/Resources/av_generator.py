#!/usr/bin/env python3
"""Unified Audio-Video Generator using mlx-video-with-audio.

This script wraps the mlx_video.generate_av module for generating
video with synchronized audio in a single pass.
"""

import sys
import os
import argparse
import json


def status_output(message: str):
    """Output status message for Swift to parse."""
    print(f"STATUS:{message}", file=sys.stderr)
    sys.stderr.flush()


def progress_output(stage: int, step: int, total_steps: int, message: str = ""):
    """Output progress for Swift to parse."""
    print(f"STAGE:{stage}:STEP:{step}:{total_steps}:{message}", file=sys.stderr)
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2 Unified Audio-Video Generation with MLX"
    )
    parser.add_argument(
        "--prompt", "-p", type=str, required=True, help="Text prompt for generation"
    )
    parser.add_argument(
        "--height",
        "-H",
        type=int,
        default=512,
        help="Output video height (must be divisible by 64)",
    )
    parser.add_argument(
        "--width",
        "-W",
        type=int,
        default=512,
        help="Output video width (must be divisible by 64)",
    )
    parser.add_argument(
        "--num-frames",
        "-n",
        type=int,
        default=65,
        help="Number of frames (should be 8n+1: 9,17,25,33,41,49,57,65,73,81,89,97)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument(
        "--output-path", "-o", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="notapalindrome/ltx2-mlx-av",
        help="Model repository ID",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Input image for image-to-video generation",
    )
    parser.add_argument(
        "--image-strength",
        type=float,
        default=1.0,
        help="Image conditioning strength (0.0-1.0)",
    )
    parser.add_argument(
        "--tiling",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "default",
            "aggressive",
            "conservative",
            "spatial",
            "temporal",
        ],
        help="Tiling mode for VAE decoding",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        default=False,
        help="Disable audio generation (video only)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Gemma prompt enhancement repetition penalty (1.0-2.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Gemma prompt enhancement top-p sampling (0.0-1.0)",
    )

    args = parser.parse_args()

    try:
        status_output("Loading unified audio-video model...")

        # Import the mlx-video-with-audio package
        from mlx_video.generate_av import generate_av

        is_i2v = args.image is not None
        mode_str = "I2V" if is_i2v else "T2V"
        status_output(
            f"Starting {mode_str} generation with audio: {args.width}x{args.height}, {args.num_frames} frames"
        )

        # Build generation kwargs
        gen_kwargs = {
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "model_repo": args.model_repo,
            "output": args.output_path,
        }

        # Add image conditioning if provided
        if args.image:
            gen_kwargs["image"] = args.image
            gen_kwargs["image_strength"] = args.image_strength
            status_output(
                f"Using source image: {args.image} (strength={args.image_strength})"
            )

        # Pass tiling mode if supported
        if args.tiling != "auto":
            gen_kwargs["tiling"] = args.tiling

        # Disable audio if requested
        if args.no_audio:
            gen_kwargs["no_audio"] = True
            status_output("Audio generation disabled")

        # Pass Gemma prompt enhancement params if non-default
        if args.repetition_penalty != 1.2:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty
        if args.top_p != 0.9:
            gen_kwargs["top_p"] = args.top_p

        audio_label = "without" if args.no_audio else "with synchronized"
        status_output(f"Generating video {audio_label} audio...")
        generate_av(**gen_kwargs)

        status_output(f"Video with audio saved to: {args.output_path}")
        print("SUCCESS", file=sys.stderr)

        # Output JSON result for Swift to parse
        result = {
            "video_path": args.output_path,
            "seed": args.seed,
            "mode": "i2v" if is_i2v else "t2v",
            "has_audio": not args.no_audio,
        }
        print(json.dumps(result))

    except ImportError as e:
        error_msg = f"mlx-video-with-audio not installed: {e}. Run: pip install git+https://github.com/james-see/mlx-video-with-audio.git"
        status_output(f"ERROR: {error_msg}")
        print(json.dumps({"success": False, "error": error_msg}))
        sys.exit(1)
    except Exception as e:
        import traceback

        error_msg = f"{e}\n{traceback.format_exc()}"
        status_output(f"ERROR: {error_msg}")
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
