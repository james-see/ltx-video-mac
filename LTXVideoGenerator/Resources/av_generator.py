#!/usr/bin/env python3
"""Unified Audio-Video Generator using mlx-video-with-audio.

Prefer: python -m mlx_video.generate_av (same flags). This script remains for
manual testing and documents the public API.
"""

import argparse
import json
import sys


def status_output(message: str):
    """Output status message for Swift to parse."""
    print(f"STATUS:{message}", file=sys.stderr)
    sys.stderr.flush()


def progress_output(stage: int, step: int, total_steps: int, message: str = ""):
    """Output progress for Swift to parse."""
    print(
        f"STAGE:{stage}:STEP:{step}:{total_steps}:{message}",
        file=sys.stderr,
    )
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(
        description="LTX-2 Unified Audio-Video Generation with MLX"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Text prompt for generation",
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
        help=(
            "Number of frames " "(should be 8n+1: 9,17,25,33,41,49,57,65,73,81,89,97)"
        ),
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second",
    )
    parser.add_argument(
        "--num-inference-steps",
        "--steps",
        dest="num_inference_steps",
        type=int,
        default=30,
        help="Total denoising steps across both stages",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt used for classifier-free guidance",
    )
    parser.add_argument(
        "--cfg-scale",
        "--guidance-scale",
        dest="cfg_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale (1.0 disables guidance)",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="output.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="notapalindrome/ltx2-mlx-av",
        help="Model repository ID (selected in app preferences)",
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
        help="Disable audio in final file (video-only MP4)",
    )

    args = parser.parse_args()

    try:
        status_output("Loading unified audio-video model...")

        from mlx_video.generate_av import generate_video_with_audio

        is_i2v = args.image is not None
        mode_str = "I2V" if is_i2v else "T2V"
        status_output(
            "Starting "
            f"{mode_str} generation with audio: "
            f"{args.width}x{args.height}, {args.num_frames} frames"
        )

        generate_video_with_audio(
            model_repo=args.model_repo,
            text_encoder_repo=None,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            fps=args.fps,
            output_path=args.output_path,
            negative_prompt=args.negative_prompt,
            cfg_scale=args.cfg_scale,
            image=args.image,
            image_strength=args.image_strength,
            tiling=args.tiling,
            num_inference_steps=args.num_inference_steps,
            verbose=True,
            no_audio=args.no_audio,
        )

        status_output(f"Video with audio saved to: {args.output_path}")
        print("SUCCESS", file=sys.stderr)

        result = {
            "video_path": args.output_path,
            "seed": args.seed,
            "mode": "i2v" if is_i2v else "t2v",
            "has_audio": not args.no_audio,
        }
        print(json.dumps(result))

    except ImportError as e:
        error_msg = (
            "mlx-video-with-audio not installed: "
            f"{e}. Run: pip install -U mlx-video-with-audio"
        )
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
