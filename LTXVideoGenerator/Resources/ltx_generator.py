#!/usr/bin/env python3
"""LTX Video Generator - MLX Backend Entry Point

This script wraps the bundled ltx_mlx module for video generation.
Progress output is sent to stderr for Swift GUI integration.
"""
import sys
import os
import argparse

# Add the Resources directory to path so ltx_mlx can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from ltx_mlx.generate import generate_video


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Video Generation with MLX")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--height", "-H", type=int, default=512, help="Output video height (must be divisible by 64)")
    parser.add_argument("--width", "-W", type=int, default=512, help="Output video width (must be divisible by 64)")
    parser.add_argument("--num-frames", "-n", type=int, default=33, help="Number of frames (should be 1 + 8*k)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--output-path", "-o", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--model-repo", type=str, default="Lightricks/LTX-2", help="Model repository ID")
    parser.add_argument("--image", "-i", type=str, default=None, help="Input image for image-to-video generation")
    parser.add_argument("--image-strength", type=float, default=1.0, help="Image conditioning strength (0.0-1.0)")
    parser.add_argument("--tiling", type=str, default="auto", 
                       choices=["auto", "none", "default", "aggressive", "conservative", "spatial", "temporal"],
                       help="Tiling mode for VAE decoding")
    
    args = parser.parse_args()
    
    try:
        generate_video(
            model_repo=args.model_repo,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            fps=args.fps,
            output_path=args.output_path,
            image=args.image,
            image_strength=args.image_strength,
            tiling=args.tiling,
        )
        print("SUCCESS", file=sys.stderr)
    except Exception as e:
        print(f"ERROR:{str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
