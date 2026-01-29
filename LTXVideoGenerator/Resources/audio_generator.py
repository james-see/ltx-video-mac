"""Audio generation using MLX Audio for Apple Silicon."""

import sys
import json


def status_output(message: str):
    """Output status message for Swift to parse."""
    print(f"STATUS:{message}", file=sys.stderr)
    sys.stderr.flush()


def progress_output(percent: float, message: str):
    """Output progress for Swift to parse."""
    print(f"PROGRESS:{percent}:{message}", file=sys.stderr)
    sys.stderr.flush()


def generate_audio(
    text: str,
    voice: str = "af_heart",
    output_path: str = "output.wav",
    speed: float = 1.0,
) -> dict:
    """
    Generate audio from text using MLX Audio.

    Args:
        text: Text to convert to speech
        voice: Voice ID (e.g., 'af_heart', 'am_adam')
        output_path: Path to save the output audio file
        speed: Speech speed multiplier (0.5 to 2.0)

    Returns:
        dict with 'success' and 'audio_path' or 'error'
    """
    try:
        status_output("Loading MLX Audio...")
        progress_output(10, "Loading model")

        from mlx_audio.tts.generate import generate_speech

        progress_output(30, "Generating speech")
        status_output(f"Generating speech with voice: {voice}")

        # Generate speech
        generate_speech(
            text=text,
            voice=voice,
            speed=speed,
            output_path=output_path,
        )

        progress_output(90, "Saving audio")
        status_output(f"Audio saved to: {output_path}")

        progress_output(100, "Complete")

        return {"success": True, "audio_path": output_path}

    except ImportError as e:
        error_msg = f"MLX Audio not installed: {e}"
        status_output(f"ERROR: {error_msg}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        error_msg = str(e)
        status_output(f"ERROR: {error_msg}")
        return {"success": False, "error": error_msg}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate audio with MLX Audio")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text to speak")
    parser.add_argument("--voice", "-v", type=str, default="af_heart", help="Voice ID")
    parser.add_argument(
        "--output", "-o", type=str, default="output.wav", help="Output path"
    )
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Speech speed")

    args = parser.parse_args()

    result = generate_audio(
        text=args.text, voice=args.voice, output_path=args.output, speed=args.speed
    )

    print(json.dumps(result))
    sys.exit(0 if result["success"] else 1)
