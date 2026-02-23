#!/usr/bin/env python3
"""Preview enhanced prompt without running full video generation.

Outputs JSON: {"enhanced_prompt": "..."} or {"error": "..."}

When enhancement returns empty (e.g. safety filter), auto-retries with suspected
filtered words replaced by placeholders, then merges originals back into the result.

Always uses MLX uncensored Gemma via mlx_lm (~7GB, no Lightricks/LTX-2 download).
"""

import argparse
import json
import re
import sys
from pathlib import Path

ENHANCER_MODEL = "TheCluster/amoral-gemma-3-12B-v2-mlx-4bit"

# Words that commonly trigger Gemma/content filters (lowercase)
SUSPECTED_FILTERED_WORDS = [
    "piss",
    "urine",
    "blood",
    "gore",
    "corpse",
    "dead body",
    "vomit",
    "vomiting",
    "naked",
    "nude",
    "sex",
    "sexual",
]


def _sanitize_prompt(prompt: str) -> tuple[str, dict[str, str]]:
    """Replace suspected filtered words with placeholders. Returns (sanitized, {placeholder: original})."""
    replacements: dict[str, str] = {}
    result = prompt
    for i, word in enumerate(SUSPECTED_FILTERED_WORDS):
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.I)
        for m in reversed(list(pattern.finditer(result))):
            ph = f"__X{i}__"
            replacements[ph] = m.group()
            result = result[: m.start()] + ph + result[m.end() :]
    return result, replacements


def _merge_back(enhanced: str, replacements: dict[str, str]) -> str:
    """Restore original words from placeholders."""
    result = enhanced
    for ph, orig in replacements.items():
        result = result.replace(ph, orig)
    return result


def _apply_chat_template(system_prompt: str, user_content: str) -> str:
    """Apply Gemma 3 chat template."""
    formatted = f"<start_of_turn>user\n{system_prompt}<end_of_turn>\n"
    formatted += f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
    formatted += "<start_of_turn>model\n"
    return formatted


def _clean_response(response: str) -> str:
    """Clean up the generated response."""
    response = response.strip()
    response = re.sub(r"^[^\w\s]+", "", response)
    return response


def _enhance_with_mlx_lm(
    prompt: str,
    model_repo: str,
    system_prompt: str | None,
    temperature: float,
    seed: int,
    max_tokens: int,
    verbose: bool,
) -> str:
    """Enhance prompt using mlx_lm with given MLX model. No Lightricks/LTX-2 download."""
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("mlx-lm not available. Install: pip install mlx-lm", file=sys.stderr)
        return prompt

    print(f"Loading prompt enhancer ({model_repo}, first run ~7GB)...", file=sys.stderr, flush=True)
    model, tokenizer = load(model_repo)

    if system_prompt is None:
        try:
            from mlx_video.models.ltx.enhance_prompt import _load_system_prompt
            system_prompt = _load_system_prompt("gemma_t2v_system_prompt.txt")
        except Exception:
            system_prompt = "You are a creative writer. Expand the user's short video prompt into a detailed, vivid description suitable for AI video generation. Include lighting, camera movement, and atmosphere."

    user_content = f"user prompt: {prompt}"
    formatted = _apply_chat_template(system_prompt, user_content)

    import mlx.core as mx
    mx.random.seed(seed)

    response = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=verbose,
    )

    del model
    mx.clear_cache()
    return _clean_response(response)


def main():
    parser = argparse.ArgumentParser(description="Preview Gemma-enhanced prompt")
    parser.add_argument("--prompt", "-p", required=True, help="User prompt to enhance")
    parser.add_argument(
        "--model-repo",
        default="notapalindrome/ltx2-mlx-av",
        help="Model repository (unified AV)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for enhancement",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image path for I2V (uses i2v system prompt if set)",
    )
    parser.add_argument(
        "--resources-path",
        default=None,
        help="App Resources path for bundled prompts (pre-flight injection)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        # Pre-flight: inject bundled prompts if mlx_video is missing them
        if args.resources_path:
            try:
                from pathlib import Path as P
                import shutil

                resources_path = P(args.resources_path)
                bundled_prompts = (
                    resources_path / "ltx_mlx" / "models" / "ltx" / "prompts"
                )
                import mlx_video.models.ltx.text_encoder as te

                target_dir = P(te.__file__).parent / "prompts"
                for name in [
                    "gemma_t2v_system_prompt.txt",
                    "gemma_i2v_system_prompt.txt",
                ]:
                    src = bundled_prompts / name
                    dst = target_dir / name
                    if src.exists() and not dst.exists():
                        target_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
            except Exception:
                pass

        # Always use MLX uncensored Gemma via mlx_lm
        model_repo = ENHANCER_MODEL

        system_prompt = None
        if args.image:
            try:
                from mlx_video.models.ltx.enhance_prompt import _load_system_prompt
                system_prompt = _load_system_prompt("gemma_i2v_system_prompt.txt")
            except Exception:
                pass

        def do_enhance(p: str):
            return _enhance_with_mlx_lm(
                p,
                model_repo=model_repo,
                system_prompt=system_prompt,
                temperature=args.temperature,
                seed=args.seed,
                max_tokens=256,
                verbose=False,
            )

        enhanced = do_enhance(args.prompt)

        # Auto-retry with sanitized prompt when enhancement returns empty (filtered)
        if not enhanced or not enhanced.strip():
            sanitized, replacements = _sanitize_prompt(args.prompt)
            if replacements:
                enhanced = do_enhance(sanitized)
                if enhanced and enhanced.strip():
                    enhanced = _merge_back(enhanced, replacements)

        print(json.dumps({"enhanced_prompt": enhanced or ""}))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
