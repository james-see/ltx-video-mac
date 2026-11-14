#!/usr/bin/env python3
"""ltx-2-mlx generate wrapper for community LTX packs.

dgrauet ltx-2-mlx only treats a pack as Gemma 4 when text_encoder.safetensors
sits at the snapshot root. notapalindrome/ltx25-mlx (and mlx-community/ltx-2.5-mlx)
ship the tower as mlx-lm layout under gemma4-12b-ltx-v1/.

baa-ai/LTX-2.3-22B-RAM-12GB-MLX is mixed-precision (2–8 bit). Stock
apply_quantization uses one bit width; we apply per-layer nn.quantize.

DurationHead loading for split q/k/v community packs is handled upstream
since 0.16 (dgrauet/ltx-2-mlx#125); no patch needed on 0.18+.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_COMMUNITY_GEMMA = "gemma4-12b-ltx-v1"
_COMMUNITY_PREFIX = "model.language_model."


def _installed_pipelines_version() -> tuple[int, ...]:
    try:
        import importlib.metadata as m

        return tuple(
            int(p) for p in m.version("ltx-pipelines-mlx").split(".") if p.isdigit()
        )
    except Exception:
        return ()


def drop_unsupported_args(argv: list[str]) -> list[str]:
    """Strip flags the installed ltx-2-mlx predates, with a warning.

    The app pins v0.18.0, but a local-repo checkout or a venv the app did
    not provision can still be older. Flags map to (min version, value
    tokens to consume): 0 = none, 1 = exactly one, -1 = TEXT plus an
    optional trailing positive int (--segment TEXT [LEN]).
    """
    ver = _installed_pipelines_version()
    flag_rules: dict[str, tuple[tuple[int, int, int], int, str]] = {
        "--no-audio": ((0, 16, 0), 0, "audio will be generated (ignored)"),
        "--dfr": ((0, 16, 3), 0, "DFR ignored; running the model's default flow"),
        "--num-generated-keyframes": (
            (0, 16, 3),
            1,
            "generated keyframe slots ignored",
        ),
        "--detailing-lora": ((0, 16, 3), 1, "detailing LoRA override ignored"),
        "--segment": ((0, 17, 0), -1, "segments ignored"),
    }
    unsupported: dict[str, tuple[int, str]] = {}
    if len(ver) >= 3:
        for flag, (min_ver, tokens, note) in flag_rules.items():
            if ver < min_ver:
                unsupported[flag] = (
                    tokens,
                    f"{flag} needs ltx-2-mlx {'.'.join(map(str, min_ver))}+; {note}",
                )
    if not unsupported:
        return argv
    out: list[str] = []
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag in unsupported:
            tokens, message = unsupported[flag]
            print(message, file=sys.stderr, flush=True)
            i += 1
            for _ in range(max(tokens, 0)):
                if i < len(argv) and not argv[i].startswith("-"):
                    i += 1
            if tokens < 0 and i < len(argv) and not argv[i].startswith("-"):
                i += 1  # segment TEXT
                if i < len(argv) and argv[i].isdigit():
                    i += 1  # optional LEN_FRAMES
            continue
        out.append(flag)
        i += 1
    return out


def community_gemma_dir(model_dir: str | Path) -> Path | None:
    root = Path(model_dir)
    if (root / "text_encoder.safetensors").is_file() and (
        root / "text_encoder_config.json"
    ).is_file():
        return None
    gemma = root / _COMMUNITY_GEMMA
    if (gemma / "model.safetensors").is_file() and (gemma / "config.json").is_file():
        return gemma
    return None


def apply_mixed_precision_quantization(model, weights, group_size=64, bits=None):
    """Per-layer mixed-precision quant (baa-ai generate.py). bits is unused."""
    import mlx.nn as nn

    layer_bits: dict[str, int] = {}
    for key in weights:
        if not key.endswith(".scales"):
            continue
        layer = key[: -len(".scales")]
        w_key = layer + ".weight"
        if w_key not in weights:
            continue
        w_cols = weights[w_key].shape[-1]
        s_cols = weights[key].shape[-1]
        detected = round(w_cols * 32 / (s_cols * group_size))
        if detected in (2, 3, 4, 5, 6, 8):
            layer_bits[layer] = detected

    if not layer_bits:
        return

    bits_to_layers: dict[int, set] = defaultdict(set)
    for layer, b in layer_bits.items():
        bits_to_layers[b].add(layer)

    for quant_bits, layers in sorted(bits_to_layers.items()):

        def _predicate(path: str, module: nn.Module, _layers=layers) -> bool:
            return path in _layers and isinstance(module, nn.Linear)

        nn.quantize(
            model, group_size=group_size, bits=quant_bits, class_predicate=_predicate
        )

    dist = {b: len(v) for b, v in sorted(bits_to_layers.items())}
    print(
        "Mixed-precision quantization: %d layers — %s" % (sum(dist.values()), dist),
        file=sys.stderr,
        flush=True,
    )


def apply_patches() -> None:
    from ltx_core_mlx.text_encoders.gemma.encoders import encoder_configurator
    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import (
        Gemma4TextEncoder,
    )
    from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextModel
    from ltx_core_mlx.utils import weights as weights_mod
    from ltx_pipelines_mlx.utils import blocks
    from ltx_pipelines_mlx.utils import _orchestration as orch
    import ltx_pipelines_mlx._base as ltx_base

    weights_mod.apply_quantization = apply_mixed_precision_quantization
    orch.apply_quantization = apply_mixed_precision_quantization
    ltx_base.apply_quantization = apply_mixed_precision_quantization

    _orig_select = encoder_configurator.select_text_encoder

    def select_text_encoder(model_dir):
        if community_gemma_dir(model_dir) is not None:
            return "gemma4"
        return _orig_select(model_dir)

    encoder_configurator.select_text_encoder = select_text_encoder
    blocks.select_text_encoder = select_text_encoder

    _orig_load_from_pack = Gemma4TextModel.load_from_pack

    @classmethod
    def load_from_pack(cls, model_dir):
        gemma = community_gemma_dir(model_dir)
        if gemma is None:
            return _orig_load_from_pack(model_dir)
        import json

        from ltx_core_mlx.text_encoders.gemma.gemma4_config import Gemma4TextConfig
        from ltx_core_mlx.utils.weights import load_split_safetensors

        with open(gemma / "config.json") as f:
            raw = json.load(f)
        config = Gemma4TextConfig.from_text_encoder_config(raw)
        model = cls(config)
        weights = load_split_safetensors(
            gemma / "model.safetensors", prefix=_COMMUNITY_PREFIX
        )
        model.load_weights(list(weights.items()))
        print(
            "Using pack Gemma 4 at %s (%d tensors)" % (gemma, len(weights)),
            file=sys.stderr,
            flush=True,
        )
        return model

    Gemma4TextModel.load_from_pack = load_from_pack

    _orig_encoder_load = Gemma4TextEncoder.load

    def encoder_load(self, model_dir):
        gemma = community_gemma_dir(model_dir)
        if gemma is None:
            return _orig_encoder_load(self, model_dir)
        self.load_tokenizer(gemma)
        self._tower = Gemma4TextModel.load_from_pack(model_dir)

    Gemma4TextEncoder.load = encoder_load

    _orig_lss = weights_mod.load_split_safetensors

    def load_split_safetensors(path, prefix=None):
        path = Path(path)
        if path.name == "text_encoder.safetensors" and not path.is_file():
            print(
                "Skipping missing %s (projections live in connector.safetensors)"
                % path.name,
                file=sys.stderr,
                flush=True,
            )
            return {}
        return _orig_lss(path, prefix=prefix)

    weights_mod.load_split_safetensors = load_split_safetensors
    blocks.load_split_safetensors = load_split_safetensors


def main() -> int:
    apply_patches()
    sys.argv = ["ltx-2-mlx"] + drop_unsupported_args(sys.argv[1:])
    from ltx_pipelines_mlx.cli import main as cli_main

    return cli_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
