#!/usr/bin/env python3
"""ltx-2-mlx generate wrapper for mlx-community/ltx-2.5-mlx.

dgrauet 0.15.2 only treats a pack as Gemma 4 when text_encoder.safetensors
sits at the snapshot root. The community pack ships the tower as mlx-lm
layout under gemma4-12b-ltx-v1/. Without this adapter, generate falls back
to Gemma 3 4-bit into a Gemma-4 connector (wrong scene, not a weak follow).

Also skips DurationHead when the pack has split q/k/v keys (0.15.2 expects
fused in_proj). We always pass --frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

_COMMUNITY_GEMMA = "gemma4-12b-ltx-v1"
_COMMUNITY_PREFIX = "model.language_model."


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


def apply_patches() -> None:
    from ltx_core_mlx.text_encoders.gemma.encoders import encoder_configurator
    from ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder import (
        Gemma4TextEncoder,
    )
    from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextModel
    from ltx_core_mlx.utils import weights as weights_mod
    from ltx_pipelines_mlx.utils import blocks
    from ltx_pipelines_mlx.utils.blocks import DurationPredictor

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

    _orig_duration = DurationPredictor.from_checkpoint

    def _safe_duration(model_dir):
        try:
            return _orig_duration(model_dir)
        except Exception as e:
            print(
                "DurationHead skipped (%s: %s); using --frames" % (type(e).__name__, e),
                file=sys.stderr,
                flush=True,
            )
            return None

    DurationPredictor.from_checkpoint = classmethod(
        lambda cls, model_dir: _safe_duration(model_dir)
    )


def main() -> int:
    apply_patches()
    from ltx_pipelines_mlx.cli import main as cli_main

    sys.argv = ["ltx-2-mlx"] + sys.argv[1:]
    return cli_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
