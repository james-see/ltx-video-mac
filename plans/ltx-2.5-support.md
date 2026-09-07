# LTX 2.5 + antirez h3.c

Status: implemented in-app. dgrauet/ltx-2-mlx **0.15.x already has LTX-2.5** — do not port 2.5 into `mlx-video-with-audio`. Keep LTX-2/2.3 on James’s library.

## Backends

`GenerationBackend` on `LTXModel`:

- `mlxVideoWithAudio` — existing 2.0/2.3 (`python -m mlx_video.generate_av`)
- `ltx2Mlx` — LTX-2.5 (`ltx-2-mlx generate`)
- `h3c` — MiniMax H3 (`./h3`)

## Catalog

| id | repo | backend | notes |
|---|---|---|---|
| `ltx23_distilled_q4` | notapalindrome/ltx23-mlx-av-q4 | mlxVideoWithAudio | default |
| `ltx25_distilled` | mlx-community/ltx-2.5-mlx (~100GB) | ltx2Mlx | Gemma 4 bundled, 8 steps, 64GB+ |
| `ltx25_dev` | same pack + overlay `ltx-2.5-22b-distilled-lora-450-bf16` | ltx2Mlx | official two-stage: fuse LoRA into dev DiT; ~8.3GB extra |
| `ltx25_distilled_ditq8` | same pack + overlay `mlx-community/ltx-2.5-mlx-ditq8` | ltx2Mlx | 32GB+ with --low-ram |
| `minimax_h3` | MiniMaxAI/MiniMax-H3 (~144GB) | h3c | native Metal |

`mlx-community/ltx-2.5-mlx-q8` is the **text encoder**, not the DiT. Do not treat it as a DiT quant.

## Install

- 2.5: git pin `v0.15.2` (`ltx-core-mlx` + `ltx-pipelines-mlx`) or `~/projects/ltx-2-mlx` + `uv run`. `mlx-lm>=0.31.2`. Only required when a 2.5 model is selected.
- H3 binary: user builds `antirez/h3.c` (`make -j8`). Discovery: Preferences path → `~/projects/h3.c/h3` → `which h3`. Not bundled in the DMG.
- H3 weights: first generate runs `huggingface_hub.snapshot_download("MiniMaxAI/MiniMax-H3")` into the configured HF cache unless a local snapshot already exists.

## Out of scope

- Porting 2.5 into mlx-video-with-audio
- Migrating 2.3 catalog onto ltx-2-mlx
- PipeNetwork Python H3 / FastH3 / ltx-2-mlx-swift / mlx-serve
- H3 Ref2VA (v1 is first-frame only)
