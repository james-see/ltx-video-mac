---
layout: default
title: Architecture
nav_order: 6
---

# Architecture Reference
{: .no_toc }

Technical details of the generation pipeline and supported models.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

LTX Video Generator is a native macOS SwiftUI application that shells out to a Python backend (`mlx-video-with-audio`) for the actual generation work. The Python library runs LTX-2 models on Apple Silicon using MLX.

```
┌──────────────────────────────┐
│  LTX Video Generator (Swift) │
│  SwiftUI · GenerationService │
│  LTXBridge · PythonEnv       │
└──────────┬───────────────────┘
           │ subprocess
           ▼
┌──────────────────────────────┐
│  mlx-video-with-audio (Py)   │
│  generate_av.py              │
│  MLX transformer · VAE       │
│  Vocoder · Text encoder      │
└──────────────────────────────┘
```

### Swift Side

| Component | File | Role |
|:----------|:-----|:-----|
| `GenerationService` | `GenerationService.swift` | Manages the generation queue, spawns Python processes |
| `LTXBridge` | `LTXBridge.swift` | Builds CLI args, manages `PYTHONPATH`, parses progress |
| `PythonEnvironment` | `PythonEnvironment.swift` | Detects Python, validates packages, handles upgrades |
| `RootView` | `LTXVideoGeneratorApp.swift` | Launch-time Python validation and upgrade consent |

### Python Side

| Component | File | Role |
|:----------|:-----|:-----|
| `generate_av.py` | `mlx_video/generate_av.py` | Main entry point — loads models, runs the 2-stage pipeline, decodes AV |
| `Vocoder` / `BigVGANVocoder` | `mlx_video/models/ltx/audio_vae/vocoder.py` | Mel spectrogram → audio waveform |
| `VocoderWithBWE` | `mlx_video/models/ltx/audio_vae/vocoder.py` | Wraps vocoder with bandwidth extension (48kHz output) |
| `LTXModel` | `mlx_video/models/ltx/ltx.py` | Transformer backbone for denoising |
| Text encoder | `mlx_video/models/ltx/text_encoder.py` | Gemma-based text encoding |

---

## Supported Models

### notapalindrome/ltx2-mlx-av (LTX-2 Unified)

| Property | Value |
|:---------|:------|
| Parameters | ~19B |
| Download size | ~42GB |
| Format | Single `model.safetensors` with all components |
| Audio | Built-in (stereo, 24kHz base → 48kHz with BWE) |
| Vocoder type | Standard `Vocoder` (LeakyReLU activations) |
| Weight layout | MLX-native — `ConvTranspose1d` weights stored as `(out_ch, kernel, in_ch)` |
| Config | No `embedded_config.json` — uses hardcoded defaults |

### dgrauet/ltx-2.3-mlx-distilled-q4 (LTX-2.3 Distilled Q4)

| Property | Value |
|:---------|:------|
| Parameters | ~19B (quantized to 4-bit) |
| Download size | ~19.4GB |
| Format | Split safetensors (`transformer.safetensors`, `vocoder.safetensors`, etc.) |
| Audio | Built-in (stereo, 24kHz base → 48kHz with BWE) |
| Vocoder type | `BigVGANVocoder` (SnakeBeta activations + anti-aliased upsampling) |
| Weight layout | PyTorch-origin — `ConvTranspose1d` weights stored as `(in_ch, out_ch, kernel)` |
| Config | Has `embedded_config.json` with vocoder/transformer/scheduler overrides |

---

## Generation Pipeline

Both models use a 2-stage generation pipeline:

### Stage 1: Low-Resolution Generation

1. Encode text prompt via Gemma text encoder
2. Generate latents at **half resolution** (e.g. 256×160 for a 512×320 target)
3. Denoising loop using the transformer backbone
4. For the unified model: fixed 8-step sigma schedule
5. For the distilled model: configurable steps with `ltx2_schedule`

### Stage 2: High-Resolution Refinement

1. Spatially upsample Stage 1 latents 2× using a learned upscaler
2. Refine at full resolution with additional denoising steps
3. For the unified model: fixed 3-step sigma schedule
4. For the distilled model: configurable steps

### Decode

1. **Video**: VAE decoder converts latents → pixel frames
   - Supports temporal tiling for long videos
2. **Audio**: Audio VAE decoder extracts mel spectrogram from audio latents
3. **Vocoder**: Converts mel spectrogram → audio waveform
   - Standard `Vocoder` for unified model
   - `BigVGANVocoder` with `SnakeBeta` activations for distilled model
   - `VocoderWithBWE` wraps either vocoder to upsample 24kHz → 48kHz
4. **Mux**: FFmpeg combines video + audio into final MP4

---

## Weight Loading Details

### ConvTranspose1d Weight Layout

This is a critical detail. MLX's `nn.ConvTranspose1d` stores weights as:

```
(out_channels, kernel_size, in_channels)
```

The two models store vocoder upsampler (`ups.*`) weights differently:

| Model | Stored layout | Needs transpose? |
|:------|:-------------|:-----------------|
| notapalindrome (unified) | `(out_ch, kernel, in_ch)` | No — already MLX format |
| dgrauet (distilled) | `(in_ch, out_ch, kernel)` | Yes — `transpose(1, 2, 0)` |

The loader in `generate_av.py` auto-detects the layout: if the last dimension is the largest (since `in_ch > out_ch` for upsamplers), the weight is already in MLX format. Otherwise it transposes.

### Unified Model Weight Prefixes

Weights in `model.safetensors` are prefixed by component:

| Prefix | Component |
|:-------|:----------|
| `transformer.*` | Transformer backbone |
| `vae_decoder.*` | Video VAE decoder |
| `vocoder.*` | Audio vocoder |
| `connector.*` | Audio-video connector |

The `load_unified_weights()` helper strips the prefix when loading into each component.

### Distilled Model (Split Files)

Each component has its own safetensors file:

| File | Component |
|:-----|:----------|
| `transformer.safetensors` | Transformer |
| `vae_decoder.safetensors` | Video VAE |
| `vocoder.safetensors` | Audio vocoder + BWE |
| `audio_vae.safetensors` | Audio VAE decoder |
| `vae_encoder.safetensors` | Video VAE encoder |
| `connector.safetensors` | Audio-video connector |

### embedded_config.json

The distilled model includes `embedded_config.json` with overrides:

```json
{
  "vocoder": {
    "upsample_initial_channel": 1536,
    "upsample_rates": [6, 2, 2, 2, 2, 2],
    "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "activation": "snakebeta",
    "resblock": "AMP1"
  },
  "transformer": { ... },
  "scheduler": { ... }
}
```

When this file is absent (unified model), hardcoded defaults are used.

---

## Python Environment Management

### Package Detection

On launch, the app validates the configured Python environment:

1. Checks Python version (3.10+ required)
2. Checks installed packages via `pip show`
3. Compares `mlx-video-with-audio` version against `mlxVideoMinVersion`
4. If packages are missing or outdated in a venv, prompts the user to upgrade

### PYTHONPATH Resolution

`LTXBridge.swift` determines whether to use the pip-installed package or a local developer checkout:

1. **Default**: Use pip-installed `mlx-video-with-audio` from site-packages
2. **Developer override**: If `~/projects/mlx-video-with-audio` exists AND either:
   - The Preferences toggle "Use local mlx-video-with-audio repo" is enabled, OR
   - The env var `LTX_FORCE_LOCAL_MLX_VIDEO=1` is set, OR
   - The local repo version is strictly newer than the pip version
3. When using pip, `PYTHONPATH` is explicitly cleared to prevent stale shell values from shadowing

### Version Cache

After a successful validation, the result is cached for 5 minutes (`generationValidationCache` in `PythonEnvironment.swift`) so that queueing multiple generations doesn't repeatedly re-validate.

---

## Audio Pipeline Details

### Vocoder Architecture

**Standard Vocoder** (unified model):
- `conv_pre` → repeated (`LeakyReLU` → `ConvTranspose1d` upsample → `ResBlock`) → `conv_post`
- `upsample_initial_channel = 1024`
- `upsample_rates = [6, 5, 2, 2, 2]`
- Stereo output (128 mel channels in, 2 audio channels)

**BigVGANVocoder** (distilled model):
- Same high-level structure but uses `SnakeBeta` activations with anti-aliased upsampling
- `SnakeBeta` applies `x + (1/α) * sin²(αx)` with learned α/β parameters stored in log-scale
- Anti-aliased activation: upsample → activate → downsample with Kaiser-sinc filter
- `upsample_initial_channel = 1536`
- `upsample_rates = [6, 2, 2, 2, 2, 2]`

### Bandwidth Extension (BWE)

Both models use `VocoderWithBWE` which wraps the base vocoder:
- Base vocoder produces 24kHz audio
- BWE generator upsamples to 48kHz using a smaller secondary vocoder
- Skip connection with linear interpolation for residual detail
- Final output: 48kHz stereo audio

### Audio Normalization

Generated audio is peak-normalized to ±0.95 before saving, ensuring consistent loudness regardless of the vocoder's raw output level.

---

## Debugging

### Log Files

```bash
# Generation log (stdout/stderr from Python)
cat /tmp/ltx_generation.log

# Check installed package version
pip show mlx-video-with-audio
```

### Common Errors

| Error | Cause | Fix |
|:------|:------|:----|
| `ValueError: [conv] Expect the input channels...` | Vocoder weight layout mismatch | Upgrade to `mlx-video-with-audio>=0.1.25` |
| `ModuleNotFoundError: mlx_video` | Package not installed | `pip install mlx-video-with-audio` |
| `Text encoder configuration mismatch` | Outdated mlx-video-with-audio | `pip install -U mlx-video-with-audio` |
| `AttributeError: module 'mlx.core' has no attribute...` | MLX version too old | `pip install -U mlx` |

### Environment Variables

| Variable | Effect |
|:---------|:-------|
| `LTX_FORCE_LOCAL_MLX_VIDEO=1` | Force use of `~/projects/mlx-video-with-audio` |
| `PYTHONPATH` | Cleared by the app unless local repo is active |

---

## Version History

See the full [CHANGELOG](https://github.com/james-see/ltx-video-mac/blob/main/CHANGELOG.md) for release details.

Key milestones:
- **v2.3.42**: Fix vocoder crash with unified model (ConvTranspose1d weight layout detection)
- **v2.3.41**: Launch-time package upgrade consent, PYTHONPATH fix
- **v2.3.40**: Resilient model download progress
- **v2.3.0+**: LTX-2.3 distilled Q4 model support, BigVGAN vocoder, SnakeBeta
