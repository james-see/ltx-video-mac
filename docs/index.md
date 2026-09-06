---
layout: default
title: Home
nav_order: 1
description: "Native macOS app for local AI video — LTX-2, LTX-2.3, LTX-2.5, and MiniMax H3"
permalink: /
---

# LTX Video Generator
{: .fs-9 }

Local text-to-video and image-to-video on Apple Silicon. LTX-2 / 2.3, LTX-2.5, and MiniMax H3.
{: .fs-6 .fw-300 }

[Download v2.3.69](https://github.com/james-see/ltx-video-mac/releases/tag/v2.3.69){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[All Releases](https://github.com/james-see/ltx-video-mac/releases){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/james-see/ltx-video-mac){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## What's New in 2.3.69

- **LTX-2.5** — Distilled (`ltx25_distilled`, ~100GB) and Distilled Q8 DiT (`ltx25_distilled_ditq8`). Gemma 4 is bundled. The app installs `dgrauet/ltx-2-mlx` 0.15+ only when you pick a 2.5 model.
- **MiniMax H3** — Native `h3.c` (`minimax_h3` BF16 ~144GB, `minimax_h3_int8` ~92GB, `minimax_h3_turbo` folded 6-step). License dialog on first select/generate; weights auto-download into the Hugging Face cache. Build `./h3` yourself (not in the DMG).
- **Unchanged 2.3 path** — LTX-2 / 2.3 stay on `mlx-video-with-audio`. REST `POST /generate` accepts the new `model_id` values.

[Release notes](https://github.com/james-see/ltx-video-mac/releases/tag/v2.3.69) · [Installation](installation) · [Architecture](architecture)

---

## Native macOS Experience

SwiftUI app. Generation is a local subprocess: `mlx-video-with-audio` for LTX-2 / 2.3, `ltx-2-mlx` for 2.5, `./h3` for MiniMax H3.

### Key Features

- **Apple Silicon Native** - MLX for LTX; Metal via `h3.c` for H3
- **LTX-2, LTX-2.3, LTX-2.5, MiniMax H3** - 2.3 Q4 default (~22GB) except ≤16GB Macs (12GB pack); 2.5 / 12GB via `ltx-2-mlx`; H3 via native `h3.c`
- **Text-to-Video** - Generate videos from text descriptions
- **Image-to-Video** - Animate images; first/last frame and multi-image keyframes on the LTX path
- **Local REST API** - `127.0.0.1:8420` (`ltx25_distilled`, `ltx25_distilled_ditq8`, `minimax_h3`, `minimax_h3_int8`, `minimax_h3_turbo`)
- **Gemma Prompt Enhancement** - Optional AI rewrites prompts for better results; uncensored enhancer avoids content filters
- **Voiceover Narration** - Add TTS audio using ElevenLabs (cloud) or MLX-Audio (local)
- **Background Music** - 54 genre presets for AI-generated instrumental music via ElevenLabs
- **Auto Package Installer** - Missing Python packages detected and installed with one click
- **Generation Queue** - Queue multiple videos and track progress in real-time
- **Smart History** - Browse, preview, and manage all your generated videos
- **Flexible Presets** - Quick access to common configurations or customize every parameter

## Quick Start

1. **Download** the app from the [Releases page](https://github.com/james-see/ltx-video-mac/releases)
2. **Open Preferences** and click Auto Detect to find Python
3. **Install packages** if prompted (one-click install available)
4. **Generate** your first video! (model downloads on first run)

## System Requirements

| Requirement | Minimum | Recommended |
|:------------|:--------|:------------|
| macOS | 14.0+ | 15.0+ |
| Processor | Apple M1 | Apple M2 Pro/M3/M4 |
| Unified Memory | 32GB (2.3 Q4) | 64GB+ (2.5 / H3) |
| Storage | 40GB free (2.3 Q4) | 150GB+ (2.5 ~100GB or H3 ~144GB) |
| Python | 3.10+ | 3.12+ |

{: .warning }
**First run**: the selected model downloads from Hugging Face on first generate (default 2.3 Distilled Q4 is ~22GB). LTX-2.5 is ~100GB; MiniMax H3 is ~144GB and requires accepting the MiniMax H3 Community License. Cache is `~/.cache/huggingface/` unless you set a Model Cache Directory.

## Sample Results

Generate videos like:
- "A river flowing through a misty forest at dawn"
- "The camera slowly pans across a futuristic cityscape"  
- "Golden leaves falling in slow motion against a blue sky"

---

## Getting Help

- [Installation Guide](installation) - Complete setup instructions
- [Usage Guide](usage) - Learn how to get the best results
- [Parameters Reference](parameters) - Understand all settings
- [Troubleshooting](troubleshooting) - Common issues and solutions
- [Architecture](architecture) - Technical details of the pipeline and models

## Contributing

LTX Video Generator is open source! Contributions, issues, and feature requests are welcome on [GitHub](https://github.com/james-see/ltx-video-mac).
