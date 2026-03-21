---
layout: default
title: Home
nav_order: 1
description: "LTX Video Generator - Native macOS app for AI video generation"
permalink: /
---

# LTX Video Generator
{: .fs-9 }

Transform text into stunning AI-generated videos on your Mac.
{: .fs-6 .fw-300 }

[Download Latest Release](https://github.com/james-see/ltx-video-mac/releases){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/james-see/ltx-video-mac){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Native macOS Experience

LTX Video Generator is a beautiful, native macOS application built with SwiftUI. It runs the LTX-2 model natively on Apple Silicon using MLX (Apple's machine learning framework) for optimal performance.

### Key Features

- **Apple Silicon Native** - Uses MLX for optimal M1/M2/M3/M4 performance
- **Two LTX-2 Models** - LTX-2 Unified (~42GB) and LTX-2.3 Distilled Q4 (~19.4GB), both with built-in audio
- **Text-to-Video** - Generate videos from text descriptions
- **Image-to-Video** - Animate images into videos
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
| Unified Memory | 32GB | 64GB+ |
| Storage | 100GB free | 150GB+ free |
| Python | 3.10+ | 3.12+ |

{: .warning }
**First Run Download**: The LTX-2 Unified model (~42GB) downloads automatically on first generation. This is a one-time download cached in `~/.cache/huggingface/`.

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
