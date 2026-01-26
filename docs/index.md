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

LTX Video Generator is a beautiful, native macOS application built with SwiftUI. It harnesses the power of Apple Silicon to generate AI videos with synchronized audio directly on your Mac using the LTX-2 model from Lightricks.

### Key Features

- **Apple Silicon Optimized** - Leverages Metal Performance Shaders (MPS) for fast GPU-accelerated generation
- **LTX-2 Model** - Latest 19B audio-video foundation model generates video with synchronized audio
- **Model Variants** - Choose between Full (best quality), Distilled (fast), or FP8 (low memory)
- **Intuitive Interface** - Clean, native macOS design that feels right at home
- **Generation Queue** - Queue multiple videos and track progress in real-time
- **Smart History** - Browse, preview, and manage all your generated videos with thumbnails
- **Flexible Presets** - Quick access to common configurations or customize every parameter

## Quick Start

1. **Download** the app from the [Releases page](https://github.com/james-see/ltx-video-mac/releases)
2. **Install Python dependencies** (see [Installation Guide](installation))
3. **Configure** your Python path in Preferences
4. **Generate** your first video!

## System Requirements

| Requirement | Minimum | Recommended |
|:------------|:--------|:------------|
| macOS | 14.0+ | 15.0+ |
| Processor | Apple M1 | Apple M2 Pro/M3/M4 |
| Unified Memory | 32GB | 64GB+ |
| Storage | 50GB free | 100GB+ free |
| Python | 3.10+ | 3.12+ |

{: .note }
LTX-2 is a 19B parameter model requiring significant unified memory. For systems with less RAM, use the **Distilled** or **FP8** model variants in Preferences.

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

## Contributing

LTX Video Generator is open source! Contributions, issues, and feature requests are welcome on [GitHub](https://github.com/james-see/ltx-video-mac).
