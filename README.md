# LTX Video Generator for Mac

[![macOS](https://img.shields.io/badge/macOS-14.0+-blue.svg)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange.svg)](https://support.apple.com/en-us/HT211814)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/james-see/ltx-video-mac)](https://github.com/james-see/ltx-video-mac/releases)

A beautiful, native macOS application for generating AI videos from text prompts using the LTX-2 model, running natively on Apple Silicon with MLX.

![screenshot](https://i.imgur.com/LfBhmJa.png)

## Features

- **Native macOS App** - Built with SwiftUI for a seamless Mac experience
- **Apple Silicon Native** - Uses MLX framework for optimal performance on M-series chips
- **Text-to-Video Generation** - Transform text prompts into video clips
- **Image-to-Video** - Animate images into videos (coming soon)
- **Auto Package Installer** - Missing Python packages are detected and can be installed with one click
- **Generation Queue** - Queue multiple generations with real-time progress tracking
- **History Management** - Browse, preview, and manage all your generated videos
- **Presets** - Save and load generation parameter presets
- **Customizable Parameters** - Fine-tune resolution, frames, steps, guidance scale, and more

## Requirements

- **macOS 14.0** or later
- **Apple Silicon** Mac (M1, M2, M3, M4 series)
- **32GB RAM** minimum (64GB+ recommended for higher resolutions)
- **Python 3.10+** installed (via Homebrew, pyenv, or system)
- **~100GB disk space** for model weights

## Installation

### 1. Download the App

Download the latest release from the [Releases page](https://github.com/james-see/ltx-video-mac/releases).

### 2. First Launch Setup

1. Open LTX Video Generator
2. Go to **Preferences** (⌘,)
3. Click **Auto Detect** to find your Python installation, or manually set the path
4. Click **Validate Setup** - the app will check for required packages

### 3. Install Python Packages

If packages are missing, the app will show an "Install Missing Packages" button. Click it to automatically install:

```
mlx mlx-vlm transformers safetensors huggingface_hub numpy opencv-python tqdm
```

Or install manually:
```bash
pip install mlx mlx-vlm transformers safetensors huggingface_hub numpy opencv-python tqdm
```

### 4. First Generation - Model Download

**Important:** On first generation, the app will download the LTX-2 model (~90GB) from Hugging Face. This is a one-time download that may take 30-60 minutes depending on your internet connection.

The model is cached in `~/.cache/huggingface/` and will not be re-downloaded on subsequent runs.

Progress is shown in the app during download.

## Usage

1. Enter a descriptive prompt in the text field
2. Adjust parameters using presets or manual controls
3. Click **Generate** to start
4. Watch progress in the Queue sidebar
5. Find completed videos in your configured output directory (default: Application Support)

### Tips for Better Results

- Be descriptive: "A river flowing through a misty forest at dawn" works better than "river forest"
- Use camera directions: "The camera slowly pans across..."
- Specify lighting: "golden hour lighting", "dramatic shadows"
- Include motion: "waves crashing", "leaves falling"

## Example

Here's an example video generated with LTX Video Generator:

https://bellwetherlabs.com/example1.mp4

**Prompt used:**
> Create a 15-second cinematic product commercial for a sleek, premium TIME MACHINE device called "ChronoShift One."
> 
> Overall style: glossy tech product ad, filmed in 4K, smooth dolly and slider shots, soft studio lighting, subtle retro‑futuristic aesthetic (think brushed aluminum, glowing rings, clean UI). The time machine looks like a compact desktop appliance about the size of a toaster: brushed metal body, circular time dial with glowing blue light, small display, and a single illuminated control knob.

## Building from Source

```bash
# Clone the repository
git clone https://github.com/james-see/ltx-video-mac.git
cd ltx-video-mac

# Open in Xcode
open LTXVideoGenerator/LTXVideoGenerator.xcodeproj

# Or build from command line
./scripts/build-local.sh
```

## Technical Details

- **Frontend**: SwiftUI
- **Python Bridge**: Subprocess execution with progress streaming
- **ML Framework**: [MLX](https://github.com/ml-explore/mlx) (Apple's machine learning framework)
- **Video Model**: [LTX-2 Distilled](https://huggingface.co/mlx-community/LTX-2-distilled-bf16) (19B parameters, 2-stage generation)
- **Precision**: bfloat16

### Architecture

Generation uses a 2-stage pipeline:
1. Stage 1: Generate at half resolution
2. Stage 2: Upsample and refine to full resolution

## Troubleshooting

### "Model download stuck"
The download progress updates every 1%. For a 90GB download, each percent is ~900MB. Be patient.

### "Out of memory"
- Reduce resolution (512x320 is fastest)
- Reduce frame count
- Close other applications
- 32GB RAM minimum, 64GB recommended

### "Python not found"
- Install Python via Homebrew: `brew install python@3.12`
- Or use pyenv: `pyenv install 3.12`
- Then click "Auto Detect" in Preferences

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for the LTX-2 model
- [MLX Community](https://huggingface.co/mlx-community) for the MLX-converted weights
- [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) for the MLX video generation code
- [Hugging Face](https://huggingface.co/) for model hosting
