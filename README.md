# LTX Video Generator for Mac

[![macOS](https://img.shields.io/badge/macOS-14.0+-blue.svg)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange.svg)](https://support.apple.com/en-us/HT211814)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/james-see/ltx-video-mac)](https://github.com/james-see/ltx-video-mac/releases)

A beautiful, native macOS application for generating AI videos from text prompts using the LTX-Video model from Lightricks, optimized for Apple Silicon.

![LTX Video Generator Screenshot]([https://private-user-images.githubusercontent.com/616585/533780147-e824ea5b-1c93-4026-9c17-da0ae03538df.png](https://private-user-images.githubusercontent.com/616585/533780147-e824ea5b-1c93-4026-9c17-da0ae03538df.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg5NzQxNDMsIm5iZiI6MTc2ODk3Mzg0MywicGF0aCI6Ii82MTY1ODUvNTMzNzgwMTQ3LWU4MjRlYTViLTFjOTMtNDAyNi05YzE3LWRhMGFlMDM1MzhkZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTIxJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEyMVQwNTM3MjNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lM2JjYzE2MjZmYjhjMDg0NjljM2NhMTVlYjdlYWU4YzM0MGM0MzkzMjI4MjRiYzk2ZDljOWE1MGZkNTc1NTU0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.jUzyGRiY5Aicman0IkofDHSshNVTo02E8FCTvxmpXCE))

## Features

- **Native macOS App** - Built with SwiftUI for a seamless Mac experience
- **Apple Silicon Optimized** - Leverages MPS (Metal Performance Shaders) for GPU acceleration
- **Text-to-Video Generation** - Transform text prompts into video clips
- **Generation Queue** - Queue multiple generations with real-time progress tracking
- **History Management** - Browse, preview, and manage all your generated videos
- **Presets** - Save and load generation parameter presets
- **Customizable Parameters** - Fine-tune resolution, frames, steps, guidance scale, and more

## Requirements

- **macOS 14.0** or later
- **Apple Silicon** Mac (M1, M2, M3, M4 series)
- **16GB RAM** minimum (32GB+ recommended for higher resolutions)
- **Python 3.10+** with PyTorch and diffusers installed
- **~15GB disk space** for model weights

## Installation

### 1. Download the App

Download the latest release from the [Releases page](https://github.com/james-see/ltx-video-mac/releases).

### 2. Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv ~/ltx-venv
source ~/ltx-venv/bin/activate

# Install required packages
pip install torch torchvision torchaudio
pip install diffusers accelerate transformers safetensors sentencepiece
pip install imageio imageio-ffmpeg opencv-python
```

### 3. Configure the App

1. Open LTX Video Generator
2. Go to **Preferences** (⌘,)
3. Set your Python executable path (e.g., `~/ltx-venv/bin/python3`)
4. Click **Validate** to verify the configuration

### 4. First Generation

The model (~15GB) will be downloaded automatically on first use. Subsequent generations will be much faster.

## Usage

1. Enter a descriptive prompt in the text field
2. Adjust parameters using presets or manual controls
3. Click **Generate** to start
4. Watch progress in the Queue sidebar
5. Find completed videos in the History tab

### Tips for Better Results

- Be descriptive: "A river flowing through a misty forest at dawn" works better than "river forest"
- Use camera directions: "The camera slowly pans across..."
- Specify lighting: "golden hour lighting", "dramatic shadows"
- Include motion: "waves crashing", "leaves falling"

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
- **Python Bridge**: Subprocess execution for stability
- **ML Framework**: PyTorch with MPS backend
- **Model**: [LTX-Video 0.9.1](https://huggingface.co/a-r-r-o-w/LTX-Video-0.9.1-diffusers) via Hugging Face diffusers
- **Precision**: bfloat16 (recommended for Apple Silicon)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for the LTX-Video model
- [Hugging Face](https://huggingface.co/) for the diffusers library
- [PythonKit](https://github.com/pvieito/PythonKit) for Swift-Python interop
