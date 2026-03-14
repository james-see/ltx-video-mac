# LTX Video Generator for Mac

[![macOS](https://img.shields.io/badge/macOS-14.0+-blue.svg)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-orange.svg)](https://support.apple.com/en-us/HT211814)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/james-see/ltx-video-mac)](https://github.com/james-see/ltx-video-mac/releases)

A beautiful, native macOS application for generating AI videos with synchronized audio from text prompts using the LTX-2 model, running natively on Apple Silicon with MLX.

![screenshot](https://i.imgur.com/LfBhmJa.png)

## Features

- **Native macOS App** - Built with SwiftUI for a seamless Mac experience
- **Apple Silicon Native** - Uses MLX framework for optimal performance on M-series chips
- **Text-to-Video Generation** - Transform text prompts into video clips
- **Image-to-Video** - Animate images into videos
- **Built-in Audio Generation** - Available model variants generate synchronized audio with video automatically
- **Voiceover Narration** - Add TTS voiceover using ElevenLabs (cloud) or MLX-Audio (local)
- **Background Music** - Generate instrumental music with 54 genre presets via ElevenLabs Music API
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
- **~20-42GB disk space** for model weights (depends on selected model)

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
mlx mlx-vlm mlx-video-with-audio transformers safetensors huggingface_hub numpy opencv-python tqdm
```

Or install manually:
```bash
pip install mlx mlx-vlm mlx-video-with-audio transformers safetensors huggingface_hub numpy opencv-python tqdm
```

The `mlx-video-with-audio` package is available on [PyPI](https://pypi.org/project/mlx-video-with-audio/) and provides the unified audio-video generation.

### 4. First Generation - Model Download

**Important:** On first generation, the app downloads your selected model from Hugging Face. This is a one-time download that may take 15-30 minutes depending on model size and internet connection.

The model is cached in `~/.cache/huggingface/` and will not be re-downloaded on subsequent runs.

Progress is shown in the app during download.

**Available models:**
- LTX-2 Unified (`notapalindrome/ltx2-mlx-av`, ~42GB)
- LTX-2.3 Distilled Q4 (`dgrauet/ltx-2.3-mlx-distilled-q4`, ~19.4GB)

## Usage

1. Enter a descriptive prompt in the text field
2. Adjust parameters using presets or manual controls
3. Click **Generate** to start
4. Watch progress in the Queue sidebar
5. Find completed videos in your configured output directory (default: Application Support)

### Gemma Prompt Enhancement

When enabled in **Settings > Generation**, Gemma rewrites your prompt before generation—expanding short descriptions into detailed, LTX-2–optimized prompts with visuals, audio, camera movement, and style. Use the **Preview enhanced prompt** button to see the rewritten prompt before generating.

> **Note:** This enhancer is optional.  
> The core text encoder used for generation embeddings is still required even when prompt enhancement is off.

1. Go to **Settings > Generation**
2. Turn on **Enable Gemma Prompt Enhancement**
3. First run downloads the Gemma enhancer (~7GB)
4. In the prompt view, expand **Prompt Enhancement (Gemma)** and adjust sliders (Repetition Penalty, Top-P) if desired
5. Click **Preview enhanced prompt** to see the enhanced version before generating
6. Generate as usual—the enhanced prompt is used automatically

If enhancement fails for any reason, generation automatically falls back to your original prompt.

### Tips for Better Results

- Be descriptive: "A river flowing through a misty forest at dawn" works better than "river forest"
- Use camera directions: "The camera slowly pans across..."
- Specify lighting: "golden hour lighting", "dramatic shadows"
- Include motion: "waves crashing", "leaves falling"

## Audio Features

### Built-in Audio (Default)

Selected models generate synchronized audio alongside video automatically. No additional configuration needed - just generate and your video will have audio.

For best speech/lip-sync alignment, use **24 FPS**.

You can still layer additional voiceover or background music on top of the built-in audio if desired.

### Voiceover / Narration

Add text-to-speech voiceover to your videos:

1. Expand **Voiceover / Narration** in the generation view
2. Choose your source: **MLX-Audio** (local, free) or **ElevenLabs** (cloud, requires API key)
3. Select a voice from the dropdown
4. Enter your narration text
5. Audio generates with your video or can be added later from History

### Background Music

Add AI-generated instrumental music (requires ElevenLabs API key):

1. Expand **Background Music** in the generation view
2. Toggle **Generate background music**
3. Choose from 54 genre presets:
   - **Electronic**: EDM, House, Techno, Ambient, Synthwave, etc.
   - **Hip-Hop/R&B**: Trap, Lo-Fi, Boom Bap, Soul, etc.
   - **Rock**: Classic, Alternative, Indie, Metal, etc.
   - **Pop**: Modern, Indie, Dance, Acoustic
   - **Jazz/Blues**: Smooth Jazz, Bebop, Lounge, Blues
   - **Classical/Cinematic**: Orchestral, Piano, Epic, Tense, Uplifting
   - **World**: Latin, Reggae, Afrobeat, Middle Eastern, Asian
   - **Country/Folk**: Modern, Classic, Acoustic, Indie
   - **Functional**: Corporate, Motivational, Relaxing, Suspense, Action, Romantic, etc.

Music automatically matches your video length and is mixed at background volume (30%) or ducked further (20%) when combined with voiceover.

### Adding Audio to Existing Videos

Right-click any video thumbnail in **Video Archive** and select **Add Audio** to add voiceover, music, or both to previously generated videos.

## Example

Here's an example video generated with LTX Video Generator:

[Open video link](https://github.com/user-attachments/assets/82031683-1763-4dff-97f9-c2b6d38f7ee8)

**Prompt used:**
> Create a 15-second cinematic product commercial for a sleek, premium TIME MACHINE device called "ChronoShift One."
>
> Overall style: glossy tech product ad, filmed in 4K, smooth dolly and slider shots, soft studio lighting, subtle retro‑futuristic aesthetic (think brushed aluminum, glowing rings, clean UI). The time machine looks like a compact desktop appliance about the size of a toaster: brushed metal body, circular time dial with glowing blue light, small display, and a single illuminated control knob.

### Example (X/Twitter Link)

And a second run produced this one:

[Open video link](https://github.com/user-attachments/assets/59e9f752-4d0c-43fd-96bf-711134e65944)

[Open X/Twitter post](https://twitter.com/jc50000000/status/2029412416472203277)

**Prompt used:**
> Scene tone: quiet, reflective, fragmented memory. Cinematic realism, muted natural colors. Overcast but DRY weather. No rain, no raindrops, no wet falling precipitation.
>
> START FRAME (0-2.5s)  
> Extreme close-up (85mm) of the elderly man's face. He breathes slowly. A tiny tremor in the lower eyelid. Strands of white hair drift gently in a light breeze.  
> Dialogue (man, barely above a whisper):  
> "I remember."
>
> Motion: micro push-in only.
>
> JUMP CUT 1 (2.5-5s)  
> Hard cut to an extreme close-up of his hands: weathered fingers rubbing a small object (a coin / pebble / ring) in his palm.  
> Dialogue (man):  
> "Not the day..."
>
> Motion: hands move slowly, deliberately.
>
> JUMP CUT 2 (5-7.5s)  
> Hard cut to close-up (50-85mm) of his boots stepping into soft mud at the lake edge. The movement is careful, almost hesitant. No splashing, just a quiet press into wet ground.  
> Dialogue (man):  
> "The feeling."
>
> Motion: one slow step, then stillness.
>
> JUMP CUT 3 (7.5-10s)  
> Hard cut to close-up of the lake surface: perfectly still water with faint ripples spreading outward (from a dropped pebble or a gentle touch).  
> Dialogue (man):  
> "It stayed."

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
- **Models**:
  - [LTX-2 Unified](https://huggingface.co/notapalindrome/ltx2-mlx-av) (~42GB, synchronized audio+video)
  - [LTX-2.3 Distilled Q4](https://huggingface.co/dgrauet/ltx-2.3-mlx-distilled-q4) (~19.4GB, synchronized audio+video)
- **Precision**: bfloat16

### Architecture

Generation uses a 2-stage pipeline:
1. Stage 1: Generate at half resolution
2. Stage 2: Upsample and refine to full resolution

## Troubleshooting

### "Model download stuck"
The download progress updates every 1%. Download time depends on selected model size (~19.4GB or ~42GB). Be patient.

### "Out of memory"
- Reduce resolution (512x320 is fastest)
- Reduce frame count (25/33/49 recommended)
- Use 24 FPS
- Set VAE tiling to aggressive
- Close other applications
- 32GB RAM minimum, 64GB recommended

### "Python not found"
- Install Python via Homebrew: `brew install python@3.12`
- Or use pyenv: `pyenv install 3.12`
- Then click "Auto Detect" in Preferences

### "LTX 2.3 conversion / LoRA compatibility"
- This app supports multiple AV model repos, including `notapalindrome/ltx2-mlx-av` and `dgrauet/ltx-2.3-mlx-distilled-q4`.
- Converting additional upstream checkpoints can require package-level updates in `mlx-video-with-audio` before they run reliably here.
- Standard LTX LoRA workflows are not guaranteed to transfer directly to the MLX-converted AV path without conversion tooling support.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Lightricks](https://www.lightricks.com/) for the LTX-2 model
- [mlx-video-with-audio](https://pypi.org/project/mlx-video-with-audio/) for unified audio-video generation
- [MLX Community](https://huggingface.co/mlx-community) for the MLX-converted weights
- [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) for the original MLX video generation code
- [Hugging Face](https://huggingface.co/) for model hosting
