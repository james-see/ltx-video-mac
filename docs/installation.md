---
layout: default
title: Installation
nav_order: 2
---

# Installation Guide
{: .no_toc }

Complete setup instructions for LTX Video Generator.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Download the App

1. Go to the [Releases page](https://github.com/james-see/ltx-video-mac/releases)
2. Download the latest `.dmg` file
3. Open the DMG and drag **LTX Video Generator** to your Applications folder
4. Right-click the app and select **Open** (required for first launch of notarized apps)

## Configure Python

LTX Video Generator needs Python 3.10+ with MLX and related packages.

### Step 1: Open Preferences

1. Launch **LTX Video Generator**
2. Open **Preferences** (⌘,)
3. Click **Auto Detect** to find your Python installation

The app will search common locations including Homebrew, pyenv, conda, and system Python.

### Step 2: Validate Setup

Click **Validate Setup** to check for required packages:
- `mlx` - Apple's machine learning framework
- `mlx-vlm` - Vision-language models for MLX
- `mlx-video-with-audio` - Unified audio-video generation (LTX-2 / 2.3)
- `ltx-2-mlx` - LTX-2.5 only (git install, not required for 2.3)
- `transformers` - Hugging Face transformers
- `safetensors` - Fast tensor serialization
- `huggingface_hub` - Model downloading
- `numpy` - Numerical computing
- `opencv-python` - Video encoding
- `tqdm` - Progress bars

### Step 3: Install Missing Packages

If packages are missing, you have two options:

**Option A: One-Click Install (Recommended)**

Click the **Install Missing Packages** button in Preferences. The app will run pip install automatically.

**Option B: Manual Install**

```bash
pip install mlx mlx-vlm mlx-video-with-audio transformers safetensors huggingface_hub numpy opencv-python tqdm
```

{: .note }
If using a virtual environment, make sure to activate it first, or point the app to the venv's Python executable.

## LTX-2.5 (`ltx-2-mlx`)

LTX-2.5 is opt-in and is **not** installed for 2.3-only users. When you select a 2.5 model, the app tries to pip-install from git into the venv:

```bash
pip install \
  "git+https://github.com/dgrauet/ltx-2-mlx.git@v0.15.2#subdirectory=packages/ltx-core-mlx" \
  "git+https://github.com/dgrauet/ltx-2-mlx.git@v0.15.2#subdirectory=packages/ltx-pipelines-mlx" \
  "mlx-lm>=0.31.2"
```

Or clone to `~/projects/ltx-2-mlx`, run `uv sync --all-extras`, and enable **Use local ltx-2-mlx repo** in Preferences.

## MiniMax H3 (`h3.c`)

H3 does not use Python. Build the binary yourself (not bundled in the DMG):

```bash
git clone https://github.com/antirez/h3.c
cd h3.c
make -j8
```

Put `./h3` at `~/projects/h3.c/h3`, on `PATH`, or set **h3 binary path** in Preferences. Selecting H3 (or the first Generate) shows an Accept / Read license dialog for the MiniMax H3 Community License. Weights then download automatically into the Hugging Face cache (`MiniMaxAI/MiniMax-H3`, ~144GB). Override with **H3 model directory** if you already have a snapshot. If the repo is gated, run `hf auth login` first. US/EU/UK/KR users may need [territory authorization](https://platform.minimax.io/h3-license).

## First Run - Model Download

{: .warning }
**Important**: On your first generation, the app downloads the model selected in Preferences from Hugging Face. This is a one-time download.

### What to Expect

1. Start a generation with any prompt
2. Progress shows "Downloading model..." with percentage
3. Download takes 30-60 minutes depending on connection speed
4. Model is cached in `~/.cache/huggingface/hub/`
5. Subsequent runs skip the download

### Download Progress

The app shows real-time download progress:
- `Downloading: 8.4GB / 42GB (20%)`
- `Downloading: 4.0GB / 19.4GB (21%)`

If download is interrupted, it will resume from where it left off.

### Storage Location

Models are cached by Hugging Face in folders such as:
```
~/.cache/huggingface/hub/models--notapalindrome--ltx2-mlx-av/
~/.cache/huggingface/hub/models--dgrauet--ltx-2.3-mlx-distilled-q4/
~/.cache/huggingface/hub/models--mlx-community--ltx-2.5-mlx/
~/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/
```

To store new downloads on another disk:

1. Open **Preferences > General > Storage**
2. Set **Model Cache Directory** to a folder on the mounted disk
3. Start generation after confirming the disk is still mounted

The app passes the selected directory to Hugging Face through `HF_HOME` and stores Hub downloads in its `hub` subfolder. Existing cache files are not moved automatically. Copy them from `~/.cache/huggingface/` into the selected directory before removing the originals.

If the selected folder is unavailable or not writable, generation stops with an error instead of silently downloading models to the internal disk.

To free up space later, you can delete this folder (the model will re-download on next use).

## Optional: Gemma Prompt Enhancement

For better results, enable **Settings > Generation > Enable Gemma Prompt Enhancement**. Gemma rewrites your prompts with vivid details before generation.

If prompts with certain words (e.g. medical terms) return empty enhancement, enable **Use uncensored enhancer**. First run downloads ~7GB (TheCluster/amoral-gemma-3-12B-v2-mlx-4bit).

## Verify Installation

To verify everything is working:

1. Enter a simple prompt: `"A river flowing through a forest"`
2. Use the **Quick Preview** preset (512x320, 49 frames)
3. Click **Generate**
4. Watch progress in the Queue sidebar
5. Your video should appear when complete

## Installing Python

If you don't have Python installed, here are your options:

### Homebrew (Recommended)

```bash
brew install python@3.12
```

Python will be at `/opt/homebrew/bin/python3.12`

### pyenv

```bash
# Install pyenv
brew install pyenv

# Install Python
pyenv install 3.12

# Set as default
pyenv global 3.12
```

### System Python

macOS includes Python, but you may need to install Xcode Command Line Tools:

```bash
xcode-select --install
```

## Troubleshooting

### "Python not found"
- Click **Auto Detect** in Preferences
- Or manually enter the path to your Python executable
- Verify it exists: `which python3`

### "Missing packages" after install
- Make sure you're using the same Python the app is configured to use
- Try: `/path/to/your/python3 -m pip install mlx mlx-vlm mlx-video-with-audio transformers safetensors huggingface_hub numpy opencv-python tqdm`

### "Out of memory" during generation
- Use smaller resolution (512x320)
- Reduce frame count
- Close other applications
- 32GB RAM minimum required

See the [Troubleshooting Guide](troubleshooting) for more solutions.
