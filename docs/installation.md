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

## Install Python Dependencies

LTX Video Generator requires Python with PyTorch and the diffusers library. We recommend using a virtual environment.

### Option 1: Virtual Environment (Recommended)

```bash
# Create a dedicated virtual environment
python3 -m venv ~/ltx-venv

# Activate it
source ~/ltx-venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install diffusers 0.36.0+ (required for LTX-2)
pip install "diffusers>=0.36.0" accelerate transformers safetensors sentencepiece

# Install video export dependencies
pip install imageio imageio-ffmpeg opencv-python numpy

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
python -c "from diffusers import LTX2Pipeline; print('LTX2Pipeline OK')"
```

### Option 2: Using pyenv

```bash
# Install Python 3.11 via pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Install packages globally or in a virtualenv
pip install torch torchvision torchaudio
pip install diffusers accelerate transformers safetensors sentencepiece
pip install imageio imageio-ffmpeg opencv-python
```

### Option 3: Conda/Miniforge

```bash
# Create conda environment
conda create -n ltx python=3.11
conda activate ltx

# Install PyTorch
pip install torch torchvision torchaudio

# Install other dependencies
pip install diffusers accelerate transformers safetensors sentencepiece
pip install imageio imageio-ffmpeg opencv-python
```

## Configure the App

1. Launch **LTX Video Generator**
2. Open **Preferences** (⌘,)
3. In the **Python Path** field, enter your Python executable:
   - Virtual env: `~/ltx-venv/bin/python3`
   - pyenv: `~/.pyenv/versions/3.11.9/bin/python3`
   - Conda: `~/miniforge3/envs/ltx/bin/python`
4. Click **Validate** to verify the configuration
5. You should see a green checkmark and version information

## First Run

On your first generation:

1. The app will download the LTX-2 model (~20-40GB depending on variant)
2. This may take 10-30 minutes depending on your internet connection
3. The model is cached locally for future use
4. Subsequent generations will start much faster

### Model Variants

You can select your preferred model in **Preferences > Model**:
- **Full (19B)** - Best quality, largest download
- **Distilled** - Same size, but only needs 8 inference steps
- **FP8** - Quantized, smaller memory footprint

### Model Storage Location

The model is cached by Hugging Face in:
```
~/.cache/huggingface/hub/
```

{: .warning }
LTX-2 is a large model. Ensure you have at least 50GB free disk space for model downloads.

## Verify Installation

To verify everything is working:

1. Enter a simple prompt: `"A river flowing through a forest"`
2. Use the **Fast Preview** preset
3. Click **Generate**
4. Watch the progress in the Queue sidebar
5. Your video should appear in History when complete

## Troubleshooting

### "Python not configured"
- Make sure you've set the Python path in Preferences
- Verify the path exists: `ls -la /path/to/python3`

### "Module not found" errors
- Activate your virtual environment before checking
- Run `pip list` to verify packages are installed

### MPS not available
- Ensure you're on Apple Silicon (M1/M2/M3/M4)
- Update to the latest PyTorch version
- Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

See the [Troubleshooting Guide](troubleshooting) for more solutions.
