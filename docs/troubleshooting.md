---
layout: default
title: Troubleshooting
nav_order: 5
---

# Troubleshooting Guide
{: .no_toc }

Solutions for common issues.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation Issues

### "Python not configured"

**Problem:** App shows Python not configured error.

**Solution:**
1. Open Preferences (⌘,)
2. Enter the full path to your Python executable
3. Click Validate
4. Ensure you see version information

**Finding your Python path:**
```bash
# Virtual environment
~/ltx-venv/bin/python3

# pyenv
~/.pyenv/versions/3.11.9/bin/python3

# Check with which
which python3
```

### "Module not found" errors

**Problem:** Missing Python packages.

**Solution:**
```bash
# Activate your environment first
source ~/ltx-venv/bin/activate

# IMPORTANT: Install diffusers from git for LTX-2 support
pip install git+https://github.com/huggingface/diffusers.git

# Install other packages
pip install torch accelerate transformers
pip install safetensors sentencepiece numpy
pip install imageio imageio-ffmpeg opencv-python
```

### "cannot import name 'LTX2Pipeline'"

**Problem:** diffusers installed from PyPI doesn't have LTX-2 support yet.

**Solution:**
```bash
# Uninstall PyPI version and install from git
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers.git
```

### "MPS not available"

**Problem:** Metal Performance Shaders not detected.

**Causes:**
- Not running on Apple Silicon
- Outdated PyTorch version
- Environment issue

**Solution:**
```bash
# Check MPS availability
python3 -c "import torch; print(torch.backends.mps.is_available())"

# If False, update PyTorch
pip install --upgrade torch torchvision torchaudio
```

---

## Generation Issues

### "Generation Failed" error

**Problem:** Generation starts but fails partway through.

**Check the logs:**
1. Look at `/tmp/ltx_generation.log`
2. Or run the Python script manually to see errors

**Common causes:**
- Out of memory (try Distilled or FP8 model variant)
- Model not fully downloaded
- Network issue during model download

**Solution:**
```bash
# Clear partially downloaded model
rm -rf ~/.cache/huggingface/hub/models--Lightricks--LTX-2*

# Retry - model will redownload
```

**Try a lighter model:**
1. Open Preferences > Model
2. Select "LTX-2 Distilled (Fast)" or "LTX-2 FP8 (Low Memory)"

### Black video output

**Problem:** Video generates but appears completely black.

**Cause:** Usually a precision/dtype issue with MPS.

**Solution:** The app uses `float16` for MPS compatibility. If you're building from source, ensure:
```python
torch_dtype=torch.float16
```

### Very slow generation

**Problem:** Generation takes much longer than expected.

**Solutions:**
1. Check that MPS is being used (not CPU)
2. Close other memory-intensive applications
3. Reduce resolution or frame count
4. Ensure you're using a supported preset

---

## Memory Issues

### "Out of memory" or crash

**Problem:** App crashes during generation.

**Solutions:**

1. **Use a lighter model variant:**
   - In Preferences > Model, select "LTX-2 FP8 (Low Memory)"
   - Or "LTX-2 Distilled (Fast)" for fewer steps
   
2. **Reduce resolution:**
   - Use 512×320 instead of higher resolutions
   
3. **Reduce frame count:**
   - Start with 49 frames instead of 121
   
4. **Close other apps:**
   - Safari, Chrome use significant memory
   - Other ML applications
   
5. **Check available memory:**
   ```bash
   # In Terminal
   vm_stat | head -5
   ```

{: .note }
LTX-2 is a 19B parameter model. For best results, 32GB+ unified memory is recommended. Macs with 16GB may struggle with the full model.

### Model download stuck

**Problem:** Model download appears stuck or very slow.

**Solutions:**
1. Check internet connection
2. Try again later (HuggingFace servers may be busy)
3. Download manually:
   ```bash
   huggingface-cli download Lightricks/LTX-2
   ```
4. Use a smaller variant first:
   ```bash
   huggingface-cli download Lightricks/LTX-2 --include "ltx-2-19b-dev-fp8/*"
   ```

---

## App Issues

### App won't launch

**Problem:** App bounces in dock but won't open.

**Solutions:**
1. Right-click and select "Open" for first launch
2. Check System Settings > Privacy & Security
3. Try moving to Applications folder
4. Verify notarization:
   ```bash
   spctl -a -vv "/Applications/LTX Video Generator.app"
   ```

### Preferences not saving

**Problem:** Settings reset after restart.

**Solution:**
1. Check write permissions to preferences:
   ```bash
   ls -la ~/Library/Preferences/com.jamescampbell.ltxvideogenerator.plist
   ```
2. Reset preferences:
   ```bash
   defaults delete com.jamescampbell.ltxvideogenerator
   ```

### Videos not appearing in History

**Problem:** Generated videos don't show in History tab.

**Check:**
1. Look in the default output folder (~/Movies/LTXVideoGenerator/)
2. Verify the video file exists
3. Check that thumbnails are being generated

---

## Getting More Help

### Debug Logging

Check the generation log:
```bash
cat /tmp/ltx_generation.log
```

### Python Environment Test

```bash
# Test your Python setup
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

from diffusers import LTX2Pipeline
print("Diffusers LTX2Pipeline: OK")

import imageio
print("imageio: OK")

print("\nAll checks passed!")
EOF
```

### Report an Issue

If problems persist:
1. Check existing [GitHub Issues](https://github.com/james-see/ltx-video-mac/issues)
2. Open a new issue with:
   - macOS version
   - Mac model (M1/M2/M3)
   - Python version
   - Error message or log output
   - Steps to reproduce
