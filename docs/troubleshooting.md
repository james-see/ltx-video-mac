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
2. Click **Auto Detect** to find Python automatically
3. Or manually enter the path to your Python executable
4. Click **Validate Setup**

**Finding your Python path:**
```bash
# Homebrew
/opt/homebrew/bin/python3

# pyenv
~/.pyenv/versions/3.12.x/bin/python3

# System
/usr/bin/python3

# Check with which
which python3
```

### "Missing packages" error

**Problem:** Required Python packages not installed.

**Solution:**

**Option 1:** Click the **Install Missing Packages** button in Preferences (recommended)

**Option 2:** Install manually:
```bash
pip install mlx mlx-vlm mlx-video-with-audio transformers safetensors huggingface_hub numpy opencv-python tqdm
```

{: .note }
Make sure you're installing to the same Python that the app is configured to use.

---

## Model Download Issues

### Download stuck or very slow

**Problem:** Model download appears stuck at a percentage.

**Explanation:** The model is ~42GB. Progress updates every 1%, so each percent is ~420MB. Even at 100Mbps, each percent takes about 35 seconds.

**Solutions:**
1. Be patient - large model downloads take time
2. Check your internet connection
3. The download will resume if interrupted

### "No space left on device"

**Problem:** Not enough disk space for model download.

**Solution:**
1. The model requires ~100GB free space
2. Clear old HuggingFace models:
   ```bash
   # See what's cached
   du -sh ~/.cache/huggingface/hub/*
   
   # Remove old/unused models
   rm -rf ~/.cache/huggingface/hub/models--OLD-MODEL-NAME
   ```

### Download interrupted

**Problem:** Download was interrupted partway through.

**Solution:** Simply retry the generation. HuggingFace automatically resumes downloads from where they left off.

To force a fresh download:
```bash
rm -rf ~/.cache/huggingface/hub/models--notapalindrome--ltx2-mlx-av
```

### "App downloaded Lightricks/LTX-2 (~150GB) - I only want the unified model"

**Problem:** Cache grew to 400GB+ because both the unified model (~42GB) and Lightricks/LTX-2 (~150GB) were downloaded. The app now uses only the unified model (`notapalindrome/ltx2-mlx-av`).

**Solution:** Remove the unused Lightricks cache to free ~150GB:
```bash
# See cache sizes
du -sh ~/.cache/huggingface/hub/*

# Remove Lightricks model (app no longer uses it)
rm -rf ~/.cache/huggingface/hub/models--Lightricks--LTX-2
```

Keep `models--notapalindrome--ltx2-mlx-av` (~42GB) - that's the model the app uses.

---

## Generation Issues

### "Generation Failed" error

**Problem:** Generation starts but fails with an error.

**Check the logs:**
```bash
cat /tmp/ltx_generation.log
```

**Common causes:**
- Out of memory - reduce resolution or frames
- Model not fully downloaded
- Corrupted model cache

**Solution:**
1. Try a smaller resolution (512x320)
2. Reduce frame count
3. If model seems corrupted, delete and re-download:
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--notapalindrome--ltx2-mlx-av
   ```

### Black video output

**Problem:** Video generates but appears completely black.

**Cause:** Usually a memory or precision issue.

**Solution:**
1. Reduce resolution
2. Reduce frame count
3. Close other applications to free memory

### Very slow generation

**Problem:** Generation takes much longer than expected.

**Solutions:**
1. First generation is slower (model loading)
2. Close other memory-intensive applications
3. Check Activity Monitor for memory pressure
4. Reduce resolution or frame count

---

## Memory Issues

### "Out of memory" or crash

**Problem:** App crashes during generation.

**Solutions:**

1. **Reduce resolution:**
   - Use 512×320 instead of higher resolutions
   
2. **Reduce frame count:**
   - Start with 49 frames instead of 97+
   
3. **Close other apps:**
   - Safari, Chrome use significant memory
   - Other AI/ML applications
   
4. **Check memory usage:**
   - Open Activity Monitor
   - Look at Memory Pressure graph
   - Should have minimal swap usage

{: .warning }
LTX-2 is a 19B parameter model. **32GB RAM minimum** required. 64GB+ recommended for higher resolutions.

### Generation slower than expected

**Problem:** Each generation takes longer than it should.

**Possible causes:**
1. Memory pressure causing swap usage
2. Thermal throttling on hot Mac
3. Background processes

**Solutions:**
1. Close other applications
2. Ensure Mac has good ventilation
3. Check Activity Monitor for CPU/memory hogs

---

## App Issues

### App won't launch

**Problem:** App bounces in dock but won't open.

**Solutions:**
1. Right-click and select "Open" for first launch
2. Check System Settings > Privacy & Security
3. Try moving to Applications folder

### Videos saving to wrong location

**Problem:** Videos not saving to configured directory.

**Solution:**
1. Open Preferences
2. Check the **Output Directory** setting
3. Click **Browse** to select your preferred folder
4. Ensure the folder exists and is writable

### Videos not appearing in History

**Problem:** Generated videos don't show in History tab.

**Check:**
1. Look in your configured output directory
2. Or check the default: `~/Library/Application Support/LTXVideoGenerator/Videos/`
3. Verify the video file exists

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
import mlx.core as mx
print(f"MLX device: {mx.default_device()}")

import transformers
print(f"Transformers: {transformers.__version__}")

from huggingface_hub import snapshot_download
print("HuggingFace Hub: OK")

import cv2
print(f"OpenCV: {cv2.__version__}")

print("\nAll checks passed!")
EOF
```

### Report an Issue

If problems persist:
1. Check existing [GitHub Issues](https://github.com/james-see/ltx-video-mac/issues)
2. Open a new issue with:
   - macOS version
   - Mac model (M1/M2/M3/M4) and RAM
   - Python version
   - Error message from `/tmp/ltx_generation.log`
   - Steps to reproduce
