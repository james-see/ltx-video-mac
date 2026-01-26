---
layout: default
title: Parameters
nav_order: 4
---

# Parameters Reference
{: .no_toc }

Detailed explanation of all generation parameters for LTX-2.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Model Variants

Select your model variant in **Preferences > Model**:

| Variant | Description | Steps | Guidance | Memory |
|:--------|:------------|:------|:---------|:-------|
| **Full (19B)** | Best quality | 40 | 4.0 | ~20GB |
| **Distilled** | Fast previews | 8 | 1.0 | ~20GB |
| **FP8** | Lower memory | 40 | 4.0 | ~12GB |

{: .note }
The **Distilled** model automatically uses CFG=1.0 and ignores negative prompts (required by the model architecture).

---

## Resolution

### Width & Height

Controls the dimensions of the generated video in pixels.

| Setting | Range | Default |
|:--------|:------|:--------|
| Width | 320-1024 | 768 |
| Height | 320-768 | 512 |

**Tips:**
- Both dimensions should be divisible by 64 for best results
- Higher resolutions require more memory and time
- Start with smaller sizes for testing

### Common Aspect Ratios

| Aspect | Width × Height | Use Case |
|:-------|:---------------|:---------|
| 16:10 | 512×320, 640×400, 768×480 | Standard |
| 16:9 | 768×432, 1024×576 | Widescreen |
| 9:16 | 384×640, 432×768 | Portrait/Mobile |
| 1:1 | 512×512 | Social media |
| 2.4:1 | 768×320 | Cinematic |

---

## Frames

### Number of Frames

Total frames to generate. More frames = longer video. Frames must be 8n+1 (9, 17, 25, 33, 41, 49... 121, etc).

| Setting | Range | Default |
|:--------|:------|:--------|
| Frames | 25-1000 | 121 |

**Duration calculation:**
```
Duration (seconds) = Frames ÷ FPS
```

Examples at 24 FPS:
- 49 frames = ~2 seconds
- 121 frames = ~5 seconds (default)
- 241 frames = ~10 seconds
- 481 frames = ~20 seconds

### FPS (Frames Per Second)

Playback speed of the generated video.

| Setting | Range | Default |
|:--------|:------|:--------|
| FPS | 8-30 | 24 |

**Tips:**
- 24 FPS: Cinematic feel
- 30 FPS: Smooth motion
- 12-15 FPS: Animation style

---

## Quality Settings

### Inference Steps

Number of denoising steps. More steps = higher quality but slower.

| Setting | Range | Default |
|:--------|:------|:--------|
| Steps | 10-100 | 40 |

**Recommendations by model:**
- **Distilled model:** 8 steps (fixed, fast previews)
- **Full/FP8 model:** 40 steps (balanced), 50+ for maximum quality

The quality improvement diminishes after ~50 steps.

### Guidance Scale

How closely the model follows your prompt (CFG - Classifier-Free Guidance).

| Setting | Range | Default |
|:--------|:------|:--------|
| Guidance | 1.0-15.0 | 4.0 |

**Effects:**
- **1.0:** No guidance (required for Distilled model)
- **3-5:** Balanced adherence (recommended for LTX-2)
- **6+:** Stronger prompt following, may reduce quality

**Recommended range:** 3.0 - 5.0 for LTX-2

---

## Seed

### Random Seed

Controls the randomness of generation.

| Setting | Range | Default |
|:--------|:------|:--------|
| Seed | 0-2147483647 | Random |

**Usage:**
- **Same seed + same parameters = identical output**
- Leave at -1 for random seed each time
- Save the seed of good results to reproduce them
- Try seeds ±1 from good results for variations

---

## Prompts

### Main Prompt

Describe what you want to see in the video.

**Best practices:**
- Be descriptive and specific
- Include camera movement if desired
- Describe motion and action
- Mention lighting and atmosphere

### Negative Prompt

Describe what you want to avoid.

**Common negative prompts:**
```
worst quality, low quality, blurry, 
jittery motion, distorted, 
watermark, text, logo,
static, still image
```

---

## Presets

Presets provide quick access to common configurations optimized for LTX-2:

### Quick Preview
```
512×320, 49 frames, 20 steps, guidance 4.0
```
Fast testing, ~1-2 minutes

### Standard (Default)
```
768×512, 121 frames, 40 steps, guidance 4.0
```
Good balance, ~5 minutes

### High Quality
```
768×512, 121 frames, 50 steps, guidance 4.0
```
Best results, ~7+ minutes

### Portrait
```
512×768, 121 frames, 40 steps, guidance 4.0
```
Vertical format for mobile

### Square
```
512×512, 121 frames, 40 steps, guidance 4.0
```
Social media format

### Cinematic 21:9
```
768×320, 121 frames, 40 steps, guidance 4.0
```
Ultra-wide aspect ratio

---

## Performance Notes

### Memory Usage

LTX-2 is a 19B parameter model requiring significant unified memory:

| Model Variant | Base Memory | With Generation |
|:--------------|:------------|:----------------|
| Full (19B) | ~20GB | ~25-30GB |
| Distilled | ~20GB | ~25-30GB |
| FP8 | ~12GB | ~15-20GB |

### Generation Time

Rough estimates on M2 Max (64GB):

| Preset | Full Model | Distilled |
|:-------|:-----------|:----------|
| Quick Preview | 2-3 min | 30-60 sec |
| Standard | 5-7 min | 1-2 min |
| High Quality | 8-12 min | 2-3 min |

Times vary based on:
- Chip (M1/M2/M3/M4) and core count
- Unified memory amount
- Other running applications
- Number of frames

### Audio Generation

LTX-2 generates synchronized audio with your video automatically. The audio is embedded in the output MP4 file.
