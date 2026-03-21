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

## Model

The app supports two LTX-2 models running on MLX (Apple's machine learning framework). Select your model in Preferences.

| Model | Parameters | Size | Vocoder | Notes |
|:------|:-----------|:-----|:--------|:------|
| LTX-2 Unified (`notapalindrome/ltx2-mlx-av`) | 19B | ~42GB | Standard | Original unified model |
| LTX-2.3 Distilled Q4 (`dgrauet/ltx-2.3-mlx-distilled-q4`) | 19B (Q4) | ~19.4GB | BigVGAN | Quantized, smaller download |

Both models use a 2-stage pipeline:
1. **Stage 1:** Generate at half resolution
2. **Stage 2:** Upsample and refine to full resolution
3. **Audio:** Synchronized stereo audio generated alongside video (24kHz → 48kHz with bandwidth extension)

See [Architecture](architecture) for full technical details on how each model is loaded and decoded.

---

## Resolution

### Width & Height

Controls the dimensions of the generated video in pixels.

| Setting | Range | Default |
|:--------|:------|:--------|
| Width | 320-1024 | 512 |
| Height | 320-768 | 320 |

**Tips:**
- Both dimensions should be divisible by 64 for best results
- Higher resolutions require more memory and time
- Start with smaller sizes (512x320) for testing

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

Total frames to generate. More frames = longer video.

| Setting | Range | Default |
|:--------|:------|:--------|
| Frames | 17-257 | 49 |

{: .note }
Frames are automatically adjusted to be 8n+1 (9, 17, 25, 33, 41, 49... 121, etc.) as required by the model.

**Duration calculation:**
```
Duration (seconds) = Frames ÷ FPS
```

Examples at 24 FPS:
- 49 frames = ~2 seconds
- 97 frames = ~4 seconds
- 121 frames = ~5 seconds

### FPS (Frames Per Second)

Playback speed of the generated video.

| Setting | Range | Default |
|:--------|:------|:--------|
| FPS | 12-30 | 24 |

**Tips:**
- 24 FPS: Cinematic feel
- 30 FPS: Smooth motion
- 12 FPS: Animation style
- For synchronized speech/lip-sync, use 24 FPS

---

## Quality Settings

### Inference Steps

Number of denoising steps per stage.

| Setting | Range | Default |
|:--------|:------|:--------|
| Steps | 8-50 | 28 |

The LTX-2 Unified model uses a fixed sigma schedule. More steps provide diminishing returns after ~30.

### Guidance Scale

How closely the model follows your prompt (CFG - Classifier-Free Guidance).

| Setting | Range | Default |
|:--------|:------|:--------|
| Guidance | 1.0-10.0 | 4.0 |

**Effects:**
- **1.0:** Minimal guidance
- **3-5:** Balanced adherence (recommended)
- **6+:** Stronger prompt following

**Recommended:** 3.5 - 5.0 for LTX-2

---

## Seed

### Random Seed

Controls the randomness of generation.

| Setting | Range | Default |
|:--------|:------|:--------|
| Seed | 0-2147483647 | Random |

**Usage:**
- **Same seed + same parameters = identical output**
- Leave empty for random seed each time
- Save the seed of good results to reproduce them
- Try nearby seeds for variations

---

## Image-to-Video Settings

When using image-to-video mode:

### Image Strength

How much the source image influences the output.

| Setting | Range | Default |
|:--------|:------|:--------|
| Strength | 0.0-1.0 | 1.0 |

- **1.0:** Strong influence, first frame closely matches image
- **0.5:** Moderate influence
- **0.0:** Image is ignored (text-to-video mode)

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

Presets provide quick access to common configurations:

### Quick Preview
```
512×320, 49 frames, 20 steps, guidance 4.0
```
Fast testing, ~2-3 minutes

### Standard
```
768×512, 97 frames, 28 steps, guidance 4.0
```
Good balance, ~5-7 minutes

### High Quality
```
768×512, 121 frames, 28 steps, guidance 4.0
```
Best results, longer duration

### Portrait
```
512×768, 97 frames, 28 steps, guidance 4.0
```
Vertical format for mobile

### Square
```
512×512, 97 frames, 28 steps, guidance 4.0
```
Social media format

---

## Audio Settings

### Voiceover Source

Choose between local or cloud-based text-to-speech.

| Source | Quality | Speed | Requirements |
|:-------|:--------|:------|:-------------|
| MLX Audio (Local) | Good | Fast | None (free) |
| ElevenLabs (Cloud) | Excellent | Fast | API key |

### Voice Options

**MLX Audio Voices:**
- Heart, Bella, Nova, Sky (US Female)
- Adam, Echo (US Male)
- Alice, Emma (UK Female)
- Daniel, George (UK Male)

**ElevenLabs Voices:**
- Rachel, Domi, Bella, Elli (Female)
- Antoni, Josh, Arnold, Adam, Sam (Male)

### Background Music

Requires ElevenLabs API key. Music is generated to match video length automatically.

| Setting | Value |
|:--------|:------|
| Volume (music only) | 30% |
| Volume (with voiceover) | 20% |
| Format | AAC 192kbps |

### Music Genres

54 presets across 9 categories. Each genre has a carefully crafted prompt optimized for ElevenLabs Music API.

**Categories:**
- Electronic (8 genres)
- Hip-Hop/R&B (6 genres)
- Rock (6 genres)
- Pop (4 genres)
- Jazz/Blues (5 genres)
- Classical/Cinematic (6 genres)
- World (5 genres)
- Country/Folk (4 genres)
- Functional/Mood (10 genres)

---

## Performance Notes

### Memory Usage

LTX-2 is a 19B parameter model requiring significant unified memory:

| Resolution | Approximate Memory |
|:-----------|:------------------|
| 512×320 | ~24-28GB |
| 768×512 | ~28-32GB |
| 1024×768 | ~35-40GB+ |

### Generation Time

Rough estimates on Apple Silicon:

| Mac | Quick Preview | Standard |
|:----|:--------------|:---------|
| M1 (16GB) | May fail | Not recommended |
| M1 Pro (32GB) | 4-5 min | 8-10 min |
| M2 Max (64GB) | 2-3 min | 5-7 min |
| M3 Max (128GB) | 1-2 min | 3-5 min |

Times vary based on:
- Chip generation and core count
- Unified memory amount
- Other running applications
- Resolution and frame count

{: .warning }
**32GB RAM minimum required.** Macs with 16GB will likely fail or be extremely slow. 64GB+ recommended for comfortable usage.
