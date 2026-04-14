---
layout: default
title: Usage Guide
nav_order: 3
---

# Usage Guide
{: .no_toc }

Learn how to get the best results from LTX Video Generator.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Basic Workflow

### 1. Write Your Prompt

Enter a descriptive text prompt in the main text field. The more detail you provide, the better the results.

**Good prompt:**
> "The camera slowly pans across a misty forest at dawn, with rays of golden sunlight filtering through the trees"

**Less effective:**
> "forest"

### 2. Choose a Preset

Select a preset from the dropdown to quickly configure parameters:

| Preset | Resolution | Frames | Steps | Best For |
|:-------|:-----------|:-------|:------|:---------|
| Quick Preview | 512×320 | 49 | 15 | Quick tests |
| Standard | 768×512 | 121 | 30 | Balanced quality |
| High Quality | 768×512 | 121 | 40 | Best results |
| Portrait | 512×768 | 97 | 30 | Vertical videos |
| Square | 512×512 | 97 | 30 | Social media |

{: .tip }
The LTX-2 Unified model uses a 2-stage generation pipeline with built-in audio.

### 3. Generate

Click the **Generate** button. Progress shows:
- Model loading status
- Stage 1 and Stage 2 denoising progress
- Video encoding and saving

### 4. View Results

- **Queue** sidebar shows real-time progress
- **History** tab displays all generated videos with thumbnails
- Videos save to your configured output directory

## Gemma Prompt Enhancement

Improve results by having Gemma rewrite your prompt with vivid details:

1. Go to **Settings > Generation** and enable **Enable Gemma Prompt Enhancement**
2. In the prompt view, expand **Prompt Enhancement (Gemma)**
3. Optionally enable **Use uncensored enhancer** to avoid content filters (first run downloads ~7GB)
4. Click **Preview enhanced prompt** to see the rewritten prompt before generating
5. Generate as usual—the enhanced prompt is used automatically

{: .note }
If enhancement returns empty (e.g. safety filter), the app auto-retries with filtered words replaced, then merges originals back.

## Image-to-Video

You can animate images into videos:

1. On the main **Generate** screen (prompt column), expand the **Image to Video** disclosure section — it sits below **Prompt Enhancement** and above **Negative Prompt**.
2. Click **Select Source Image...** and pick an image file; it becomes the conditioned first frame.
3. Optionally adjust **Image Strength** (1.0 = full influence, lower = more motion freedom).
4. Write a prompt that describes the motion; then generate as usual.

## Adding Audio

### Voiceover / Narration

Add text-to-speech voiceover to your videos:

1. Expand the **Voiceover / Narration** section
2. Choose your source:
   - **MLX Audio (Local)** - Free, runs on-device, good quality
   - **ElevenLabs (Cloud)** - High quality, requires API key
3. Select a voice from the dropdown (10 voices for MLX, 9 for ElevenLabs)
4. Enter your narration text
5. Generate your video - audio will be added automatically

{: .tip }
You can also add audio later by right-clicking any video thumbnail in the **Video Archive**.

### Background Music

Add AI-generated instrumental music (requires ElevenLabs API key):

1. Expand the **Background Music** section
2. Toggle **Generate background music** on
3. Choose a genre from the dropdown:
   - Organized into 9 categories with 54 total presets
   - Preview the prompt by hovering over the selection
4. Music automatically matches your video length

### Music Genre Categories

| Category | Genres |
|:---------|:-------|
| Electronic | EDM, House, Techno, Ambient, Chillwave, Synthwave, D&B, Trance |
| Hip-Hop/R&B | Trap, Lo-Fi, Boom Bap, Slow R&B, Modern R&B, Soul |
| Rock | Classic, Alternative, Indie, Metal, Punk, Acoustic |
| Pop | Modern, Indie, Dance, Acoustic |
| Jazz/Blues | Smooth, Bebop, Lounge, Electric Blues, Acoustic Blues |
| Classical/Cinematic | Orchestral, Piano, Chamber, Epic, Tense, Uplifting |
| World | Latin, Reggae, Afrobeat, Middle Eastern, Asian |
| Country/Folk | Modern, Classic, Acoustic Folk, Indie Folk |
| Functional | Corporate, Motivational, Relaxing, Suspense, Action, Romantic, Happy, Sad, Dramatic, Mystery |

### Audio from History

Add audio to previously generated videos:

1. Go to **Video Archive**
2. Right-click any video thumbnail
3. Select **Add Audio** (or **Replace Audio** if it already has audio)
4. Choose from three tabs:
   - **Voiceover** - Add narration only
   - **Music** - Add background music only
   - **Both** - Add voiceover and music together

When combining voiceover and music, the music is automatically ducked to 20% volume so the voice remains clear.

## Writing Effective Prompts

### Include Camera Movement

```
"The camera slowly pans left revealing..."
"A drone shot flying over..."
"Close-up tracking shot of..."
"The camera pushes in toward..."
```

### Describe Motion

```
"waves gently crashing on the shore"
"leaves falling in slow motion"
"clouds drifting across the sky"
"a person walking through..."
```

### Specify Lighting

```
"at golden hour with warm lighting"
"under dramatic storm clouds"
"illuminated by neon city lights"
"in soft diffused morning light"
```

### Add Atmosphere

```
"with fog rolling through the valley"
"rain drops falling on the window"
"dust particles floating in sunbeams"
"snow gently falling"
```

## Using the Queue

### Add Multiple Generations

1. Write your prompt
2. Click **Add to Queue** (instead of Generate)
3. Modify the prompt or parameters
4. Add more to the queue
5. Videos generate one after another

### Batch Variations

Click the batch menu (stack icon) to:
- Generate 3 variations
- Generate 5 variations
- Each uses a random seed for different results

### Queue Management

- **Cancel** the current generation with the X button
- **Remove** pending items from the queue
- **Clear** the entire queue with the Clear button

## History Features

### Browse Videos

- Thumbnails show a frame from each video
- Sort by newest, oldest, or prompt alphabetically
- Search prompts to find specific videos

### Video Details

Click a video to see:
- Full video preview (loops automatically)
- Original prompt
- All generation parameters
- Timestamp and generation duration
- Seed value for reproducibility

### Actions

- **Show in Finder** - Reveal the video file
- **Share** - Share via macOS share sheet
- **Reuse Prompt** - Copy prompt back to input
- **Delete** - Remove video

## Tips for Best Results

### Start Small

- Use **Quick Preview** preset first
- Iterate on prompts quickly
- Only increase quality for final renders

### Use Negative Prompts

Click the disclosure arrow to add negative prompts:
```
worst quality, blurry, jittery, distorted, watermark
```

### Reproducible Results

- Note the seed value of good generations
- Enter the same seed to reproduce results
- Useful for making variations with slight prompt changes

### Memory Management

- Higher resolutions use more memory
- Close other apps if you encounter issues
- 32GB RAM minimum, 64GB recommended
