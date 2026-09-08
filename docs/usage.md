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

Enter a descriptive text prompt. For LTX-2.5, write one flowing present-tense paragraph (shot, lighting, action, character, camera, audio). Spoken lines only in quotes.

**Good prompt:**
> Handheld medium close-up, cool overcast light on a street corner. A busker in his late 40s — weathered face, grey stubble, battered acoustic guitar — stops mid-riff and looks past the camera. Soft traffic hum. He speaks in a low urgent whisper, "They're coming." He pauses and holds the silence.

**Less effective:**
> "busker scary" / `Beat.` / `JUMP CUT 1 (2.5–5s): …`

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
The LTX-2 Unified model uses a 2-stage generation pipeline with built-in audio. ltx-2-mlx distilled (LTX-2.5 Distilled / Q8 DiT and the 12GB 2.3 pack) is a single 8-step pass. `ltx25_dev` is half-res dev+CFG then distilled-LoRA refine. MiniMax H3 runs the native `h3` binary (24 fps, frames snap to 5+17n).

### MiniMax H3 variants

All three stay on Metal `h3.c`. No MLX-H3, ANE, GGUF, or ComfyUI runtime.

| `model_id` | Weights | Disk | RAM | Steps | Notes |
|:-----------|:--------|:-----|:----|:------|:------|
| `minimax_h3` | Official MiniMaxAI BF16 | ~144GB | 32GB+ | 4 / 20 / 50 presets | Stock `antirez/h3.c` |
| `minimax_h3_int8` | Comfy-Org int8 DiT + MiniMaxAI TE/VAE | ~92GB (~20GB extra if BF16 is cached) | 24GB+ | Same presets | `h3.c-int8` (GPU ConvRot de-rot). Slight quality drop. |
| `minimax_h3_turbo` | Official BF16 + folded Turbo v4 LoRA | ~144GB + ~3GB CoW | 32GB+ | Fixed 6 | First generate bakes the LoRA. Do not combine with reuse. |

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

## Writing Effective Prompts (LTX-2.5)

Follow the official LTX-2.5 guides: [prompt guide](https://ltx.io/blog/ltx-2-5-prompt-guide) · [docs](https://docs.ltx.io/api-documentation/implementation-guides/prompting-guide). Copy-paste examples: [EXAMPLES.md](https://github.com/james-see/ltx-video-mac/blob/main/EXAMPLES.md).

### Six ingredients

1. **Shot** — scale / lens / genre look  
2. **Scene** — lighting, palette, atmosphere  
3. **Action** — present-tense verbs that move  
4. **Characters** — age, clothes, physical emotion cues (not abstract labels)  
5. **Camera** — how and when it moves  
6. **Audio** — ambience, music, speech  

### Structure

- **Single take:** one flowing paragraph, ~4–8 sentences, present tense.  
- **Dialogue:** only spoken words in `"quotes"`; name language / accent / delivery.  
- **Pauses:** write as action (*he pauses*, *holds the silence*). Avoid bare `Beat.` — the model may speak it.  
- **Multishot:** one chronological paragraph; name cuts in prose (*A hard cut transitions to…*); re-ID subjects and state audio continuity. Do **not** use `START FRAME` / numbered JUMP CUT lists.  
- **Mac / Distilled–Dev:** prefer a single continuous take for dialogue + face; native multishot / Prompt Relay / DFR is incomplete on `ltx-2-mlx`.

### Leave enhancement off when dense

A well-formed 2.5 prompt often needs no rewrite. Gemma / `--enhance-prompt` helps short or foreign-model prompts; leave it off when you already wrote the full cinematic paragraph.

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
