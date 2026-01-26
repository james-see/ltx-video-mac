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
| Quick Preview | 512×320 | 49 | 20 | Quick tests |
| Standard | 768×512 | 121 | 40 | Balanced quality |
| High Quality | 768×512 | 121 | 50 | Best results |
| Portrait | 512×768 | 121 | 40 | Vertical videos |
| Square | 512×512 | 121 | 40 | Social media |
| Cinematic 21:9 | 768×320 | 121 | 40 | Wide format |

{: .tip }
For faster iteration, select the **Distilled** model variant in Preferences. It generates videos in ~8 steps instead of 40.

### 3. Generate

Click the **Generate** button. The button shows:
- **Spinner** while generating
- **"Complete!"** (green) when finished

### 4. View Results

- **Queue** sidebar shows real-time progress with step count
- **History** tab displays all generated videos with thumbnails
- Click a video to preview it with looping playback

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
- **Reorder** items via right-click context menu
- **Clear** the entire queue with the Clear button

## History Features

### Browse Videos

- Thumbnails show a frame from each video
- Sort by newest, oldest, or prompt alphabetically
- Search prompts to find specific videos

### Video Details

Click a video to see:
- Full video preview (loops automatically)
- Original prompt and negative prompt
- All generation parameters
- Timestamp and generation duration
- Seed value for reproducibility

### Actions

- **Show in Finder** - Reveal the video file
- **Share** - Share via macOS share sheet
- **Reuse Prompt** - Copy prompt to clipboard
- **Regenerate** - Create again with same seed
- **Delete** - Remove video and thumbnail

## Tips for Best Results

### Start Small

- Use **Fast Preview** preset first
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
- The app automatically uses memory-efficient techniques
