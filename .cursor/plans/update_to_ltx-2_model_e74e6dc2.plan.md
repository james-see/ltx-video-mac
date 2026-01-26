---
name: Update to LTX-2 Model
overview: Fix the LTX model integration by updating from the incorrect LTXPipeline to the proper LTX2Pipeline for the LTX-2 model, with appropriate parameter and export handling changes.
todos:
  - id: update-imports
    content: Update imports to use LTX2Pipeline and encode_video from diffusers
    status: completed
  - id: update-model-loading
    content: Change from LTXPipeline to LTX2Pipeline with bfloat16 dtype
    status: completed
  - id: update-defaults
    content: Update default parameters (steps=40, guidance=4.0, frames=121)
    status: completed
  - id: handle-audio
    content: Update generate() to handle audio output from LTX-2
    status: completed
  - id: update-export
    content: Use encode_video utility for video export with audio
    status: completed
  - id: update-requirements
    content: Update requirements.txt with latest diffusers and transformers
    status: completed
  - id: test-mps
    content: Test on MPS and add memory optimization if needed
    status: completed
isProject: false
---

# Update to LTX-2 Model

## Problem

GitHub issue #3 asks about model version "0.9.1". The current code in [ltx_generator.py](LTXVideoGenerator/Resources/ltx_generator.py) has a mismatch:

- References `"Lightricks/LTX-2"` (the newest model)
- But uses `LTXPipeline` (for older LTX-Video 0.9.x models)

LTX-2 requires `LTX2Pipeline` which has different parameters and outputs (including audio).

## Changes to [ltx_generator.py](LTXVideoGenerator/Resources/ltx_generator.py)

### 1. Update imports

```python
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
```

### 2. Update model loading

```python
self.pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2",
    torch_dtype=torch.bfloat16  # LTX-2 uses bfloat16, not float16
)
```

Note: MPS (Apple Silicon) may have limited bfloat16 support. May need fallback to float32 or use `enable_model_cpu_offload()`.

### 3. Update default parameters

- `num_inference_steps`: 50 -> 40 (LTX-2 default)
- `guidance_scale`: 5.0 -> 4.0 (LTX-2 default)
- `num_frames`: 97 -> 121 (LTX-2 default)

### 4. Handle audio output

LTX-2 returns both video and audio:

```python
video, audio = self.pipe(
    prompt=prompt,
    # ... other params
    return_dict=False,
)
```

### 5. Update video export

Use the new `encode_video` utility that handles audio:

```python
encode_video(
    video[0],
    fps=fps,
    audio=audio[0].float().cpu(),
    audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
    output_path=output_path,
)
```

## VRAM Considerations

LTX-2 is a 19B parameter model. For Mac with limited VRAM:

- Use `pipe.enable_model_cpu_offload()` to reduce memory
- Consider the distilled version `ltx-2-19b-distilled` (8 steps, faster)
- Or quantized versions (`fp8`, `fp4`) if available for MPS

## Requirements Update

Update [requirements.txt](LTXVideoGenerator/requirements.txt):

- Ensure diffusers is latest version (0.36.0+ or install from source for LTX2Pipeline)
- Add transformers for Gemma text encoder