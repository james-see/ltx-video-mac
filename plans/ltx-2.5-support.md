# LTX 2.5 Support Plan

## Context

The ltx-video-mac app currently supports LTX-2, LTX-2.3, and LTX-2.3 Distilled Q4 via `mlx-video-with-audio` (James's fork of Blaizzy/mlx-video). LTX-2.5 was released Aug 11, 2026 by Lightricks with significant improvements: multi-shot scenes, real footage editing, cinema-grade EXR export, auto-predicted duration via DurationHead, and the DFR (Detailing/Finishing Rounds) pipeline.

The "m3" James heard about is MiniMax H3 — a 33B omni-modal video+audio model with amazing quality. Initial assessment was too dismissive; see the MiniMax H3 Research section below for distilled/quantized variants that make it more feasible (4-bit + Turbo LoRA can run on 64GB Macs, though generation is slow: ~35 min for a 5s clip).

## Key Finding: Two Separate MLX Codebases

There are **two independent MLX implementations** of the LTX-2 pipeline:

1. **`mlx-video-with-audio`** (James's fork of Blaizzy/mlx-video) — what the app uses now. Has LTX-2.0 and LTX-2.3. NOT upgradable to 2.5 without major work — the upstream Blaizzy/mlx-video doesn't have 2.5 support either.

2. **`ltx-2-mlx`** (dgrauet/ltx-2-mlx) — a separate, more advanced port with full LTX-2.5 support (Gemma 4 encoder, DurationHead, DiffVAE, connector module, 6 model packs including int4/int8 for 16-64GB Macs, block streaming for low-ram, modality tiling). NOT on PyPI — install is `git clone + uv sync`. 72 Python files vs our fork's 32.

The critical question: **do we add 2.5 support to `mlx-video-with-audio`, or do we switch the app to use `ltx-2-mlx`?**

## Recommendation: Phase 1 (Fast) — Use dgrauet's ltx-2-mlx directly

The fastest path to LTX 2.5 support is to make the app use `ltx-2-mlx` (dgrauet's repo) as the backend for 2.5 models, while keeping `mlx-video-with-audio` for 2.0/2.3 models. This avoids porting ~40 new Python files (Gemma 4 encoder, connector, DurationHead, DiffVAE, etc.) into our fork.

`ltx-2-mlx` has a CLI (`ltx-2-mlx generate --prompt ... --model <repo> --distilled -o output.mp4`) that's similar to what our app already calls (`python -m mlx_video.generate_av --prompt ... --model-repo <repo> --output-path <path>`).

## Phase 1: App Changes (ltx-video-mac)

### 1.1 Add a "backend" concept to LTXModel

In `GenerationRequest.swift`, add a `backend` field to `LTXModel`:

```swift
enum GenerationBackend: String, Codable {
    case mlxVideoWithAudio  // existing: python -m mlx_video.generate_av
    case ltx2Mlx            // new: ltx-2-mlx generate
}
```

Each model catalog entry specifies which backend it uses. The app's `LTXBridge.swift` dispatches to the right CLI based on the model's backend.

### 1.2 Add LTX 2.5 models to the catalog

```swift
// LTX-2.5 bf16 (128GB Macs)
LTXModel(
    id: "ltx25_distilled",
    repo: "mlx-community/ltx-2.5-mlx",
    displayName: "LTX-2.5 Distilled (bf16)",
    downloadSize: "~44GB",
    supportsBuiltInAudio: true,
    qualityWarning: "Requires 64GB+ RAM. Best quality.",
    recommendedStepsLower: 8,   // distilled is fixed 8-step
    recommendedStepsUpper: 8,
    tips: "LTX-2.5 distilled uses a fixed 8-step schedule. Multi-shot scenes, auto-duration prediction."
)

// LTX-2.5 8-bit (64GB Macs)
LTXModel(
    id: "ltx25_distilled_q8",
    repo: "mlx-community/ltx-2.5-mlx-ditq8",
    displayName: "LTX-2.5 Distilled Q8 (Beta)",
    downloadSize: "~24GB",
    supportsBuiltInAudio: true,
    qualityWarning: "8-bit quantized DiT. Needs 64GB+ RAM.",
    recommendedStepsLower: 8,
    recommendedStepsUpper: 8,
    tips: "8-bit quantized. Fits 64GB Macs."
)
```

### 1.3 Add Gemma 4 text encoder entries

LTX 2.5 uses Gemma-4-unified (not Gemma-3). The encoder ships *inside* the model pack (gemma4-12b-ltx-v1/ subdir), so no separate text encoder repo is needed — `ltx-2-mlx` auto-loads it from the pack.

Add a text encoder entry:
```swift
LTXTextEncoder(
    id: "gemma4_12b_pack",
    repo: "",  // loaded from model pack, not a separate repo
    displayName: "Gemma 4 12B (bundled with LTX-2.5 pack)",
    downloadSize: "included in model",
    qualityWarning: nil,
    tips: "Gemma-4-unified text encoder, bundled in the LTX-2.5 model pack."
)
```

### 1.4 Modify LTXBridge.swift to support ltx-2-mlx backend

Add a branch in `generate()` that calls `ltx-2-mlx generate` instead of `python -m mlx_video.generate_av` when the model's backend is `.ltx2Mlx`.

Key CLI differences:
- `ltx-2-mlx generate --prompt "..." --model <repo> --distilled -H <h> -W <w> -f <frames> --seed <s> -o <path>`
- No `--text-encoder-repo` (Gemma 4 is bundled in the pack)
- No `--tiling` (uses `--tile-frames N --tile-spatial M` instead)
- `--no-audio` → `--no-audio` (same)
- `--image` → `--image` (same)
- Auto-duration: omit `-f` on 2.5 packs (DurationHead predicts it)
- `--low-ram` for block streaming (16GB Macs)

### 1.5 Install ltx-2-mlx in the Python environment

The app's Python environment setup needs to install `ltx-2-mlx` (git clone + uv sync or pip install from git). Since it's not on PyPI, the app would need to either:
- Install from git: `pip install git+https://github.com/dgrauet/ltx-2-mlx.git#subdirectory=packages/ltx-pipelines-mlx`
- Or bundle a local checkout

### 1.6 Memory warnings in the UI

LTX 2.5 needs significantly more RAM:
- bf16: 62.4 GB peak (128GB Mac recommended)
- Q8: ~40 GB peak (64GB Mac recommended)
- Q8 + --low-ram: ~20 GB (32GB Mac)
- int4 + --low-ram: ~12 GB (16GB Mac)

Add warnings when user selects 2.5 models on low-RAM Macs.

## Phase 2 (Later) — Port 2.5 into mlx-video-with-audio

If we want to keep a single backend, we'd need to port into `mlx-video-with-audio`:

### Required new code (~40 new Python files):

1. **Gemma 4 text encoder** (~8 files)
   - `gemma4.py`, `gemma4_config.py`, `gemma4_encoder.py`
   - `embeddings_connector.py` (text embeddings connector + aggregate projections)
   - Alternating sliding/full attention, different RMSNorm (no +1), proportional rope
   - 49 hidden states (all layers) vs Gemma-3's approach

2. **DurationHead** (1 file)
   - `duration_head.py` — prompt→duration predictor
   - Small regression head on connector outputs

3. **DiffVAE** (1 file)
   - 1-step x0 video decoder with NA attention

4. **Config changes** (modify existing `config.py`)
   - Add: `ff_bias`, `audio_ff_bias`, `use_prompt_adaln_single`, `use_keyframes_abs_pos_embedding`, `double_precision_rope` (from `frequencies_precision: float64`)
   - `from_checkpoint_config()` classmethod to read from embedded_config.json

5. **FeedForward** (modify existing)
   - Already has `bias` parameter ✓ (no change needed)

6. **RoPE** (modify existing)
   - Already has `double_precision` support ✓ (no change needed)

7. **Weight loading/sanitization** (modify existing `convert.py`)
   - Support split-component file layout (transformer-distilled.safetensors, connector.safetensors, etc. separate files vs monolithic)
   - Handle ff_bias=False (no bias tensors in 2.5 weights)
   - Handle key name differences for 2.5

8. **Pipeline changes** (modify existing `generate_av.py`)
   - Auto-detect 2.5 pack via `ff_bias=False` in config
   - Load Gemma 4 encoder from pack subdirectory
   - Load connector weights
   - Optional: DurationHead for auto-duration
   - Optional: DiffVAE for 1-step decoding
   - Distilled 2.5 uses fixed 8-step schedule (CFG=1)

### Estimate: 2-3 days of focused work for Phase 2

## MiniMax H3 Research (Distilled & Quantized Variants)

H3 is more feasible than the initial assessment suggested. There's a whole ecosystem of distilled and quantized variants that dramatically shrink the requirements.

### Architecture

- 33B joint video+audio diffusion transformer (50 blocks, hidden 5376, SwiGLU ffn 14336, 3D MM-RoPE)
- Text encoder: Qwen3-VL-32B (frozen, H3 reads unnormalized hidden state after layer 50 of 64 — 14 layers never evaluated)
- Video VAE: ViT+CNN KL VAE, 16x spatial / 4x temporal, 24 latent channels (10.4 GB)
- Audio VAE: DAC/BigVGAN stereo 32 kHz (0.6 GB)
- Full self-attention over one packed 1-D sequence: `[ text | keyframe conditions | target audio | target video ]`
- CFG-distilled: no guidance scale, no negative prompt, no guider — one forward per step
- Output: up to 768p (2K via H3-Regenerate-2K post-processing), 4-15 seconds, 24 FPS, stereo audio

### Structural optimizations (before any quantization)

PipeNetwork's MLX port has two structural memory savings:

1. **AdaLN precompute**: 13B of the 33B params are per-block `adaln_proj.linear` projections whose only input is the timestep embedding. For a fixed sampler schedule, all modulation tensors can be computed once up front and the projections dropped. **Saves 25.3 GB** (66.3 GB → 40.3 GB + 745 MB cache).

2. **Encoder truncation**: H3 reads only layers 0-49 of Qwen3-VL-32B's 64 layers. The LM head, final norm, and layers 50-63 are never touched. **Saves 16.4 GB** (66.7 GB → 50.3 GB).

Together: resident pipeline drops from 144 GB to ~102 GB **before any quantization**.

### Published MLX quants (PipeNetwork collection on HuggingFace)

| Build | On disk | Resident (DiT only) | PSNR vs bf16 | Notes |
|-------|---------|---------------------|--------------|-------|
| f32 | 132.5 GB | 80.5 GB | — | Lossless widening, for fine-tuning not generation |
| bf16 | 66.3 GB | 40.3 GB | reference | Faithful conversion, mixed precision |
| 8-bit | 35.3 GB | 21.5 GB | 27.6 dB | Good quality, velocity rel-L2 0.033 |
| 6-bit | 30.3 GB | 16.5 GB | — | Velocity rel-L2 0.061 |
| 4-bit | 25.3 GB | 11.5 GB | 22.0 dB* | Usable at full resolution (*256x256 test is misleading) |

AdaLN held at 8-bit in every build (adds 0.25% error, saves 12.2 GB per download).

Plus text encoder (layers 0-49 only): ~50 GB bf16, ~26 GB 8-bit, ~15 GB 4-bit.
Plus VAEs: ~11 GB (video) + 0.6 GB (audio).

### Total resident memory estimates (with all components)

| Config | DiT | Text encoder (truncated) | VAEs | Total resident | Target Mac |
|--------|-----|--------------------------|------|----------------|------------|
| bf16 + AdaLN precompute | 40.3 GB | 50.3 GB bf16 | 11 GB | ~102 GB | 128GB+ |
| 8-bit DiT + 8-bit encoder | 21.5 GB | ~26 GB | 11 GB | ~59 GB | 64GB |
| 4-bit DiT + 4-bit encoder | 11.5 GB | ~15 GB | 11 GB | ~38 GB | 64GB (tight) |
| 4-bit DiT + 4-bit encoder + encoder eviction | 11.5 GB | load→evict→0 | 11 GB | ~23 GB | 32GB (experimental) |

### Distilled / Turbo variants

**FastVideo FastH3** (FastVideo/FastVideo-Minimax-FastH3-Preview-v0.1)
- 4-step distillation via data-free DMD2 (base model uses 50 steps)
- Cuts generation time ~12x (50 steps → 4 steps)
- Preview release, ComfyUI workflows available

**ModelTC/Minimax-H3-Turbo** (also on GitHub as groxaxo/minimax-h3-turbo)
- 4-step FL2V Turbo LoRA checkpoints
- ComfyUI workflows + diffusers batch inference
- Multiple LoRA variants from Lightx2v team:
  - `minimax_h3_fl2v_turbo_4step_v0.1` — 4-step
  - `minimax_h3_fl2v_turbo_8step_v1.0` — 8-step
  - Turbo-SLA: 4-step + Sparse-Linear Attention for even faster inference

**Fused Turbo single-file** (Comfy-Org)
- `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` — 21 GB
- INT8 pruned + Turbo LoRA baked in, one file load with stock UNETLoader
- Pruned INT8 FL2VA: 21 GB, pruned FP8: ~17 GB

**Comfy-Org quantized variants** (ComfyUI ecosystem, not MLX):
- BF16, FP8, INT8, INT4, NVFP4, GGUF (Q3_K_M=15.6GB, Q4_K_M=19.9GB)
- Pruned models: ~21 GB (INT8) vs ~34 GB (full INT8)
- INT4 diffusion model quality is poor — community recommends swapping only the text encoder to INT4, keep DiT at INT8+
- NVFP4 text encoder (xdkings/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4): 15.7 GB, fits 16 GB card

### How "faster than live" H3 videos work — three separate things

People posting "faster than real-time" H3 are using one of three very different setups, **none of which are the MLX port on Apple Silicon**:

**1. H3 Max (fal.ai, hosted API) — the "3 seconds for 5s clip" claims**
- This is a **cloud-hosted, post-trained variant** by fal.ai, NOT local generation
- fal took the open H3 weights, post-trained for better aesthetics/prompt adherence, then co-optimized with their custom inference stack (proprietary kernel optimizations, likely on H100/B200 clusters)
- 5s 768p clip in <3 seconds, 15s clip in ~15 seconds — genuinely faster than real-time
- 35x throughput of the official MiniMax H3 endpoint
- This is an API service, not something you can run locally. Not relevant to our app.

**2. FastH3 Preview v1 (FastVideo, open weights) — the "14x speedup" claims**
- Open-weight 4-step DMD2 distillation of H3, released Aug 27-28, 2026
- **Two key innovations**:
  - **4-step distillation**: base H3 uses 50 steps; FastH3 uses 4 (12.5x fewer forwards)
  - **90% Video Sparse Attention (VSA)**: keeps only 10% of video-to-video attention tiles (64-token blocks), text and audio stay dense. The sparse student was trained against a dense-attention teacher so it learns to approximate full attention. **This is the real speedup** — it attacks the attention FLOPs bottleneck that quantization can't.
- Benchmarks on NVIDIA Blackwell B200:
  - 15s clip at 1344x768: base H3 = 678.7s → FastH3 = **47.2s** on one B200 (14x)
  - 15s clip on 4xB200: **15.5s** (roughly real-time)
  - 15s clip on 8xB200: **12.88s** (faster than real-time)
- **Open weights**, ComfyUI workflows available. But designed for NVIDIA Blackwell (B200). RTX 5090 also benefits but less dramatically.

**3. Turbo LoRAs + SageAttention on RTX 5090 (local, consumer GPU)**
- Community Turbo LoRAs (Lightx2v, ModelTC, larryvrh): 4-step or 8-step distillation, no sparse attention
- SageAttention 2.2.0: INT8/FP8 quantized attention kernel, 2.7x faster than FlashAttention2 on RTX 5090
- Combined on RTX 5090: 5s clip at 1344x768 with 4-step Turbo + SageAttention = **~34-48 seconds** (vs 4+ min without optimizations)
- Still slower than real-time but usable for iteration

### Why the MLX port on Apple Silicon is so much slower

The PipeNetwork MLX port numbers I cited (8.8 min/step for 5s clip on M3 Ultra) are for the **base H3 with full dense attention, no Turbo LoRA, no sparse attention, no SageAttention**. The gap between "35 min on MLX" and "3 seconds on fal.ai" is explained by:

| Factor | MLX port (Apple Silicon) | FastH3 / H3 Max (NVIDIA) |
|--------|--------------------------|--------------------------|
| Steps | 8 (base distilled) | 4 (FastH3 DMD2) |
| Attention | Full dense (MiniMax hasn't released sparse) | 90% sparse VSA (FastH3) or proprietary (H3 Max) |
| Attention kernel | MLX flash attention | SageAttention 2.2.0 (INT8 quantized) or custom |
| Hardware | M3 Ultra (GPU ~27 TFLOPS fp16) | B200 (~2000 TFLOPS fp8) or H100 clusters |
| Post-training | None | fal.ai custom inference stack |

**The sparse attention is the key unlock.** The MLX port can't use it because:
1. MiniMax hasn't open-sourced their sparse attention implementation
2. FastH3's VSA is open-weight but designed for NVIDIA CUDA, not Metal
3. No MLX port of SageAttention exists yet (there's a GitHub repo `pythongiant/SageMLXAttention` and `mlx-mfa` Metal Flash Attention, but neither is integrated into the H3 MLX port)

There IS a project called **H3-metal** (`antirez/h3.c` on GitHub) — and it's a much bigger deal than I initially thought. It's by **Salvatore Sanfilippo (antirez)**, the creator of Redis. 2.6k stars, 199 forks, 122 commits, MIT licensed.

**What it is**: A from-scratch C + Metal implementation of the entire MiniMax H3 pipeline for Apple Silicon. Not MLX, not Python — pure C with Metal shaders. Single binary, `make -j8` to build.

**What works**: Prompt-to-video/audio, first/last-frame conditioning (FL2VA), ordered Ref2VA image/video/audio references — all end to end. Interactive REPL session with live frame preview during denoising.

**Performance** (from antirez's HN comments and README):
- On M5 Max 128GB: "a few minutes" for a 9-second 480x864 clip at 20 steps (vs "a bit over an hour" via ComfyUI+GGUF on M5 Pro 64GB)
- 4-step fox test at 512x512x22 frames: **3.5 seconds** denoise on M5 Max (vs 26.4s for 29-step reference)
- SSD streaming mode: 2.0 GiB DiT residency (vs 36.5 GiB full residency) with only 84% slowdown at 512 square, 26% at 864x480. Byte-identical output.
- End-to-end image+audio render on 128GB M5 Max: 74.58 seconds, ~40.1 GB peak footprint

**Key innovations** (antirez's own, not from MiniMax or FastVideo):
- `--layers N`: run fewer than 50 transformer blocks (e.g. 45 or 40)
- `--reuse N`: reuse whole denoiser velocities between steps (20 steps with reuse 2 = only 11 fresh DiT evaluations)
- `--core-reuse N`: keep timestep-dependent patch/output heads fresh every step but reuse the expensive core less often
- `--token-reduction`: pair adjacent horizontal video tokens in middle blocks (28.3% denoise speedup)
- `--render-width/--render-height`: run DiT/VAE at smaller internal canvas, upscale with vImage
- `--ssd-streaming`: stream DiT blocks from SSD, keep only 2 blocks in memory (2 GiB vs 36.5 GiB)
- `--use-int8-row-fc2`: M5-specific INT8 MLP kernels via Metal TensorOps
- Metal 4 + TensorOps paths for SwiGLU fused gate/up projections
- Live terminal frame preview via Kitty/Ghostty/iTerm2 graphical protocols

**This changes the H3 viability calculus for our app significantly.** A native C/Metal implementation by antirez is likely to be much faster than the PipeNetwork MLX port (Python overhead, MLX graph construction) and could potentially be integrated as a subprocess backend similar to how the app currently calls Python. The SSD streaming mode means even 32GB Macs could potentially run H3 (2 GiB DiT + VAEs + prompt encoding phases run separately).

### What this means for our app

The "faster than live" claims come from three places:
- **fal.ai H3 Max** (cloud API, NVIDIA clusters) — not local, but proves the model can be fast
- **FastH3** (open weights, NVIDIA B200) — 14x speedup via 90% sparse attention + 4-step distillation
- **antirez/h3.c** (local, Apple Silicon) — native C/Metal, 3.5s denoise for 512x512 4-step on M5 Max

The h3.c path is the one that matters for our app. It's not "faster than real-time" yet (that needs sparse attention, which MiniMax hasn't open-sourced), but it's fast enough to be practical — minutes, not hours. And it works on Macs we already target.

### MLX port status

PipeNetwork/minimax-h3-mlx: working, validated against diffusers reference. Apache-2.0 license. 72 stars, 16 forks. Can generate end-to-end including audio. Published quants on HuggingFace (pipenetwork/MiniMax-H3-MLX-8bit, -4bit, -6bit, -bf16).

BUT: not a pip-installable package. Clone + run scripts. Not integrated with any app. The pipeline is a standalone script (`scripts/generate.py`).

### H3 vs LTX 2.5 comparison

| Feature | LTX 2.5 | MiniMax H3 (4-bit + Turbo) |
|---------|---------|---------------------------|
| Max resolution | 4K (with upscaler) | 768p (2K via post-processing) |
| Max duration | configurable | 15 seconds |
| Audio | stereo 48kHz | stereo 32kHz |
| Steps (distilled) | 8 (fixed) | 4 (Turbo LoRA) or 8 |
| DiT size | 22B | 33B |
| Text encoder | Gemma-4 12B (bundled) | Qwen3-VL-32B (truncated to 50 layers) |
| MLX support | dgrauet/ltx-2-mlx (mature) | PipeNetwork MLX (working) + antirez/h3.c (fast native C/Metal) |
| 16GB Mac viable? | Yes (int4 + --low-ram) | No (VAEs alone need 11GB) |
| 32GB Mac viable? | Yes (int4 or q8 + --low-ram) | Yes with h3.c SSD streaming (2 GiB DiT) |
| 64GB Mac viable? | Yes (q8) | Yes (h3.c full residency, ~40 GB peak) |
| 128GB Mac? | Comfortable (bf16) | Fast (h3.c: 3.5s denoise for 512x512 4-step on M5 Max) |
| Generation speed (512x512, 4 steps) | ~90s on RTX 5090 | ~3.5s denoise on M5 Max via h3.c |
| Multi-shot scenes | Yes (native) | No (single shot) |
| Omni-reference | No | Yes (9 images + 3 videos + 3 audio) |
| License | LTX-2.x Community (>$10M rev needs paid) | MiniMax H3 Community (more restrictive) |

## Phase 3 — MiniMax H3 Support (via antirez/h3.c)

The discovery of `antirez/h3.c` (h3-metal) changes this from "future maybe" to "viable next step." It's a native C + Metal H3 implementation by the creator of Redis — fast, lightweight, MIT licensed, and already working end-to-end.

### Why h3.c instead of PipeNetwork MLX port

| Factor | PipeNetwork MLX | antirez/h3.c |
|--------|----------------|--------------|
| Language | Python + MLX | C + Metal shaders |
| Speed (M5 Max, 512x512, 4 steps) | Unknown (M3 Ultra: 8.8 min/step) | **3.5 seconds** denoise |
| Memory (DiT only) | 11.5-40.3 GB depending on quant | 2.0 GiB with SSD streaming |
| Binary | Python scripts | Single compiled binary |
| Integration | Would need pip packaging | Subprocess call, like current Python |
| Maturity | Validated, published quants | 2.6k stars, 122 commits, working |
| License | Apache-2.0 | MIT |

h3.c is dramatically faster because it's native Metal with no Python/MLX overhead, and antirez implemented several speed optimizations (layer thinning, velocity reuse, token reduction, internal canvas scaling) that the MLX port doesn't have.

### What would be needed:

1. **Backend integration**: Add `h3c` as a third `GenerationBackend`. The app calls `./h3 -d ./MiniMax-H3 -p "prompt" --width W --height H --frames N --steps S --seed S -o output.mp4` as a subprocess — same pattern as the current Python backend.

2. **Build/bundling**: h3.c is a `make -j8` C project. Could either:
   - Ask users to clone and build (like the current Python env setup)
   - Pre-build and bundle the binary in the app (it's a single executable)
   - Distribute via Homebrew

3. **Model download**: Point to MiniMax-H3 HuggingFace repo (MiniMaxAI/MiniMax-H3, ~144 GB for FL2VA). The app already handles large model downloads via hf.

4. **Memory tiering**: 
   - SSD streaming mode (`--ssd-streaming`): 2 GiB DiT residency — viable on 32GB Macs
   - Full residency: ~40 GB peak — viable on 64GB Macs
   - 16GB Macs: not viable (VAEs + prompt encoding need ~15 GB even with SSD streaming)

5. **Speed/quality presets**: h3.c has a rich preset system that maps well to the app's UI:
   - Fast preview: `--steps 4 --layers 50 --reuse 1` (~3.5s denoise on M5 Max)
   - Default: `--steps 20 --layers 45 --reuse 2` (balanced)
   - Reference: `--steps 50 --layers 50 --reuse 1` (highest quality)

6. **Ref2VA support**: h3.c supports ordered image/video/audio references — could expose in the UI as "reference images" (up to 9 images, 3 video clips, 3 audio clips).

7. **License handling**: MiniMax H3 Community License for the weights; h3.c itself is MIT. Need to review H3 license terms before shipping.

### Realistic targets (h3.c, not MLX):

| Mac | Mode | 5s clip (107 frames, 512x512) | 10s clip (243 frames) |
|-----|------|-------------------------------|----------------------|
| 32GB | SSD streaming, 4 steps | ~30-60s | ~2-5 min |
| 64GB | Full residency, 20 steps w/ reuse 2 | ~2-5 min | ~10-20 min |
| 128GB (M5 Max) | Full residency, 4 steps | ~3.5s denoise + overhead | ~30s-2 min |
| 128GB (M5 Max) | Full residency, 20 steps | ~15-30s denoise + overhead | ~5-10 min |

### Estimate: 3-5 days of work
- Add h3c backend to LTXBridge.swift (subprocess call, similar to existing Python path)
- Add H3 models to catalog with memory tier warnings
- Handle model download (MiniMax-H3 from HuggingFace)
- Map h3.c CLI flags to app parameters
- Test on M4 Max 128GB (James's machine)

## Decision Points

1. **Phase 1 vs Phase 2**: Phase 1 is faster (days vs weeks) but introduces a dependency on dgrauet's repo (not on PyPI, could break). Phase 2 keeps everything in-house but requires porting ~40 files.

2. **Publish ltx-2-mlx to PyPI**: Could fork dgrauet's repo and publish to PyPI as a package, or ask dgrauet to publish. This would simplify the install story.

3. **Merge with upstream**: Blaizzy/mlx-video (our upstream) is adding Wan2.2 support but shows no signs of 2.5. We could try to upstream our 2.5 changes once Phase 2 is done.

4. **Phase 3 via h3.c**: antirez's h3.c is the strongest H3 path for Apple Silicon — native C/Metal, 3.5s denoise on M5 Max, SSD streaming for 32GB Macs, MIT licensed. Could be Phase 3 immediately after Phase 1 (skip Phase 2 entirely). The backend integration is simpler than the MLX path (subprocess call to a single binary vs Python packaging).

5. **Multi-backend architecture**: If we're going to support LTX-2.0/2.3 (mlx-video-with-audio), LTX-2.5 (ltx-2-mlx), and H3 (h3.c), the `GenerationBackend` enum from Phase 1 becomes the critical architectural decision. Getting this right in Phase 1 means Phase 3 is just adding another backend case.