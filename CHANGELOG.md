# Changelog

All notable changes to LTX Video Generator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.54] - 2026-04-23

### Fixed
- **Issue #45 — text encoder config validation** — Require `mlx-video-with-audio>=0.1.34` which relaxes `_looks_like_text_config` to accept Gemma `model_type`-based configs and flat configs without `vocab_size`, auto-retries with a fresh HF download when the cached text encoder config looks corrupt, and distinguishes AV-model-misrouted configs with a specific error message. Updated LTXBridge hint text and added `huggingface-cli delete-cache` guidance.

## [2.3.53] - 2026-04-22

### Fixed
- **MLX Audio voiceover** — Kokoro / mlx-audio can print status lines to Python stdout before the app's JSON result. The client now parses the last JSON line with a `success` field, so successful local TTS is no longer misreported as `Failed to parse output` ([#42](https://github.com/james-see/ltx-video-mac/issues/42)).

### Changed
- **Docs** — Image-to-Video steps in `docs/usage.md` match the app (disclosure location and **Select Source Image...** label).

## [2.3.52] - 2026-04-07

### Fixed
- **LTX-2.3 garbled/mosaic output** — Require `mlx-video-with-audio>=0.1.33` which forces all unified models through the legacy fixed-sigma Euler sampler (the `ltx2_schedule` dynamic sigmas produced garbled output with MLX-converted 2.3 weights), disables decoder residual connections that caused blockiness, and guards the encoder skip connection against shape mismatch on I2V. LTX-2 Unified is unaffected.

## [2.3.51] - 2026-04-07

### Added
- **Image persistence** — Selected source image for Image-to-Video now persists between renders and across app launches. History regenerate also restores the original source image.

## [2.3.50] - 2026-03-23

### Fixed
- **LTX-2.3 quality regression** — Reverted scheduler and AdaIN changes from v2.3.49 that introduced new artifacts and audio regression. Now only disables CFG (the core fix) without changing the scheduler or adding experimental filters.

### Changed
- Require `mlx-video-with-audio>=0.1.32`.

## [2.3.49] - 2026-03-23

### Fixed
- **LTX-2.3 garbled/datamoshed video** — Disabled double-guidance (CFG) on distilled models where guidance is already baked in, switched from SD3-style scheduler to the correct LinearQuadratic scheduler per the model's embedded config, and added AdaIN latent normalization after upsampling. Also roughly halves generation time since only one transformer pass per step is needed.

### Changed
- Require `mlx-video-with-audio>=0.1.31`.

## [2.3.48] - 2026-03-23

### Fixed
- **LTX-2.3 Image-to-Video crash** - VAE encoder now reads the correct block topology from `embedded_config.json` for 2.3 models instead of using the LTX-2 default, fixing a shape mismatch (`1024 vs 2048`) in the encoder residual connection.

### Changed
- Require `mlx-video-with-audio>=0.1.30`.

## [2.3.47] - 2026-03-22

### Added
- **Runtime version diagnostics** - Generation logs now print the `mlx-video-with-audio` version at startup and emit an `MLX_VIDEO_VERSION` token, making it easier to diagnose 2.3 model issues from user logs.

### Changed
- Require `mlx-video-with-audio>=0.1.29`.

## [2.3.46] - 2026-03-21

### Fixed
- **Image-to-video startup crash** - Updated app minimum dependency to pull in `mlx-video-with-audio` fix for missing unified VAE metadata fallback, resolving `KeyError: 0` during VAE encoder load on I2V runs.

### Changed
- Require `mlx-video-with-audio>=0.1.28`.

## [2.3.45] - 2026-03-21

### Fixed
- **LTX-2.3 garbled video output** - Fixed two bugs in `mlx-video-with-audio` that caused garbled mosaic output for LTX 2.3 models: upsampler weights not loading from unified format (prefix not stripped), and VAE latent statistics missing for Q4 split format.

### Changed
- Require `mlx-video-with-audio>=0.1.27`.

## [2.3.44] - 2026-03-21

### Added
- **Model-specific tips in Parameters sidebar** - The right sidebar now shows contextual recommendations based on the selected model, including optimal inference step ranges and memory guidance. LTX-2.3 models display a blue info banner with distilled-model tips, and quantized models show an orange quality warning.

## [2.3.43] - 2026-03-21

### Added
- **LTX-2.3 models (Beta)** - Two new model options: "LTX-2.3 Unified (Beta)" (~48GB bf16) and "LTX-2.3 Distilled Q4 (Beta)" (~22GB quantized). Based on the 22B-parameter LTX 2.3 architecture with gated attention, 8-layer connector, and BigVGAN vocoder.

### Changed
- Require `mlx-video-with-audio>=0.1.26`.

## [2.3.42] - 2026-03-21

### Fixed
- **Vocoder crash with notapalindrome/ltx2-mlx-av** - ConvTranspose1d weight transpose was applied unconditionally, corrupting weights for unified models that already store them in MLX layout. Now auto-detects the layout from weight shape.

### Changed
- Require `mlx-video-with-audio>=0.1.25`.

## [2.3.41] - 2026-03-21

### Added
- **Launch-time Python package consent** - On startup, if your configured venv needs installs/upgrades, the app asks before running `pip`, lists **Install** / **Upgrade** lines, then shows **Update complete** when finished.
- **Preferences: “Use local mlx-video-with-audio repo (dev)”** - Optional override when `~/projects/mlx-video-with-audio` exists; by default the app prefers **newer pip** over an older local checkout so `pip install -U` actually applies.

### Fixed
- **Launch validation task** - `hasCheckedPython` is only set after validation finishes so a cancelled `.task` cannot skip checks permanently.
- **Generation preflight** - Runs the same environment check/upgrade path as validation before each job (cached briefly) so users are not forced to open Preferences after changing packages.
- **Bridge PYTHONPATH** - Child processes clear inherited `PYTHONPATH` unless explicitly using a local repo; avoids stale shell paths shadowing site-packages.

### Changed
- Require `mlx-video-with-audio>=0.1.24` (unchanged minimum; launch flow makes upgrades visible).
- Download stall watchdog extended for very large model downloads.

## [2.3.37] - 2026-03-18

### Fixed
- **Distilled AV decode parity** - Integrated the distilled BWE vocoder execution path, corrected LTX-2.3 upsampler checkpoint key/layout mapping, and tightened unified VAE Conv3d layout handling to avoid incorrect transpose paths.

### Changed
- Require `mlx-video-with-audio>=0.1.20`.

## [2.3.36] - 2026-03-18

### Fixed
- **Strict 2.3 upsampler selection** - 2.3 distilled runs now resolve only the LTX-2.3 spatial upsampler path for stage-2 latent refinement, preventing incompatible fallback weights from producing corrupted output.

### Changed
- Require `mlx-video-with-audio>=0.1.19`.

## [2.3.35] - 2026-03-18

### Fixed
- **Distilled output corruption** - Align app runtime with `mlx-video-with-audio` VAE topology fix so distilled unified checkpoints decode using their embedded `decoder_blocks` graph rather than a mismatched static decoder path.

### Changed
- Require `mlx-video-with-audio>=0.1.18`.

## [2.3.34] - 2026-03-18

### Fixed
- **Distilled Q4 generation path** - Added support for gated-attention transformer blocks (`9`-parameter scale-shift tables + prompt scale/shift modulation), BigVGAN/SnakeBeta vocoder loading, and targeted unified vocoder transpose handling for ConvTranspose upsamplers.
- **Distilled VAE fallback compatibility** - Unified repos with unsupported VAE decoder block layouts now automatically fall back to `Lightricks/LTX-2` VAE for decode/encode paths, preventing decoder shape/broadcast crashes.

### Changed
- Require `mlx-video-with-audio>=0.1.17`.

## [2.3.33] - 2026-03-18

### Fixed
- **Distilled/quantized model support** - Models using split-weight format (e.g. `dgrauet/ltx-2.3-mlx-distilled-q4`) now load correctly. Previously failed with "Text encoder configuration mismatch" because the model detection only recognized single-file `model.safetensors` layouts.

### Changed
- Require `mlx-video-with-audio>=0.1.16` (split-weight model format support for quantized/distilled repos).

## [2.3.31] - 2026-03-15

### Fixed
- **Live GUI progress finally restored** - Root cause: Python's `BufferedReader.read(8192)` blocked until 8192 bytes accumulated, starving the progress loop for the entire generation. Replaced with unbuffered `os.read()` so every `STAGE:` / `STATUS:` line reaches the UI immediately. Also added `flush=True` on the catch-all stderr relay.

## [2.3.30] - 2026-03-15

### Fixed
- **Live GUI status/progress updates** - Fixed chunked stderr parsing so progress tokens (`STAGE:`, `STATUS:`, `DOWNLOAD:*`) are buffered into complete lines before parsing, preventing dropped updates and frozen "Running MLX generation..." status during active runs.

## [2.3.29] - 2026-03-15

### Fixed
- **Progress reporting accuracy** - Removed synthetic progress bumps and now advance generation progress strictly from real parsed logs (`STAGE:`, `STATUS:`, `DOWNLOAD:*`, and tqdm download lines).
- **Stage parsing resilience** - Added support for raw `stage 1/2 (n/m)` log lines so UI progress updates remain grounded in emitted model output.

### Changed
- **Memory risk warning threshold** - Warning now scales to machine memory (70% threshold floor of 36GB) to reduce false warnings on high-memory Macs while keeping guardrails for smaller systems.

## [2.3.28] - 2026-03-15

### Fixed
- **Issue #24 - instant generation failure after cache reset** - Kept the embedded runtime Python script path safe from carriage-return tokenization that could trigger `IndentationError` before model download starts.
- **About version display** - About tab now reads app version/build from bundle metadata instead of a hardcoded stale string.
- **Model status wording** - Sidebar now reflects environment readiness explicitly and clarifies that model files download on first generation when cache is missing.

## [2.3.24] - 2026-03-14

### Fixed
- **Issue #20 - text encoder config failure** - Detects `text_config` schema mismatch early and surfaces actionable guidance instead of opaque `mlx_video.generate_av failed with code 1`.

### Changed
- Require `mlx-video-with-audio>=0.1.15` (structured text-encoder path/config diagnostics + missing `text_config` guardrails).

## [2.3.23] - 2026-03-14

### Changed
- Require `mlx-video-with-audio>=0.1.14` (native prompt-enhancer fallback + `ENHANCER_FALLBACK` token support).
- Startup Python validation/auto-upgrade flow now enforces the new minimum version automatically via existing package checks.

## [2.3.22] - 2026-03-14

### Fixed
- **Issue #21 – Prompt enhancement stability** - Prompt enhancement is now resolved before generation and failures automatically fall back to the original prompt instead of failing the run.

### Changed
- **Issue #21 – 24 FPS speech sync guardrail** - Added UI guidance and runtime status hints that synchronized speech/lip-sync works best at 24 FPS.
- Prompt enhancement preference text now explicitly states fallback behavior when enhancement fails.
- README + parameter docs updated with 24 FPS sync guidance and current LTX 2.3 conversion/LoRA support boundaries.

## [2.3.21] - 2026-03-13

### Fixed
- **Issue #22 – First-download progress / stall timeout** – Chunk-based read of subprocess output so tqdm/huggingface_hub progress is detected even when output is sparse or `\r`-updated; stall timer only runs after download has started and no data received (5 min timeout). Prevents false 180s timeout while model is still downloading.
- **Missing bundled script** – `av_generator.py` is now included in app resources so "Disable audio" generation no longer fails with "Missing bundled script" when running from DMG or release build.

### Changed
- Clearer stall error message with next steps: check network, `hf login` for gated models, optional manual `hf download notapalindrome/ltx2-mlx-av`.

## [2.3.20] - 2026-03-05

### Changed
- Require `mlx-video-with-audio>=0.1.13` for unified VAE decoder layout compatibility.

### Fixed
- Decoder channel mismatch crashes now resolved via upstream `mlx-video-with-audio` 0.1.13.

## [2.3.11] - 2026-02-23

### Fixed
- **IndentationError** - Fix embedded Python script indentation for `if enable_enhancement:` block.

## [2.3.10] - 2026-02-22

### Fixed
- **mlx-lm 0.25+ compatibility** - Prompt enhancement now uses `sampler` instead of deprecated `temp` kwarg. Fixes "generate_step() got an unexpected keyword argument 'temp'" error.

### Changed
- mlx-video-with-audio>=0.1.6 (includes same mlx-lm fix for generation).

## [2.3.9] - 2026-02-22

### Added
- **Auto-upgrade mlx-video-with-audio** - App checks installed version and auto-upgrades to >=0.1.5 if older.
- **Improved download progress** - Parses huggingface_hub tqdm output for file count and bytes (e.g. "Downloading: 5.2GB / 5.4GB (file 1/13, 96%)").

### Changed
- **Prompt enhancement simplified** - Single toggle: Enable Prompt Enhancement on/off. Always uses MLX uncensored Gemma (TheCluster/amoral-gemma-3-12B-v2-mlx-4bit, ~7GB). Removed "Use uncensored enhancer" sub-toggle.
- **MLX-only enhancer** - Preview and generation use mlx_lm with amoral-gemma; no Lightricks/LTX-2 download for enhancement.
- Progress handler now receives accumulated stderr for accurate parsing across chunked output.

## [2.3.8] - 2026-02-22

### Added
- **Save audio track separately** - Settings > Generation > Output: toggle to keep .wav alongside video (default: off, audio only in mp4).

### Changed
- mlx-video-with-audio 0.1.5: --save-audio-separately flag; no longer saves .wav by default.
- requirements.txt: mlx-video-with-audio>=0.1.5

## [2.3.7] - 2026-02-22

### Added
- **Uncensored prompt enhancer** - Settings > Generation: "Use uncensored enhancer" toggle (when Gemma enhancement is on). Uses TheCluster/amoral-gemma-3-12B-v2-mlx-4bit to avoid content filters on words like urine, blood, etc. First run downloads ~7GB. Progress UI shows "Loading uncensored prompt enhancer (first run may download ~7GB)..." during generation.

### Changed
- mlx-video-with-audio: New `enhance_prompt.py` module and `--use-uncensored-enhancer` CLI flag for generate_av.
- Preview button uses uncensored model when the setting is on.

## [2.3.6] - 2026-02-22

### Added
- **Filtered-word retry** - When Gemma prompt enhancement returns empty (e.g. safety filter on words like "urine", "piss"), the preview script now auto-retries with suspected filtered words replaced by placeholders, then merges the originals back into the enhanced result.

### Changed
- `enhance_prompt_preview.py`: Maintains a list of commonly filtered words; on empty enhancement, sanitizes prompt, retries, and restores original wording.

## [2.3.5] - 2026-02-22

### Removed
- Legacy LTX-2 Distilled model option (~90GB, video only). App now uses only LTX-2 Unified.

### Changed
- Model variant picker removed from Preferences. Single model: LTX-2 Unified (audio+video).
- Removed ltx_generator.py and legacy generation path from LTXBridge.
- Fix enhanced prompt preview text_config error: use Lightricks/LTX-2 for text encoder when model is unified MLX format.

## [2.3.4] - 2026-02-22

### Added
- **Enhanced Prompt Preview** - Preview the Gemma-rewritten prompt before generating. Button appears in the Gemma section when enhancement is enabled (unified AV model only). First run may take 30–60s while the model loads.

### Changed
- README: Added Gemma Prompt Enhancement section with usage steps
- Settings > Generation: Tooltip on Enable Gemma Prompt Enhancement toggle

## [2.3.3] - 2026-02-22

### Added
- **Gemma Prompt Enhancement Toggle** - New setting in Preferences > Generation to enable/disable prompt rewriting. When off (default), the Gemma section is grayed out with "Turn on in Settings" note. Fixes FileNotFoundError when mlx-video-with-audio package omits system prompt files.

### Changed
- Gemma prompt enhancement now gated by Settings toggle instead of slider values
- Pre-flight injection: when enhancement is enabled, bundled system prompts are copied into mlx_video package if missing
- Upgraded bundled system prompts to upstream versions (LTX-2 best practices: visuals, audio, camera, style)

### Fixed
- Second video generation failing with FileNotFoundError for gemma_t2v_system_prompt.txt when prompt enhancement was used

## [2.3.2] - 2026-02-12

### Added
- **Enhanced Prompt Display** - When Gemma prompt enhancement is used, the rewritten prompt is now captured and displayed in the video archive detail view with a sparkle icon and purple label
- Enhanced prompt is preserved in generation history and persists across sessions

### Technical
- Python `generate.py` emits `ENHANCED_PROMPT:` on stderr for structured capture by Swift bridge
- `LTXBridge` parses enhanced prompt from stderr output (supports both custom and mlx_video formats)
- `GenerationResult` model extended with optional `enhancedPrompt` field (backward-compatible via Codable)
- All `GenerationResult` construction sites updated across `GenerationService`, `AudioService`, and `HistoryManager`

## [2.3.1] - 2026-02-12

### Fixed
- Unified AV model generation failing due to unsupported `--repetition-penalty` and `--top-p` CLI args
- Gemma prompt enhancement params now correctly map to the package's `--enhance-prompt` and `--temperature` flags

## [2.3.0] - 2026-02-12

### Added
- **Gemma Prompt Enhancement Controls** - Collapsible section in prompt view with sliders for:
  - Repetition Penalty (1.0–2.0) — reduces repeated phrases in enhanced prompts
  - Top-P (0.0–1.0) — controls focus/creativity of prompt rewriting
- **Image Strength Slider** - Control how strongly the source image influences generation (0.0–1.0) in image-to-video mode
- **Audio Disable Toggle** - Skip audio generation on the unified AV model for faster silent video output
- **VAE Tiling Mode Picker** - Dropdown in parameters panel with 7 modes: Auto, None, Default, Aggressive, Conservative, Spatial Only, Temporal Only
  - Controls memory vs speed tradeoff during VAE decoding
- Prompt enhancement pipeline in bundled `ltx_mlx` using Gemma model with system prompts

### Changed
- Default inference steps changed from 40 to 30
- Default guidance scale changed from 4.0 to 3.0
- All built-in presets updated to new defaults (Quick Preview: 15 steps, High Quality: 40 steps)

### Technical
- `GenerationParameters` extended with `vaeTilingMode` and `imageStrength`
- `GenerationRequest` extended with `disableAudio`, `gemmaRepetitionPenalty`, `gemmaTopP`
- LTXBridge passes new params as CLI args to both unified AV and distilled paths
- Python entry points (`av_generator.py`, `ltx_generator.py`, `generate.py`) accept new arguments

## [2.2.0] - 2026-01-31

### Added
- **Unified Audio-Video Model** - New default model generates synchronized audio with video automatically
  - Uses `notapalindrome/ltx2-mlx-av` from Hugging Face
  - Smaller download size (~42GB vs ~90GB for legacy model)
  - No additional configuration needed for audio
- **Audio Included Banner** - Shows in prompt view when unified model is selected
- **Model Audio Indicator** - Preferences shows which models support built-in audio
- **Real-time Progress** - Stage-by-stage denoising progress display (requires mlx-video-with-audio 0.1.3+)
- `mlx-video-with-audio` package dependency (available on [PyPI](https://pypi.org/project/mlx-video-with-audio/))

### Changed
- Default model changed from `mlx-community/LTX-2-distilled-bf16` to `notapalindrome/ltx2-mlx-av`
- Model picker now shows audio capability and download size for each variant
- Status messages indicate when generating "with audio"
- Users can still layer voiceover/music on top of built-in audio

### Technical
- LTXBridge routes to correct generator based on model variant
- GenerationService detects unified model and adjusts audio workflow
- Progress parsing supports STAGE:X:STEP:Y:Z format from mlx-video-with-audio

## [2.1.3] - 2026-01-29

### Added
- British voices: Alice, Lily, Charlotte, Daniel, George, Harry
- Australian voices: Charlie, Matilda
- ElevenLabs formatting help tooltip with pause/emphasis syntax
  - `<break time="1s" />` for pauses up to 3 seconds
  - `...` for hesitation
  - CAPS for emphasis
  - Dashes for short pauses

### Changed
- Voice picker now shows accent labels (e.g., "Rachel (US Female)")
- Expanded US voice selection: Rachel, Sarah, Jessica, Aria, Adam, Josh, Brian, Eric

## [2.1.2] - 2026-01-29

### Fixed
- "Open Preferences" button in Python setup alert now correctly opens Settings
- Added delay for alert dismissal and macOS version compatibility

## [2.1.1] - 2026-01-29

### Fixed
- Voiceover and music now generate during video creation when enabled
- Previously only worked in post-processing (Add Audio view)

## [2.1.0] - 2026-01-29

### Added
- **Background Music Generation** - Generate instrumental music using ElevenLabs Music API
  - 54 genre presets across 9 categories (Electronic, Hip-Hop/R&B, Rock, Pop, Jazz/Blues, Classical/Cinematic, World, Country/Folk, Functional/Mood)
  - Music automatically matches video length
  - Ducked under voiceover when both are present (20% volume)
- **Voiceover Source Selection** - Choose between ElevenLabs (cloud) or MLX-Audio (local) for TTS
  - Voice picker for both engines in generation view
  - API key validation with helpful warnings
- **Enhanced Add Audio View** - Three-tab interface:
  - Voiceover: Add narration to videos
  - Music: Add background music by genre
  - Both: Add voiceover and music together
- Genre preview shows ElevenLabs prompt before generation

### Changed
- GenerationRequest model extended with voiceoverSource, voiceoverVoice, musicEnabled, musicGenre
- GenerationResult model extended with voiceoverSource, voiceoverVoice, musicPath, musicGenre
- Default audio source changed to MLX-Audio (local)
- AddAudioView expanded from 500x550 to 550x650

## [2.0.9] - 2026-01-28

### Fixed
- Audio files now save to the same directory as the video (respects custom output folder setting)
- Previously audio was incorrectly saving to default Application Support folder

## [2.0.8] - 2026-01-28

### Added
- Voiceover/Narration input field in generation view
- voiceoverText stored with results and pre-fills audio dialog

### Fixed
- Fixed mlx-audio API to use correct Kokoro model interface
- Fixed empty Add Audio sheet bug (race condition with sheet presentation)
- Fixed truncated buttons in detail view (now icon-only with tooltips)

## [2.0.7] - 2026-01-28

### Added
- **Audio Generation** - Add voiceover narration to videos
  - ElevenLabs (cloud) and MLX-Audio (local) TTS options
  - Voice selection with 9+ preset voices per engine
  - FFmpeg integration for audio/video merging
- **Add Audio from History** - Right-click any video thumbnail to add audio
- ElevenLabs API key configuration in Preferences > Audio
- Default audio source preference setting

### Changed
- History detail view buttons now icon-only with tooltips for better layout

## [2.0.0] - 2026-01-26

### Changed
- **BREAKING**: Migrated from PyTorch/diffusers to native Apple MLX framework
- Bundled MLX generation code (inspired by [mlx-video](https://github.com/Blaizzy/mlx-video))
- Two-stage generation pipeline: half-resolution → upscale → full-resolution refinement
- Model: `Lightricks/LTX-2` distilled variant (~10GB download, cached in ~/.cache/huggingface/)
- Updated UI to show "MLX" badge instead of "MPS"
- Python validation now checks for MLX dependencies (mlx, mlx-vlm, transformers, etc.)
- Stage-aware progress display (Stage 1: 0-50%, Stage 2: 50-100%)

### Added
- Bundled `ltx_mlx` Python module with all generation code
- Automatic model download progress display
- Info note in Preferences: "Currently, only the distilled variant is supported"

### Removed
- Removed PyTorch/diffusers dependency
- Removed MPS patching code (no longer needed with MLX)
- Removed FP8 and dev model variants (distilled only for now)

### Fixed
- Closes GitHub issue #6 - Native Apple Silicon support via MLX

## [1.0.14] - 2026-01-26

### Fixed
- Improved MPS patch robustness - now uses regex to handle any whitespace/indentation
- Patch properly matches ltx2-mps approach for connectors.py and transformer_ltx2.py

## [1.0.13] - 2026-01-26

### Fixed
- MPS float64 error in LTX-2 rotary position embeddings - now auto-patches diffusers during validation
- Based on fix from Pocket-science/ltx2-mps - patches connectors.py and transformer_ltx2.py

## [1.0.12] - 2026-01-26

### Fixed
- MPS float64 error in LTX-2 rotary position embeddings (TypeError: Cannot convert MPS Tensor to float64)
- Force float32 default dtype and disable double_precision on RoPE modules for MPS compatibility

## [1.0.11] - 2026-01-26

### Added
- Model variant indicator in sidebar showing current model (Full/Distilled/FP8)
- MPS badge confirming Metal GPU acceleration is active
- Device verification logging to confirm pipeline is on MPS

### Fixed
- Enforce pure MPS execution with `device_map=None` to prevent CPU offloading
- Text-to-video mode no longer passes `image` parameter (was causing pipeline error)
- Progress UI now distinguishes model download from generation steps

## [1.0.10] - 2026-01-26

### Added
- PyAV (`av`) dependency for LTX-2 video export with audio support
- Pillow (`PIL`) dependency validation for image-to-video mode

### Changed
- Preferences validation now checks for `av` and `PIL` packages
- Updated requirements.txt with `av>=10.0.0` and `Pillow>=10.0.0`

### Fixed
- Build error in PromptInputView.swift (NSImage.size is not optional)

## [1.0.6] - 2025-01-20

### Added
- REST API server (toggle in sidebar) for MCP integration on port 8420
  - GET / - API info
  - GET /status - Server and model status
  - GET /queue - Current generation queue
  - POST /generate - Submit generation request
  - DELETE /queue/:id - Cancel queued request
- VRAM display now refreshes every 5 seconds

## [1.0.5] - 2025-01-20

### Added
- Display available VRAM in bottom right of parameters panel
- Estimated VRAM usage calculation

### Changed
- Max frames set to 505 (step of 8)

## [1.0.4] - 2025-01-09

### Changed
- Increased max frames from 193 to 1000 for longer videos
- Added 20 FPS option
- Renamed "History" tab to "Video Archive"
- Added real-time video length estimate that updates with FPS/frames changes

## [1.0.3] - 2025-01-09

### Fixed
- Fixed crash when viewing videos in history (AVKit SwiftUI crash on macOS 26)
- Replaced SwiftUI VideoPlayer with NSViewRepresentable AVPlayerView wrapper

## [1.0.2] - 2025-01-09

### Fixed
- App icon now properly included in build (was missing from Xcode project)

## [1.0.1] - 2025-01-09

### Added
- Custom pixel art app icon (created with Aseprite)
- Python setup prompt on first launch if not configured
- Alert guides users to Preferences when Python path is missing

### Changed
- Improved first-run user experience

## [1.0.0] - 2025-01-09

### Added
- Initial release of LTX Video Generator for macOS
- Native SwiftUI interface optimized for Apple Silicon
- Text-to-video generation using LTX-Video 0.9.1 model
- MPS (Metal Performance Shaders) GPU acceleration
- Generation queue with real-time progress tracking (step count display)
- Video history management with thumbnails
- Parameter presets (Fast Preview, Standard, High Quality, Portrait, Square, Cinematic)
- Customizable generation parameters:
  - Resolution (width/height)
  - Number of frames
  - Inference steps
  - Guidance scale
  - FPS
  - Seed control
- Video preview with looping playback
- Export and sharing functionality
- Python environment configuration in Preferences

### Technical
- Uses bfloat16 precision for optimal Apple Silicon compatibility
- Subprocess-based Python execution for stability
- Automatic thumbnail generation from video frames
- Clean environment isolation for reliable MPS execution
