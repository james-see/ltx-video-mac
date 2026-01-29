# Changelog

All notable changes to LTX Video Generator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
