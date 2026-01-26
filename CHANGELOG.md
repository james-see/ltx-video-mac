# Changelog

All notable changes to LTX Video Generator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
