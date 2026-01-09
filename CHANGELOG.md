# Changelog

All notable changes to LTX Video Generator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
