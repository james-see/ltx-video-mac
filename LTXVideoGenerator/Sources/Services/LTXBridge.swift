import Foundation

enum LTXError: LocalizedError, Equatable {
    case pythonNotConfigured
    case modelLoadFailed(String)
    case generationFailed(String)
    case exportFailed(String)
    case cancelled
    
    var errorDescription: String? {
        switch self {
        case .pythonNotConfigured:
            return "Python environment not configured. Please check Preferences."
        case .modelLoadFailed(let msg):
            return "Failed to load LTX model: \(msg)"
        case .generationFailed(let msg):
            return "Generation failed: \(msg)"
        case .exportFailed(let msg):
            return "Failed to export video: \(msg)"
        case .cancelled:
            return "Generation was cancelled"
        }
    }
}

// Use subprocess to run MLX-based generation
class LTXBridge {
    static let shared = LTXBridge()
    
    private(set) var isModelLoaded = false
    private var pythonHome: String?
    private var pythonExecutable: String?
    
    private init() {
        setupPythonPaths()
    }
    
    private func setupPythonPaths() {
        // Get Python path from user defaults
        guard let savedPath = UserDefaults.standard.string(forKey: "pythonPath"),
              !savedPath.isEmpty else {
            pythonExecutable = nil
            pythonHome = nil
            return
        }
        
        // Use PythonEnvironment's path detection to handle both executable and dylib paths
        let pathType = PythonEnvironment.shared.detectPathType(savedPath)
        
        switch pathType {
        case .executable:
            pythonExecutable = savedPath
            if let dylib = PythonEnvironment.shared.executableToDylib(savedPath),
               let home = PythonEnvironment.shared.extractPythonHome(from: dylib) {
                pythonHome = home
            } else {
                let execURL = URL(fileURLWithPath: savedPath)
                pythonHome = execURL.deletingLastPathComponent().deletingLastPathComponent().path
            }
            
        case .dylib:
            if let exec = PythonEnvironment.shared.dylibToExecutable(savedPath) {
                pythonExecutable = exec
            }
            if let home = PythonEnvironment.shared.extractPythonHome(from: savedPath) {
                pythonHome = home
                if pythonExecutable == nil {
                    let standardExec = "\(home)/bin/python3"
                    if FileManager.default.isExecutableFile(atPath: standardExec) {
                        pythonExecutable = standardExec
                    }
                }
            }
            
        case .unknown:
            if FileManager.default.isExecutableFile(atPath: savedPath) {
                pythonExecutable = savedPath
                let execURL = URL(fileURLWithPath: savedPath)
                pythonHome = execURL.deletingLastPathComponent().deletingLastPathComponent().path
            } else {
                pythonExecutable = nil
                pythonHome = nil
            }
        }
    }
    
    func loadModel(progressHandler: @escaping (String) -> Void) async throws {
        setupPythonPaths()
        
        guard pythonExecutable != nil else {
            throw LTXError.pythonNotConfigured
        }
        
        progressHandler("Checking MLX environment...")
        
        // Test that MLX and required packages are installed
        let testScript = """
        import mlx.core as mx
        import mlx_vlm
        import transformers
        print("OK")
        """
        
        let result = try await runPython(script: testScript)
        if !result.contains("OK") {
            throw LTXError.pythonNotConfigured
        }
        
        let selectedModel = LTXModelCatalog.selectedModel()
        progressHandler("MLX environment ready. Model will download on first generation (\(selectedModel.downloadSize)).")
        isModelLoaded = true
    }
    
    func generate(
        request: GenerationRequest,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> (videoPath: String, seed: Int, enhancedPrompt: String?) {
        setupPythonPaths()
        
        guard let _ = pythonExecutable else {
            throw LTXError.pythonNotConfigured
        }
        
        let params = request.parameters
        let seed = params.seed ?? Int.random(in: 0..<Int(Int32.max))
        
        let selectedModel = LTXModelCatalog.resolvedModel(id: request.modelId)
        let modelRepo = selectedModel.repo
        let oomRecoveryHint = "Metal ran out of memory during generation. Retry with safer settings: 512x320 resolution, 25/33/49 frames, 24 FPS, and tiling set to aggressive. Close memory-heavy apps, then retry."
        let isImageToVideo = request.isImageToVideo
        let modeDescription = isImageToVideo ? "image-to-video" : "text-to-video"
        progressHandler(0.1, "Starting \(modeDescription) (\(selectedModel.displayName))...")
        if selectedModel.supportsBuiltInAudio && !request.disableAudio && params.fps != 24 {
            progressHandler(0.1, "Sync tip: speech alignment works best at 24 FPS (current: \(params.fps))")
        }
        
        let enableGemmaPromptEnhancement = UserDefaults.standard.bool(forKey: "enableGemmaPromptEnhancement")
        let saveAudioTrackSeparately = UserDefaults.standard.bool(forKey: "saveAudioTrackSeparately")

        // Apply prompt enhancement up-front so generation can continue safely even
        // when upstream enhancer internals fail.
        var preEnhancedPrompt: String? = nil
        var generationPrompt = request.prompt
        if enableGemmaPromptEnhancement {
            progressHandler(0.06, "Enhancing prompt...")
            do {
                if let enhanced = try await previewEnhancedPrompt(
                    prompt: request.prompt,
                    modelRepo: modelRepo,
                    temperature: request.gemmaTopP,
                    sourceImagePath: request.sourceImagePath,
                    progressHandler: { status in
                    progressHandler(0.06, status)
                    }
                ), !enhanced.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    generationPrompt = enhanced
                    preEnhancedPrompt = enhanced
                    progressHandler(0.07, "Prompt enhanced with Gemma")
                } else {
                    progressHandler(0.07, "Prompt enhancement returned empty text; using original prompt")
                }
            } catch {
                progressHandler(0.07, "Prompt enhancement failed; using original prompt")
            }
        }

        // Escape the prompt for Python
        let escapedPrompt = generationPrompt
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        
        // Escape source image path if provided
        let escapedImagePath = request.sourceImagePath?
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"") ?? ""
        
        // Log file path
        let logFile = "/tmp/ltx_generation.log"
        
        // Ensure dimensions are divisible by 64 for MLX
        let genWidth = (params.width / 64) * 64
        let genHeight = (params.height / 64) * 64
        
        let resourcesPath = Bundle.main.bundlePath + "/Contents/Resources"
        
        let script: String
        // LTX-2 Unified - uses mlx-video-with-audio package
        script = """
import os
import sys
import json
import subprocess
import time
import select

# Set up file logging
log_file = open("\(logFile)", "w")
def log(msg):
    print(msg, file=log_file, flush=True)
    print(msg, file=sys.stderr, flush=True)

try:
    log("=== LTX-2 Unified AV Generation Started ===")
    log(f"Python: {sys.executable}")
    
    # Check MLX
    import mlx.core as mx
    log(f"MLX device: Apple Silicon")
    
    model_repo = "\(modelRepo)"
    log(f"Model: {model_repo}")
    
    # Image-to-video mode
    source_image_path = "\(escapedImagePath)" if "\(escapedImagePath)" else None
    mode = "image-to-video" if source_image_path else "text-to-video"
    
    prompt = '''\(escapedPrompt)'''
    log(f"Prompt: {prompt[:100]}...")
    log(f"Size: \(genWidth)x\(genHeight), \(params.numFrames) frames")
    log(f"Seed: \(seed)")
    
    disable_audio = \(request.disableAudio ? "True" : "False")
    resources_path = "\(resourcesPath)"
    wrapper_script = os.path.join(resources_path, "av_generator.py")
    use_wrapper = disable_audio
    if use_wrapper and not os.path.exists(wrapper_script):
        raise FileNotFoundError(f"Missing bundled script: {wrapper_script}")
    
    # Build command; use bundled wrapper when no-audio is requested
    if use_wrapper:
        cmd = [
            sys.executable, wrapper_script,
            "--prompt", prompt,
            "--height", str(\(genHeight)),
            "--width", str(\(genWidth)),
            "--num-frames", str(\(params.numFrames)),
            "--seed", str(\(seed)),
            "--fps", str(\(params.fps)),
            "--output-path", "\(outputPath)",
            "--model-repo", model_repo,
            "--tiling", "\(params.vaeTilingMode)",
            "--no-audio",
        ]
        log(f"Mode: {mode} (audio disabled)")
    else:
        cmd = [
            sys.executable, "-m", "mlx_video.generate_av",
            "--prompt", prompt,
            "--height", str(\(genHeight)),
            "--width", str(\(genWidth)),
            "--num-frames", str(\(params.numFrames)),
            "--seed", str(\(seed)),
            "--fps", str(\(params.fps)),
            "--output-path", "\(outputPath)",
            "--model-repo", model_repo,
            "--tiling", "\(params.vaeTilingMode)",
        ]
        log(f"Mode: {mode} (with audio)")
    
    # Add image conditioning if provided
    if source_image_path:
        cmd.extend(["--image", source_image_path])
        cmd.extend(["--image-strength", str(\(params.imageStrength))])
        log(f"Image conditioning: {source_image_path}")
    
    if (not disable_audio) and \(saveAudioTrackSeparately ? "True" : "False"):
        cmd.append("--save-audio-separately")
        log("Saving audio track separately")
    
    log("Starting generation...")
    log(f"Command: {' '.join(cmd)}")
    
    # Run the CLI module and stream combined output (binary read so we see tqdm \\r updates)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Chunk-based read: tqdm/huggingface_hub often update in place with \\r, so readline() can block.
    # Any received data during download phase counts as activity to avoid false stall timeouts.
    line_buf = ""
    download_in_progress = False
    last_download_activity = None  # None = not in download phase yet; set when first download line seen
    download_stall_timeout = 300  # 5 min without any data once download has started
    chunk_size = 8192
    while True:
        if process.stdout is None:
            break
        ready, _, _ = select.select([process.stdout], [], [], 1.0)
        if ready:
            try:
                raw = process.stdout.read(chunk_size)
            except (ValueError, OSError):
                raw = b""
            if not raw:
                if process.poll() is not None:
                    break
                continue
            # Decode and treat any received data as activity when we're in download phase
            try:
                chunk = raw.decode("utf-8", errors="replace")
            except Exception:
                chunk = ""
            if download_in_progress and last_download_activity is not None:
                last_download_activity = time.time()
            line_buf += chunk
            # Partial tqdm line (e.g. "  3%|") also counts as download activity
            if "%" in line_buf and "|" in line_buf:
                download_in_progress = True
                if last_download_activity is None:
                    last_download_activity = time.time()
            _nl = "\\n"
            _cr = "\\r"
            while _nl in line_buf or _cr in line_buf:
                line, sep, rest = line_buf.partition(_nl)
                if not sep:
                    line, sep, rest = line_buf.partition(_cr)
                line_buf = rest if sep else line_buf
                if not sep:
                    break
                line = line.strip()
                if not line:
                    continue
                log(line)
                low = line.lower()
                if ("fetching" in low) or ("downloading" in low) or line.startswith("DOWNLOAD:") or ("%" in line and "|" in line):
                    download_in_progress = True
                    if last_download_activity is None:
                        last_download_activity = time.time()
                if line.startswith("STAGE:") or "generation..." in low or "decoding" in low:
                    download_in_progress = False
                    last_download_activity = None
                # Emit explicit phase statuses so UI doesn't look frozen after denoising
                if "decoding video" in low:
                    print("STATUS:Decoding video...", file=sys.stderr, flush=True)
                elif "video encoded" in low:
                    print("STATUS:Saving video frames...", file=sys.stderr, flush=True)
                elif "decoding audio" in low:
                    print("STATUS:Decoding audio...", file=sys.stderr, flush=True)
                elif "combining video and audio" in low:
                    print("STATUS:Saving final video...", file=sys.stderr, flush=True)
                elif "saved video with audio" in low:
                    print("STATUS:Saving final video...", file=sys.stderr, flush=True)
                print(line, file=sys.stderr)
        else:
            if process.poll() is not None:
                break
            # Only enforce stall when we've seen download start and then no data for timeout
            if download_in_progress and last_download_activity is not None:
                stalled_for = int(time.time() - last_download_activity)
                if stalled_for >= download_stall_timeout:
                    print(f"DOWNLOAD:STALL:{stalled_for}", file=sys.stderr, flush=True)
                    log(f"ERROR: model download stalled for {stalled_for}s (no data received)")
                    process.kill()
                    raise TimeoutError(f"Model download stalled for {stalled_for}s")
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"mlx_video.generate_av failed with code {process.returncode}")
    
    log(f"Video with audio saved to: \(outputPath)")
    log("Generation complete!")
    log_file.close()
    print(json.dumps({"video_path": "\(outputPath)", "seed": \(seed), "mode": mode, "has_audio": not disable_audio}))
except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    log(traceback.format_exc())
    log_file.close()
    sys.exit(1)
"""
        
        progressHandler(0.05, "Running MLX generation...")
        
        // Thread-safe capture of enhanced prompt from stderr
        let enhancedPromptLock = NSLock()
        var capturedEnhancedPrompt: String? = preEnhancedPrompt
        let failureHintLock = NSLock()
        var capturedFailureHint: String? = nil
        
        let output: String
        do {
            output = try await runPython(script: script, timeout: 3600) { stderr in
            // Capture enhanced prompt from stderr
            // Our generate.py emits "ENHANCED_PROMPT:..." and mlx_video may emit "Enhanced prompt: ..."
            for line in stderr.components(separatedBy: "\n") {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                var extracted: String? = nil
                if trimmed.hasPrefix("ENHANCED_PROMPT:") {
                    extracted = String(trimmed.dropFirst("ENHANCED_PROMPT:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                } else if trimmed.lowercased().hasPrefix("enhanced prompt:") {
                    extracted = String(trimmed.dropFirst("enhanced prompt:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                }
                if let text = extracted, !text.isEmpty {
                    enhancedPromptLock.lock()
                    capturedEnhancedPrompt = text
                    enhancedPromptLock.unlock()
                }
            }
            
            DispatchQueue.main.async {
                // Parse line-by-line so chunked stderr reads never lose progress updates.
                for raw in stderr.components(separatedBy: "\n") {
                    let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                    if line.isEmpty { continue }
                    let lower = line.lowercased()
                    
                    if line.hasPrefix("DOWNLOAD:STALL:") {
                        let seconds = String(line.dropFirst("DOWNLOAD:STALL:".count))
                        progressHandler(0.01, "Download stalled for \(seconds)s. Stopping generation.")
                        failureHintLock.lock()
                        capturedFailureHint = "No download data received for \(seconds)s—connection may have stalled. Check your network; run `hf login` in Terminal if using gated models; then retry. To download the model manually, use: hf download \(modelRepo) (saves to ~/.cache/huggingface)."
                        failureHintLock.unlock()
                    } else if line.hasPrefix("TEXT_ENCODER_CONFIG_ERROR:") {
                        let detail = String(line.dropFirst("TEXT_ENCODER_CONFIG_ERROR:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                        failureHintLock.lock()
                        capturedFailureHint = "Text encoder configuration mismatch detected. \(detail) Update with: pip install -U \"mlx-video-with-audio>=0.1.15\" and retry."
                        failureHintLock.unlock()
                    } else if lower.contains("keyerror: 'text_config'") {
                        failureHintLock.lock()
                        capturedFailureHint = "Text encoder config mismatch (`text_config` missing). This usually means an outdated or misconfigured `mlx-video-with-audio` install. Update with: pip install -U \"mlx-video-with-audio>=0.1.15\" and retry."
                        failureHintLock.unlock()
                    } else if lower.contains("valueerror: [conv] expect the input channels") {
                        failureHintLock.lock()
                        capturedFailureHint = "Detected MLX VAE channel mismatch during decoding. This is an upstream `mlx-video-with-audio` model/package issue; please update the package and retry."
                        failureHintLock.unlock()
                    } else if lower.contains("kiogpucommandbuffercallbackerroroutofmemory")
                                || lower.contains("insufficient memory")
                                || lower.contains("std::bad_alloc")
                                || (lower.contains("command buffer execution failed") && lower.contains("memory"))
                                || (lower.contains("metal") && lower.contains("out of memory")) {
                        failureHintLock.lock()
                        capturedFailureHint = oomRecoveryHint
                        failureHintLock.unlock()
                        progressHandler(0.01, "Generation stopped: GPU memory limit reached")
                    }
                    
                    if line.hasPrefix("STAGE:") {
                        // Parse stage-aware progress: STAGE:1:STEP:3:8:Denoising
                        let parts = line.components(separatedBy: ":")
                        if parts.count >= 5,
                           let stage = Int(parts[1]),
                           let step = Int(parts[3]),
                           let total = Int(parts[4]) {
                            let stageProgress = Double(step) / Double(total)
                            let mappedProgress: Double
                            let message: String
                            
                            if stage == 1 {
                                mappedProgress = 0.1 + (stageProgress * 0.4)
                                message = "Stage 1 (\(step)/\(total)): Generating at half resolution"
                            } else {
                                mappedProgress = 0.5 + (stageProgress * 0.4)
                                message = "Stage 2 (\(step)/\(total)): Refining at full resolution"
                            }
                            progressHandler(mappedProgress, message)
                        }
                    } else if line.hasPrefix("STATUS:") {
                        let message = String(line.dropFirst(7))
                        if message.contains("Stage 1") {
                            progressHandler(0.1, message)
                        } else if message.contains("Stage 2") || message.contains("Upsampling") {
                            progressHandler(0.5, message)
                        } else if message.contains("Decoding") {
                            progressHandler(0.9, message)
                        } else if message.contains("Saving") {
                            progressHandler(0.95, message)
                        } else if message.contains("Loading") {
                            progressHandler(0.08, message)
                        } else {
                            progressHandler(0.05, message)
                        }
                    } else if line.hasPrefix("MODEL:CACHED:") {
                        let repo = String(line.dropFirst(13))
                        progressHandler(0.08, "Model cached: \(repo)")
                    } else if line.hasPrefix("DOWNLOAD:START:") {
                        let repo = String(line.dropFirst(15))
                        progressHandler(0.01, "Downloading model: \(repo)")
                    } else if line.hasPrefix("DOWNLOAD:PROGRESS:") {
                        let parts = line.dropFirst(18).split(separator: ":")
                        if parts.count >= 3 {
                            let currentBytes = Double(parts[0]) ?? 0
                            let totalBytes = Double(parts[1]) ?? 1
                            let pctStr = String(parts[2]).replacingOccurrences(of: "%", with: "")
                            let pct = Int(pctStr) ?? 0
                            let currentGB = currentBytes / 1_000_000_000
                            let totalGB = totalBytes / 1_000_000_000
                            let mappedProgress = 0.01 + (Double(pct) / 100.0 * 0.07)
                            progressHandler(mappedProgress, String(format: "Downloading: %.1fGB / %.1fGB (%d%%)", currentGB, totalGB, pct))
                        }
                    } else if line.hasPrefix("DOWNLOAD:COMPLETE:") {
                        progressHandler(0.08, "Model download complete")
                    } else if line.contains("Downloading") || line.contains("Fetching") {
                        // huggingface_hub tqdm output
                        let fileCountPattern = #/(\d+)%\|[^|]*\|\s*(\d+)/(\d+)/#
                        if let match = line.firstMatch(of: fileCountPattern) {
                            let currentFile = Int(match.2) ?? 0
                            let totalFiles = Int(match.3) ?? 1
                            var filePercent = Double(currentFile) / Double(max(totalFiles, 1))
                            var message = "Downloading: \(currentFile)/\(totalFiles) files"
                            if let bytesMatch = line.firstMatch(of: #/\|\s*([\d.]+)([KMG]?)B?\/([\d.]+)([KMG]?)B?/#) {
                                let curVal = Double(bytesMatch.1) ?? 0
                                let totVal = Double(bytesMatch.3) ?? 1
                                let unit = String(bytesMatch.2)
                                let scale: Double = unit == "G" ? 1 : (unit == "M" ? 0.001 : 0.000001)
                                let curGB = curVal * scale
                                let totGB = totVal * scale
                                let pct = totVal > 0 ? Int(100 * curVal / totVal) : 0
                                filePercent = (Double(currentFile) + Double(pct) / 100.0) / Double(max(totalFiles, 1))
                                message = String(format: "Downloading: %.1fGB / %.1fGB (file %d/%d, %d%%)", curGB, totGB, currentFile + 1, totalFiles, pct)
                            }
                            let mappedProgress = 0.01 + (filePercent * 0.07)
                            progressHandler(mappedProgress, message)
                        }
                    }
                }
            }
            }
        } catch {
            failureHintLock.lock()
            let hint = capturedFailureHint
            failureHintLock.unlock()
            if let hint, !hint.isEmpty {
                throw LTXError.generationFailed(hint)
            }
            throw error
        }
        
        // Parse JSON output - extract JSON from output (may have other text before it)
        // Look for JSON object starting with { and ending with }
        if let jsonStart = output.range(of: "{\"video_path\""),
           let jsonEnd = output.range(of: "}", range: jsonStart.lowerBound..<output.endIndex) {
            let jsonString = String(output[jsonStart.lowerBound...jsonEnd.lowerBound])
            if let data = jsonString.data(using: String.Encoding.utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let videoPath = json["video_path"] as? String,
               let resultSeed = json["seed"] as? Int {
                progressHandler(1.0, "Complete!")
                // Safe to read without lock: runPython has completed, no more stderr callbacks
                return (videoPath, resultSeed, capturedEnhancedPrompt)
            }
        }
        
        throw LTXError.generationFailed("Failed to parse generation output: \(output)")
    }
    
    func unloadModel() async {
        isModelLoaded = false
    }

    /// Preview enhanced prompt without running generation. Returns enhanced text or nil on error.
    func previewEnhancedPrompt(
        prompt: String,
        modelRepo: String,
        temperature: Double,
        sourceImagePath: String?,
        progressHandler: @escaping (String) -> Void
    ) async throws -> String? {
        setupPythonPaths()
        guard let python = pythonExecutable else {
            throw LTXError.pythonNotConfigured
        }
        let resourcesPath = Bundle.main.bundlePath + "/Contents/Resources"
        let scriptPath = resourcesPath + "/enhance_prompt_preview.py"
        guard FileManager.default.fileExists(atPath: scriptPath) else {
            throw LTXError.generationFailed("Preview script not found")
        }
        var args = [
            scriptPath,
            "--prompt", prompt,
            "--model-repo", modelRepo,
            "--temperature", String(temperature),
            "--resources-path", resourcesPath,
        ]
        if let img = sourceImagePath, !img.isEmpty {
            args.append(contentsOf: ["--image", img])
        }
        progressHandler("Loading prompt enhancer (first run may download ~7GB)...")
        let output = try await runPythonScript(executable: python, arguments: args, timeout: 300)
        if let data = output.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let enhanced = json["enhanced_prompt"] as? String, !enhanced.isEmpty {
                return enhanced
            }
            if let err = json["error"] as? String {
                throw LTXError.generationFailed(err)
            }
        }
        return nil
    }

    private func runPythonScript(
        executable: String,
        arguments: [String],
        timeout: TimeInterval = 60
    ) async throws -> String {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: executable)
                process.arguments = arguments
                var env: [String: String] = [:]
                let pythonBin = URL(fileURLWithPath: executable).deletingLastPathComponent().path
                env["PATH"] = "\(pythonBin):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
                env["HOME"] = ProcessInfo.processInfo.environment["HOME"] ?? ""
                env["USER"] = ProcessInfo.processInfo.environment["USER"] ?? ""
                env["TMPDIR"] = ProcessInfo.processInfo.environment["TMPDIR"] ?? "/tmp"
                process.environment = env
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe
                do {
                    try process.run()
                    process.waitUntilExit()
                    let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let output = String(data: outputData, encoding: .utf8) ?? ""
                    let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
                    if process.terminationStatus != 0 {
                        let errData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                        let errStr = String(data: errData, encoding: .utf8) ?? ""
                        continuation.resume(throwing: LTXError.generationFailed(errStr.isEmpty ? "Exit code \(process.terminationStatus)" : errStr))
                    } else {
                        continuation.resume(returning: trimmed)
                    }
                } catch {
                    continuation.resume(throwing: LTXError.generationFailed(error.localizedDescription))
                }
            }
        }
    }

    private func runPython(
        script: String,
        timeout: TimeInterval = 60,
        stderrHandler: ((String) -> Void)? = nil
    ) async throws -> String {
        guard let python = pythonExecutable else {
            throw LTXError.pythonNotConfigured
        }
        
        let logFile = "/tmp/ltx_generation.log"
        
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: python)
                process.arguments = ["-c", script]
                
                // Clean environment for MLX
                var env: [String: String] = [:]
                
                let pythonBin = URL(fileURLWithPath: python).deletingLastPathComponent().path
                env["PATH"] = "\(pythonBin):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"
                env["HOME"] = ProcessInfo.processInfo.environment["HOME"] ?? ""
                env["USER"] = ProcessInfo.processInfo.environment["USER"] ?? ""
                env["TMPDIR"] = ProcessInfo.processInfo.environment["TMPDIR"] ?? "/tmp"
                
                // MLX uses Metal - inherit any Metal-related env vars
                if let metalDevice = ProcessInfo.processInfo.environment["MTL_DEVICE_WRAPPER_TYPE"] {
                    env["MTL_DEVICE_WRAPPER_TYPE"] = metalDevice
                }
                
                process.environment = env
                
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe
                
                var stderrAccumulated = ""
                let stderrLock = NSLock()
                
                stderrPipe.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                        stderrLock.lock()
                        stderrAccumulated += str
                        stderrLock.unlock()
                        
                        if let logData = ("[STDERR] " + str).data(using: .utf8) {
                            if FileManager.default.fileExists(atPath: logFile) {
                                if let handle = FileHandle(forWritingAtPath: logFile) {
                                    handle.seekToEndOfFile()
                                    handle.write(logData)
                                    handle.closeFile()
                                }
                            } else {
                                try? logData.write(to: URL(fileURLWithPath: logFile))
                            }
                        }
                        
                        // Send only the latest chunk; caller parses line-by-line.
                        stderrHandler?(str)
                    }
                }
                
                do {
                    let startLog = "=== LTX MLX Process Started ===\nPython: \(python)\nTime: \(Date())\n"
                    try? startLog.write(toFile: logFile, atomically: false, encoding: .utf8)
                    
                    try process.run()
                    process.waitUntilExit()
                    
                    stderrPipe.fileHandleForReading.readabilityHandler = nil
                    
                    let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let output = String(data: outputData, encoding: .utf8) ?? ""
                    
                    let outputLog = "\n[STDOUT] \(output)\n[EXIT CODE] \(process.terminationStatus)\n"
                    if let handle = FileHandle(forWritingAtPath: logFile) {
                        handle.seekToEndOfFile()
                        handle.write(outputLog.data(using: .utf8)!)
                        handle.closeFile()
                    }
                    
                    let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)
                    if let data = trimmedOutput.data(using: .utf8),
                       let _ = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        continuation.resume(returning: trimmedOutput)
                        return
                    }
                    
                    if process.terminationStatus != 0 {
                        stderrLock.lock()
                        let stderr = stderrAccumulated
                        stderrLock.unlock()
                        
                        let harmlessPatterns = ["UserWarning", "FutureWarning"]
                        let isOnlyHarmless = harmlessPatterns.allSatisfy { stderr.contains($0) } ||
                                            stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        
                        if !trimmedOutput.isEmpty && isOnlyHarmless {
                            continuation.resume(returning: trimmedOutput)
                        } else {
                            continuation.resume(throwing: LTXError.generationFailed("Exit code \(process.terminationStatus). Check /tmp/ltx_generation.log"))
                        }
                    } else {
                        continuation.resume(returning: trimmedOutput)
                    }
                } catch {
                    let errorLog = "\n[ERROR] \(error.localizedDescription)\n"
                    if let handle = FileHandle(forWritingAtPath: logFile) {
                        handle.seekToEndOfFile()
                        handle.write(errorLog.data(using: .utf8)!)
                        handle.closeFile()
                    }
                    continuation.resume(throwing: LTXError.generationFailed(error.localizedDescription))
                }
            }
        }
    }
}
