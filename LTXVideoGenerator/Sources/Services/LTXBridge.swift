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
        
        guard let python = pythonExecutable else {
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
        
        progressHandler("MLX environment ready. Model will download on first generation (~90GB).")
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
        
        let modelRepo = LTXModelVariant.modelRepo
        let isImageToVideo = request.isImageToVideo
        let modeDescription = isImageToVideo ? "image-to-video" : "text-to-video"
        progressHandler(0.1, "Starting \(modeDescription) with audio (\(LTXModelVariant.displayName))...")
        
        // Escape the prompt for Python
        let escapedPrompt = request.prompt
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
        let enableGemmaPromptEnhancement = UserDefaults.standard.bool(forKey: "enableGemmaPromptEnhancement")
        let saveAudioTrackSeparately = UserDefaults.standard.bool(forKey: "saveAudioTrackSeparately")
        // LTX-2 Unified - uses mlx-video-with-audio package
        script = """
import os
import sys
import json
import subprocess

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
    log(f"Mode: {mode} (with audio)")
    
    prompt = '''\(escapedPrompt)'''
    log(f"Prompt: {prompt[:100]}...")
    log(f"Size: \(genWidth)x\(genHeight), \(params.numFrames) frames")
    log(f"Seed: \(seed)")
    
    # Build CLI args for mlx_video.generate_av module
    # Note: mlx_video.generate_av uses --enhance-prompt/--temperature
    # instead of --repetition-penalty/--top-p
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
    
    # Add --enhance-prompt only when enabled in Settings
    enable_enhancement = \(enableGemmaPromptEnhancement ? "True" : "False")
    if enable_enhancement:
        cmd.append("--enhance-prompt")
        cmd.append("--use-uncensored-enhancer")
        cmd.extend(["--temperature", str(\(request.gemmaTopP))])
        # Pre-flight: copy bundled prompts into mlx_video if missing (pip package omits them)
        try:
            from pathlib import Path
            import shutil
            resources_path = Path("\(resourcesPath)")
            bundled_prompts = resources_path / "ltx_mlx" / "models" / "ltx" / "prompts"
            import mlx_video.models.ltx.text_encoder as te
            target_dir = Path(te.__file__).parent / "prompts"
            for name in ["gemma_t2v_system_prompt.txt", "gemma_i2v_system_prompt.txt"]:
                src = bundled_prompts / name
                dst = target_dir / name
                if src.exists() and not dst.exists():
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    log(f"Injected missing prompt: {name}")
        except Exception as inj_err:
            log(f"Warning: could not inject prompts: {inj_err}")
    
    # Add image conditioning if provided
    if source_image_path:
        cmd.extend(["--image", source_image_path])
        cmd.extend(["--image-strength", str(\(params.imageStrength))])
        log(f"Image conditioning: {source_image_path}")
    
    # Disable audio if requested
    disable_audio = \(request.disableAudio ? "True" : "False")
    if disable_audio:
        cmd.append("--no-audio")
        log("Audio generation disabled")
    elif \(saveAudioTrackSeparately ? "True" : "False"):
        cmd.append("--save-audio-separately")
        log("Saving audio track separately")
    
    log("Starting generation...")
    log(f"Command: {' '.join(cmd)}")
    
    # Run the CLI module and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Stream stderr for progress updates
    for line in process.stderr:
        line = line.strip()
        if line:
            log(line)
            print(line, file=sys.stderr)
    
    # Wait for completion
    stdout, _ = process.communicate()
    
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
        var capturedEnhancedPrompt: String? = nil
        
        let output = try await runPython(script: script, timeout: 3600) { stderr in
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
                // Parse structured progress output from generate.py
                // Format: STAGE:X:STEP:Y:Z:message or STATUS:message or DOWNLOAD:START/COMPLETE:repo
                
                if stderr.hasPrefix("STAGE:") {
                    // Parse stage-aware progress: STAGE:1:STEP:3:8:Denoising
                    // Stage 1 maps to 0.1-0.5, Stage 2 maps to 0.5-0.9
                    let parts = stderr.components(separatedBy: ":")
                    if parts.count >= 5,
                       let stage = Int(parts[1]),
                       let step = Int(parts[3]),
                       let total = Int(parts[4]) {
                        let stageProgress = Double(step) / Double(total)
                        let mappedProgress: Double
                        let message: String
                        
                        if stage == 1 {
                            // Stage 1: 0.1 to 0.5 (half resolution)
                            mappedProgress = 0.1 + (stageProgress * 0.4)
                            message = "Stage 1 (\(step)/\(total)): Generating at half resolution"
                        } else {
                            // Stage 2: 0.5 to 0.9 (full resolution)
                            mappedProgress = 0.5 + (stageProgress * 0.4)
                            message = "Stage 2 (\(step)/\(total)): Refining at full resolution"
                        }
                        progressHandler(mappedProgress, message)
                    }
                } else if stderr.hasPrefix("STATUS:") {
                    // Parse status message: STATUS:Loading model...
                    let message = String(stderr.dropFirst(7))
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
                } else if stderr.hasPrefix("MODEL:CACHED:") {
                    let repo = String(stderr.dropFirst(13))
                    progressHandler(0.08, "Model cached: \(repo)")
                } else if stderr.hasPrefix("DOWNLOAD:START:") {
                    let repo = String(stderr.dropFirst(15))
                    progressHandler(0.01, "Downloading model: \(repo)")
                } else if stderr.hasPrefix("DOWNLOAD:PROGRESS:") {
                    // Format: DOWNLOAD:PROGRESS:currentBytes:totalBytes:pct%
                    let parts = stderr.dropFirst(18).split(separator: ":")
                    if parts.count >= 3 {
                        let currentBytes = Double(parts[0]) ?? 0
                        let totalBytes = Double(parts[1]) ?? 1
                        let pctStr = String(parts[2]).replacingOccurrences(of: "%", with: "")
                        let pct = Int(pctStr) ?? 0
                        // Convert bytes to GB for display
                        let currentGB = currentBytes / 1_000_000_000
                        let totalGB = totalBytes / 1_000_000_000
                        // Map download progress to 1-8% of total progress
                        let mappedProgress = 0.01 + (Double(pct) / 100.0 * 0.07)
                        progressHandler(mappedProgress, String(format: "Downloading: %.1fGB / %.1fGB (%d%%)", currentGB, totalGB, pct))
                    }
                } else if stderr.hasPrefix("DOWNLOAD:COMPLETE:") {
                    progressHandler(0.08, "Model download complete")
                } else if stderr.contains("Downloading") || stderr.contains("Fetching") {
                    // huggingface_hub tqdm: "Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]"
                    // Per-file bytes: "model.safetensors:  45%|████▌     | 2.3G/5.1G" - parse bytes if present
                    let fileCountPattern = #/(\d+)%\|[^|]*\|\s*(\d+)/(\d+)/#
                    if let match = stderr.firstMatch(of: fileCountPattern) {
                        let currentFile = Int(match.2) ?? 0
                        let totalFiles = Int(match.3) ?? 1
                        var filePercent = Double(currentFile) / Double(max(totalFiles, 1))
                        var message = "Downloading: \(currentFile)/\(totalFiles) files"
                        // Bytes progress: "| 2.3G/5.1G" or "| 45MB/100MB"
                        if let bytesMatch = stderr.firstMatch(of: #/\|\s*([\d.]+)([KMG]?)B?\/([\d.]+)([KMG]?)B?/#) {
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
        let output = try await runPythonScript(executable: python, arguments: args, timeout: 120)
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
                        let accumulated = stderrAccumulated
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
                        
                        stderrHandler?(accumulated)
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
