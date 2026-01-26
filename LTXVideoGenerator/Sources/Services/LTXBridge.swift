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
        
        progressHandler("MLX environment ready. Model will download on first generation (~10GB).")
        isModelLoaded = true
    }
    
    func generate(
        request: GenerationRequest,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> (videoPath: String, seed: Int) {
        setupPythonPaths()
        
        guard let _ = pythonExecutable else {
            throw LTXError.pythonNotConfigured
        }
        
        let params = request.parameters
        let seed = params.seed ?? Int.random(in: 0..<Int(Int32.max))
        
        // Get selected model variant from preferences
        let modelVariantRaw = UserDefaults.standard.string(forKey: "selectedModelVariant") ?? "distilled"
        let modelVariant = LTXModelVariant(rawValue: modelVariantRaw) ?? .distilled
        let modelRepo = modelVariant.modelRepo
        let isDistilled = modelVariant.isDistilled
        
        let isImageToVideo = request.isImageToVideo
        let modeDescription = isImageToVideo ? "image-to-video" : "text-to-video"
        progressHandler(0.1, "Starting \(modeDescription) with \(modelVariant.displayName)...")
        
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
        
        // Get the bundled ltx_generator.py script path
        let resourcesPath = Bundle.main.bundlePath + "/Contents/Resources"
        let generatorScript = resourcesPath + "/ltx_generator.py"
        
        // Build arguments for the generator script
        var scriptArgs = [
            generatorScript,
            "--prompt", request.prompt,
            "--height", String(genHeight),
            "--width", String(genWidth),
            "--num-frames", String(params.numFrames),
            "--seed", String(seed),
            "--fps", String(params.fps),
            "--output-path", outputPath,
            "--model-repo", modelRepo,
            "--tiling", "auto"
        ]
        
        // Add image conditioning if provided
        if isImageToVideo, let imagePath = request.sourceImagePath {
            scriptArgs.append(contentsOf: ["--image", imagePath])
        }
        
        let script = """
import os
import sys
import json

# Set up file logging
log_file = open("\(logFile)", "w")
def log(msg):
    print(msg, file=log_file, flush=True)
    print(msg, file=sys.stderr, flush=True)

try:
    log("=== LTX-2 MLX Generation Started ===")
    log(f"Python: {sys.executable}")
    
    # Add Resources directory to path for bundled ltx_mlx module
    resources_dir = "\(resourcesPath)"
    sys.path.insert(0, resources_dir)
    
    # Check MLX
    import mlx.core as mx
    log(f"MLX device: Apple Silicon")
    
    # Import bundled generation module
    from ltx_mlx.generate import generate_video
    
    model_repo = "\(modelRepo)"
    log(f"Model: {model_repo}")
    
    # Image-to-video mode
    source_image_path = "\(escapedImagePath)" if "\(escapedImagePath)" else None
    mode = "image-to-video" if source_image_path else "text-to-video"
    log(f"Mode: {mode}")
    
    prompt = "\(escapedPrompt)"
    log(f"Prompt: {prompt[:100]}...")
    log(f"Size: \(genWidth)x\(genHeight), \(params.numFrames) frames")
    log(f"Seed: \(seed)")
    
    # Build generation kwargs
    gen_kwargs = {
        "model_repo": model_repo,
        "prompt": prompt,
        "height": \(genHeight),
        "width": \(genWidth),
        "num_frames": \(params.numFrames),
        "seed": \(seed),
        "fps": \(params.fps),
        "output_path": "\(outputPath)",
        "tiling": "auto",
    }
    
    # Add image conditioning if provided
    if source_image_path:
        gen_kwargs["image"] = source_image_path
        gen_kwargs["image_strength"] = 1.0
        gen_kwargs["image_frame_idx"] = 0
        log(f"Image conditioning: {source_image_path}")
    
    log("Starting generation...")
    generate_video(**gen_kwargs)
    
    log(f"Video saved to: \(outputPath)")
    log("Generation complete!")
    log_file.close()
    print(json.dumps({"video_path": "\(outputPath)", "seed": \(seed), "mode": mode}))
except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    log(traceback.format_exc())
    log_file.close()
    sys.exit(1)
"""
        
        progressHandler(0.05, "Running MLX generation...")
        
        let output = try await runPython(script: script, timeout: 3600) { stderr in
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
                } else if stderr.hasPrefix("DOWNLOAD:START:") {
                    let repo = String(stderr.dropFirst(15))
                    progressHandler(0.01, "Downloading model: \(repo) (~90GB)")
                } else if stderr.hasPrefix("DOWNLOAD:PROGRESS:") {
                    // Format: DOWNLOAD:PROGRESS:current:total:pct%
                    let parts = stderr.dropFirst(18).split(separator: ":")
                    if parts.count >= 3 {
                        let current = Int(parts[0]) ?? 0
                        let total = Int(parts[1]) ?? 1
                        let pctStr = String(parts[2]).replacingOccurrences(of: "%", with: "")
                        let pct = Int(pctStr) ?? 0
                        // Map download progress to 1-8% of total progress
                        let mappedProgress = 0.01 + (Double(pct) / 100.0 * 0.07)
                        progressHandler(mappedProgress, "Downloading: \(current)/\(total) files (\(pct)%)")
                    }
                } else if stderr.hasPrefix("DOWNLOAD:COMPLETE:") {
                    progressHandler(0.08, "Model download complete")
                } else if stderr.contains("Downloading") || stderr.contains("Fetching") {
                    // Fallback for tqdm/huggingface_hub progress (legacy)
                    if let match = stderr.firstMatch(of: /(\d+)%\|[^|]*\|\s*(\d+)\/(\d+)/) {
                        let currentFile = Int(match.2) ?? 0
                        let totalFiles = Int(match.3) ?? 1
                        let percent = Double(currentFile) / Double(totalFiles)
                        let mappedProgress = 0.01 + (percent * 0.07)
                        progressHandler(mappedProgress, "Downloading: \(currentFile)/\(totalFiles) files")
                    }
                }
            }
        }
        
        // Parse JSON output
        if let data = output.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let videoPath = json["video_path"] as? String,
           let resultSeed = json["seed"] as? Int {
            progressHandler(1.0, "Complete!")
            return (videoPath, resultSeed)
        }
        
        throw LTXError.generationFailed("Failed to parse generation output: \(output)")
    }
    
    func unloadModel() async {
        isModelLoaded = false
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
