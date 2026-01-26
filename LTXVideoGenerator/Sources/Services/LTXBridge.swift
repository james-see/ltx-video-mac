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

// Use subprocess to avoid PythonKit threading issues
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
            // Direct executable path
            pythonExecutable = savedPath
            // Try to find python home
            if let dylib = PythonEnvironment.shared.executableToDylib(savedPath),
               let home = PythonEnvironment.shared.extractPythonHome(from: dylib) {
                pythonHome = home
            } else {
                // Fallback: assume standard layout
                let execURL = URL(fileURLWithPath: savedPath)
                pythonHome = execURL.deletingLastPathComponent().deletingLastPathComponent().path
            }
            
        case .dylib:
            // Dylib path - extract executable
            if let exec = PythonEnvironment.shared.dylibToExecutable(savedPath) {
                pythonExecutable = exec
            }
            if let home = PythonEnvironment.shared.extractPythonHome(from: savedPath) {
                pythonHome = home
                // If we couldn't find executable, try standard location
                if pythonExecutable == nil {
                    let standardExec = "\(home)/bin/python3"
                    if FileManager.default.isExecutableFile(atPath: standardExec) {
                        pythonExecutable = standardExec
                    }
                }
            }
            
        case .unknown:
            // Try to use it as executable if it exists and is executable
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
        
        progressHandler("Checking Python environment...")
        
        // Test that we can import the required modules for LTX-2
        // LTX2Pipeline requires diffusers installed from git
        let testScript = """
        import torch
        from diffusers import LTX2Pipeline
        from diffusers.pipelines.ltx2.export_utils import encode_video
        print("OK")
        """
        
        let result = try await runPython(script: testScript)
        if !result.contains("OK") {
            throw LTXError.pythonNotConfigured
        }
        
        progressHandler("Python environment ready. Model will load on first generation.")
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
        let modelVariantRaw = UserDefaults.standard.string(forKey: "selectedModelVariant") ?? "full"
        let modelVariant = LTXModelVariant(rawValue: modelVariantRaw) ?? .full
        let subfolder = modelVariant.subfolder
        let isDistilled = modelVariant == .distilled
        
        let isImageToVideo = request.isImageToVideo
        let modeDescription = isImageToVideo ? "image-to-video" : "text-to-video"
        progressHandler(0.1, "Starting \(modeDescription) with \(modelVariant.displayName)...")
        
        // Escape the prompt for Python
        let escapedPrompt = request.prompt
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        
        let escapedNegative = request.negativePrompt
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        
        // Escape source image path if provided
        let escapedImagePath = request.sourceImagePath?
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"") ?? ""
        
        // Log file path
        let logFile = "/tmp/ltx_generation.log"
        
        // Adjust guidance for distilled model (must be 1.0)
        let effectiveGuidance = isDistilled ? 1.0 : params.guidanceScale
        
        let script = """
import os
import sys
import json

import torch
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video

# Set up file logging
log_file = open("\(logFile)", "w")
def log(msg):
    print(msg, file=log_file, flush=True)
    print(msg, file=sys.stderr, flush=True)

try:
    log("=== LTX-2 Generation Started ===")
    log(f"Python: {sys.executable}")
    log(f"Torch version: {torch.__version__}")
    log(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Image-to-video: load source image if provided
    source_image_path = "\(escapedImagePath)"
    source_image = None
    mode = "text-to-video"
    if source_image_path:
        from PIL import Image
        source_image = Image.open(source_image_path).convert("RGB")
        mode = "image-to-video"
        log(f"Source image loaded: {source_image_path} ({source_image.size[0]}x{source_image.size[1]})")
    
    log(f"Mode: {mode}")
    log("Loading model...")
    
    model_repo = "Lightricks/LTX-2"
    subfolder = "\(subfolder)"
    log(f"Loading LTX-2 pipeline: {model_repo} / {subfolder}")
    
    # Use float16 for best MPS compatibility on Apple Silicon
    # device_map=None prevents automatic CPU offloading - we want pure MPS
    pipe = LTX2Pipeline.from_pretrained(
        model_repo,
        subfolder=subfolder,
        torch_dtype=torch.float16,
        device_map=None,
    )
    
    log("Moving to MPS (no CPU offload)...")
    pipe.to("mps")
    
    # Verify pipeline is on MPS
    log(f"Pipeline device: {pipe.device}")
    log(f"Transformer device: {pipe.transformer.device}")
    
    log("Pipeline ready")
    log("Generating video with audio...")
    
    # Ensure dimensions are divisible by 32 for proper alignment
    gen_width = (\(params.width) // 32) * 32
    gen_height = (\(params.height) // 32) * 32
    # Frames must be 8n+1 for this model
    gen_frames = ((\(params.numFrames) - 1) // 8) * 8 + 1
    
    log(f"Setting up generator with seed \(seed)...")
    generator = torch.Generator(device="mps")
    generator.manual_seed(\(seed))
    
    prompt = "\(escapedPrompt)"
    negative_prompt = "\(escapedNegative)" if "\(escapedNegative)" else None
    
    # Distilled model uses CFG=1
    is_distilled = \(isDistilled ? "True" : "False")
    guidance = \(effectiveGuidance) if not is_distilled else 1.0
    
    log(f"Prompt: {prompt}")
    log(f"Model: {subfolder}, Distilled: {is_distilled}")
    log(f"Settings: steps=\(params.numInferenceSteps), guidance={guidance}, size={gen_width}x{gen_height}, frames={gen_frames}")
    log("Starting pipeline...")
    
    # Synchronize MPS before generation
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    # Build pipeline arguments - only include image for image-to-video mode
    pipe_kwargs = {
        "prompt": prompt,
        "num_inference_steps": \(params.numInferenceSteps),
        "guidance_scale": guidance,
        "width": gen_width,
        "height": gen_height,
        "num_frames": gen_frames,
        "frame_rate": float(\(params.fps)),
        "generator": generator,
        "output_type": "np",
        "return_dict": False,
    }
    
    # Add optional parameters only when needed
    if negative_prompt and not is_distilled:
        pipe_kwargs["negative_prompt"] = negative_prompt
    if source_image is not None:
        pipe_kwargs["image"] = source_image
    
    # LTX-2 returns video and audio
    video, audio = pipe(**pipe_kwargs)
    
    # Synchronize after generation
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    log(f"Pipeline complete. Video shape: {video.shape}")
    
    # Convert video to tensor format for encode_video
    video = (video * 255).round().astype("uint8")
    video_tensor = torch.from_numpy(video)
    
    log(f"Exporting video with audio to \(outputPath)...")
    
    # Export video with synchronized audio using LTX-2's encode_video
    encode_video(
        video_tensor[0],
        fps=\(params.fps),
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path="\(outputPath)",
    )
    
    log("Export complete!")
    log_file.close()
    print(json.dumps({"video_path": "\(outputPath)", "seed": \(seed), "mode": mode}))
except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    log(traceback.format_exc())
    log_file.close()
    sys.exit(1)
"""
        
        progressHandler(0.2, "Running generation script...")
        
        let output = try await runPython(script: script, timeout: 600) { stderr in
            DispatchQueue.main.async {
                if stderr.contains("Loading pipeline") || stderr.contains("Loading checkpoint") {
                    progressHandler(0.1, "Loading model...")
                } else if stderr.contains("Moving to MPS") {
                    progressHandler(0.2, "Moving to GPU...")
                } else if stderr.contains("Fetching") {
                    // Model download progress: "Fetching 55 files:  7%|▋ | 4/55"
                    if let match = stderr.firstMatch(of: /(\d+)%\|[^|]*\|\s*(\d+)\/(\d+)/) {
                        let currentFile = Int(match.2) ?? 0
                        let totalFiles = Int(match.3) ?? 1
                        let percent = Double(currentFile) / Double(totalFiles)
                        // Map to 0.05-0.15 range (download is early phase)
                        let mappedProgress = 0.05 + (percent * 0.1)
                        progressHandler(mappedProgress, "Downloading model: \(currentFile)/\(totalFiles) files")
                    }
                } else if stderr.contains("Starting pipeline") {
                    progressHandler(0.25, "Starting generation...")
                } else if let match = stderr.firstMatch(of: /(\d+)%\|[^|]*\|\s*(\d+)\/(\d+)/) {
                    // Generation progress like "12%|█▏        | 3/25"
                    let currentStep = Int(match.2) ?? 0
                    let totalSteps = Int(match.3) ?? 1
                    let percent = Double(currentStep) / Double(totalSteps)
                    // Map to 0.3-0.9 range (leaving room for load/export)
                    let mappedProgress = 0.3 + (percent * 0.6)
                    progressHandler(mappedProgress, "Generating: \(currentStep)/\(totalSteps) steps")
                } else if stderr.contains("Exporting") {
                    progressHandler(0.95, "Exporting video...")
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
                
                // Use a CLEAN environment like terminal - GUI app environment can interfere with MPS
                var env: [String: String] = [:]
                
                // Essential paths only
                let pythonBin = URL(fileURLWithPath: python).deletingLastPathComponent().path
                env["PATH"] = "\(pythonBin):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
                env["HOME"] = ProcessInfo.processInfo.environment["HOME"] ?? ""
                env["USER"] = ProcessInfo.processInfo.environment["USER"] ?? ""
                env["TMPDIR"] = ProcessInfo.processInfo.environment["TMPDIR"] ?? "/tmp"
                
                // Inherit library paths for dynamic libraries
                if let dylibPath = ProcessInfo.processInfo.environment["DYLD_LIBRARY_PATH"] {
                    env["DYLD_LIBRARY_PATH"] = dylibPath
                }
                
                // Metal/MPS - use defaults, don't inherit app's GPU state
                env["MTL_ENABLE_DEBUG_INFO"] = "0"
                
                // Allow PyTorch MPS to use all available unified memory (disable 70% limit)
                env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
                
                process.environment = env
                
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe
                
                // Accumulate stderr for logging
                var stderrAccumulated = ""
                let stderrLock = NSLock()
                
                // Handle stderr for progress and logging
                stderrPipe.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                        stderrLock.lock()
                        stderrAccumulated += str
                        stderrLock.unlock()
                        
                        // Write to log file
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
                    // Log start
                    let startLog = "=== LTX Process Started ===\nPython: \(python)\nTime: \(Date())\n"
                    try? startLog.write(toFile: logFile, atomically: false, encoding: .utf8)
                    
                    try process.run()
                    process.waitUntilExit()
                    
                    stderrPipe.fileHandleForReading.readabilityHandler = nil
                    
                    let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let output = String(data: outputData, encoding: .utf8) ?? ""
                    
                    // Log output
                    let outputLog = "\n[STDOUT] \(output)\n[EXIT CODE] \(process.terminationStatus)\n"
                    if let handle = FileHandle(forWritingAtPath: logFile) {
                        handle.seekToEndOfFile()
                        handle.write(outputLog.data(using: .utf8)!)
                        handle.closeFile()
                    }
                    
                    // Check for valid JSON output first - if we got valid output, ignore exit code
                    let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)
                    if let data = trimmedOutput.data(using: .utf8),
                       let _ = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        // Valid JSON output - success regardless of exit code (warnings may cause non-zero)
                        continuation.resume(returning: trimmedOutput)
                        return
                    }
                    
                    if process.terminationStatus != 0 {
                        // Filter out harmless warnings
                        stderrLock.lock()
                        let stderr = stderrAccumulated
                        stderrLock.unlock()
                        
                        // If stderr only contains known harmless warnings, check if we have output
                        let harmlessPatterns = ["resource_tracker", "semaphore", "UserWarning"]
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
                    // Log error
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
