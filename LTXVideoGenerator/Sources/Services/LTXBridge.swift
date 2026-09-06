import Foundation
import Darwin

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

/// Thread-safe holder so a Task-cancellation handler can terminate the running
/// generation subprocess even though it is created on a background dispatch queue.
private final class CancellableProcessHolder: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var cancelledFlag = false

    /// Store the process. Returns false if cancellation already happened, in
    /// which case the caller should not start the process.
    func storeIfNotCancelled(_ p: Process) -> Bool {
        lock.lock(); defer { lock.unlock() }
        if cancelledFlag { return false }
        process = p
        return true
    }

    /// Mark cancelled and terminate the process (SIGTERM). The Python wrapper's
    /// signal handler forwards SIGKILL to the whole child process group.
    func cancel() {
        lock.lock()
        cancelledFlag = true
        let p = process
        lock.unlock()
        if let p {
            let pid = p.processIdentifier
            if pid > 0 {
                kill(pid, SIGTERM)
            }
            p.terminate()
        }
    }

    var wasCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return cancelledFlag
    }
}

// Use subprocess to run MLX-based generation
class LTXBridge {
    static let shared = LTXBridge()

    private static let metalInteractivityUserHint =
        "GPU command buffer was killed by the system because a single Metal kernel ran too long (typical with VAE tiling set to Aggressive on small resolutions or older hardware). Try tiling Auto or Conservative, lower resolution, or fewer frames. Disabling Aggressive tiling is the first thing to try.\n\n"
        + "Update mlx-video-with-audio if you are on an older build. Full log: /tmp/ltx_generation.log"

    private static func jetsamSigKillUserHint(modelRepo: String, textEncoderRepo: String) -> String {
        let phys = MacOSSystemMemory.physicalMemoryGBFormatted()
        let avail = MacOSSystemMemory.approximateAvailableMemoryGBFormatted()
        return "macOS killed the generator process due to memory pressure. Try a smaller text encoder (for example mlx-community/gemma-3-4b-it-bf16 or a 4-bit quant), enable aggressive VAE tiling, lower resolution or frame count, or close other apps.\n\n"
            + "Model: \(modelRepo). Text encoder: \(textEncoderRepo). This Mac reports about \(phys) GB physical memory and roughly \(avail) GB available (approximate).\n\n"
            + "Open Preferences → General and use the Text Encoder picker (Gemma 4B bf16 or 4-bit). Full log: /tmp/ltx_generation.log"
    }

    private static func stderrIndicatesMetalInteractivity(_ stderr: String) -> Bool {
        let low = stderr.lowercased()
        return low.contains("diagnostic_metal_interactivity")
            || low.contains("impacting interactivity")
            || low.contains("kiogpucommandbuffercallbackerrorimpactinginteractivity")
    }

    /// Interprets the outer Python wrapper exit / stderr after generation.
    private static func diagnoseRunnerFailure(
        exitCode: Int32,
        stderr: String,
        modelRepo: String?,
        textEncoderRepo: String?
    ) -> String? {
        let low = stderr.lowercased()
        let model = modelRepo ?? "(unknown model)"
        let enc = textEncoderRepo ?? "(unknown text encoder)"

        if low.contains("diagnostic_metal_interactivity")
            || low.contains("impacting interactivity")
            || low.contains("kiogpucommandbuffercallbackerrorimpactinginteractivity")
            || low.contains("kIOGPUCommandBufferCallbackErrorImpactingInteractivity".lowercased())
        {
            return metalInteractivityUserHint
        }

        // Do not match a bare "-9" — UUIDs like -9996 plus a RuntimeError traceback
        // were misread as jetsam after LTX-2.5 DurationHead KeyError.
        let looksLikeSigKill = low.contains("sigkill")
            || low.contains("exit code -9")
            || low.contains("killed by signal 9")
            || low.contains("terminated by signal 9")
        if looksLikeSigKill || exitCode == 137 || exitCode == 9 {
            return jetsamSigKillUserHint(modelRepo: model, textEncoderRepo: enc)
        }
        return nil
    }
    
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
        let selectedModel = LTXModelCatalog.selectedModel()
        if selectedModel.backend == .h3c {
            let variant = H3Variant.from(modelId: selectedModel.id)
            if let binary = H3Engine.resolvedBinary(for: variant) {
                progressHandler("\(variant.displayName) ready (\(binary)). Weights download on first generation (\(selectedModel.downloadSize)).")
            } else {
                progressHandler("\(variant.displayName) will clone/build h3 on first generation. Weights download then too (\(selectedModel.downloadSize)).")
            }
            isModelLoaded = true
            return
        }

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
        
        progressHandler("MLX environment ready. Model will download on first generation (\(selectedModel.downloadSize)).")
        isModelLoaded = true
    }
    
    func generate(
        request: GenerationRequest,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> (videoPath: String, seed: Int, enhancedPrompt: String?) {
        let selectedModel = LTXModelCatalog.resolvedModel(id: request.modelId)
        switch selectedModel.backend {
        case .mlxVideoWithAudio:
            return try await generateWithMlxVideo(
                request: request,
                outputPath: outputPath,
                progressHandler: progressHandler
            )
        case .ltx2Mlx:
            return try await generateWithLtx2Mlx(
                request: request,
                outputPath: outputPath,
                progressHandler: progressHandler
            )
        case .h3c:
            return try await generateWithH3(
                request: request,
                outputPath: outputPath,
                progressHandler: progressHandler
            )
        }
    }

    private func generateWithMlxVideo(
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
        let selectedTextEncoder = LTXTextEncoderCatalog.resolvedTextEncoder(id: request.textEncoderId)
        let textEncoderRepo = selectedTextEncoder.repo
        guard !textEncoderRepo.isEmpty else {
            throw LTXError.generationFailed(
                "Set a text encoder Hugging Face repo in Preferences → General (pick a preset or fill in Custom)."
            )
        }
        let (effectiveTilingMode, appliedTilingRecovery) = GenerationFailureRecovery.effectiveTilingMode(
            requested: params.vaeTilingMode
        )
        if appliedTilingRecovery {
            progressHandler(0.05, "VAE tiling set to Auto after a previous Metal timeout with Aggressive tiling.")
        }
        let oomRecoveryHint = "Metal ran out of memory during generation. Retry with safer settings: 512x320 resolution, 25/33/49 frames, 24 FPS, and VAE tiling set to aggressive. Close memory-heavy apps, then retry."
        let isImageToVideo = request.isImageToVideo
        let modeDescription = isImageToVideo ? "image-to-video" : "text-to-video"
        progressHandler(0.1, "Starting \(modeDescription) (\(selectedModel.displayName))...")
        if selectedModel.supportsBuiltInAudio && !request.disableAudio && params.fps != 24 {
            progressHandler(0.1, "Sync tip: speech alignment works best at 24 FPS (current: \(params.fps))")
        }
        
        let enableGemmaPromptEnhancement = UserDefaults.standard.bool(forKey: "enableGemmaPromptEnhancement")
        let saveAudioTrackSeparately = UserDefaults.standard.bool(forKey: "saveAudioTrackSeparately")
        let useLocalMlxVideoRepoPref = UserDefaults.standard.bool(forKey: "useLocalMlxVideoRepo")

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
        let escapedNegativePrompt = request.negativePrompt
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        
        // Build conditioning keyframes (primary source image + extra keyframes).
        // Each is encoded as "path|latentFrameIdx|strength"; "|" never appears in
        // macOS file paths so it is a safe delimiter for the library CLI.
        let latentFrameCount = 1 + (params.numFrames - 1) / 8
        let lastLatentIdx = max(0, latentFrameCount - 1)
        func latentIndex(forFraction f: Double) -> Int {
            let idx = Int((f * Double(lastLatentIdx)).rounded())
            return min(max(idx, 0), lastLatentIdx)
        }
        func pyEscape(_ s: String) -> String {
            s.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
        }
        var keyframeSpecs: [String] = []
        if let primary = request.sourceImagePath, !primary.isEmpty {
            let idx = request.parameters.imageFramePosition == "last" ? lastLatentIdx : 0
            keyframeSpecs.append("\(pyEscape(primary))|\(idx)|\(params.imageStrength)")
        }
        for kf in params.keyframes where !kf.imagePath.isEmpty {
            let idx = latentIndex(forFraction: kf.position)
            keyframeSpecs.append("\(pyEscape(kf.imagePath))|\(idx)|\(kf.strength)")
        }
        let keyframesPyList = "[" + keyframeSpecs.map { "\"\($0)\"" }.joined(separator: ", ") + "]"

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
import signal

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
    text_encoder_repo = "\(textEncoderRepo)"
    log(f"Model: {model_repo}")
    log(f"Text encoder: {text_encoder_repo}")
    local_mlx_video_repo = os.path.expanduser("~/projects/mlx-video-with-audio")
    local_has_mlx = os.path.exists(os.path.join(local_mlx_video_repo, "mlx_video", "generate_av.py"))

    def _mlx_version_subprocess(extra_env):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        for k, v in extra_env.items():
            env[k] = v
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mlx_video.version; print(mlx_video.version.__version__)",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        if r.returncode != 0:
            return None
        return (r.stdout or "").strip() or None

    def _parse_version_tuple(s):
        if not s:
            return None
        parts = []
        for seg in s.split("."):
            digits = "".join(c for c in seg if c.isdigit())
            try:
                parts.append(int(digits) if digits else 0)
            except Exception:
                parts.append(0)
        return tuple(parts)

    pip_ver = _mlx_version_subprocess({})
    local_ver = _mlx_version_subprocess({"PYTHONPATH": local_mlx_video_repo}) if local_has_mlx else None
    use_local_pref = \(useLocalMlxVideoRepoPref ? "True" : "False")
    force_local = os.environ.get("LTX_FORCE_LOCAL_MLX_VIDEO") == "1" or use_local_pref

    if not local_has_mlx:
        use_local_mlx_video_repo = False
    elif force_local:
        use_local_mlx_video_repo = True
    elif pip_ver and local_ver:
        use_local_mlx_video_repo = _parse_version_tuple(local_ver) > _parse_version_tuple(pip_ver)
    else:
        use_local_mlx_video_repo = bool(local_ver) and not pip_ver

    log(
        "mlx-video-with-audio versions: pip=%r local_repo=%r -> use_local_repo=%s"
        % (pip_ver, local_ver, use_local_mlx_video_repo)
    )
    if local_has_mlx and not use_local_mlx_video_repo and pip_ver and local_ver:
        if _parse_version_tuple(pip_ver) >= _parse_version_tuple(local_ver):
            log(
                "Using pip/site-packages (newer or same as ~/projects/mlx-video-with-audio). "
                "Preferences: enable 'Use local mlx-video-with-audio repo' or set LTX_FORCE_LOCAL_MLX_VIDEO=1 to override."
            )

    # Conditioning keyframes (primary source image + any extra keyframes)
    keyframe_specs = \(keyframesPyList)
    mode = "image-to-video" if keyframe_specs else "text-to-video"
    
    prompt = '''\(escapedPrompt)'''
    negative_prompt = '''\(escapedNegativePrompt)'''
    log(f"Prompt: {prompt[:100]}...")
    log(f"Size: \(genWidth)x\(genHeight), \(params.numFrames) frames")
    log(f"Seed: \(seed)")
    
    disable_audio = \(request.disableAudio ? "True" : "False")
    
    cmd = [
        sys.executable, "-m", "mlx_video.generate_av",
        "--prompt", prompt,
        "--height", str(\(genHeight)),
        "--width", str(\(genWidth)),
        "--num-frames", str(\(params.numFrames)),
        "--seed", str(\(seed)),
        "--fps", str(\(params.fps)),
        "--steps", str(\(params.numInferenceSteps)),
        "--cfg-scale", str(\(params.guidanceScale)),
        "--output-path", "\(outputPath)",
        "--model-repo", model_repo,
        "--text-encoder-repo", text_encoder_repo,
        "--tiling", "\(effectiveTilingMode)",
    ]
    if negative_prompt.strip():
        cmd.extend(["--negative-prompt", negative_prompt])
    if disable_audio:
        cmd.append("--no-audio")
        log(f"Mode: {mode} (audio disabled; video-only mux)")
    else:
        log(f"Mode: {mode} (with audio)")
    
    # Add conditioning keyframes if provided (path|latent_frame_idx|strength).
    # latent_frame_idx is already computed app-side from each keyframe's timeline
    # position and the frame count.
    for spec in keyframe_specs:
        cmd.extend(["--keyframe", spec])
        log(f"Keyframe: {spec}")
    
    if (not disable_audio) and \(saveAudioTrackSeparately ? "True" : "False"):
        cmd.append("--save-audio-separately")
        log("Saving audio track separately")
    
    log("Starting generation...")
    log(f"Command: {' '.join(cmd)}")
    child_env = os.environ.copy()
    # Drop inherited PYTHONPATH so venv site-packages wins unless we explicitly use a local checkout.
    child_env.pop("PYTHONPATH", None)
    if use_local_mlx_video_repo:
        child_env["PYTHONPATH"] = local_mlx_video_repo
    
    # Run the CLI module and stream combined output (binary read so we see tqdm \\r updates).
    # start_new_session=True puts the child (and anything it spawns) in its own
    # process group so we can terminate the whole tree atomically on cancel.
    process = subprocess.Popen(
        cmd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # When the app cancels generation it sends this wrapper SIGTERM. Forward that
    # to the entire child process group (SIGKILL) so the heavy mlx_video process
    # actually dies instead of being orphaned and holding the GPU/memory.
    def _terminate_child(signum, frame):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        os._exit(1)

    signal.signal(signal.SIGTERM, _terminate_child)
    signal.signal(signal.SIGINT, _terminate_child)

    # Unbuffered read: use os.read() on the raw fd so every line the inner process
    # flushes is available immediately.  process.stdout.read(n) uses Python's
    # BufferedReader which blocks until n bytes accumulate, starving the progress loop.
    line_buf = ""
    interactivity_watchdog = False
    download_in_progress = False
    last_download_activity = None  # None = not in download phase yet; set when first download line seen
    # Large models can go quiet between tqdm updates; heartbeats + long window avoid false kills.
    download_stall_timeout = 7200  # 2 hours without *any* subprocess output while downloading
    stdout_fd = process.stdout.fileno()
    while True:
        if process.stdout is None:
            break
        ready, _, _ = select.select([process.stdout], [], [], 1.0)
        if ready:
            try:
                raw = os.read(stdout_fd, 8192)
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
                if ("impacting interactivity" in low) or ("kiogpucommandbuffercallbackerrorimpactinginteractivity" in low):
                    interactivity_watchdog = True
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
                print(line, file=sys.stderr, flush=True)
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
        if process.returncode < 0:
            signal_num = -process.returncode
            signal_name = signal.Signals(signal_num).name if signal_num in signal.Signals._value2member_map_ else f"signal {signal_num}"
            if signal_num == signal.SIGKILL:
                log("DIAGNOSTIC_SIGKILL: mlx_video.generate_av killed by SIGKILL (exit code -9); likely macOS memory pressure during text encoder / model eval.")
                raise RuntimeError(
                    "mlx_video.generate_av was killed by SIGKILL (exit code -9). "
                    "macOS often sends SIGKILL under unified memory pressure (jetsam). "
                    "Try a smaller text encoder in Preferences, aggressive VAE tiling, lower resolution or frames, or close other apps. "
                    "Full output is in /tmp/ltx_generation.log."
                )
            if signal_num == signal.SIGABRT:
                if interactivity_watchdog:
                    log("DIAGNOSTIC_METAL_INTERACTIVITY: SIGABRT after Metal Impacting Interactivity watchdog.")
                    raise RuntimeError(
                        "DIAGNOSTIC_METAL_INTERACTIVITY: mlx_video.generate_av aborted with SIGABRT (code -6) after "
                        "[METAL] Impacting Interactivity / kIOGPUCommandBufferCallbackErrorImpactingInteractivity. "
                        "Try VAE tiling auto or conservative instead of aggressive; reduce resolution or frames."
                    )
                raise RuntimeError(
                    "mlx_video.generate_av aborted with SIGABRT (code -6). "
                    "This is usually a native MLX/Metal abort, often from peak unified-memory pressure "
                    "or the macOS Metal watchdog terminating a long-running command buffer. "
                    "Update mlx-video-with-audio, then retry with lower-memory settings if needed. "
                    "Full subprocess output is in /tmp/ltx_generation.log."
                )
            raise RuntimeError(
                f"mlx_video.generate_av was terminated by {signal_name} (code {process.returncode}). "
                "Full subprocess output is in /tmp/ltx_generation.log."
            )
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
        let stderrLineBufferLock = NSLock()
        var stderrLineBuffer = ""
        
        let output: String
        do {
            output = try await runPython(
                script: script,
                timeout: 3600,
                generationDiagnostics: (modelRepo: modelRepo, textEncoderRepo: textEncoderRepo),
                originalVaeTilingMode: request.parameters.vaeTilingMode
            ) { stderrChunk in
            // Build complete logical lines from chunked stderr reads so STAGE/STATUS tokens
            // are never dropped when a token is split across read boundaries.
            stderrLineBufferLock.lock()
            stderrLineBuffer += stderrChunk.replacingOccurrences(of: "\r", with: "\n")
            let completeLines = stderrLineBuffer.components(separatedBy: "\n")
            stderrLineBuffer = completeLines.last ?? ""
            let linesToParse = Array(completeLines.dropLast())
            stderrLineBufferLock.unlock()

            guard !linesToParse.isEmpty else { return }

            // Capture enhanced prompt from stderr
            // Our generate.py emits "ENHANCED_PROMPT:..." and mlx_video may emit "Enhanced prompt: ..."
            for line in linesToParse {
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
                // Parse complete lines only; chunk boundary handling is done above.
                for raw in linesToParse {
                    let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                    if line.isEmpty { continue }
                    let cleanLine = line.replacingOccurrences(
                        of: #"\u{001B}\[[0-9;]*[A-Za-z]"#,
                        with: "",
                        options: .regularExpression
                    )
                    let lower = cleanLine.lowercased()
                    
                    if cleanLine.hasPrefix("DOWNLOAD:STALL:") {
                        let seconds = String(cleanLine.dropFirst("DOWNLOAD:STALL:".count))
                        progressHandler(0.01, "Download stalled for \(seconds)s. Stopping generation.")
                        failureHintLock.lock()
                        capturedFailureHint = "No download data received for \(seconds)s—connection may have stalled. Check your network; run `hf auth login` in Terminal if using gated models; then retry. To download the model manually, use: hf download \(modelRepo) (saves to ~/.cache/huggingface)."
                        failureHintLock.unlock()
                    } else if cleanLine.hasPrefix("TEXT_ENCODER_CONFIG_ERROR:") {
                        let detail = String(cleanLine.dropFirst("TEXT_ENCODER_CONFIG_ERROR:".count)).trimmingCharacters(in: .whitespacesAndNewlines)
                        failureHintLock.lock()
                        capturedFailureHint = "Text encoder configuration mismatch detected. \(detail) Update with: pip install -U \"mlx-video-with-audio>=0.1.34\" and retry. If it persists, clear the Hugging Face cache: huggingface-cli delete-cache"
                        failureHintLock.unlock()
                    } else if lower.contains("keyerror: 'text_config'") || lower.contains("missing expected keys") || lower.contains("missing `text_config`") {
                        failureHintLock.lock()
                        capturedFailureHint = "Text encoder config mismatch (`text_config` missing). Update with: pip install -U \"mlx-video-with-audio>=0.1.34\" and retry. If it persists, clear the Hugging Face cache: huggingface-cli delete-cache"
                        failureHintLock.unlock()
                    } else if lower.contains("mlx-video-with-audio not installed") || lower.contains("cannot import name 'generate_av'") {
                        failureHintLock.lock()
                        capturedFailureHint = "The mlx-video-with-audio package is installed but incomplete or outdated. Update with: pip install -U mlx-video-with-audio and retry."
                        failureHintLock.unlock()
                    } else if lower.contains("valueerror: [conv] expect the input channels") {
                        failureHintLock.lock()
                        capturedFailureHint = "Detected MLX VAE channel mismatch during decoding. Update with: pip install -U \"mlx-video-with-audio>=0.1.36\". If it persists, the selected Hugging Face model snapshot may be incomplete or stale. Try Preferences → General → Model and select LTX-2.3 Distilled Q4, clear the cached model under ~/.cache/huggingface/hub, or pre-download a fresh copy with: hf download \(modelRepo). Full log: /tmp/ltx_generation.log"
                        failureHintLock.unlock()
                    } else if lower.contains("diagnostic_sigkill")
                                || (lower.contains("sigkill") && lower.contains("-9"))
                                || lower.contains("code -9")
                                || lower.contains("signal 9") {
                        failureHintLock.lock()
                        capturedFailureHint = Self.jetsamSigKillUserHint(
                            modelRepo: modelRepo,
                            textEncoderRepo: textEncoderRepo
                        )
                        failureHintLock.unlock()
                    } else if lower.contains("diagnostic_metal_interactivity")
                                || lower.contains("kiogpucommandbuffercallbackerrorimpactinginteractivity")
                                || lower.contains("impacting interactivity") {
                        failureHintLock.lock()
                        capturedFailureHint = Self.metalInteractivityUserHint
                        failureHintLock.unlock()
                        progressHandler(0.01, "Generation stopped: Metal watchdog timeout")
                        if request.parameters.vaeTilingMode == "aggressive" {
                            GenerationFailureRecovery.recordMetalInteractivityFailureWithAggressiveTiling()
                        }
                    } else if lower.contains("sigabrt") || lower.contains("failed with code -6") || lower.contains("aborted with code -6") {
                        failureHintLock.lock()
                        let interactivityLine = lower.contains("impacting interactivity")
                            || lower.contains("kiogpucommandbuffercallbackerrorimpactinginteractivity")
                            || (lower.contains("libc++abi") && lower.contains("impacting interactivity"))
                        if interactivityLine {
                            capturedFailureHint = Self.metalInteractivityUserHint
                            if request.parameters.vaeTilingMode == "aggressive" {
                                GenerationFailureRecovery.recordMetalInteractivityFailureWithAggressiveTiling()
                            }
                        } else {
                            capturedFailureHint = "The MLX generation process aborted with SIGABRT (code -6). Update with: pip install -U \"mlx-video-with-audio>=0.1.36\" and retry. If it still fails, try 512x320 resolution, 25/33/49 frames, 24 FPS, and tuning VAE tiling, then attach /tmp/ltx_generation.log to the GitHub issue."
                        }
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
                    
                    if cleanLine.hasPrefix("STAGE:") {
                        // Parse stage-aware progress: STAGE:1:STEP:3:8:Denoising
                        let parts = cleanLine.components(separatedBy: ":")
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
                    } else if cleanLine.hasPrefix("STATUS:") {
                        let message = String(cleanLine.dropFirst(7))
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
                    } else if let stageMatch = lower.firstMatch(of: #/stage\s+([12])\s*\((\d+)\/(\d+)\)/#),
                              let stage = Int(stageMatch.1),
                              let step = Int(stageMatch.2),
                              let total = Int(stageMatch.3),
                              total > 0 {
                        let stageProgress = Double(step) / Double(total)
                        let mappedProgress = stage == 1
                            ? 0.1 + (stageProgress * 0.4)
                            : 0.5 + (stageProgress * 0.4)
                        let message = stage == 1
                            ? "Stage 1 (\(step)/\(total)): Generating at half resolution"
                            : "Stage 2 (\(step)/\(total)): Refining at full resolution"
                        progressHandler(mappedProgress, message)
                    } else if cleanLine.hasPrefix("MLX_VIDEO_VERSION:") {
                        let version = String(cleanLine.dropFirst(18))
                        print("[LTXBridge] mlx-video-with-audio v\(version)")
                    } else if cleanLine.hasPrefix("MODEL:CACHED:") {
                        let repo = String(cleanLine.dropFirst(13))
                        progressHandler(0.08, "Model cached: \(repo)")
                    } else if cleanLine.hasPrefix("DOWNLOAD:START:") {
                        let repo = String(cleanLine.dropFirst(15))
                        progressHandler(0.01, "Downloading model: \(repo)")
                    } else if cleanLine.hasPrefix("DOWNLOAD:HEARTBEAT:") {
                        let repo = String(cleanLine.dropFirst(18))
                        progressHandler(0.04, "Downloading \(repo)… (still working — large files can look idle)")
                    } else if cleanLine.hasPrefix("DOWNLOAD:PROGRESS:") {
                        let parts = cleanLine.dropFirst(18).split(separator: ":")
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
                    } else if cleanLine.hasPrefix("DOWNLOAD:COMPLETE:") {
                        progressHandler(0.08, "Model download complete")
                    } else if cleanLine.contains("Downloading") || cleanLine.contains("Fetching") {
                        // huggingface_hub tqdm output
                        let fileCountPattern = #/(\d+)%\|[^|]*\|\s*(\d+)/(\d+)/#
                        if let match = cleanLine.firstMatch(of: fileCountPattern) {
                            let currentFile = Int(match.2) ?? 0
                            let totalFiles = Int(match.3) ?? 1
                            var filePercent = Double(currentFile) / Double(max(totalFiles, 1))
                            var message = "Downloading: \(currentFile)/\(totalFiles) files"
                            if let bytesMatch = cleanLine.firstMatch(of: #/\|\s*([\d.]+)([KMG]?)B?\/([\d.]+)([KMG]?)B?/#) {
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
            let excerpt = LTXGenerationLogSummary.userFacingExcerpt()
            if let hint, !hint.isEmpty {
                LTXGenerationLogSummary.appendToLog(
                    lines: ["", "=== Swift failure summary ===", hint, "", excerpt]
                )
                var full = hint
                if !excerpt.isEmpty {
                    full += "\n\n--- Log excerpt ---\n" + excerpt
                }
                throw LTXError.generationFailed(full)
            }
            LTXGenerationLogSummary.appendToLog(
                lines: ["", "=== Swift failure (no stderr hint) ===", error.localizedDescription, "", excerpt]
            )
            let extra = excerpt.isEmpty ? "" : "\n\n--- Log excerpt ---\n" + excerpt
            throw LTXError.generationFailed(error.localizedDescription + extra)
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

    private func generateWithLtx2Mlx(
        request: GenerationRequest,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> (videoPath: String, seed: Int, enhancedPrompt: String?) {
        setupPythonPaths()
        guard pythonExecutable != nil else {
            throw LTXError.pythonNotConfigured
        }

        let params = request.parameters
        let seed = params.seed ?? Int.random(in: 0..<Int(Int32.max))
        let selectedModel = LTXModelCatalog.resolvedModel(id: request.modelId)
        let modelRepo = selectedModel.repo
        let ditRepo = selectedModel.ditRepo ?? ""
        let enableEnhance = UserDefaults.standard.bool(forKey: "enableGemmaPromptEnhancement")
        let useLocalLtx2 = UserDefaults.standard.bool(forKey: "useLocalLtx2MlxRepo")
        let forceLowRam = UserDefaults.standard.bool(forKey: "ltx2MlxLowRam")
        let physGB = Int(MacOSSystemMemory.physicalMemoryBytes / 1_073_741_824)
        let autoLowRam = selectedModel.minRecommendedRAMGB.map { physGB < $0 } ?? false
        let useLowRam = forceLowRam || autoLowRam || selectedModel.id == LTXModelCatalog.lowRAMModelID
        let gemmaRepo = selectedModel.usesExternalTextEncoder
            ? LTXTextEncoderCatalog.resolvedTextEncoder(id: request.textEncoderId).repo
            : ""

        let genWidth = (params.width / 64) * 64
        let genHeight = (params.height / 64) * 64
        let escapedPrompt = request.prompt
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        let imagePath = request.sourceImagePath ?? ""
        let extraKeyframes = params.keyframes.contains { !$0.imagePath.isEmpty }
        let adapterPath = Bundle.main.bundlePath + "/Contents/Resources/ltx25_community_adapter.py"

        progressHandler(0.1, "Starting \(request.isImageToVideo ? "image-to-video" : "text-to-video") (\(selectedModel.displayName))...")
        if extraKeyframes {
            progressHandler(0.11, "Extra timeline keyframes are not forwarded on the ltx-2-mlx backend yet; using the primary image only.")
        }

        let tilingArgs: String
        switch params.vaeTilingMode {
        case "aggressive", "spatial":
            tilingArgs = "cmd.extend(['--tile-spatial', '2'])"
        case "temporal":
            tilingArgs = "cmd.extend(['--tile-frames', '2'])"
        default:
            tilingArgs = "pass"
        }

        let script = """
import os
import sys
import json
import shutil
import subprocess
import time
import select
import signal

log_file = open("/tmp/ltx_generation.log", "w")
def log(msg):
    print(msg, file=log_file, flush=True)
    print(msg, file=sys.stderr, flush=True)

try:
    log("=== LTX-2.5 ltx-2-mlx Generation Started ===")
    log(f"Python: {sys.executable}")
    prompt = '''\(escapedPrompt)'''
    output_path = "\(outputPath)"
    model_repo = "\(modelRepo)"
    dit_repo = "\(ditRepo)"
    gemma_repo = "\(gemmaRepo)"
    image_path = "\(imagePath)"
    local_repo = os.path.expanduser("~/projects/ltx-2-mlx")
    use_local = \(useLocalLtx2 ? "True" : "False") and os.path.isdir(local_repo)

    def resolve_prefix():
        if use_local:
            uv = shutil.which("uv")
            if uv:
                log(f"Using local ltx-2-mlx via uv: {local_repo}")
                return [uv, "run", "--directory", local_repo, "ltx-2-mlx"]
            raise RuntimeError(
                "Preferences: Use local ltx-2-mlx repo is on, but uv was not found. "
                "Install uv or turn the toggle off and pip-install ltx-2-mlx."
            )
        venv_bin = os.path.dirname(sys.executable)
        candidate = os.path.join(venv_bin, "ltx-2-mlx")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return [candidate]
        which = shutil.which("ltx-2-mlx")
        if which:
            return [which]
        raise RuntimeError(
            "ltx-2-mlx is not installed. In the app venv run: "
            "pip install \\"git+https://github.com/dgrauet/ltx-2-mlx.git@v0.15.2#subdirectory=packages/ltx-core-mlx\\" "
            "\\"git+https://github.com/dgrauet/ltx-2-mlx.git@v0.15.2#subdirectory=packages/ltx-pipelines-mlx\\" "
            "or clone https://github.com/dgrauet/ltx-2-mlx to ~/projects/ltx-2-mlx and enable "
            "'Use local ltx-2-mlx repo' in Preferences."
        )

    # Resume incomplete HF blobs; no-op when the snapshot is already complete.
    log(f"Ensuring snapshot {model_repo} (resumes incomplete files)...")
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(repo_id=model_repo)
    log(f"Model snapshot ready: {model_path}")

    # mlx-community pack is mlx-lm Gemma 4 (gemma4-12b-ltx-v1/), not
    # dgrauet's root text_encoder.safetensors. Adapter loads that folder
    # and skips DurationHead fused-in_proj mismatch (we pass --frames).
    adapter = "\(adapterPath)"
    if not os.path.isfile(adapter):
        raise RuntimeError(
            "Missing ltx25_community_adapter.py in the app bundle. Rebuild the app."
        )
    cmd = [sys.executable, adapter, "generate"] + [
        "--prompt", prompt,
        "--model", model_path,
        "--distilled",
        "-H", str(\(genHeight)),
        "-W", str(\(genWidth)),
        "-f", str(\(params.numFrames)),
        "--frame-rate", str(\(params.fps)),
        "-s", str(\(seed)),
        "-o", output_path,
    ]
    if dit_repo:
        # v0.15.2 generate has no --dit; keep the pack default transformer.
        log(f"Skipping DiT override {dit_repo}: this ltx-2-mlx generate does not accept --dit")
    if gemma_repo:
        cmd.extend(["--gemma", gemma_repo])
        log(f"Gemma text encoder: {gemma_repo}")
    if image_path:
        cmd.extend(["--image", image_path])
        log(f"I2V image: {image_path}")
    if \(request.disableAudio ? "True" : "False"):
        log("Generate Audio off is ignored on ltx-2-mlx 0.15.2 (no --no-audio flag)")
    if \(enableEnhance ? "True" : "False"):
        cmd.append("--enhance-prompt")
    if \(useLowRam ? "True" : "False"):
        cmd.append("--low-ram")
        log("Using --low-ram block streaming")
    \(tilingArgs)

    log(f"Command: {' '.join(cmd)}")
    child_env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def _terminate_child(signum, frame):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        os._exit(1)

    signal.signal(signal.SIGTERM, _terminate_child)
    signal.signal(signal.SIGINT, _terminate_child)

    line_buf = ""
    stdout_fd = process.stdout.fileno()
    while True:
        if process.stdout is None:
            break
        ready, _, _ = select.select([process.stdout], [], [], 1.0)
        if ready:
            try:
                raw = os.read(stdout_fd, 8192)
            except (ValueError, OSError):
                raw = b""
            if not raw:
                if process.poll() is not None:
                    break
                continue
            chunk = raw.decode("utf-8", errors="replace")
            line_buf += chunk
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
                print(line, file=sys.stderr, flush=True)
        elif process.poll() is not None:
            break

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ltx-2-mlx generate failed with code {process.returncode}")
    if not os.path.isfile(output_path):
        raise RuntimeError(f"ltx-2-mlx finished but output is missing: {output_path}")
    log("Generation complete!")
    log_file.close()
    print(json.dumps({"video_path": output_path, "seed": \(seed), "mode": "ltx-2.5", "has_audio": \(request.disableAudio ? "False" : "True")}))
except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    log(traceback.format_exc())
    log_file.close()
    sys.exit(1)
"""

        progressHandler(0.05, "Running ltx-2-mlx generation...")
        let output = try await runPython(
            script: script,
            timeout: 14400,
            generationDiagnostics: (modelRepo: modelRepo, textEncoderRepo: "gemma4-bundled")
        ) { stderrChunk in
            let lines = stderrChunk.replacingOccurrences(of: "\r", with: "\n")
                .components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            DispatchQueue.main.async {
                for line in lines {
                    let lower = line.lowercased()
                    if lower.contains("download") || lower.contains("fetching") {
                        progressHandler(0.04, line)
                    } else if lower.contains("denois") || lower.contains("step") {
                        progressHandler(0.4, line)
                    } else if lower.contains("decod") {
                        progressHandler(0.9, line)
                    } else if lower.contains("saving") || lower.contains("wrote") {
                        progressHandler(0.95, line)
                    }
                }
            }
        }

        if let jsonStart = output.range(of: "{\"video_path\""),
           let jsonEnd = output.range(of: "}", range: jsonStart.lowerBound..<output.endIndex) {
            let jsonString = String(output[jsonStart.lowerBound...jsonEnd.lowerBound])
            if let data = jsonString.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let videoPath = json["video_path"] as? String,
               let resultSeed = json["seed"] as? Int {
                progressHandler(1.0, "Complete!")
                return (videoPath, resultSeed, nil)
            }
        }
        throw LTXError.generationFailed("Failed to parse ltx-2-mlx output: \(output)")
    }

    private func generateWithH3(
        request: GenerationRequest,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> (videoPath: String, seed: Int, enhancedPrompt: String?) {
        if H3Engine.physicalMemoryGB() < H3Engine.minimumRAMGB {
            throw LTXError.generationFailed(
                "MiniMax H3 needs more than \(H3Engine.minimumRAMGB)GB RAM (VAEs alone are ~11GB). This Mac reports about \(H3Engine.physicalMemoryGB())GB."
            )
        }
        guard H3Engine.licenseAccepted else {
            throw LTXError.generationFailed(H3Engine.licenseNotice())
        }
        let variant = H3Variant.from(modelId: request.modelId)
        let binary: String
        if let existing = H3Engine.resolvedBinary(for: variant) {
            binary = existing
        } else {
            progressHandler(0.01, "Cloning and building \(variant.usesInt8Binary ? "h3.c-int8" : "h3.c")…")
            do {
                binary = try await H3Engine.ensureBinary(for: variant) { msg in
                    progressHandler(0.01, msg)
                }
            } catch {
                throw LTXError.generationFailed(error.localizedDescription)
            }
        }
        let modelDir: String
        if let existing = H3Engine.resolvedModelDirectory(for: variant) {
            modelDir = existing
        } else {
            switch variant {
            case .bf16:
                progressHandler(0.02, "Downloading MiniMax-H3 (~144GB) into the model cache…")
                modelDir = try await downloadH3Snapshot(progressHandler: progressHandler)
            case .int8:
                progressHandler(0.02, "Assembling H3 int8 tree (Comfy DiT + MiniMaxAI TE/VAE)…")
                modelDir = try await assembleH3Int8Tree(progressHandler: progressHandler)
            case .turbo:
                progressHandler(0.02, "Preparing H3 Turbo (official BF16 + folded LoRA)…")
                modelDir = try await assembleH3TurboTree(progressHandler: progressHandler)
            }
        }

        let params = request.parameters
        let seed = params.seed ?? Int.random(in: 0..<Int(Int32.max))
        let frames = H3Engine.snapFrameCount(params.numFrames)
        let (width, height) = H3Engine.clampCanvas(width: params.width, height: params.height)
        let preset = H3SpeedPreset.schedule(for: variant, inferenceSteps: params.numInferenceSteps)
        var args = [
            "-d", modelDir,
            "-p", request.prompt,
            "--width", String(width),
            "--height", String(height),
            "--frames", String(frames),
            "--steps", String(preset.steps),
            "--layers", String(preset.layers),
            "--reuse", String(preset.reuse),
            "--seed", String(seed),
            "-o", outputPath,
        ]
        if H3Engine.shouldUseSSDStreaming() {
            args.append("--ssd-streaming")
        }
        if let image = request.sourceImagePath, !image.isEmpty {
            args.append(contentsOf: ["--first-frame", image])
        }

        progressHandler(0.1, "Starting \(variant.displayName) (\(preset.displayName), \(width)×\(height), \(frames) frames)...")
        try await runExternalProcess(
            executable: binary,
            arguments: args,
            timeout: 7200
        ) { line in
            let lower = line.lowercased()
            if lower.contains("download") {
                progressHandler(0.05, line)
            } else if lower.contains("denois") || lower.contains("step") {
                progressHandler(0.4, line)
            } else if lower.contains("decode") || lower.contains("vae") {
                progressHandler(0.85, line)
            } else if lower.contains("ffmpeg") || lower.contains("encod") {
                progressHandler(0.95, line)
            } else {
                progressHandler(0.2, line)
            }
        }

        guard FileManager.default.fileExists(atPath: outputPath) else {
            throw LTXError.generationFailed("h3 finished but output is missing: \(outputPath). Full log: /tmp/ltx_generation.log")
        }
        progressHandler(1.0, "Complete!")
        return (outputPath, seed, nil)
    }

    private func downloadH3Snapshot(
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> String {
        setupPythonPaths()
        guard pythonExecutable != nil else {
            throw LTXError.generationFailed(H3Engine.downloadRequiresPythonHint())
        }

        let repo = H3Engine.huggingfaceRepo
        let script = """
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
from huggingface_hub import snapshot_download

repo = "\(repo)"
print(f"DOWNLOAD:START:{repo}", file=sys.stderr, flush=True)
path = None
last_err = None
for attempt in range(1, 8):
    try:
        path = snapshot_download(repo_id=repo, max_workers=4)
        last_err = None
        break
    except Exception as e:
        last_err = e
        print(f"DOWNLOAD:RETRY:{attempt}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(min(60, 2 ** attempt))
if last_err is not None:
    print(f"ERROR: {last_err}", file=sys.stderr, flush=True)
    raise last_err
print(f"DOWNLOAD:COMPLETE:{repo}", file=sys.stderr, flush=True)
print(json.dumps({"snapshot_path": path}))
"""

        let output = try await runPython(
            script: script,
            timeout: 14400,
            generationDiagnostics: (modelRepo: repo, textEncoderRepo: "bundled")
        ) { stderrChunk in
            let lines = stderrChunk.replacingOccurrences(of: "\r", with: "\n")
                .components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            DispatchQueue.main.async {
                for line in lines {
                    let lower = line.lowercased()
                    if line.hasPrefix("DOWNLOAD:START:") {
                        progressHandler(0.02, "Downloading MiniMax-H3 (~144GB)…")
                    } else if line.hasPrefix("DOWNLOAD:RETRY:") {
                        progressHandler(0.03, "MiniMax-H3 download stalled; resuming…")
                    } else if line.hasPrefix("DOWNLOAD:COMPLETE:") {
                        progressHandler(0.08, "MiniMax-H3 download complete")
                    } else if lower.contains("gated") || lower.contains("401") || lower.contains("unauthorized") {
                        progressHandler(0.02, "Hugging Face auth required for MiniMax-H3")
                    } else if lower.contains("fetching") || lower.contains("download") || (line.contains("%") && line.contains("|")) {
                        progressHandler(0.04, line)
                    }
                }
            }
        }

        if let jsonStart = output.range(of: "{\"snapshot_path\""),
           let jsonEnd = output.range(of: "}", range: jsonStart.lowerBound..<output.endIndex) {
            let jsonString = String(output[jsonStart.lowerBound...jsonEnd.lowerBound])
            if let data = jsonString.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let path = json["snapshot_path"] as? String,
               H3Engine.isOfficialSnapshot(path) {
                return path
            }
        }
        if let existing = H3Engine.resolvedOfficialSnapshot() {
            return existing
        }
        throw LTXError.generationFailed(H3Engine.missingModelHint())
    }

    private func assembleH3Int8Tree(
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> String {
        setupPythonPaths()
        guard pythonExecutable != nil else {
            throw LTXError.generationFailed(H3Engine.downloadRequiresPythonHint())
        }

        let dest = H3Engine.int8ModelDirectory
        let official = H3Engine.resolvedOfficialAuxiliaries() ?? ""
        let comfyRepo = H3Engine.comfyOrgRepo
        let comfyFile = H3Engine.comfyInt8DitFile
        let officialRepo = H3Engine.huggingfaceRepo
        let script = """
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
from huggingface_hub import hf_hub_download, snapshot_download

dest = Path(r"\(dest)")
fl2va = dest / "FL2VA"
transformer = fl2va / "transformer"
transformer.mkdir(parents=True, exist_ok=True)

print("DOWNLOAD:START:\(comfyRepo)", file=sys.stderr, flush=True)
dit = None
last_err = None
for attempt in range(1, 8):
    try:
        dit = hf_hub_download(repo_id="\(comfyRepo)", filename="\(comfyFile)")
        last_err = None
        break
    except Exception as e:
        last_err = e
        print(f"DOWNLOAD:RETRY:{attempt}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(min(60, 2 ** attempt))
if last_err is not None:
    raise last_err
print("DOWNLOAD:COMPLETE:\(comfyRepo)", file=sys.stderr, flush=True)

dit_dst = transformer / Path(dit).name
if dit_dst.exists() or dit_dst.is_symlink():
    dit_dst.unlink()
os.symlink(dit, dit_dst)

official = r"\(official)"
if not official or not Path(official, "FL2VA", "text_encoder").exists():
    print("DOWNLOAD:START:\(officialRepo)", file=sys.stderr, flush=True)
    last_err = None
    for attempt in range(1, 8):
        try:
            official = snapshot_download(
                repo_id="\(officialRepo)",
                allow_patterns=[
                    "FL2VA/text_encoder/**",
                    "FL2VA/video_vae/**",
                    "FL2VA/audio_vae/**",
                    "FL2VA/tokenizer/**",
                    "FL2VA/processor/**",
                    "FL2VA/transformer/config.json",
                    "FL2VA/model_index.json",
                ],
                max_workers=4,
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"DOWNLOAD:RETRY:{attempt}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            time.sleep(min(60, 2 ** attempt))
    if last_err is not None:
        raise last_err
    print("DOWNLOAD:COMPLETE:\(officialRepo)", file=sys.stderr, flush=True)

official_fl2 = Path(official) / "FL2VA"
for name in ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor"):
    src = official_fl2 / name
    dst = fl2va / name
    if not src.exists():
        continue
    if dst.exists() or dst.is_symlink():
        continue
    os.symlink(src, dst)

cfg_src = official_fl2 / "transformer" / "config.json"
cfg_dst = transformer / "config.json"
if cfg_src.exists() and not (cfg_dst.exists() or cfg_dst.is_symlink()):
    os.symlink(cfg_src, cfg_dst)

print(json.dumps({"snapshot_path": str(dest)}))
"""

        let output = try await runPython(
            script: script,
            timeout: 14400,
            generationDiagnostics: (modelRepo: comfyRepo, textEncoderRepo: "bundled")
        ) { stderrChunk in
            let lines = stderrChunk.replacingOccurrences(of: "\r", with: "\n")
                .components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            DispatchQueue.main.async {
                for line in lines {
                    let lower = line.lowercased()
                    if line.hasPrefix("DOWNLOAD:START:Comfy-Org") {
                        progressHandler(0.03, "Downloading Comfy-Org int8 DiT (~20GB)…")
                    } else if line.hasPrefix("DOWNLOAD:START:") {
                        progressHandler(0.04, "Downloading MiniMaxAI text encoder/VAEs…")
                    } else if line.hasPrefix("DOWNLOAD:RETRY:") {
                        progressHandler(0.04, "H3 int8 download stalled; resuming…")
                    } else if line.hasPrefix("DOWNLOAD:COMPLETE:") {
                        progressHandler(0.08, "H3 int8 component download complete")
                    } else if lower.contains("gated") || lower.contains("401") || lower.contains("unauthorized") {
                        progressHandler(0.02, "Hugging Face auth required")
                    } else if lower.contains("fetching") || lower.contains("download") || (line.contains("%") && line.contains("|")) {
                        progressHandler(0.05, line)
                    }
                }
            }
        }

        if H3Engine.isInt8TreeReady(dest) {
            return dest
        }
        if let jsonStart = output.range(of: "{\"snapshot_path\""),
           let jsonEnd = output.range(of: "}", range: jsonStart.lowerBound..<output.endIndex) {
            let jsonString = String(output[jsonStart.lowerBound...jsonEnd.lowerBound])
            if let data = jsonString.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let path = json["snapshot_path"] as? String,
               H3Engine.isInt8TreeReady(path) {
                return path
            }
        }
        throw LTXError.generationFailed(
            "H3 int8 tree is incomplete at \(dest). Need Comfy int8 DiT plus MiniMaxAI text encoder/VAEs. Full log: /tmp/ltx_generation.log"
        )
    }

    private func assembleH3TurboTree(
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> String {
        setupPythonPaths()
        guard let python = pythonExecutable else {
            throw LTXError.generationFailed(H3Engine.downloadRequiresPythonHint())
        }

        let official: String
        if let existing = H3Engine.resolvedOfficialSnapshot() {
            official = existing
        } else {
            progressHandler(0.02, "Downloading MiniMax-H3 BF16 (~144GB) for Turbo fold…")
            official = try await downloadH3Snapshot(progressHandler: progressHandler)
        }
        guard let transformer = H3Engine.officialTransformerDirectory(),
              H3Engine.hasOfficialTransformer(official) else {
            throw LTXError.generationFailed(
                "H3 Turbo needs the official BF16 transformer shards. Download MiniMax H3 BF16 first. Full log: /tmp/ltx_generation.log"
            )
        }

        let dest = H3Engine.turboModelDirectory
        if H3Engine.isTurboTreeReady(dest) {
            return dest
        }

        progressHandler(0.06, "Downloading Turbo LoRA (larryvrh v4)…")
        let loraRepo = H3Engine.turboLoraRepo
        let loraFile = H3Engine.turboLoraFile
        let loraScript = """
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
from huggingface_hub import hf_hub_download

print("DOWNLOAD:START:\(loraRepo)", file=sys.stderr, flush=True)
path = None
last_err = None
for attempt in range(1, 8):
    try:
        path = hf_hub_download(repo_id="\(loraRepo)", filename="\(loraFile)")
        last_err = None
        break
    except Exception as e:
        last_err = e
        print(f"DOWNLOAD:RETRY:{attempt}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(min(60, 2 ** attempt))
if last_err is not None:
    raise last_err
print("DOWNLOAD:COMPLETE:\(loraRepo)", file=sys.stderr, flush=True)
print(json.dumps({"lora_path": path}))
"""
        let loraOutput = try await runPython(
            script: loraScript,
            timeout: 3600,
            generationDiagnostics: (modelRepo: loraRepo, textEncoderRepo: "bundled")
        ) { stderrChunk in
            let lines = stderrChunk.replacingOccurrences(of: "\r", with: "\n")
                .components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            DispatchQueue.main.async {
                for line in lines {
                    if line.hasPrefix("DOWNLOAD:START:") {
                        progressHandler(0.06, "Downloading Turbo LoRA…")
                    } else if line.hasPrefix("DOWNLOAD:RETRY:") {
                        progressHandler(0.06, "Turbo LoRA download stalled; resuming…")
                    } else if line.contains("%") && line.contains("|") {
                        progressHandler(0.07, line)
                    }
                }
            }
        }
        guard let jsonStart = loraOutput.range(of: "{\"lora_path\""),
              let jsonEnd = loraOutput.range(of: "}", range: jsonStart.lowerBound..<loraOutput.endIndex) else {
            throw LTXError.generationFailed("Turbo LoRA download did not return a path. Full log: /tmp/ltx_generation.log")
        }
        let jsonString = String(loraOutput[jsonStart.lowerBound...jsonEnd.lowerBound])
        guard let data = jsonString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let loraPath = json["lora_path"] as? String else {
            throw LTXError.generationFailed("Could not parse Turbo LoRA path. Full log: /tmp/ltx_generation.log")
        }

        let resourcesPath = Bundle.main.bundlePath + "/Contents/Resources"
        let foldScript = resourcesPath + "/fold_turbo_lora.py"
        guard FileManager.default.fileExists(atPath: foldScript) else {
            throw LTXError.generationFailed("Missing fold_turbo_lora.py in the app bundle. Rebuild the app.")
        }

        let destURL = URL(fileURLWithPath: dest)
        let destFL2 = destURL.appendingPathComponent("FL2VA", isDirectory: true)
        let destTransformer = destFL2.appendingPathComponent("transformer", isDirectory: true)
        try FileManager.default.createDirectory(at: destFL2, withIntermediateDirectories: true)

        progressHandler(0.08, "Folding Turbo LoRA into BF16 transformer (APFS CoW)…")
        try await runExternalProcess(
            executable: python,
            arguments: [
                foldScript,
                "--checkpoint", transformer,
                "--lora", loraPath,
                "--out", destTransformer.path,
            ],
            timeout: 7200
        ) { line in
            progressHandler(0.08, line)
        }

        let officialFL2 = URL(fileURLWithPath: official).appendingPathComponent("FL2VA", isDirectory: true)
        for name in ["text_encoder", "video_vae", "audio_vae", "tokenizer", "processor"] {
            let src = officialFL2.appendingPathComponent(name)
            let dst = destFL2.appendingPathComponent(name)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: src.path, isDirectory: &isDir) else { continue }
            if FileManager.default.fileExists(atPath: dst.path) { continue }
            try FileManager.default.createSymbolicLink(at: dst, withDestinationURL: src)
        }
        try "folded".write(
            to: destTransformer.appendingPathComponent(".turbo-folded"),
            atomically: true,
            encoding: .utf8
        )

        guard H3Engine.isTurboTreeReady(dest) else {
            throw LTXError.generationFailed(
                "H3 Turbo fold finished but the tree is incomplete at \(dest). Full log: /tmp/ltx_generation.log"
            )
        }
        return dest
    }

    private func runExternalProcess(
        executable: String,
        arguments: [String],
        timeout: TimeInterval,
        stdoutHandler: ((String) -> Void)?
    ) async throws {
        if let cacheError = HuggingFaceCacheConfiguration.availabilityError() {
            throw LTXError.generationFailed(cacheError)
        }

        let logFile = "/tmp/ltx_generation.log"
        let holder = CancellableProcessHolder()

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    let process = Process()
                    process.executableURL = URL(fileURLWithPath: executable)
                    process.arguments = arguments
                    let execDir = URL(fileURLWithPath: executable).deletingLastPathComponent()
                    process.currentDirectoryURL = execDir
                    var env: [String: String] = [:]
                    env["PATH"] = "\(execDir.path):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
                    env["HOME"] = ProcessInfo.processInfo.environment["HOME"] ?? ""
                    env["USER"] = ProcessInfo.processInfo.environment["USER"] ?? ""
                    env["TMPDIR"] = ProcessInfo.processInfo.environment["TMPDIR"] ?? "/tmp"
                    HuggingFaceCacheConfiguration.apply(to: &env)
                    process.environment = env

                    let stdoutPipe = Pipe()
                    process.standardOutput = stdoutPipe
                    process.standardError = stdoutPipe

                    stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
                        let data = handle.availableData
                        guard !data.isEmpty, let str = String(data: data, encoding: .utf8) else { return }
                        if let logData = str.data(using: .utf8),
                           let fh = FileHandle(forWritingAtPath: logFile) {
                            fh.seekToEndOfFile()
                            fh.write(logData)
                            fh.closeFile()
                        } else {
                            try? str.write(toFile: logFile, atomically: false, encoding: .utf8)
                        }
                        for line in str.replacingOccurrences(of: "\r", with: "\n")
                            .components(separatedBy: "\n")
                            .map({ $0.trimmingCharacters(in: .whitespacesAndNewlines) })
                            .filter({ !$0.isEmpty }) {
                            DispatchQueue.main.async { stdoutHandler?(line) }
                        }
                    }

                    guard holder.storeIfNotCancelled(process) else {
                        continuation.resume(throwing: CancellationError())
                        return
                    }

                    do {
                        let header = "=== h3.c Process Started ===\nBinary: \(executable)\nArgs: \(arguments.joined(separator: " "))\nTime: \(Date())\n"
                        try? header.write(toFile: logFile, atomically: false, encoding: .utf8)
                        try process.run()
                        process.waitUntilExit()
                        stdoutPipe.fileHandleForReading.readabilityHandler = nil
                        if holder.wasCancelled {
                            continuation.resume(throwing: CancellationError())
                            return
                        }
                        if process.terminationStatus != 0 {
                            continuation.resume(throwing: LTXError.generationFailed(
                                "h3 exited with code \(process.terminationStatus). Full log: /tmp/ltx_generation.log"
                            ))
                            return
                        }
                        continuation.resume()
                    } catch {
                        continuation.resume(throwing: LTXError.generationFailed(error.localizedDescription))
                    }
                }
            }
        } onCancel: {
            holder.cancel()
        }
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
        if let cacheError = HuggingFaceCacheConfiguration.availabilityError() {
            throw LTXError.generationFailed(cacheError)
        }

        return try await withCheckedThrowingContinuation { continuation in
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
                HuggingFaceCacheConfiguration.apply(to: &env)
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
        generationDiagnostics: (modelRepo: String, textEncoderRepo: String)? = nil,
        originalVaeTilingMode: String? = nil,
        stderrHandler: ((String) -> Void)? = nil
    ) async throws -> String {
        guard let python = pythonExecutable else {
            throw LTXError.pythonNotConfigured
        }

        if let cacheError = HuggingFaceCacheConfiguration.availabilityError() {
            throw LTXError.generationFailed(cacheError)
        }
        
        let logFile = "/tmp/ltx_generation.log"

        let holder = CancellableProcessHolder()

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
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
                HuggingFaceCacheConfiguration.apply(to: &env)
                
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
                
                // If the task was cancelled before we could start, abort now.
                guard holder.storeIfNotCancelled(process) else {
                    continuation.resume(throwing: CancellationError())
                    return
                }

                do {
                    let startLog = "=== LTX MLX Process Started ===\nPython: \(python)\nTime: \(Date())\n"
                    try? startLog.write(toFile: logFile, atomically: false, encoding: .utf8)

                    try process.run()
                    process.waitUntilExit()

                    // User-initiated cancel: surface as cancellation, not a failure.
                    if holder.wasCancelled {
                        stderrPipe.fileHandleForReading.readabilityHandler = nil
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    
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
                            let excerpt = LTXGenerationLogSummary.userFacingExcerpt()
                            let diagnosed = Self.diagnoseRunnerFailure(
                                exitCode: process.terminationStatus,
                                stderr: stderr,
                                modelRepo: generationDiagnostics?.modelRepo,
                                textEncoderRepo: generationDiagnostics?.textEncoderRepo
                            )
                            if Self.stderrIndicatesMetalInteractivity(stderr),
                               originalVaeTilingMode == "aggressive" {
                                GenerationFailureRecovery.recordMetalInteractivityFailureWithAggressiveTiling()
                            }
                            var message = diagnosed
                                ?? "Exit code \(process.terminationStatus). Check /tmp/ltx_generation.log"
                            if !stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                               diagnosed == nil || stderr.count < 12_000 {
                                message += "\n\n--- Recent stderr ---\n"
                                    + String(stderr.suffix(8000))
                            }
                            if !excerpt.isEmpty {
                                message += "\n\n--- Log excerpt ---\n" + excerpt
                            }
                            LTXGenerationLogSummary.appendToLog(
                                lines: ["", "=== Swift runner summary ===", message]
                            )
                            continuation.resume(throwing: LTXError.generationFailed(message))
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
        } onCancel: {
            holder.cancel()
        }
    }
}
