import Foundation

enum AudioError: LocalizedError, Equatable {
    case elevenLabsKeyNotSet
    case elevenLabsApiFailed(String)
    case mlxAudioFailed(String)
    case ffmpegFailed(String)
    case pythonNotConfigured
    
    var errorDescription: String? {
        switch self {
        case .elevenLabsKeyNotSet:
            return "ElevenLabs API key not configured. Please add your API key in Preferences > Audio."
        case .elevenLabsApiFailed(let msg):
            return "ElevenLabs API error: \(msg)"
        case .mlxAudioFailed(let msg):
            return "MLX Audio generation failed: \(msg)"
        case .ffmpegFailed(let msg):
            return "FFmpeg merge failed: \(msg)"
        case .pythonNotConfigured:
            return "Python environment not configured."
        }
    }
}

enum AudioSource: String, CaseIterable, Identifiable {
    case elevenLabs = "elevenlabs"
    case mlxAudio = "mlx-audio"
    
    var id: String { rawValue }
    
    var displayName: String {
        switch self {
        case .elevenLabs: return "ElevenLabs (Cloud)"
        case .mlxAudio: return "MLX Audio (Local)"
        }
    }
    
    var description: String {
        switch self {
        case .elevenLabs: return "High-quality cloud TTS, requires API key"
        case .mlxAudio: return "On-device TTS, runs locally on Apple Silicon"
        }
    }
}

struct ElevenLabsVoice: Identifiable, Codable {
    let voice_id: String
    let name: String
    
    var id: String { voice_id }
    
    static let defaultVoices: [ElevenLabsVoice] = [
        ElevenLabsVoice(voice_id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel"),
        ElevenLabsVoice(voice_id: "AZnzlk1XvdvUeBnXmlld", name: "Domi"),
        ElevenLabsVoice(voice_id: "EXAVITQu4vr4xnSDxMaL", name: "Bella"),
        ElevenLabsVoice(voice_id: "ErXwobaYiN019PkySvjV", name: "Antoni"),
        ElevenLabsVoice(voice_id: "MF3mGyEYCl7XYWbV9V6O", name: "Elli"),
        ElevenLabsVoice(voice_id: "TxGEqnHWrfWFTfGW9XjX", name: "Josh"),
        ElevenLabsVoice(voice_id: "VR6AewLTigWG4xSOukaG", name: "Arnold"),
        ElevenLabsVoice(voice_id: "pNInz6obpgDQGcFmaJgB", name: "Adam"),
        ElevenLabsVoice(voice_id: "yoZ06aMxZJJ28mfd3POQ", name: "Sam")
    ]
}

struct MLXAudioVoice: Identifiable {
    let id: String
    let name: String
    
    static let defaultVoices: [MLXAudioVoice] = [
        MLXAudioVoice(id: "af_heart", name: "Heart (Female)"),
        MLXAudioVoice(id: "af_bella", name: "Bella (Female)"),
        MLXAudioVoice(id: "af_nicole", name: "Nicole (Female)"),
        MLXAudioVoice(id: "af_sarah", name: "Sarah (Female)"),
        MLXAudioVoice(id: "af_sky", name: "Sky (Female)"),
        MLXAudioVoice(id: "am_adam", name: "Adam (Male)"),
        MLXAudioVoice(id: "am_michael", name: "Michael (Male)"),
        MLXAudioVoice(id: "bf_emma", name: "Emma (British Female)"),
        MLXAudioVoice(id: "bm_george", name: "George (British Male)")
    ]
}

@MainActor
class AudioService: ObservableObject {
    static let shared = AudioService()
    
    @Published var isGenerating = false
    @Published var progress: Double = 0
    @Published var statusMessage: String = ""
    @Published var error: AudioError?
    
    private init() {}
    
    // MARK: - ElevenLabs
    
    var elevenLabsApiKey: String {
        UserDefaults.standard.string(forKey: "elevenLabsApiKey") ?? ""
    }
    
    var isElevenLabsConfigured: Bool {
        !elevenLabsApiKey.isEmpty
    }
    
    func generateWithElevenLabs(
        text: String,
        voiceId: String,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> URL {
        guard isElevenLabsConfigured else {
            throw AudioError.elevenLabsKeyNotSet
        }
        
        progressHandler(0.1, "Connecting to ElevenLabs...")
        
        let url = URL(string: "https://api.elevenlabs.io/v1/text-to-speech/\(voiceId)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(elevenLabsApiKey, forHTTPHeaderField: "xi-api-key")
        
        let body: [String: Any] = [
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": [
                "stability": 0.5,
                "similarity_boost": 0.75
            ]
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        progressHandler(0.3, "Generating audio...")
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AudioError.elevenLabsApiFailed("Invalid response")
        }
        
        if httpResponse.statusCode != 200 {
            let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw AudioError.elevenLabsApiFailed("Status \(httpResponse.statusCode): \(errorMessage)")
        }
        
        progressHandler(0.8, "Saving audio file...")
        
        let outputURL = URL(fileURLWithPath: outputPath)
        try data.write(to: outputURL)
        
        progressHandler(1.0, "Audio generated")
        
        return outputURL
    }
    
    // MARK: - MLX Audio
    
    func generateWithMLXAudio(
        text: String,
        voice: String,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> URL {
        guard let pythonPath = UserDefaults.standard.string(forKey: "pythonPath"),
              !pythonPath.isEmpty else {
            throw AudioError.pythonNotConfigured
        }
        
        progressHandler(0.1, "Starting MLX Audio generation...")
        
        let resourcesPath = Bundle.main.bundlePath + "/Contents/Resources"
        let generatorScript = resourcesPath + "/audio_generator.py"
        
        // Escape text for Python
        let escapedText = text
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        
        let script = """
        import sys
        import json
        
        # Add Resources directory to path
        sys.path.insert(0, "\(resourcesPath)")
        
        try:
            from audio_generator import generate_audio
            
            result = generate_audio(
                text="\(escapedText)",
                voice="\(voice)",
                output_path="\(outputPath)"
            )
            
            print(json.dumps({"success": True, "audio_path": "\(outputPath)"}))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}), file=sys.stderr)
            sys.exit(1)
        """
        
        let output = try await runPython(
            executable: pythonPath,
            script: script,
            timeout: 300
        ) { stderr in
            DispatchQueue.main.async {
                if stderr.hasPrefix("PROGRESS:") {
                    let parts = stderr.dropFirst(9).split(separator: ":")
                    if parts.count >= 2,
                       let pct = Double(parts[0]) {
                        let msg = String(parts[1...].joined(separator: ":"))
                        progressHandler(pct / 100.0, msg)
                    }
                } else if stderr.hasPrefix("STATUS:") {
                    let msg = String(stderr.dropFirst(7))
                    progressHandler(0.5, msg)
                }
            }
        }
        
        // Parse result
        if let data = output.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let success = json["success"] as? Bool, success {
            progressHandler(1.0, "Audio generated")
            return URL(fileURLWithPath: outputPath)
        }
        
        throw AudioError.mlxAudioFailed("Failed to parse output: \(output)")
    }
    
    // MARK: - FFmpeg Merge
    
    func mergeAudioWithVideo(
        videoURL: URL,
        audioURL: URL,
        outputURL: URL,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws {
        progressHandler(0.1, "Merging audio with video...")
        
        // Find ffmpeg
        let ffmpegPaths = [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg"
        ]
        
        guard let ffmpegPath = ffmpegPaths.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw AudioError.ffmpegFailed("FFmpeg not found. Install with: brew install ffmpeg")
        }
        
        // FFmpeg command: combine video and audio
        // -y: overwrite output
        // -i: input files
        // -c:v copy: copy video stream without re-encoding
        // -c:a aac: encode audio as AAC
        // -shortest: stop when shortest stream ends
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpegPath)
        process.arguments = [
            "-y",
            "-i", videoURL.path,
            "-i", audioURL.path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            outputURL.path
        ]
        
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        process.standardOutput = Pipe()
        
        try process.run()
        
        progressHandler(0.5, "Processing...")
        
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            let errorData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let errorOutput = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw AudioError.ffmpegFailed(errorOutput)
        }
        
        progressHandler(1.0, "Merge complete")
    }
    
    // MARK: - Full Pipeline
    
    func addAudioToVideo(
        result: GenerationResult,
        text: String,
        source: AudioSource,
        voiceId: String,
        historyManager: HistoryManager,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> GenerationResult {
        isGenerating = true
        progress = 0
        statusMessage = "Starting..."
        error = nil
        
        defer {
            isGenerating = false
        }
        
        // Generate audio file path
        let audioFileName = "\(result.id.uuidString)_audio"
        let audioExtension = source == .elevenLabs ? "mp3" : "wav"
        let audioPath = historyManager.videosDirectory
            .appendingPathComponent("\(audioFileName).\(audioExtension)").path
        
        // Generate audio
        progressHandler(0.1, "Generating audio...")
        
        let audioURL: URL
        switch source {
        case .elevenLabs:
            audioURL = try await generateWithElevenLabs(
                text: text,
                voiceId: voiceId,
                outputPath: audioPath
            ) { pct, msg in
                progressHandler(0.1 + pct * 0.4, msg)
            }
        case .mlxAudio:
            audioURL = try await generateWithMLXAudio(
                text: text,
                voice: voiceId,
                outputPath: audioPath
            ) { pct, msg in
                progressHandler(0.1 + pct * 0.4, msg)
            }
        }
        
        // Merge with video
        progressHandler(0.5, "Merging audio with video...")
        
        let outputFileName = "\(result.id.uuidString)_with_audio.mp4"
        let outputPath = historyManager.videosDirectory.appendingPathComponent(outputFileName)
        
        try await mergeAudioWithVideo(
            videoURL: result.videoURL,
            audioURL: audioURL,
            outputURL: outputPath
        ) { pct, msg in
            progressHandler(0.5 + pct * 0.4, msg)
        }
        
        // Update result with new video path (with audio)
        progressHandler(0.95, "Updating history...")
        
        let updatedResult = GenerationResult(
            id: result.id,
            requestId: result.requestId,
            prompt: result.prompt,
            negativePrompt: result.negativePrompt,
            parameters: result.parameters,
            videoPath: outputPath.path,
            thumbnailPath: result.thumbnailPath,
            audioPath: audioPath,
            createdAt: result.createdAt,
            completedAt: result.completedAt,
            duration: result.duration,
            seed: result.seed
        )
        
        // Clean up original video file (optional - keep backup)
        // try? FileManager.default.removeItem(at: result.videoURL)
        
        progressHandler(1.0, "Complete!")
        
        return updatedResult
    }
    
    // MARK: - Python Runner
    
    private func runPython(
        executable: String,
        script: String,
        timeout: TimeInterval,
        stderrHandler: ((String) -> Void)? = nil
    ) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: executable)
                process.arguments = ["-c", script]
                
                var env: [String: String] = [:]
                let pythonBin = URL(fileURLWithPath: executable).deletingLastPathComponent().path
                env["PATH"] = "\(pythonBin):/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"
                env["HOME"] = ProcessInfo.processInfo.environment["HOME"] ?? ""
                env["USER"] = ProcessInfo.processInfo.environment["USER"] ?? ""
                env["TMPDIR"] = ProcessInfo.processInfo.environment["TMPDIR"] ?? "/tmp"
                process.environment = env
                
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe
                
                stderrPipe.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                        stderrHandler?(str)
                    }
                }
                
                do {
                    try process.run()
                    process.waitUntilExit()
                    
                    stderrPipe.fileHandleForReading.readabilityHandler = nil
                    
                    let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let output = String(data: outputData, encoding: .utf8) ?? ""
                    
                    if process.terminationStatus != 0 {
                        continuation.resume(throwing: AudioError.mlxAudioFailed("Process exited with code \(process.terminationStatus)"))
                    } else {
                        continuation.resume(returning: output.trimmingCharacters(in: .whitespacesAndNewlines))
                    }
                } catch {
                    continuation.resume(throwing: AudioError.mlxAudioFailed(error.localizedDescription))
                }
            }
        }
    }
}
