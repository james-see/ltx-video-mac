import Foundation

struct GenerationRequest: Identifiable, Codable, Equatable {
    let id: UUID
    let prompt: String
    let negativePrompt: String
    let voiceoverText: String  // Optional voiceover narration text
    let voiceoverSource: String  // "elevenlabs" or "mlx-audio"
    let voiceoverVoice: String   // Voice ID for TTS
    let sourceImagePath: String?  // For image-to-video mode
    let musicEnabled: Bool       // Whether to generate background music
    let musicGenre: String?      // Music genre raw value
    let disableAudio: Bool       // Skip audio in unified AV model
    let gemmaRepetitionPenalty: Double  // Gemma prompt enhancement repetition penalty
    let gemmaTopP: Double              // Gemma prompt enhancement top-p sampling
    var parameters: GenerationParameters
    let createdAt: Date
    var status: GenerationStatus
    
    /// True if this is an image-to-video request
    var isImageToVideo: Bool {
        sourceImagePath != nil && !sourceImagePath!.isEmpty
    }
    
    /// True if voiceover text is provided
    var hasVoiceover: Bool {
        !voiceoverText.isEmpty
    }
    
    /// True if music generation is enabled
    var hasMusic: Bool {
        musicEnabled && musicGenre != nil
    }
    
    init(
        id: UUID = UUID(),
        prompt: String,
        negativePrompt: String = "",
        voiceoverText: String = "",
        voiceoverSource: String = "mlx-audio",
        voiceoverVoice: String = "af_heart",
        sourceImagePath: String? = nil,
        musicEnabled: Bool = false,
        musicGenre: String? = nil,
        disableAudio: Bool = false,
        gemmaRepetitionPenalty: Double = 1.2,
        gemmaTopP: Double = 0.9,
        parameters: GenerationParameters = .default,
        createdAt: Date = Date(),
        status: GenerationStatus = .pending
    ) {
        self.id = id
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.voiceoverText = voiceoverText
        self.voiceoverSource = voiceoverSource
        self.voiceoverVoice = voiceoverVoice
        self.sourceImagePath = sourceImagePath
        self.musicEnabled = musicEnabled
        self.musicGenre = musicGenre
        self.disableAudio = disableAudio
        self.gemmaRepetitionPenalty = gemmaRepetitionPenalty
        self.gemmaTopP = gemmaTopP
        self.parameters = parameters
        self.createdAt = createdAt
        self.status = status
    }
}

enum GenerationStatus: String, Codable, Equatable {
    case pending
    case processing
    case completed
    case failed
    case cancelled
}

struct GenerationParameters: Codable, Equatable, Hashable {
    var numInferenceSteps: Int
    var guidanceScale: Double
    var width: Int
    var height: Int
    var numFrames: Int
    var fps: Int
    var seed: Int?
    var vaeTilingMode: String
    var imageStrength: Double
    
    // Default for LTX-2 on Apple Silicon
    static let `default` = GenerationParameters(
        numInferenceSteps: 30,
        guidanceScale: 3.0,
        width: 768,
        height: 512,
        numFrames: 121,
        fps: 24,
        seed: nil,
        vaeTilingMode: "auto",
        imageStrength: 1.0
    )
    
    // Quick preview - fewer frames and steps
    static let preview = GenerationParameters(
        numInferenceSteps: 15,
        guidanceScale: 3.0,
        width: 512,
        height: 320,
        numFrames: 49,
        fps: 24,
        seed: nil,
        vaeTilingMode: "auto",
        imageStrength: 1.0
    )
    
    // High quality - more steps
    static let highQuality = GenerationParameters(
        numInferenceSteps: 40,
        guidanceScale: 3.0,
        width: 768,
        height: 512,
        numFrames: 121,
        fps: 24,
        seed: nil,
        vaeTilingMode: "auto",
        imageStrength: 1.0
    )
    
    var estimatedDuration: String {
        // Rough estimate based on parameters
        let baseTime = Double(numInferenceSteps) * 0.5
        let sizeMultiplier = Double(width * height) / (768.0 * 512.0)
        let frameMultiplier = Double(numFrames) / 97.0
        let totalSeconds = baseTime * sizeMultiplier * frameMultiplier
        
        if totalSeconds < 60 {
            return "\(Int(totalSeconds))s"
        } else {
            let minutes = Int(totalSeconds) / 60
            let seconds = Int(totalSeconds) % 60
            return "\(minutes)m \(seconds)s"
        }
    }
    
    var videoLength: String {
        let totalSeconds = Double(numFrames) / Double(fps)
        if totalSeconds < 60 {
            return String(format: "%.1fs", totalSeconds)
        } else {
            let minutes = Int(totalSeconds) / 60
            let seconds = totalSeconds.truncatingRemainder(dividingBy: 60)
            return String(format: "%dm %.1fs", minutes, seconds)
        }
    }
    
    var estimatedVRAM: Int {
        // LTX-2 is a 19B model - requires significant unified memory
        // Base ~20GB for model, plus ~200MB per frame at 768x512
        let baseVRAM = 20.0 + Double(numFrames) * 0.2
        let resolutionScale = Double(width * height) / (768.0 * 512.0)
        return Int(baseVRAM * resolutionScale)
    }
}
