import Foundation

struct LTXModel: Identifiable, Codable, Hashable {
    let id: String
    let repo: String
    let displayName: String
    let downloadSize: String
    let supportsBuiltInAudio: Bool
    let qualityWarning: String?
}

enum LTXModelCatalog {
    static let selectedModelIDKey = "selectedModelID"
    // Keep default on Unified unless the user explicitly changes it.
    static let defaultModelID = "ltx2_unified"

    static let all: [LTXModel] = [
        LTXModel(
            id: "ltx2_unified",
            repo: "notapalindrome/ltx2-mlx-av",
            displayName: "LTX-2 Unified",
            downloadSize: "~42GB",
            supportsBuiltInAudio: true,
            qualityWarning: nil
        ),
        LTXModel(
            id: "ltx23_unified",
            repo: "notapalindrome/ltx23-mlx-av",
            displayName: "LTX-2.3 Unified (Beta)",
            downloadSize: "~48GB",
            supportsBuiltInAudio: true,
            qualityWarning: nil
        ),
        LTXModel(
            id: "ltx23_distilled_q4",
            repo: "notapalindrome/ltx23-mlx-av-q4",
            displayName: "LTX-2.3 Distilled Q4 (Beta)",
            downloadSize: "~22GB",
            supportsBuiltInAudio: true,
            qualityWarning: "Quantized: lower memory footprint with some quality tradeoffs versus bf16."
        ),
    ]

    static var defaultModel: LTXModel {
        all.first { $0.id == defaultModelID } ?? all[0]
    }

    static func model(id: String) -> LTXModel? {
        all.first { $0.id == id }
    }

    static func model(repo: String) -> LTXModel? {
        all.first { $0.repo == repo }
    }

    static func resolvedModel(id: String?) -> LTXModel {
        guard let id, let model = model(id: id) else { return defaultModel }
        return model
    }

    static func selectedModel(userDefaults: UserDefaults = .standard) -> LTXModel {
        let id = userDefaults.string(forKey: selectedModelIDKey) ?? defaultModelID
        return resolvedModel(id: id)
    }
}

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
    let modelId: String                // Selected model ID from catalog
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
        modelId: String = LTXModelCatalog.defaultModelID,
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
        self.modelId = modelId
        self.parameters = parameters
        self.createdAt = createdAt
        self.status = status
    }

    enum CodingKeys: String, CodingKey {
        case id
        case prompt
        case negativePrompt
        case voiceoverText
        case voiceoverSource
        case voiceoverVoice
        case sourceImagePath
        case musicEnabled
        case musicGenre
        case disableAudio
        case gemmaRepetitionPenalty
        case gemmaTopP
        case modelId
        case parameters
        case createdAt
        case status
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        prompt = try container.decode(String.self, forKey: .prompt)
        negativePrompt = try container.decode(String.self, forKey: .negativePrompt)
        voiceoverText = try container.decode(String.self, forKey: .voiceoverText)
        voiceoverSource = try container.decode(String.self, forKey: .voiceoverSource)
        voiceoverVoice = try container.decode(String.self, forKey: .voiceoverVoice)
        sourceImagePath = try container.decodeIfPresent(String.self, forKey: .sourceImagePath)
        musicEnabled = try container.decode(Bool.self, forKey: .musicEnabled)
        musicGenre = try container.decodeIfPresent(String.self, forKey: .musicGenre)
        disableAudio = try container.decode(Bool.self, forKey: .disableAudio)
        gemmaRepetitionPenalty = try container.decode(Double.self, forKey: .gemmaRepetitionPenalty)
        gemmaTopP = try container.decode(Double.self, forKey: .gemmaTopP)
        modelId = try container.decodeIfPresent(String.self, forKey: .modelId) ?? LTXModelCatalog.defaultModelID
        parameters = try container.decode(GenerationParameters.self, forKey: .parameters)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        status = try container.decode(GenerationStatus.self, forKey: .status)
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
