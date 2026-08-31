import Foundation

struct LTXModel: Identifiable, Codable, Hashable {
    let id: String
    let repo: String
    let displayName: String
    let downloadSize: String
    let supportsBuiltInAudio: Bool
    let qualityWarning: String?
    let recommendedStepsLower: Int?
    let recommendedStepsUpper: Int?
    let tips: String?

    var recommendedSteps: ClosedRange<Int>? {
        guard let lo = recommendedStepsLower, let hi = recommendedStepsUpper else { return nil }
        return lo...hi
    }
}

struct LTXTextEncoder: Identifiable, Codable, Hashable {
    let id: String
    let repo: String
    let displayName: String
    let downloadSize: String
    let qualityWarning: String?
    let tips: String?
}

enum LTXModelCatalog {
    static let selectedModelIDKey = "selectedModelID"
    // Prefer the smaller LTX-2.3 Q4 model for new installs; existing user preferences still win.
    static let defaultModelID = "ltx23_distilled_q4"

    static let all: [LTXModel] = [
        LTXModel(
            id: "ltx2_unified",
            repo: "notapalindrome/ltx2-mlx-av",
            displayName: "LTX-2 Unified",
            downloadSize: "~42GB",
            supportsBuiltInAudio: true,
            qualityWarning: nil,
            recommendedStepsLower: 20,
            recommendedStepsUpper: 40,
            tips: nil
        ),
        LTXModel(
            id: "ltx23_unified",
            repo: "notapalindrome/ltx23-mlx-av",
            displayName: "LTX-2.3 Unified (Beta)",
            downloadSize: "~48GB",
            supportsBuiltInAudio: true,
            qualityWarning: nil,
            recommendedStepsLower: 15,
            recommendedStepsUpper: 30,
            tips: "Distilled model — 15-30 steps is optimal. More steps won't improve quality."
        ),
        LTXModel(
            id: "ltx23_distilled_q4",
            repo: "notapalindrome/ltx23-mlx-av-q4",
            displayName: "LTX-2.3 Distilled Q4 (Beta)",
            downloadSize: "~22GB",
            supportsBuiltInAudio: true,
            qualityWarning: "Quantized: lower memory footprint with some quality tradeoffs versus bf16.",
            recommendedStepsLower: 15,
            recommendedStepsUpper: 30,
            tips: "Distilled model — 15-30 steps is optimal. Uses ~30GB RAM vs ~50GB for bf16."
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

enum LTXTextEncoderCatalog {
    static let selectedTextEncoderIDKey = "selectedTextEncoderID"
    /// When the "Custom" preset is selected, this repo id is passed as `--text-encoder-repo`.
    static let customTextEncoderRepoKey = "customTextEncoderRepo"
    // Match upstream mlx-video-with-audio default unless the user explicitly opts into Q4.
    static let defaultTextEncoderID = "gemma3_12b_bf16"

    static let all: [LTXTextEncoder] = [
        LTXTextEncoder(
            id: "gemma3_12b_bf16",
            repo: "mlx-community/gemma-3-12b-it-bf16",
            displayName: "Gemma 12B bf16 (max quality, ~24 GB RAM)",
            downloadSize: "~24GB",
            qualityWarning: nil,
            tips: "Quality-first upstream default. Uses substantially more memory during text encoding."
        ),
        LTXTextEncoder(
            id: "gemma3_4b_bf16",
            repo: "mlx-community/gemma-3-4b-it-bf16",
            displayName: "Gemma 4B bf16 (balanced, ~10 GB RAM)",
            downloadSize: "~10GB",
            qualityWarning: nil,
            tips: "Lower memory than 12B bf16; good balance for unified LTX models on 32 GB Macs."
        ),
        LTXTextEncoder(
            id: "gemma3_12b_4bit",
            repo: "mlx-community/gemma-3-12b-it-4bit",
            displayName: "Gemma 12B 4-bit (lowest RAM among presets, ~7 GB)",
            downloadSize: "~7GB",
            qualityWarning: "Quantized: much lower memory use with possible prompt-conditioning quality tradeoffs.",
            tips: "Recommended for 16 GB Macs or when the 12B bf16 text encoder is killed by the system (SIGKILL)."
        ),
        LTXTextEncoder(
            id: "custom",
            repo: "",
            displayName: "Custom Hugging Face repo…",
            downloadSize: "varies",
            qualityWarning: nil,
            tips: "Enter any compatible Gemma MLX repo id below. Nothing is downloaded until you run a generation."
        ),
    ]

    static var defaultTextEncoder: LTXTextEncoder {
        all.first { $0.id == defaultTextEncoderID } ?? all[0]
    }

    static func textEncoder(id: String) -> LTXTextEncoder? {
        all.first { $0.id == id }
    }

    static func textEncoder(repo: String) -> LTXTextEncoder? {
        all.first { $0.repo == repo }
    }

    static func resolvedTextEncoder(id: String?) -> LTXTextEncoder {
        guard let id else { return defaultTextEncoder }
        if id == "custom" {
            let raw = UserDefaults.standard.string(forKey: customTextEncoderRepoKey)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return LTXTextEncoder(
                id: "custom",
                repo: raw,
                displayName: raw.isEmpty ? "Custom (not set)" : "Custom (\(raw))",
                downloadSize: "varies",
                qualityWarning: raw.isEmpty
                    ? "Set a repository id in Preferences → General (Custom text encoder field)."
                    : nil,
                tips: nil
            )
        }
        guard let textEncoder = textEncoder(id: id) else { return defaultTextEncoder }
        return textEncoder
    }

    static func selectedTextEncoder(userDefaults: UserDefaults = .standard) -> LTXTextEncoder {
        let id = userDefaults.string(forKey: selectedTextEncoderIDKey) ?? defaultTextEncoderID
        return resolvedTextEncoder(id: id)
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
    let textEncoderId: String          // Selected Gemma text encoder ID from catalog
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
        textEncoderId: String = LTXTextEncoderCatalog.defaultTextEncoderID,
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
        self.textEncoderId = textEncoderId
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
        case textEncoderId
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
        textEncoderId = try container.decodeIfPresent(String.self, forKey: .textEncoderId) ?? LTXTextEncoderCatalog.defaultTextEncoderID
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

/// A single conditioning image pinned to a position along the video timeline.
/// `position` is a fraction 0.0...1.0 (0 = first frame, 1 = last frame); the
/// generation bridge maps it to a latent frame index based on `numFrames`.
struct Keyframe: Codable, Equatable, Hashable, Identifiable {
    var id: UUID = UUID()
    var imagePath: String
    var position: Double = 1.0
    var strength: Double = 1.0
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
    /// Which frame the source image anchors: "first" (default) or "last".
    /// Only meaningful for image-to-video generation.
    var imageFramePosition: String = "first"
    /// Additional conditioning keyframes beyond the primary source image.
    /// Each pins an image to a position along the timeline.
    var keyframes: [Keyframe] = []

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

extension GenerationParameters {
    // Custom decoder so older saved presets/history/profiles (which predate
    // `imageFramePosition`) still decode. Declared in an extension to preserve
    // the synthesized memberwise initializer used throughout the app.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numInferenceSteps = try c.decode(Int.self, forKey: .numInferenceSteps)
        guidanceScale = try c.decode(Double.self, forKey: .guidanceScale)
        width = try c.decode(Int.self, forKey: .width)
        height = try c.decode(Int.self, forKey: .height)
        numFrames = try c.decode(Int.self, forKey: .numFrames)
        fps = try c.decode(Int.self, forKey: .fps)
        seed = try c.decodeIfPresent(Int.self, forKey: .seed)
        vaeTilingMode = try c.decode(String.self, forKey: .vaeTilingMode)
        imageStrength = try c.decode(Double.self, forKey: .imageStrength)
        imageFramePosition = try c.decodeIfPresent(String.self, forKey: .imageFramePosition) ?? "first"
        keyframes = try c.decodeIfPresent([Keyframe].self, forKey: .keyframes) ?? []
    }
}
