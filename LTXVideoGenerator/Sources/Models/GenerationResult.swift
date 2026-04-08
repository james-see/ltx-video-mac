import Foundation

struct GenerationResult: Identifiable, Codable {
    let id: UUID
    let requestId: UUID
    let prompt: String
    let enhancedPrompt: String?  // AI-enhanced prompt (if Gemma enhancement was used)
    let negativePrompt: String
    let voiceoverText: String  // Voiceover narration text (for audio generation)
    let voiceoverSource: String  // "elevenlabs" or "mlx-audio"
    let voiceoverVoice: String   // Voice ID for TTS
    let modelId: String
    let parameters: GenerationParameters
    let videoPath: String
    let thumbnailPath: String?
    let audioPath: String?       // Path to voiceover audio
    let musicPath: String?       // Path to background music
    let musicGenre: String?      // Music genre used
    let sourceImagePath: String?  // Source image used for I2V
    let createdAt: Date
    let completedAt: Date
    let duration: TimeInterval
    let seed: Int
    
    var videoURL: URL {
        URL(fileURLWithPath: videoPath)
    }
    
    var thumbnailURL: URL? {
        thumbnailPath.map { URL(fileURLWithPath: $0) }
    }
    
    var audioURL: URL? {
        audioPath.map { URL(fileURLWithPath: $0) }
    }
    
    var musicURL: URL? {
        musicPath.map { URL(fileURLWithPath: $0) }
    }
    
    var hasAudio: Bool {
        audioPath != nil
    }
    
    var hasMusic: Bool {
        musicPath != nil
    }
    
    var formattedDuration: String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        if minutes > 0 {
            return "\(minutes)m \(seconds)s"
        }
        return "\(seconds)s"
    }
    
    var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: completedAt)
    }

    var model: LTXModel {
        LTXModelCatalog.resolvedModel(id: modelId)
    }

    init(
        id: UUID,
        requestId: UUID,
        prompt: String,
        enhancedPrompt: String?,
        negativePrompt: String,
        voiceoverText: String,
        voiceoverSource: String,
        voiceoverVoice: String,
        modelId: String,
        parameters: GenerationParameters,
        videoPath: String,
        thumbnailPath: String?,
        audioPath: String?,
        musicPath: String?,
        musicGenre: String?,
        sourceImagePath: String? = nil,
        createdAt: Date,
        completedAt: Date,
        duration: TimeInterval,
        seed: Int
    ) {
        self.id = id
        self.requestId = requestId
        self.prompt = prompt
        self.enhancedPrompt = enhancedPrompt
        self.negativePrompt = negativePrompt
        self.voiceoverText = voiceoverText
        self.voiceoverSource = voiceoverSource
        self.voiceoverVoice = voiceoverVoice
        self.modelId = modelId
        self.parameters = parameters
        self.videoPath = videoPath
        self.thumbnailPath = thumbnailPath
        self.audioPath = audioPath
        self.musicPath = musicPath
        self.musicGenre = musicGenre
        self.sourceImagePath = sourceImagePath
        self.createdAt = createdAt
        self.completedAt = completedAt
        self.duration = duration
        self.seed = seed
    }

    enum CodingKeys: String, CodingKey {
        case id
        case requestId
        case prompt
        case enhancedPrompt
        case negativePrompt
        case voiceoverText
        case voiceoverSource
        case voiceoverVoice
        case modelId
        case parameters
        case videoPath
        case thumbnailPath
        case audioPath
        case musicPath
        case musicGenre
        case sourceImagePath
        case createdAt
        case completedAt
        case duration
        case seed
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        requestId = try container.decode(UUID.self, forKey: .requestId)
        prompt = try container.decode(String.self, forKey: .prompt)
        enhancedPrompt = try container.decodeIfPresent(String.self, forKey: .enhancedPrompt)
        negativePrompt = try container.decode(String.self, forKey: .negativePrompt)
        voiceoverText = try container.decode(String.self, forKey: .voiceoverText)
        voiceoverSource = try container.decode(String.self, forKey: .voiceoverSource)
        voiceoverVoice = try container.decode(String.self, forKey: .voiceoverVoice)
        modelId = try container.decodeIfPresent(String.self, forKey: .modelId) ?? LTXModelCatalog.defaultModelID
        parameters = try container.decode(GenerationParameters.self, forKey: .parameters)
        videoPath = try container.decode(String.self, forKey: .videoPath)
        thumbnailPath = try container.decodeIfPresent(String.self, forKey: .thumbnailPath)
        audioPath = try container.decodeIfPresent(String.self, forKey: .audioPath)
        musicPath = try container.decodeIfPresent(String.self, forKey: .musicPath)
        musicGenre = try container.decodeIfPresent(String.self, forKey: .musicGenre)
        sourceImagePath = try container.decodeIfPresent(String.self, forKey: .sourceImagePath)
        createdAt = try container.decode(Date.self, forKey: .createdAt)
        completedAt = try container.decode(Date.self, forKey: .completedAt)
        duration = try container.decode(TimeInterval.self, forKey: .duration)
        seed = try container.decode(Int.self, forKey: .seed)
    }
}

extension GenerationResult {
    static func preview() -> GenerationResult {
        GenerationResult(
            id: UUID(),
            requestId: UUID(),
            prompt: "A cinematic shot of a majestic eagle soaring through mountains",
            enhancedPrompt: "A breathtaking cinematic aerial shot captures a majestic bald eagle soaring gracefully through snow-capped mountain peaks at golden hour",
            negativePrompt: "",
            voiceoverText: "",
            voiceoverSource: "mlx-audio",
            voiceoverVoice: "af_heart",
            modelId: LTXModelCatalog.defaultModelID,
            parameters: .default,
            videoPath: "/tmp/preview.mp4",
            thumbnailPath: nil,
            audioPath: nil,
            musicPath: nil,
            musicGenre: nil,
            sourceImagePath: nil,
            createdAt: Date().addingTimeInterval(-120),
            completedAt: Date(),
            duration: 45.5,
            seed: 42
        )
    }
}
