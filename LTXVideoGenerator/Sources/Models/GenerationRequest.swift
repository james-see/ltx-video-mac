import Foundation

enum GenerationBackend: String, Codable {
    case mlxVideoWithAudio
    case ltx2Mlx
    case h3c
}

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
    let backend: GenerationBackend
    let minRecommendedRAMGB: Int?
    let ditRepo: String?
    let usesBundledGemma4: Bool
    /// Official LTX-2.5 production path: `--two-stage` + `transformer-dev`.
    let usesDevTwoStage: Bool

    var recommendedSteps: ClosedRange<Int>? {
        guard let lo = recommendedStepsLower, let hi = recommendedStepsUpper else { return nil }
        return lo...hi
    }

    var usesExternalTextEncoder: Bool {
        switch backend {
        case .mlxVideoWithAudio: return true
        case .ltx2Mlx: return !usesBundledGemma4
        case .h3c: return false
        }
    }

    var bundledTextEncoderId: String? {
        switch backend {
        case .mlxVideoWithAudio: return nil
        case .ltx2Mlx: return usesBundledGemma4 ? "gemma4_12b_pack" : nil
        case .h3c: return "h3_qwen3_vl"
        }
    }

    func resolvedTextEncoderId(_ selectedId: String) -> String {
        bundledTextEncoderId ?? selectedId
    }

    init(
        id: String,
        repo: String,
        displayName: String,
        downloadSize: String,
        supportsBuiltInAudio: Bool,
        qualityWarning: String?,
        recommendedStepsLower: Int?,
        recommendedStepsUpper: Int?,
        tips: String?,
        backend: GenerationBackend = .mlxVideoWithAudio,
        minRecommendedRAMGB: Int? = nil,
        ditRepo: String? = nil,
        usesBundledGemma4: Bool = false,
        usesDevTwoStage: Bool = false
    ) {
        self.id = id
        self.repo = repo
        self.displayName = displayName
        self.downloadSize = downloadSize
        self.supportsBuiltInAudio = supportsBuiltInAudio
        self.qualityWarning = qualityWarning
        self.recommendedStepsLower = recommendedStepsLower
        self.recommendedStepsUpper = recommendedStepsUpper
        self.tips = tips
        self.backend = backend
        self.minRecommendedRAMGB = minRecommendedRAMGB
        self.ditRepo = ditRepo
        self.usesBundledGemma4 = usesBundledGemma4
        self.usesDevTwoStage = usesDevTwoStage
    }

    enum CodingKeys: String, CodingKey {
        case id, repo, displayName, downloadSize, supportsBuiltInAudio
        case qualityWarning, recommendedStepsLower, recommendedStepsUpper, tips
        case backend, minRecommendedRAMGB, ditRepo, usesBundledGemma4, usesDevTwoStage
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        repo = try c.decode(String.self, forKey: .repo)
        displayName = try c.decode(String.self, forKey: .displayName)
        downloadSize = try c.decode(String.self, forKey: .downloadSize)
        supportsBuiltInAudio = try c.decode(Bool.self, forKey: .supportsBuiltInAudio)
        qualityWarning = try c.decodeIfPresent(String.self, forKey: .qualityWarning)
        recommendedStepsLower = try c.decodeIfPresent(Int.self, forKey: .recommendedStepsLower)
        recommendedStepsUpper = try c.decodeIfPresent(Int.self, forKey: .recommendedStepsUpper)
        tips = try c.decodeIfPresent(String.self, forKey: .tips)
        backend = try c.decodeIfPresent(GenerationBackend.self, forKey: .backend) ?? .mlxVideoWithAudio
        minRecommendedRAMGB = try c.decodeIfPresent(Int.self, forKey: .minRecommendedRAMGB)
        ditRepo = try c.decodeIfPresent(String.self, forKey: .ditRepo)
        usesBundledGemma4 = try c.decodeIfPresent(Bool.self, forKey: .usesBundledGemma4) ?? false
        usesDevTwoStage = try c.decodeIfPresent(Bool.self, forKey: .usesDevTwoStage) ?? false
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
    static let lowRAMModelID = "ltx23_12gb"
    static let lowRAMThresholdGB = 16

    static var physicalMemoryGB: Int {
        Int(MacOSSystemMemory.physicalMemoryBytes / 1_073_741_824)
    }

    static var isLowRAMMac: Bool {
        physicalMemoryGB <= lowRAMThresholdGB
    }

    /// New installs only (`@AppStorage` / UserDefaults when the key is unset).
    static var defaultModelID: String {
        isLowRAMMac ? lowRAMModelID : "ltx23_distilled_q4"
    }

    static func lowRAMRecommendationBanner(selectedModelID: String) -> String? {
        guard isLowRAMMac, selectedModelID != lowRAMModelID else { return nil }
        return "Your Mac has ~\(physicalMemoryGB)GB RAM. The baa-ai 12GB-optimized model is recommended for best results on your system."
    }

    static let all: [LTXModel] = [
        LTXModel(
            id: "ltx2_unified",
            repo: "notapalindrome/ltx2-mlx-av",
            displayName: "LTX-2 Unified",
            downloadSize: "~42GB",
            supportsBuiltInAudio: true,
            qualityWarning: nil,
            recommendedStepsLower: 11,
            recommendedStepsUpper: 11,
            tips: "Fixed 8 stage-1 + 3 stage-2 steps (11 total). The inference-steps slider is ignored."
        ),
        LTXModel(
            id: "ltx23_unified",
            repo: "notapalindrome/ltx23-mlx-av",
            displayName: "LTX-2.3 Unified (Beta)",
            downloadSize: "~48GB",
            supportsBuiltInAudio: true,
            qualityWarning: nil,
            recommendedStepsLower: 11,
            recommendedStepsUpper: 11,
            tips: "Distilled schedule is fixed at 8 stage-1 + 3 stage-2 steps (11 total). The inference-steps slider is ignored."
        ),
        LTXModel(
            id: "ltx23_distilled_q4",
            repo: "notapalindrome/ltx23-mlx-av-q4",
            displayName: "LTX-2.3 Distilled Q4 (Beta)",
            downloadSize: "~22GB",
            supportsBuiltInAudio: true,
            qualityWarning: "Quantized: lower memory footprint with some quality tradeoffs versus bf16.",
            recommendedStepsLower: 11,
            recommendedStepsUpper: 11,
            tips: "Distilled schedule is fixed at 8 stage-1 + 3 stage-2 steps (11 total). The inference-steps slider is ignored. Uses ~30GB RAM vs ~50GB for bf16."
        ),
        LTXModel(
            id: "ltx23_12gb",
            repo: "baa-ai/LTX-2.3-22B-RAM-12GB-MLX",
            displayName: "LTX-2.3 12GB RAM Optimized (Beta)",
            downloadSize: "~19GB",
            supportsBuiltInAudio: true,
            qualityWarning: "Optimized for low-RAM Macs (16GB). May have quality tradeoffs vs bf16 models.",
            recommendedStepsLower: 8,
            recommendedStepsUpper: 8,
            tips: "Mixed-precision distilled pack via ltx-2-mlx. Vendor claims ~14GB unified memory. Default for new installs on ≤16GB Macs.",
            backend: .ltx2Mlx,
            minRecommendedRAMGB: 16
        ),
        LTXModel(
            id: "ltx25_distilled",
            repo: "notapalindrome/ltx25-mlx",
            displayName: "LTX-2.5 Distilled (bf16)",
            downloadSize: "~100GB",
            supportsBuiltInAudio: true,
            qualityWarning: "Gemma 4 is bundled in the pack. Needs 64GB+ RAM (128GB comfortable). Distilled 8-step, CFG=1.",
            recommendedStepsLower: 8,
            recommendedStepsUpper: 8,
            tips: "LTX-2.5 distilled uses a fixed 8-step schedule. Gemma 4 encoder is inside the pack. Requires ltx-2-mlx 0.15+. Reuses a complete mlx-community/ltx-2.5-mlx cache if present.",
            backend: .ltx2Mlx,
            minRecommendedRAMGB: 64,
            usesBundledGemma4: true
        ),
        LTXModel(
            id: "ltx25_dev",
            repo: "notapalindrome/ltx25-mlx",
            displayName: "LTX-2.5 Dev (two-stage)",
            downloadSize: "~110GB (LoRA bundled)",
            supportsBuiltInAudio: true,
            qualityWarning: "Official production path: half-res dev+CFG, upscale, fuse distilled LoRA into the dev DiT. LoRA ships in notapalindrome/ltx25-mlx (no second download). Much slower. 64GB+; 128GB comfortable.",
            recommendedStepsLower: 20,
            recommendedStepsUpper: 40,
            tips: "Uses transformer-dev.safetensors + bundled distilled LoRA for Stage 2. Slider is stage-1 steps (default 30). Stage-2 is 3. CFG default 3. Requires ltx-2-mlx 0.15+. Reuses mlx-community/dgrauet caches when present.",
            backend: .ltx2Mlx,
            minRecommendedRAMGB: 64,
            usesBundledGemma4: true,
            usesDevTwoStage: true
        ),
        LTXModel(
            id: "ltx25_distilled_ditq8",
            repo: "notapalindrome/ltx25-mlx",
            displayName: "LTX-2.5 Distilled Q8 DiT",
            downloadSize: "~100GB + ~21GB DiT",
            supportsBuiltInAudio: true,
            qualityWarning: "8-bit quantized DiT on the 2.5 pack. 64GB recommended; 32GB may work with --low-ram.",
            recommendedStepsLower: 8,
            recommendedStepsUpper: 8,
            tips: "Same 2.5 pack; overlays notapalindrome/ltx25-mlx-ditq8 transformer-distilled.safetensors (0.15.2 has no --dit). Not a text-encoder quant.",
            backend: .ltx2Mlx,
            minRecommendedRAMGB: 32,
            ditRepo: "notapalindrome/ltx25-mlx-ditq8",
            usesBundledGemma4: true
        ),
        LTXModel(
            id: "minimax_h3",
            repo: "MiniMaxAI/MiniMax-H3",
            displayName: "MiniMax H3 BF16 (h3.c)",
            downloadSize: "~144GB",
            supportsBuiltInAudio: true,
            qualityWarning: "MiniMax H3 Community License. Native C/Metal via antirez/h3.c. 32GB+ with SSD streaming; 16GB is not viable.",
            recommendedStepsLower: 4,
            recommendedStepsUpper: 50,
            tips: "Official BF16 snapshot. Frames snap to 5+17n. FPS is 24.",
            backend: .h3c,
            minRecommendedRAMGB: 32
        ),
        LTXModel(
            id: "minimax_h3_int8",
            repo: "Comfy-Org/MiniMax-H3",
            displayName: "MiniMax H3 int8 (h3.c)",
            downloadSize: "~92GB",
            supportsBuiltInAudio: true,
            qualityWarning: "Comfy-Org int8 DiT + MiniMaxAI text encoder/VAEs. Slight detail loss vs BF16 (SSIM ~0.92). DiT peak ~16.8GB. ~20GB extra if BF16 is already cached.",
            recommendedStepsLower: 4,
            recommendedStepsUpper: 50,
            tips: "Uses Shedrackeze002/h3.c-int8 (GPU ConvRot de-rot). Same 5+17n / 24 fps rules. Do not point this at the official 13-shard DiT.",
            backend: .h3c,
            minRecommendedRAMGB: 24,
            ditRepo: "Comfy-Org/MiniMax-H3"
        ),
        LTXModel(
            id: "minimax_h3_turbo",
            repo: "MiniMaxAI/MiniMax-H3",
            displayName: "MiniMax H3 Turbo (h3.c)",
            downloadSize: "~144GB + ~3GB fold",
            supportsBuiltInAudio: true,
            qualityWarning: "Folded larryvrh Turbo v4 LoRA on official BF16. Fixed 6 steps. Fast motion can smear at 4 steps. Adapter is a MiniMax H3 Community License derivative.",
            recommendedStepsLower: 6,
            recommendedStepsUpper: 6,
            tips: "Bakes W'=W+B@A offline (no LoRA runtime). Forced 6/50/1 — do not combine with --reuse. First generate folds ~2–3GB APFS CoW.",
            backend: .h3c,
            minRecommendedRAMGB: 32
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
    // 12B bf16 unless this is a new install on a ≤16GB Mac.
    static var defaultTextEncoderID: String {
        LTXModelCatalog.isLowRAMMac ? "gemma3_12b_4bit" : "gemma3_12b_bf16"
    }

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
        LTXTextEncoder(
            id: "gemma4_12b_pack",
            repo: "",
            displayName: "Gemma 4 12B (bundled with LTX-2.5 pack)",
            downloadSize: "included in model",
            qualityWarning: nil,
            tips: "Gemma-4-unified text encoder, bundled in the LTX-2.5 model pack. Not selectable for LTX-2/2.3."
        ),
        LTXTextEncoder(
            id: "h3_qwen3_vl",
            repo: "",
            displayName: "Qwen3-VL (bundled with MiniMax H3)",
            downloadSize: "included in model",
            qualityWarning: nil,
            tips: "H3 reads the Qwen3-VL encoder from the MiniMax-H3 snapshot. Not used for LTX models."
        ),
    ]

    static let selectableEncoderIDs: Set<String> = [
        "gemma3_12b_bf16", "gemma3_4b_bf16", "gemma3_12b_4bit", "custom",
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
/// `position` is a fraction 0.0...1.0 (0 = first frame, 1 = last frame).
/// `mlxVideoWithAudio` maps to a latent frame index; `ltx2Mlx` maps to a pixel frame index.
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
