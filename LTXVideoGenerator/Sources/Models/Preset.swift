import Foundation

struct Preset: Identifiable, Codable, Equatable, Hashable {
    let id: UUID
    var name: String
    var parameters: GenerationParameters
    var isBuiltIn: Bool
    let createdAt: Date
    
    init(
        id: UUID = UUID(),
        name: String,
        parameters: GenerationParameters,
        isBuiltIn: Bool = false,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.parameters = parameters
        self.isBuiltIn = isBuiltIn
        self.createdAt = createdAt
    }
    
    // All presets optimized for LTX-2 on Apple Silicon
    static let builtInPresets: [Preset] = [
        Preset(
            name: "Quick Preview",
            parameters: .preview,
            isBuiltIn: true
        ),
        Preset(
            name: "Standard",
            parameters: .default,
            isBuiltIn: true
        ),
        Preset(
            name: "High Quality",
            parameters: .highQuality,
            isBuiltIn: true
        ),
        Preset(
            name: "Portrait",
            parameters: GenerationParameters(
                numInferenceSteps: 40,
                guidanceScale: 4.0,
                width: 512,
                height: 768,
                numFrames: 121,
                fps: 24,
                seed: nil
            ),
            isBuiltIn: true
        ),
        Preset(
            name: "Square",
            parameters: GenerationParameters(
                numInferenceSteps: 40,
                guidanceScale: 4.0,
                width: 512,
                height: 512,
                numFrames: 121,
                fps: 24,
                seed: nil
            ),
            isBuiltIn: true
        ),
        Preset(
            name: "Cinematic 21:9",
            parameters: GenerationParameters(
                numInferenceSteps: 40,
                guidanceScale: 4.0,
                width: 768,
                height: 320,
                numFrames: 121,
                fps: 24,
                seed: nil
            ),
            isBuiltIn: true
        )
    ]
}
