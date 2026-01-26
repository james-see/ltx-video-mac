import Foundation

struct GenerationRequest: Identifiable, Codable, Equatable {
    let id: UUID
    let prompt: String
    let negativePrompt: String
    var parameters: GenerationParameters
    let createdAt: Date
    var status: GenerationStatus
    
    init(
        id: UUID = UUID(),
        prompt: String,
        negativePrompt: String = "",
        parameters: GenerationParameters = .default,
        createdAt: Date = Date(),
        status: GenerationStatus = .pending
    ) {
        self.id = id
        self.prompt = prompt
        self.negativePrompt = negativePrompt
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
    
    // Default for LTX-2 on Apple Silicon
    static let `default` = GenerationParameters(
        numInferenceSteps: 40,
        guidanceScale: 4.0,
        width: 768,
        height: 512,
        numFrames: 121,
        fps: 24,
        seed: nil
    )
    
    // Quick preview - fewer frames and steps
    static let preview = GenerationParameters(
        numInferenceSteps: 20,
        guidanceScale: 4.0,
        width: 512,
        height: 320,
        numFrames: 49,
        fps: 24,
        seed: nil
    )
    
    // High quality - more steps
    static let highQuality = GenerationParameters(
        numInferenceSteps: 50,
        guidanceScale: 4.0,
        width: 768,
        height: 512,
        numFrames: 121,
        fps: 24,
        seed: nil
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
