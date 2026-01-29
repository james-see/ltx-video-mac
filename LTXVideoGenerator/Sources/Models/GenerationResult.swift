import Foundation

struct GenerationResult: Identifiable, Codable {
    let id: UUID
    let requestId: UUID
    let prompt: String
    let negativePrompt: String
    let parameters: GenerationParameters
    let videoPath: String
    let thumbnailPath: String?
    let audioPath: String?
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
    
    var hasAudio: Bool {
        audioPath != nil
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
}

extension GenerationResult {
    static func preview() -> GenerationResult {
        GenerationResult(
            id: UUID(),
            requestId: UUID(),
            prompt: "A cinematic shot of a majestic eagle soaring through mountains",
            negativePrompt: "",
            parameters: .default,
            videoPath: "/tmp/preview.mp4",
            thumbnailPath: nil,
            audioPath: nil,
            createdAt: Date().addingTimeInterval(-120),
            completedAt: Date(),
            duration: 45.5,
            seed: 42
        )
    }
}
