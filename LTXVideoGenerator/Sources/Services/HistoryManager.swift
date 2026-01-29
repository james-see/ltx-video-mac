import Foundation
import SwiftUI
import AVFoundation

@MainActor
class HistoryManager: ObservableObject {
    @Published private(set) var results: [GenerationResult] = []
    @Published var selectedResult: GenerationResult?
    
    let videosDirectory: URL
    let thumbnailsDirectory: URL
    private let historyFile: URL
    
    nonisolated init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let appDir = appSupport.appendingPathComponent("LTXVideoGenerator", isDirectory: true)
        
        videosDirectory = appDir.appendingPathComponent("Videos", isDirectory: true)
        thumbnailsDirectory = appDir.appendingPathComponent("Thumbnails", isDirectory: true)
        historyFile = appDir.appendingPathComponent("history.json")
        
        // Create directories
        try? FileManager.default.createDirectory(at: videosDirectory, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: thumbnailsDirectory, withIntermediateDirectories: true)
    }
    
    func loadInitialData() {
        loadHistory()
        
        // Regenerate missing thumbnails in background
        Task {
            await regenerateMissingThumbnails()
        }
    }
    
    // MARK: - Persistence
    
    private func loadHistory() {
        guard FileManager.default.fileExists(atPath: historyFile.path) else { return }
        
        do {
            let data = try Data(contentsOf: historyFile)
            results = try JSONDecoder().decode([GenerationResult].self, from: data)
            // Sort by date, newest first
            results.sort { $0.completedAt > $1.completedAt }
        } catch {
            print("Failed to load history: \(error)")
        }
    }
    
    private func saveHistory() {
        do {
            let data = try JSONEncoder().encode(results)
            try data.write(to: historyFile)
        } catch {
            print("Failed to save history: \(error)")
        }
    }
    
    // MARK: - Management
    
    func addResult(_ result: GenerationResult) {
        results.insert(result, at: 0)
        saveHistory()
    }
    
    func deleteResult(_ result: GenerationResult) {
        // Delete video file
        try? FileManager.default.removeItem(at: result.videoURL)
        
        // Delete thumbnail
        if let thumbnailURL = result.thumbnailURL {
            try? FileManager.default.removeItem(at: thumbnailURL)
        }
        
        results.removeAll { $0.id == result.id }
        
        if selectedResult?.id == result.id {
            selectedResult = nil
        }
        
        saveHistory()
    }
    
    func deleteResults(_ resultsToDelete: Set<GenerationResult>) {
        for result in resultsToDelete {
            try? FileManager.default.removeItem(at: result.videoURL)
            if let thumbnailURL = result.thumbnailURL {
                try? FileManager.default.removeItem(at: thumbnailURL)
            }
        }
        
        results.removeAll { resultsToDelete.contains($0) }
        
        if let selected = selectedResult, resultsToDelete.contains(selected) {
            selectedResult = nil
        }
        
        saveHistory()
    }
    
    func clearHistory() {
        for result in results {
            try? FileManager.default.removeItem(at: result.videoURL)
            if let thumbnailURL = result.thumbnailURL {
                try? FileManager.default.removeItem(at: thumbnailURL)
            }
        }
        
        results.removeAll()
        selectedResult = nil
        saveHistory()
    }
    
    // MARK: - Thumbnails
    
    /// Generate thumbnail and return the path
    func generateThumbnail(for result: GenerationResult) async -> String? {
        let asset = AVAsset(url: result.videoURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 400, height: 400)
        
        // Try to get a frame from the middle of the video for better thumbnail
        let duration = try? await asset.load(.duration)
        let midpoint = CMTimeMultiplyByFloat64(duration ?? CMTime(seconds: 1, preferredTimescale: 600), multiplier: 0.3)
        
        do {
            let cgImage = try await generator.image(at: midpoint).image
            let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
            
            let thumbnailURL = thumbnailsDirectory.appendingPathComponent("\(result.id.uuidString).jpg")
            
            if let tiffData = nsImage.tiffRepresentation,
               let bitmap = NSBitmapImageRep(data: tiffData),
               let jpegData = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) {
                try jpegData.write(to: thumbnailURL)
                return thumbnailURL.path
            }
        } catch {
            print("Failed to generate thumbnail: \(error)")
        }
        return nil
    }
    
    /// Regenerate missing thumbnails for all results
    func regenerateMissingThumbnails() async {
        for (index, result) in results.enumerated() {
            // Check if thumbnail exists
            if let thumbPath = result.thumbnailPath,
               FileManager.default.fileExists(atPath: thumbPath) {
                continue // Thumbnail exists
            }
            
            // Generate thumbnail
            if let newPath = await generateThumbnail(for: result) {
                var updated = result
                updated = GenerationResult(
                    id: result.id,
                    requestId: result.requestId,
                    prompt: result.prompt,
                    negativePrompt: result.negativePrompt,
                    parameters: result.parameters,
                    videoPath: result.videoPath,
                    thumbnailPath: newPath,
                    audioPath: result.audioPath,
                    createdAt: result.createdAt,
                    completedAt: result.completedAt,
                    duration: result.duration,
                    seed: result.seed
                )
                results[index] = updated
            }
        }
        saveHistory()
    }
    
    // MARK: - Update
    
    func updateResult(_ updatedResult: GenerationResult) {
        if let index = results.firstIndex(where: { $0.id == updatedResult.id }) {
            results[index] = updatedResult
            if selectedResult?.id == updatedResult.id {
                selectedResult = updatedResult
            }
            saveHistory()
        }
    }
    
    // MARK: - Export
    
    func exportVideo(_ result: GenerationResult, to destination: URL) throws {
        try FileManager.default.copyItem(at: result.videoURL, to: destination)
    }
    
    func revealInFinder(_ result: GenerationResult) {
        NSWorkspace.shared.selectFile(result.videoPath, inFileViewerRootedAtPath: "")
    }
}

extension GenerationResult: Hashable {
    static func == (lhs: GenerationResult, rhs: GenerationResult) -> Bool {
        lhs.id == rhs.id
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
