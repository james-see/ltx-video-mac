import Foundation
import SwiftUI
import Combine

@MainActor
class GenerationService: ObservableObject {
    @Published private(set) var queue: [GenerationRequest] = []
    @Published private(set) var currentRequest: GenerationRequest?
    @Published private(set) var progress: Double = 0
    @Published private(set) var statusMessage: String = ""
    @Published private(set) var isModelLoaded = false
    @Published private(set) var isProcessing = false
    @Published var error: LTXError?
    
    private let historyManager: HistoryManager
    private let bridge = LTXBridge.shared
    private var processingTask: Task<Void, Never>?
    
    nonisolated init(historyManager: HistoryManager) {
        self.historyManager = historyManager
    }
    
    // MARK: - Queue Management
    
    func addToQueue(_ request: GenerationRequest) {
        queue.append(request)
        processNextIfNeeded()
    }
    
    func addBatch(_ requests: [GenerationRequest]) {
        queue.append(contentsOf: requests)
        processNextIfNeeded()
    }
    
    func removeFromQueue(_ request: GenerationRequest) {
        queue.removeAll { $0.id == request.id }
    }
    
    func clearQueue() {
        queue.removeAll { $0.status == .pending }
    }
    
    func cancelCurrent() {
        processingTask?.cancel()
        if var request = currentRequest {
            request.status = .cancelled
            currentRequest = nil
        }
        isProcessing = false
        progress = 0
        statusMessage = ""
        processNextIfNeeded()
    }
    
    func moveUp(_ request: GenerationRequest) {
        guard let index = queue.firstIndex(where: { $0.id == request.id }),
              index > 0 else { return }
        queue.swapAt(index, index - 1)
    }
    
    func moveDown(_ request: GenerationRequest) {
        guard let index = queue.firstIndex(where: { $0.id == request.id }),
              index < queue.count - 1 else { return }
        queue.swapAt(index, index + 1)
    }
    
    // MARK: - Model Management
    
    func loadModel() async {
        guard !isModelLoaded else { return }
        
        statusMessage = "Loading model..."
        isProcessing = true
        
        do {
            try await bridge.loadModel { [weak self] message in
                DispatchQueue.main.async {
                    self?.statusMessage = message
                }
            }
            isModelLoaded = bridge.isModelLoaded
            statusMessage = "Model ready"
        } catch let error as LTXError {
            self.error = error
            statusMessage = error.localizedDescription ?? "Unknown error"
        } catch {
            self.error = .modelLoadFailed(error.localizedDescription)
            statusMessage = error.localizedDescription
        }
        
        isProcessing = false
    }
    
    func unloadModel() async {
        await bridge.unloadModel()
        isModelLoaded = bridge.isModelLoaded
        statusMessage = "Model unloaded"
    }
    
    // MARK: - Processing
    
    private func processNextIfNeeded() {
        guard !isProcessing,
              let nextIndex = queue.firstIndex(where: { $0.status == .pending }) else {
            return
        }
        
        processingTask = Task {
            await processRequest(at: nextIndex)
        }
    }
    
    private func processRequest(at index: Int) async {
        guard index < queue.count else { return }
        
        isProcessing = true
        progress = 0
        
        // Load model if needed
        if !isModelLoaded {
            await loadModel()
            guard isModelLoaded else {
                isProcessing = false
                return
            }
        }
        
        // Update status
        queue[index].status = .processing
        currentRequest = queue[index]
        let request = queue[index]
        let startTime = Date()
        
        // Generate output path - use user preference if set, otherwise default location
        let userOutputDir = UserDefaults.standard.string(forKey: "outputDirectory") ?? ""
        let outputDir: URL
        if userOutputDir.isEmpty {
            outputDir = historyManager.videosDirectory
        } else {
            outputDir = URL(fileURLWithPath: userOutputDir)
            // Ensure directory exists
            try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
        }
        let filename = "\(request.id.uuidString).mp4"
        let outputPath = outputDir.appendingPathComponent(filename).path
        
        do {
            let result = try await bridge.generate(
                request: request,
                outputPath: outputPath
            ) { [weak self] prog, message in
                DispatchQueue.main.async {
                    self?.progress = prog
                    self?.statusMessage = message
                }
            }
            
            // Create result
            let completedAt = Date()
            var generationResult = GenerationResult(
                id: UUID(),
                requestId: request.id,
                prompt: request.prompt,
                negativePrompt: request.negativePrompt,
                voiceoverText: request.voiceoverText,
                parameters: request.parameters,
                videoPath: result.videoPath,
                thumbnailPath: nil,
                audioPath: nil,
                createdAt: request.createdAt,
                completedAt: completedAt,
                duration: completedAt.timeIntervalSince(startTime),
                seed: result.seed
            )
            
            // Generate thumbnail and update result with path
            if let thumbnailPath = await historyManager.generateThumbnail(for: generationResult) {
                generationResult = GenerationResult(
                    id: generationResult.id,
                    requestId: generationResult.requestId,
                    prompt: generationResult.prompt,
                    negativePrompt: generationResult.negativePrompt,
                    voiceoverText: generationResult.voiceoverText,
                    parameters: generationResult.parameters,
                    videoPath: generationResult.videoPath,
                    thumbnailPath: thumbnailPath,
                    audioPath: generationResult.audioPath,
                    createdAt: generationResult.createdAt,
                    completedAt: generationResult.completedAt,
                    duration: generationResult.duration,
                    seed: generationResult.seed
                )
            }
            
            // Save to history
            historyManager.addResult(generationResult)
            
            // Update queue
            queue[index].status = .completed
            
        } catch is CancellationError {
            queue[index].status = .cancelled
            error = .cancelled
        } catch let err as LTXError {
            queue[index].status = .failed
            error = err
        } catch {
            queue[index].status = .failed
            self.error = .generationFailed(error.localizedDescription)
        }
        
        currentRequest = nil
        isProcessing = false
        progress = 0
        statusMessage = ""
        
        // Remove completed/failed/cancelled from queue
        queue.removeAll { $0.status != .pending }
        
        // Process next
        processNextIfNeeded()
    }
}
