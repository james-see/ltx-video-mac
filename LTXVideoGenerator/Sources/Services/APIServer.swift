import Foundation
import Network
import UniformTypeIdentifiers

@MainActor
class APIServer: ObservableObject {
    static let shared = APIServer()
    
    @Published var isRunning = false
    @Published var port: UInt16 = 8420
    
    private var listener: NWListener?
    private var generationServiceRef: GenerationService?
    
    private init() {}
    
    func start(generationService: GenerationService) {
        guard !isRunning else { return }
        
        self.generationServiceRef = generationService
        
        do {
            let params = NWParameters.tcp
            params.allowLocalEndpointReuse = true
            let endpointPort = NWEndpoint.Port(rawValue: port)!
            params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: endpointPort)

            // The API accepts local file paths, so it must not be exposed to the LAN.
            listener = try NWListener(using: params)
            
            listener?.stateUpdateHandler = { [weak self] state in
                Task { @MainActor in
                    switch state {
                    case .ready:
                        self?.isRunning = true
                        print("API Server running on http://localhost:\(self?.port ?? 8420)")
                    case .failed(let error):
                        print("API Server failed: \(error)")
                        self?.isRunning = false
                    case .cancelled:
                        self?.isRunning = false
                    default:
                        break
                    }
                }
            }
            
            listener?.newConnectionHandler = { [weak self] connection in
                Task { @MainActor in
                    self?.handleConnection(connection)
                }
            }
            
            listener?.start(queue: .global(qos: .userInitiated))
        } catch {
            print("Failed to start API server: \(error)")
        }
    }
    
    func stop() {
        listener?.cancel()
        listener = nil
        generationServiceRef = nil
        isRunning = false
    }
    
    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: .global(qos: .userInitiated))
        receiveRequest(connection)
    }
    
    private func receiveRequest(_ connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            if let data = data, !data.isEmpty {
                Task { @MainActor in
                    self?.processHTTPRequest(data, connection: connection)
                }
            }
            
            if isComplete || error != nil {
                connection.cancel()
            }
        }
    }
    
    private func processHTTPRequest(_ data: Data, connection: NWConnection) {
        guard let generationService = generationServiceRef else {
            sendResponse(connection, status: 500, body: ["error": "Service not available"])
            return
        }
        guard let request = String(data: data, encoding: .utf8) else {
            sendResponse(connection, status: 400, body: ["error": "Invalid request"])
            return
        }
        
        let lines = request.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else {
            sendResponse(connection, status: 400, body: ["error": "Invalid request"])
            return
        }
        
        let parts = requestLine.components(separatedBy: " ")
        guard parts.count >= 2 else {
            sendResponse(connection, status: 400, body: ["error": "Invalid request"])
            return
        }
        
        let method = parts[0]
        let path = parts[1]
        
        // Extract body for POST requests
        var body: [String: Any]?
        if method == "POST", let bodyStart = request.range(of: "\r\n\r\n") {
            let bodyString = String(request[bodyStart.upperBound...])
            if let bodyData = bodyString.data(using: .utf8) {
                body = try? JSONSerialization.jsonObject(with: bodyData) as? [String: Any]
            }
        }
        
        // Route requests
        switch (method, path) {
        case ("GET", "/"):
            sendResponse(connection, status: 200, body: [
                "service": "LTX Video Generator",
                "version": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown",
                "endpoints": [
                    "GET /status": "Server and generation status",
                    "GET /queue": "Current generation queue",
                    "POST /generate": "Submit generation request (optional source_image_path, model_id, text_encoder_id)",
                    "DELETE /queue/:id": "Cancel a queued request"
                ]
            ])
            
        case ("GET", "/status"):
            let status: [String: Any] = [
                "server": "running",
                "model_loaded": generationService.isModelLoaded,
                "queue_count": generationService.queue.count,
                "current_progress": generationService.progress
            ]
            sendResponse(connection, status: 200, body: status)
            
        case ("GET", "/queue"):
            let queue = generationService.queue.map { request -> [String: Any] in
                let model = LTXModelCatalog.resolvedModel(id: request.modelId)
                let textEncoder = LTXTextEncoderCatalog.resolvedTextEncoder(id: request.textEncoderId)
                return [
                    "id": request.id.uuidString,
                    "prompt": request.prompt,
                    "mode": request.isImageToVideo ? "image-to-video" : "text-to-video",
                    "source_image_name": request.sourceImagePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? NSNull(),
                    "status": request.status.rawValue,
                    "created_at": ISO8601DateFormatter().string(from: request.createdAt),
                    "model": [
                        "id": model.id,
                        "repo": model.repo,
                        "display_name": model.displayName,
                    ],
                    "text_encoder": [
                        "id": textEncoder.id,
                        "repo": textEncoder.repo,
                        "display_name": textEncoder.displayName,
                    ],
                    "parameters": [
                        "width": request.parameters.width,
                        "height": request.parameters.height,
                        "num_frames": request.parameters.numFrames,
                        "fps": request.parameters.fps,
                        "num_inference_steps": request.parameters.numInferenceSteps,
                        "guidance_scale": request.parameters.guidanceScale
                    ]
                ]
            }
            sendResponse(connection, status: 200, body: ["queue": queue])
            
        case ("POST", "/generate"):
            guard let body = body,
                  let prompt = body["prompt"] as? String else {
                sendResponse(connection, status: 400, body: ["error": "Missing required field: prompt"])
                return
            }
            
            let negativePrompt = body["negative_prompt"] as? String ?? ""
            let sourceImageValidation = validateSourceImagePath(body["source_image_path"])
            if let validationError = sourceImageValidation.error {
                sendResponse(connection, status: 400, body: ["error": validationError])
                return
            }
            let voiceoverText = body["voiceover_text"] as? String ?? ""
            let voiceoverSource = body["voiceover_source"] as? String ?? "mlx-audio"
            let voiceoverVoice = body["voiceover_voice"] as? String ?? "af_heart"
            let musicEnabled = body["music_enabled"] as? Bool ?? false
            let musicGenre = body["music_genre"] as? String
            let requestedModelID = body["model_id"] as? String
            let requestedModelRepo = body["model_repo"] as? String
            let requestedTextEncoderID = body["text_encoder_id"] as? String
            let requestedTextEncoderRepo = body["text_encoder_repo"] as? String
            let resolvedModel: LTXModel
            if let requestedModelID, let byID = LTXModelCatalog.model(id: requestedModelID) {
                resolvedModel = byID
            } else if let requestedModelRepo, let byRepo = LTXModelCatalog.model(repo: requestedModelRepo) {
                resolvedModel = byRepo
            } else {
                resolvedModel = LTXModelCatalog.selectedModel()
            }
            let resolvedTextEncoder: LTXTextEncoder
            if let requestedTextEncoderID, let byID = LTXTextEncoderCatalog.textEncoder(id: requestedTextEncoderID) {
                resolvedTextEncoder = byID
            } else if let requestedTextEncoderRepo, let byRepo = LTXTextEncoderCatalog.textEncoder(repo: requestedTextEncoderRepo) {
                resolvedTextEncoder = byRepo
            } else {
                resolvedTextEncoder = LTXTextEncoderCatalog.selectedTextEncoder()
            }
            
            var params = GenerationParameters.default
            if let p = body["parameters"] as? [String: Any] {
                if let width = p["width"] as? Int { params.width = width }
                if let height = p["height"] as? Int { params.height = height }
                if let numFrames = p["num_frames"] as? Int { params.numFrames = numFrames }
                if let fps = p["fps"] as? Int { params.fps = fps }
                if let steps = p["num_inference_steps"] as? Int { params.numInferenceSteps = steps }
                if let guidance = p["guidance_scale"] as? Double { params.guidanceScale = guidance }
                if let seed = p["seed"] as? Int { params.seed = seed }
                if let vaeTilingMode = p["vae_tiling_mode"] as? String { params.vaeTilingMode = vaeTilingMode }
                if let imageStrength = p["image_strength"] as? Double {
                    guard (0.0...1.0).contains(imageStrength) else {
                        sendResponse(connection, status: 400, body: ["error": "parameters.image_strength must be between 0.0 and 1.0"])
                        return
                    }
                    params.imageStrength = imageStrength
                }
            }
            
            let request = GenerationRequest(
                prompt: prompt,
                negativePrompt: negativePrompt,
                voiceoverText: voiceoverText,
                voiceoverSource: voiceoverSource,
                voiceoverVoice: voiceoverVoice,
                sourceImagePath: sourceImageValidation.path,
                musicEnabled: musicEnabled,
                musicGenre: musicGenre,
                modelId: resolvedModel.id,
                textEncoderId: resolvedTextEncoder.id,
                parameters: params
            )
            
            generationService.addToQueue(request)
            sendResponse(connection, status: 201, body: [
                "id": request.id.uuidString,
                "status": "queued",
                "mode": request.isImageToVideo ? "image-to-video" : "text-to-video",
                "source_image_name": request.sourceImagePath.map { URL(fileURLWithPath: $0).lastPathComponent } ?? NSNull(),
                "model_id": request.modelId,
                "model_repo": resolvedModel.repo,
                "text_encoder_id": request.textEncoderId,
                "text_encoder_repo": resolvedTextEncoder.repo,
                "message": "Generation request added to queue"
            ])
            
        case ("DELETE", _) where path.hasPrefix("/queue/"):
            let idString = String(path.dropFirst("/queue/".count))
            guard let uuid = UUID(uuidString: idString) else {
                sendResponse(connection, status: 400, body: ["error": "Invalid ID format"])
                return
            }
            
            if let request = generationService.queue.first(where: { $0.id == uuid }) {
                generationService.removeFromQueue(request)
                sendResponse(connection, status: 200, body: ["status": "cancelled"])
            } else {
                sendResponse(connection, status: 404, body: ["error": "Request not found"])
            }
            
        default:
            sendResponse(connection, status: 404, body: ["error": "Not found"])
        }
    }

    /// Resolves and validates an optional local image path before it reaches the Python bridge.
    private func validateSourceImagePath(_ value: Any?) -> (path: String?, error: String?) {
        guard let value else { return (nil, nil) }
        guard let rawPath = value as? String else {
            return (nil, "source_image_path must be a string")
        }

        let trimmedPath = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty else { return (nil, nil) }

        let expandedPath = NSString(string: trimmedPath).expandingTildeInPath
        guard expandedPath.hasPrefix("/") else {
            return (nil, "source_image_path must be an absolute path")
        }

        let imageURL = URL(fileURLWithPath: expandedPath)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: imageURL.path, isDirectory: &isDirectory),
              !isDirectory.boolValue,
              FileManager.default.isReadableFile(atPath: imageURL.path) else {
            return (nil, "source_image_path does not point to a readable file")
        }

        guard let fileType = UTType(filenameExtension: imageURL.pathExtension),
              fileType.conforms(to: .image) else {
            return (nil, "source_image_path must point to a supported image file")
        }

        return (imageURL.path, nil)
    }
    
    private func sendResponse(_ connection: NWConnection, status: Int, body: [String: Any]) {
        let statusText: String
        switch status {
        case 200: statusText = "OK"
        case 201: statusText = "Created"
        case 400: statusText = "Bad Request"
        case 404: statusText = "Not Found"
        case 500: statusText = "Internal Server Error"
        default: statusText = "Unknown"
        }
        
        let jsonData = (try? JSONSerialization.data(withJSONObject: body, options: .prettyPrinted)) ?? Data()
        let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
        
        let response = """
        HTTP/1.1 \(status) \(statusText)\r
        Content-Type: application/json\r
        Content-Length: \(jsonData.count)\r
        Access-Control-Allow-Origin: *\r
        Connection: close\r
        \r
        \(jsonString)
        """
        
        connection.send(content: response.data(using: .utf8), completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
}
