// Compile with APIServer.swift, GenerationRequest.swift, and MacOSSystemMemory.swift.
// Runs the real HTTP handler against an inert queue, never loading a model.
import Foundation
import Combine

@MainActor
final class GenerationService {
    var isModelLoaded = false
    var progress = 0.0
    var queue: [GenerationRequest] = []
    func addToQueue(_ request: GenerationRequest) { queue.append(request) }
    func removeFromQueue(_ request: GenerationRequest) { queue.removeAll { $0.id == request.id } }
}

// The handler references the H3 scheduling API; this harness only exercises LTX.
enum H3Variant { case test; static func from(modelId: String) -> Self { .test } }
struct H3SpeedPreset {
    var steps = 4
    static let `default` = Self()
    static func schedule(for variant: H3Variant, inferenceSteps: Int) -> Self { Self(steps: inferenceSteps) }
}
enum H3Engine {
    static func snapFrameCount(_ frames: Int) -> Int { frames }
    static func clampCanvas(width: Int, height: Int) -> (Int, Int) { (width, height) }
}

@main
struct APIKeyframeSmoke {
    static func main() async {
        do { try await run() }
        catch { fputs("FAIL: \(error)\n", stderr); exit(1) }
    }

    @MainActor
    static func run() async throws {
        guard CommandLine.arguments.count == 2 else {
            throw NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "Pass an existing image fixture path"])
        }
        let image = CommandLine.arguments[1]
        let server = APIServer.shared
        let service = GenerationService()
        server.port = 18420
        server.start(generationService: service)
        defer { server.stop() }
        for _ in 0..<100 {
            if server.isRunning { break }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        try check(server.isRunning, "HTTP listener ready")

        func post(_ params: [String: Any]) async throws -> (Int, [String: Any]) {
            let body = try JSONSerialization.data(withJSONObject: [
                "prompt": "Robot FLF HTTP regression test",
                "model_id": "ltx23_distilled_q4",
                "source_image_path": image,
                "parameters": params
            ])
            // Send header/body together, as curl does for the existing local API.
            // URLSession can split them across receives; that is a separate
            // pre-existing limitation of the server's single-receive HTTP parser.
            let output = try await Task.detached {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/curl")
                process.arguments = ["-sS", "--max-time", "5", "-w", "\n%{http_code}",
                    "-H", "Content-Type: application/json", "--data-binary", String(data: body, encoding: .utf8)!,
                    "http://127.0.0.1:18420/generate"]
                let pipe = Pipe()
                process.standardOutput = pipe
                try process.run()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                return String(data: data, encoding: .utf8)!
            }.value
            let split = output.lastIndex(of: "\n")!
            let code = Int(output[output.index(after: split)...])!
            let data = Data(output[..<split].utf8)
            let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            return (code, json)
        }

        var (code, _) = try await post([:])
        try check(code == 201 && service.queue.count == 1, "Legacy first-frame request still queues")
        try check(service.queue[0].parameters.imageFramePosition == "first" && service.queue[0].parameters.keyframes.isEmpty, "Legacy defaults preserved")

        (code, _) = try await post([
            "num_frames": 177, "image_frame_position": "first",
            "keyframes": [["image_path": image, "position": 1.0]]
        ])
        try check(code == 201, "FLF request accepted")
        let flf = service.queue[1].parameters
        try check(flf.imageFramePosition == "first" && flf.keyframes.count == 1, "Both anchors forwarded")
        try check(flf.keyframes[0].position == 1 && flf.keyframes[0].strength == 1, "Last endpoint and default strength preserved")

        (code, _) = try await post([
            "image_frame_position": "last",
            "keyframes": [["image_path": image, "position": 0.0, "strength": 0.75]]
        ])
        try check(code == 201 && service.queue[2].parameters.imageFramePosition == "last", "Primary last-frame anchor forwarded")
        try check(service.queue[2].parameters.keyframes[0].position == 0 && service.queue[2].parameters.keyframes[0].strength == 0.75, "First endpoint and explicit strength preserved")

        let bad: [[String: Any]] = [
            ["image_frame_position": "middle"],
            ["image_frame_position": 1],
            ["keyframes": "bad"],
            ["keyframes": [["position": 1.0]]],
            ["keyframes": [["image_path": "/missing/fixture.png", "position": 1.0]]],
            ["keyframes": [["image_path": image, "position": 100]]],
            ["keyframes": [["image_path": image, "position": -0.1]]],
            ["keyframes": [["image_path": image, "position": true]]],
            ["keyframes": [["image_path": image, "position": 1.0, "strength": 1.5]]],
            ["keyframes": [["image_path": image, "position": 1.0, "strength": false]]],
            ["keyframes": [["image_path": image, "position": 0.0], ["image_path": image, "position": 2.0]]]
        ]
        for (index, params) in bad.enumerated() {
            let before = service.queue.count
            let (status, body) = try await post(params)
            try check(status == 400 && body["error"] != nil && service.queue.count == before, "Invalid request \(index + 1) rejected without queuing")
        }
        let (data, _) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:18420/queue")!)
        let queue = (try JSONSerialization.jsonObject(with: data) as! [String: Any])["queue"] as! [[String: Any]]
        let echoed = queue[1]["parameters"] as! [String: Any]
        try check(echoed["image_frame_position"] as? String == "first" && (echoed["keyframes"] as? [[String: Any]])?.count == 1, "GET /queue exposes accepted anchors")
        print("PASS: real HTTP API regression checks; no model or generation started")
    }

    static func check(_ condition: Bool, _ message: String) throws {
        if !condition {
            throw NSError(domain: "APIKeyframeSmoke", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
        }
        print("PASS: \(message)")
    }
}
