import AppKit
import Foundation
import Darwin

enum H3Variant: String {
    case bf16 = "minimax_h3"
    case int8 = "minimax_h3_int8"
    case turbo = "minimax_h3_turbo"

    static func from(modelId: String) -> H3Variant {
        H3Variant(rawValue: modelId) ?? .bf16
    }

    var usesInt8Binary: Bool { self == .int8 }

    var displayName: String {
        switch self {
        case .bf16: return "H3 BF16"
        case .int8: return "H3 int8"
        case .turbo: return "H3 Turbo"
        }
    }
}

enum H3SpeedPreset: String, CaseIterable, Identifiable {
    case fast
    case `default`
    case reference

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .fast: return "Fast (4 steps)"
        case .default: return "Default (20 steps, reuse 2)"
        case .reference: return "Reference (50 steps)"
        }
    }

    var steps: Int {
        switch self {
        case .fast: return 4
        case .default: return 20
        case .reference: return 50
        }
    }

    var layers: Int {
        switch self {
        case .fast: return 50
        case .default: return 45
        case .reference: return 50
        }
    }

    var reuse: Int {
        switch self {
        case .fast: return 1
        case .default: return 2
        case .reference: return 1
        }
    }

    static func from(inferenceSteps: Int) -> H3SpeedPreset {
        if inferenceSteps <= 7 { return .fast }
        if inferenceSteps <= 30 { return .default }
        return .reference
    }

    static func schedule(
        for variant: H3Variant,
        inferenceSteps: Int
    ) -> (steps: Int, layers: Int, reuse: Int, displayName: String) {
        if variant == .turbo {
            return (6, 50, 1, "Turbo (6 steps)")
        }
        let preset = from(inferenceSteps: inferenceSteps)
        return (preset.steps, preset.layers, preset.reuse, preset.displayName)
    }
}

struct H3EngineError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

enum H3Engine {
    static let binaryPathKey = "h3BinaryPath"
    static let modelDirectoryKey = "h3ModelDirectory"
    static let licenseAcceptedKey = "h3LicenseAccepted"
    static let huggingfaceRepo = "MiniMaxAI/MiniMax-H3"
    static let sourceRepoURL = "https://github.com/antirez/h3.c.git"
    static let int8SourceRepoURL = "https://github.com/Shedrackeze002/h3.c-int8.git"
    static let int8SourcePin = "200eed4792c33be187c566cb649982246fd3c58a"
    static let comfyOrgRepo = "Comfy-Org/MiniMax-H3"
    static let comfyInt8DitFile = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    static let turboLoraRepo = "larryvrh/MiniMax-H3-Turbo-Lora"
    static let turboLoraFile = "minimax_h3_turbo_v4_step600_ema.safetensors"
    static let legalFrameCounts = [22, 39, 56, 107, 243, 362]
    static let maxPixelProduct = 768 * 1344
    static let minimumRAMGB = 16

    static var licenseAccepted: Bool {
        UserDefaults.standard.bool(forKey: licenseAcceptedKey)
    }

    static func acceptLicense() {
        UserDefaults.standard.set(true, forKey: licenseAcceptedKey)
    }

    static func physicalMemoryGB() -> Int {
        Int(MacOSSystemMemory.physicalMemoryBytes / 1_073_741_824)
    }

    static func shouldUseSSDStreaming() -> Bool {
        physicalMemoryGB() < 64
    }

    static func snapFrameCount(_ requested: Int) -> Int {
        let legal = legalFrameCounts
        return legal.min(by: { abs($0 - requested) < abs($1 - requested) }) ?? 22
    }

    static func secondsLabel(forFrames frames: Int) -> String {
        String(format: "%.1fs", Double(frames) / 24.0)
    }

    static func snapDimension(_ value: Int) -> Int {
        max(32, (value / 32) * 32)
    }

    static func clampCanvas(width: Int, height: Int) -> (Int, Int) {
        var w = snapDimension(width)
        var h = snapDimension(height)
        if w * h > maxPixelProduct {
            let scale = sqrt(Double(maxPixelProduct) / Double(w * h))
            w = snapDimension(Int(Double(w) * scale))
            h = snapDimension(Int(Double(h) * scale))
            while w * h > maxPixelProduct {
                if w >= h { w -= 32 } else { h -= 32 }
                w = max(32, w)
                h = max(32, h)
            }
        }
        return (w, h)
    }

    static var appSupportRoot: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("LTXVideoGenerator", isDirectory: true)
    }

    static var defaultSourceDirectory: String {
        appSupportRoot.appendingPathComponent("h3.c", isDirectory: true).path
    }

    static var int8SourceDirectory: String {
        appSupportRoot.appendingPathComponent("h3.c-int8", isDirectory: true).path
    }

    static var int8ModelDirectory: String {
        appSupportRoot.appendingPathComponent("h3-int8", isDirectory: true).path
    }

    static var turboModelDirectory: String {
        appSupportRoot.appendingPathComponent("H3Turbo", isDirectory: true).path
    }

    static var defaultBinaryPath: String {
        (defaultSourceDirectory as NSString).appendingPathComponent("h3")
    }

    static var int8BinaryPath: String {
        (int8SourceDirectory as NSString).appendingPathComponent("h3")
    }

    static func resolvedBinary(
        for variant: H3Variant = .bf16,
        userDefaults: UserDefaults = .standard
    ) -> String? {
        let fm = FileManager.default
        if variant.usesInt8Binary {
            return fm.isExecutableFile(atPath: int8BinaryPath) ? int8BinaryPath : nil
        }

        let preferred = userDefaults.string(forKey: binaryPathKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !preferred.isEmpty, fm.isExecutableFile(atPath: preferred) {
            return preferred
        }

        let home = fm.homeDirectoryForCurrentUser.path
        let candidates = [
            defaultBinaryPath,
            "\(home)/projects/h3.c/h3",
            "\(home)/p/h3.c/h3",
        ]
        for path in candidates where fm.isExecutableFile(atPath: path) {
            return path
        }

        if let which = which("h3") {
            return which
        }
        return nil
    }

    /// Clone and `make` h3.c when no executable is on disk. First generate, not a pre-req.
    static func ensureBinary(
        for variant: H3Variant = .bf16,
        userDefaults: UserDefaults = .standard,
        progress: @escaping (String) -> Void
    ) async throws -> String {
        if let existing = resolvedBinary(for: variant, userDefaults: userDefaults) {
            return existing
        }
        return try await Task.detached(priority: .userInitiated) {
            try bootstrapBinary(for: variant, userDefaults: userDefaults, progress: progress)
        }.value
    }

    private static func bootstrapBinary(
        for variant: H3Variant,
        userDefaults: UserDefaults,
        progress: @escaping (String) -> Void
    ) throws -> String {
        guard which("git") != nil else {
            throw H3EngineError(message: "git is required to fetch h3.c. Install Xcode Command Line Tools (`xcode-select --install`) and retry Generate.")
        }
        guard which("make") != nil else {
            throw H3EngineError(message: "make is required to build h3.c. Install Xcode Command Line Tools (`xcode-select --install`) and retry Generate.")
        }

        let source = variant.usesInt8Binary ? int8SourceDirectory : defaultSourceDirectory
        try cloneOrUpdate(
            into: source,
            repoURL: variant.usesInt8Binary ? int8SourceRepoURL : sourceRepoURL,
            pin: variant.usesInt8Binary ? int8SourcePin : nil,
            label: variant.usesInt8Binary ? "h3.c-int8" : "h3.c",
            progress: progress
        )
        try build(in: source, progress: progress)

        let binary = (source as NSString).appendingPathComponent("h3")
        guard FileManager.default.isExecutableFile(atPath: binary) else {
            throw H3EngineError(message: "h3 build finished but \(binary) is not executable.")
        }

        if !variant.usesInt8Binary {
            let preferred = userDefaults.string(forKey: binaryPathKey)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if preferred.isEmpty {
                userDefaults.set(binary, forKey: binaryPathKey)
            }
        }
        return binary
    }

    private static func cloneOrUpdate(
        into source: String,
        repoURL: String,
        pin: String?,
        label: String,
        progress: @escaping (String) -> Void
    ) throws {
        let fm = FileManager.default
        let gitDir = (source as NSString).appendingPathComponent(".git")
        let makefile = (source as NSString).appendingPathComponent("Makefile")

        if fm.fileExists(atPath: gitDir) {
            if let pin {
                let head = (try? gitOutput(["-C", source, "rev-parse", "HEAD"]))?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                if head.hasPrefix(pin) || pin.hasPrefix(head) {
                    progress("\(label) already at \(pin.prefix(8))")
                    return
                }
                progress("Checking out \(label) \(pin.prefix(8))…")
                do {
                    try runTool("git", arguments: ["-C", source, "fetch", "--depth", "1", "origin", pin], directory: nil, progress: progress)
                    try runTool("git", arguments: ["-C", source, "checkout", "--detach", pin], directory: nil, progress: progress)
                } catch {
                    progress("pin checkout skipped (\(error.localizedDescription)); building the existing tree")
                }
                return
            }
            progress("Updating \(label)…")
            do {
                try runTool(
                    "git",
                    arguments: ["-C", source, "pull", "--ff-only"],
                    directory: nil,
                    progress: progress
                )
            } catch {
                progress("git pull skipped (\(error.localizedDescription)); building the existing tree")
            }
            return
        }

        if fm.fileExists(atPath: makefile), pin == nil {
            progress("Using existing \(label) sources at \(source)")
            return
        }

        if fm.fileExists(atPath: source) {
            try fm.removeItem(atPath: source)
        }
        try fm.createDirectory(
            at: URL(fileURLWithPath: source).deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        progress("Cloning \(label)…")
        var cloneArgs = ["clone"]
        if pin == nil {
            cloneArgs.append("--depth")
            cloneArgs.append("1")
        }
        cloneArgs.append(contentsOf: [repoURL, source])
        try runTool("git", arguments: cloneArgs, directory: nil, progress: progress)
        if let pin {
            try runTool("git", arguments: ["-C", source, "checkout", "--detach", pin], directory: nil, progress: progress)
        }
    }

    private static func gitOutput(_ arguments: [String]) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw H3EngineError(message: "git \(arguments.joined(separator: " ")) failed")
        }
        return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    }

    private static func build(in source: String, progress: @escaping (String) -> Void) throws {
        let jobs = max(1, ProcessInfo.processInfo.activeProcessorCount)
        progress("Building h3 (`make -j\(jobs)`)…")
        do {
            try runTool(
                "make",
                arguments: ["-j\(jobs)"],
                directory: source,
                progress: progress
            )
        } catch {
            throw H3EngineError(
                message: "h3.c build failed. Install Xcode Command Line Tools (`xcode-select --install`) if clang/make are missing. \(error.localizedDescription)"
            )
        }
    }

    private static func runTool(
        _ name: String,
        arguments: [String],
        directory: String?,
        progress: @escaping (String) -> Void
    ) throws {
        guard let executable = which(name) ?? ["/usr/bin/\(name)", "/opt/homebrew/bin/\(name)", "/usr/local/bin/\(name)"].first(where: {
            FileManager.default.isExecutableFile(atPath: $0)
        }) else {
            throw H3EngineError(message: "\(name) not found on PATH")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        if let directory {
            process.currentDirectoryURL = URL(fileURLWithPath: directory)
        }
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (env["PATH"] ?? "")
        process.environment = env

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()

        let handle = pipe.fileHandleForReading
        var leftover = Data()
        while process.isRunning {
            let chunk = handle.availableData
            if chunk.isEmpty {
                Thread.sleep(forTimeInterval: 0.05)
                continue
            }
            leftover.append(chunk)
            leftover = emitLines(from: leftover, progress: progress)
        }
        leftover.append(handle.readDataToEndOfFile())
        _ = emitLines(from: leftover, progress: progress)
        process.waitUntilExit()
        if process.terminationStatus != 0 {
            throw H3EngineError(message: "\(name) exited \(process.terminationStatus)")
        }
    }

    private static func emitLines(from data: Data, progress: @escaping (String) -> Void) -> Data {
        var data = data
        while let range = data.range(of: Data([0x0A])) {
            let lineData = data[..<range.lowerBound]
            data.removeSubrange(..<(range.upperBound))
            if let line = String(data: lineData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !line.isEmpty {
                progress(line)
            }
        }
        return data
    }

    static func resolvedModelDirectory(
        for variant: H3Variant = .bf16,
        userDefaults: UserDefaults = .standard
    ) -> String? {
        switch variant {
        case .int8:
            return isInt8TreeReady(int8ModelDirectory) ? int8ModelDirectory : nil
        case .turbo:
            return isTurboTreeReady(turboModelDirectory) ? turboModelDirectory : nil
        case .bf16:
            return resolvedOfficialSnapshot(userDefaults: userDefaults)
        }
    }

    static func resolvedOfficialSnapshot(userDefaults: UserDefaults = .standard) -> String? {
        let fm = FileManager.default
        let preferred = userDefaults.string(forKey: modelDirectoryKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !preferred.isEmpty, isOfficialSnapshot(preferred) {
            return preferred
        }

        let home = fm.homeDirectoryForCurrentUser.path
        let candidates = [
            "\(home)/MiniMax-H3",
            "\(home)/projects/MiniMax-H3",
        ]
        for path in candidates where isOfficialSnapshot(path) {
            return path
        }

        let cacheRoot = HuggingFaceCacheConfiguration.effectiveDirectory(userDefaults: userDefaults)
        let hub = URL(fileURLWithPath: cacheRoot, isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
            .appendingPathComponent("models--MiniMaxAI--MiniMax-H3", isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
        if let snapshots = try? fm.contentsOfDirectory(atPath: hub.path) {
            for snap in snapshots.sorted() {
                let path = hub.appendingPathComponent(snap).path
                if isOfficialSnapshot(path) {
                    return path
                }
            }
        }
        return nil
    }

    /// Text encoder + VAEs + tokenizer, with or without the 13-shard BF16 DiT.
    static func resolvedOfficialAuxiliaries(userDefaults: UserDefaults = .standard) -> String? {
        if let full = resolvedOfficialSnapshot(userDefaults: userDefaults) {
            return full
        }
        let cacheRoot = HuggingFaceCacheConfiguration.effectiveDirectory(userDefaults: userDefaults)
        let hub = URL(fileURLWithPath: cacheRoot, isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
            .appendingPathComponent("models--MiniMaxAI--MiniMax-H3", isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
        if let snapshots = try? FileManager.default.contentsOfDirectory(atPath: hub.path) {
            for snap in snapshots.sorted() {
                let path = hub.appendingPathComponent(snap).path
                if hasOfficialAuxiliaries(path) {
                    return path
                }
            }
        }
        return nil
    }

    static func officialTransformerDirectory(userDefaults: UserDefaults = .standard) -> String? {
        guard let root = resolvedOfficialSnapshot(userDefaults: userDefaults) else { return nil }
        return (root as NSString).appendingPathComponent("FL2VA/transformer")
    }

    static func isModelDirectory(_ path: String) -> Bool {
        isOfficialSnapshot(path) || isInt8TreeReady(path) || isTurboTreeReady(path)
    }

    static func isOfficialSnapshot(_ path: String) -> Bool {
        hasOfficialAuxiliaries(path) && hasOfficialTransformer(path)
    }

    static func hasOfficialTransformer(_ path: String) -> Bool {
        let transformer = (path as NSString).appendingPathComponent("FL2VA/transformer")
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: transformer)) ?? []
        return entries.contains { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }
    }

    static func hasOfficialAuxiliaries(_ path: String) -> Bool {
        let fl2va = (path as NSString).appendingPathComponent("FL2VA")
        let required = ["text_encoder", "video_vae", "audio_vae", "tokenizer"]
        return required.allSatisfy { name in
            directoryHasWeights((fl2va as NSString).appendingPathComponent(name))
        } && FileManager.default.fileExists(
            atPath: (fl2va as NSString).appendingPathComponent("transformer/config.json")
        )
    }

    static func isInt8TreeReady(_ path: String) -> Bool {
        guard hasOfficialAuxiliaries(path) else { return false }
        let transformer = (path as NSString).appendingPathComponent("FL2VA/transformer")
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: transformer)) ?? []
        return entries.contains { $0.contains("int8") && $0.hasSuffix(".safetensors") }
    }

    static func isTurboTreeReady(_ path: String) -> Bool {
        guard hasOfficialAuxiliaries(path), hasOfficialTransformer(path) else { return false }
        return FileManager.default.fileExists(
            atPath: (path as NSString).appendingPathComponent("FL2VA/transformer/.turbo-folded")
        )
    }

    private static func directoryHasWeights(_ path: String) -> Bool {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: path)) ?? []
        return entries.contains { $0.hasSuffix(".safetensors") || $0 == "config.json" || $0 == "tokenizer.json" }
    }

    static let licensePageURL = URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3")!
    static let licenseTextURL = URL(string: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE")!
    static let licenseRequestURL = URL(string: "https://platform.minimax.io/h3-license")!

    static func licenseNotice() -> String {
        "MiniMax H3 weights are under the MiniMax H3 Community License (h3.c itself is MIT). BF16 downloads MiniMaxAI/MiniMax-H3 (~144GB). Int8 mixes Comfy-Org DiT (~20GB) with MiniMaxAI text encoder/VAEs. Turbo folds larryvrh Turbo LoRA into BF16 (same license, derivative). US/EU/UK/KR users may need authorization at platform.minimax.io/h3-license. Commercial use over $20M/year needs written permission."
    }

    static func openLicensePage() {
        NSWorkspace.shared.open(licensePageURL)
    }

    static func openLicenseText() {
        NSWorkspace.shared.open(licenseTextURL)
    }

    static func missingBinaryHint() -> String {
        "Could not build h3.c. Need git, make, and Xcode Command Line Tools (`xcode-select --install`). Or set Preferences → General → h3 binary path to an existing binary."
    }

    static func missingModelHint() -> String {
        "MiniMax-H3 download failed. Incomplete files stay in the Hugging Face cache — Generate again to resume. If the repo is gated, run `hf auth login`. Or set Preferences → General → H3 model directory to a local BF16 snapshot. Full log: /tmp/ltx_generation.log"
    }

    static func downloadRequiresPythonHint() -> String {
        "H3 downloads weights into the Hugging Face cache (huggingface_hub). Set a Python path in Preferences, or for BF16 point H3 model directory at an existing snapshot."
    }

    private static func which(_ name: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [name]
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (env["PATH"] ?? "")
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let path = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return path.isEmpty ? nil : path
        } catch {
            return nil
        }
    }
}
