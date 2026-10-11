import AppKit
import SwiftUI

@main
struct LTXVideoGeneratorApp: App {
    
    init() {
        // Don't configure Python here - defer until after subprocess validation
        // This prevents crashes from PythonKit trying to load invalid Python
        RestoredWindowFrameSanitizer.moveFramesOntoAttachedDisplays()
    }
    
    var body: some Scene {
        WindowGroup {
            RootView()
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unified(showsTitle: false))
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
        
        Settings {
            SettingsRootView()
        }
    }
}

struct RootView: View {
    @Environment(\.openSettings) private var openSettingsAction
    @StateObject private var historyManager: HistoryManager
    @StateObject private var presetManager: PresetManager
    @StateObject private var characterProfileManager: CharacterProfileManager
    @StateObject private var generationService: GenerationService
    @State private var showPythonSetupAlert = false
    @State private var hasCheckedPython = false
    @State private var pythonCheckMessage = ""
    @State private var showLaunchPackageUpgradePrompt = false
    @State private var pendingLaunchUpgradeDetails: PythonDetails?
    @State private var launchPythonPathForUpgrade = ""
    @State private var showPythonUpdateCompleteAlert = false
    
    init() {
        let history = HistoryManager()
        _historyManager = StateObject(wrappedValue: history)
        _presetManager = StateObject(wrappedValue: PresetManager())
        _characterProfileManager = StateObject(wrappedValue: CharacterProfileManager())
        _generationService = StateObject(wrappedValue: GenerationService(historyManager: history))
    }
    
    /// Check if Python is configured by verifying the saved path exists
    private var hasPythonPath: Bool {
        guard let path = UserDefaults.standard.string(forKey: "pythonPath"),
              !path.isEmpty else {
            return false
        }
        return FileManager.default.fileExists(atPath: path)
    }
    
    var body: some View {
        ContentView()
            .environmentObject(historyManager)
            .environmentObject(presetManager)
            .environmentObject(characterProfileManager)
            .environmentObject(generationService)
            .task {
                historyManager.loadInitialData()
                presetManager.loadInitialData()
                characterProfileManager.loadInitialData()
                
                // One-time launch check: must NOT set hasCheckedPython until work finishes — if we set it
                // before `await validateWithSubprocess`, a cancelled .task would skip validation forever.
                guard !hasCheckedPython else { return }
                
                if !hasPythonPath {
                    // No path configured - prompt to set up
                    pythonCheckMessage = "LTX Video Generator requires Python with PyTorch and diffusers installed.\n\nPlease set your Python path in Preferences or use Auto Detect to find your Python installation."
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    showPythonSetupAlert = true
                    hasCheckedPython = true
                } else {
                    let path = UserDefaults.standard.string(forKey: "pythonPath")!
                    let result = await PythonEnvironment.shared.validateWithSubprocess(
                        path: path,
                        automaticInstallAndUpgrade: false
                    )
                    hasCheckedPython = true
                    
                    if result.success, let details = result.details {
                        PythonEnvironment.shared.configureForPythonKit(details: details)
                        PythonEnvironment.shared.applyValidatedDetailsForGeneration(path: path, details: details)
                    } else if result.pendingUserConsent, let details = result.details {
                        pendingLaunchUpgradeDetails = details
                        launchPythonPathForUpgrade = path
                        showLaunchPackageUpgradePrompt = true
                    } else if !result.success {
                        pythonCheckMessage = result.message + "\n\nPlease check your Python configuration in Preferences."
                        showPythonSetupAlert = true
                    }
                }
            }
            .alert("Update Python packages?", isPresented: $showLaunchPackageUpgradePrompt) {
                Button("Not Now", role: .cancel) {}
                Button("Upgrade") {
                    Task { await performLaunchPackageUpgrade() }
                }
            } message: {
                Text(launchPackageUpgradeMessage)
            }
            .alert("Update complete", isPresented: $showPythonUpdateCompleteAlert) {
                Button("OK", role: .cancel) {}
            } message: {
                Text("Python packages were updated successfully.")
            }
            .alert("Python Setup", isPresented: $showPythonSetupAlert) {
                Button("Open Preferences") {
                    openSettings()
                }
                Button("Later", role: .cancel) {}
            } message: {
                Text(pythonCheckMessage)
            }
    }
    
    private var launchPackageUpgradeMessage: String {
        guard let d = pendingLaunchUpgradeDetails else {
            return "Some packages need to be installed or upgraded in your Python environment."
        }
        var lines: [String] = []
        if !d.missingPackages.isEmpty {
            lines.append("Install: \(d.missingPackages.joined(separator: ", "))")
        }
        if !d.packagesNeedingUpgrade.isEmpty {
            lines.append("Upgrade: \(d.packagesNeedingUpgrade.joined(separator: ", "))")
        }
        let body = lines.isEmpty ? "Packages need attention." : lines.joined(separator: "\n")
        return body + "\n\nUpgrade now using pip in the Python path set in Preferences (virtualenv recommended)."
    }
    
    /// Runs after user confirms the launch-time upgrade prompt.
    private func performLaunchPackageUpgrade() async {
        guard let details = pendingLaunchUpgradeDetails else { return }
        let path = launchPythonPathForUpgrade
        let pkgs = details.missingPackages + details.packagesNeedingUpgrade
        guard !pkgs.isEmpty else {
            showPythonUpdateCompleteAlert = true
            return
        }
        let installResult = await PythonEnvironment.shared.installPackages(
            pythonExecutable: details.executablePath,
            packages: pkgs,
            upgrade: true
        )
        if installResult.success {
            let v = await PythonEnvironment.shared.validateWithSubprocess(path: path, automaticInstallAndUpgrade: true)
            if v.success, let det = v.details {
                PythonEnvironment.shared.configureForPythonKit(details: det)
                PythonEnvironment.shared.applyValidatedDetailsForGeneration(path: path, details: det)
            }
            showPythonUpdateCompleteAlert = true
        } else {
            pythonCheckMessage = "Package update failed: \(installResult.message)\n\nPlease check your Python configuration in Preferences."
            showPythonSetupAlert = true
        }
    }
    
    private func openSettings() {
        // Open settings on the next run loop so the alert dismisses first.
        DispatchQueue.main.async {
            openSettingsAction()
        }
    }
}

struct SettingsRootView: View {
    var body: some View {
        PreferencesView()
    }
}

/// macOS 27 traps (`NSApplication._crashOnException`) if a restored window is
/// placed on a display that is no longer attached. AppKit raises from
/// `_postWindowNeedsUpdateConstraints` while the SwiftUI hosting view is still
/// inside its first constraint pass. Move those autosaved frames onto the
/// current display before the window is created. Deleting the key instead
/// also crashes: the saved split-view columns then get applied to a window
/// with no frame.
enum RestoredWindowFrameSanitizer {
    static func moveFramesOntoAttachedDisplays() {
        guard let bundleID = Bundle.main.bundleIdentifier,
              let domain = UserDefaults.standard.persistentDomain(forName: bundleID),
              let target = NSScreen.main ?? NSScreen.screens.first else { return }
        let screens = NSScreen.screens.map(\.frame)
        guard !screens.isEmpty else { return }

        let defaults = UserDefaults.standard
        for (key, value) in domain {
            guard key.hasPrefix("NSWindow Frame "),
                  let string = value as? String,
                  let parts = frameParts(string) else { continue }
            let window = CGRect(x: parts[0], y: parts[1], width: parts[2], height: parts[3])
            let screen = CGRect(x: parts[4], y: parts[5], width: parts[6], height: parts[7])
            let screenAttached = screens.contains { $0.intersects(screen) }
            let windowVisible = screens.contains { $0.intersects(window) }
            if screenAttached && windowVisible { continue }
            defaults.set(frameString(relocated(window, onto: target), on: target), forKey: key)
        }
    }

    /// Keep the saved size, clamped to the visible frame, and center it.
    private static func relocated(_ window: CGRect, onto screen: NSScreen) -> CGRect {
        let visible = screen.visibleFrame
        let width = min(max(window.width, 0), visible.width)
        let height = min(max(window.height, 0), visible.height)
        let size = CGSize(
            width: width > 1 ? width : min(1280, visible.width),
            height: height > 1 ? height : min(800, visible.height)
        )
        return CGRect(
            x: visible.midX - size.width / 2,
            y: visible.midY - size.height / 2,
            width: size.width,
            height: size.height
        )
    }

    /// `"x y width height screenX screenY screenWidth screenHeight"`
    private static func frameParts(_ value: String) -> [CGFloat]? {
        let parts = value.split(separator: " ").compactMap { Double($0) }.map { CGFloat($0) }
        guard parts.count >= 8 else { return nil }
        return parts
    }

    private static func frameString(_ window: CGRect, on screen: NSScreen) -> String {
        let screenFrame = screen.frame
        let nums = [
            window.origin.x, window.origin.y, window.width, window.height,
            screenFrame.origin.x, screenFrame.origin.y, screenFrame.width, screenFrame.height,
        ]
        return nums.map { String(format: "%.0f", $0) }.joined(separator: " ") + " "
    }
}
