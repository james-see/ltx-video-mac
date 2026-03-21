import SwiftUI

@main
struct LTXVideoGeneratorApp: App {
    
    init() {
        // Don't configure Python here - defer until after subprocess validation
        // This prevents crashes from PythonKit trying to load invalid Python
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
            .environmentObject(generationService)
            .task {
                historyManager.loadInitialData()
                presetManager.loadInitialData()
                
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
