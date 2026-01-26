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
    @StateObject private var historyManager: HistoryManager
    @StateObject private var presetManager: PresetManager
    @StateObject private var generationService: GenerationService
    @State private var showPythonSetupAlert = false
    @State private var hasCheckedPython = false
    @State private var pythonCheckMessage = ""
    
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
                
                // Check Python configuration on first launch
                if !hasCheckedPython {
                    hasCheckedPython = true
                    
                    if !hasPythonPath {
                        // No path configured - prompt to set up
                        pythonCheckMessage = "LTX Video Generator requires Python with PyTorch and diffusers installed.\n\nPlease set your Python path in Preferences or use Auto Detect to find your Python installation."
                        // Small delay to let the UI settle
                        try? await Task.sleep(nanoseconds: 500_000_000)
                        showPythonSetupAlert = true
                    } else {
                        // Path exists - validate it in background using safe subprocess
                        let path = UserDefaults.standard.string(forKey: "pythonPath")!
                        let result = await PythonEnvironment.shared.validateWithSubprocess(path: path)
                        
                        if !result.success {
                            // Python validation failed - show alert with specific message
                            pythonCheckMessage = result.message + "\n\nPlease check your Python configuration in Preferences."
                            showPythonSetupAlert = true
                        } else if let details = result.details {
                            // Configure PythonKit for later use (this is safe now that we validated)
                            PythonEnvironment.shared.configureForPythonKit(details: details)
                        }
                    }
                }
            }
            .alert("Python Setup", isPresented: $showPythonSetupAlert) {
                Button("Open Preferences") {
                    NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
                }
                Button("Later", role: .cancel) {}
            } message: {
                Text(pythonCheckMessage)
            }
    }
}

struct SettingsRootView: View {
    var body: some View {
        PreferencesView()
    }
}
