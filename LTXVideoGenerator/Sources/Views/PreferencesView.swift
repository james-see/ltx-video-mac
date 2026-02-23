import SwiftUI

/// Single model: LTX-2 Unified (audio+video). Legacy distilled model removed.
enum LTXModelVariant {
    static let modelRepo = "notapalindrome/ltx2-mlx-av"
    static let displayName = "LTX-2 Unified"
    static let downloadSize = "~42GB"
    static let supportsBuiltInAudio = true
}

struct PreferencesView: View {
    @AppStorage("pythonPath") private var pythonPath = ""
    @AppStorage("outputDirectory") private var outputDirectory = ""
    @AppStorage("autoLoadModel") private var autoLoadModel = false
    @AppStorage("keepCompletedInQueue") private var keepCompletedInQueue = false
    @AppStorage("elevenLabsApiKey") private var elevenLabsApiKey = ""
    @AppStorage("defaultAudioSource") private var defaultAudioSource = "elevenlabs"
    @AppStorage("enableGemmaPromptEnhancement") private var enableGemmaPromptEnhancement = false
    @AppStorage("saveAudioTrackSeparately") private var saveAudioTrackSeparately = false

    @State private var pythonStatus: (success: Bool, message: String)?
    @State private var pythonDetails: PythonDetails?
    @State private var isValidating = false
    @State private var isDetecting = false
    @State private var isInstalling = false
    @State private var detectedPaths: [String] = []
    @State private var showPathPicker = false
    @State private var installMessage: String?
    @State private var isTestingElevenLabs = false
    @State private var elevenLabsTestResult: (success: Bool, message: String)?
    
    var body: some View {
        TabView {
            // General
            Form {
                Section("Python Environment") {
                    HStack {
                        TextField("Python Path", text: $pythonPath)
                            .textFieldStyle(.roundedBorder)
                        
                        Button("Browse...") {
                            selectPythonPath()
                        }
                        
                        Button("Auto Detect") {
                            detectPython()
                        }
                        .disabled(isDetecting)
                    }
                    
                    // Path type indicator
                    if !pythonPath.isEmpty {
                        let pathType = PythonEnvironment.shared.detectPathType(pythonPath)
                        HStack {
                            Image(systemName: pathTypeIcon(pathType))
                                .foregroundStyle(pathTypeColor(pathType))
                            Text(pathTypeDescription(pathType))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    
                    // Status display
                    if isValidating || isDetecting || isInstalling {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.7)
                            Text(isDetecting ? "Searching for Python installations..." : 
                                 isInstalling ? "Installing MLX packages..." :
                                 "Validating Python setup...")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } else if let status = pythonStatus {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(alignment: .top) {
                                Image(systemName: status.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                                    .foregroundStyle(status.success ? .green : .red)
                                Text(status.message)
                                    .font(.caption)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            
                            // Show details if available
                            if let details = pythonDetails {
                                VStack(alignment: .leading, spacing: 2) {
                                    if !details.executablePath.isEmpty {
                                        Text("Executable: \(details.executablePath)")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                    if details.hasMLX {
                                        HStack(spacing: 4) {
                                            Image(systemName: "checkmark.seal.fill")
                                                .foregroundStyle(.green)
                                            Text("MLX Ready")
                                                .font(.caption2)
                                                .foregroundStyle(.green)
                                        }
                                        .padding(.leading, 20)
                                    }
                                }
                                .padding(.leading, 20)
                                
                                // Offer to install missing packages
                                if !details.missingPackages.isEmpty {
                                    let isVenv = PythonEnvironment.shared.isVirtualEnvironment(details.executablePath)
                                    
                                    VStack(alignment: .leading, spacing: 8) {
                                        if isVenv {
                                            // Can install directly to venv
                                            Text("Install missing packages:")
                                                .font(.caption.bold())
                                            
                                            HStack {
                                                Text("pip install \(details.missingPackages.joined(separator: " "))")
                                                    .font(.caption.monospaced())
                                                    .textSelection(.enabled)
                                                    .padding(6)
                                                    .background(Color.secondary.opacity(0.1))
                                                    .cornerRadius(4)
                                                
                                                Button("Install") {
                                                    installMissingPackages(pythonPath: details.executablePath, packages: details.missingPackages)
                                                }
                                                .buttonStyle(.borderedProminent)
                                                .disabled(isInstalling)
                                            }
                                        } else {
                                            // System Python - need to create venv first
                                            HStack(alignment: .top, spacing: 8) {
                                                Image(systemName: "exclamationmark.triangle.fill")
                                                    .foregroundStyle(.orange)
                                                VStack(alignment: .leading, spacing: 4) {
                                                    Text("System Python Detected")
                                                        .font(.caption.bold())
                                                    Text("This Python doesn't allow global pip installs. Create a virtual environment to install packages.")
                                                        .font(.caption)
                                                        .foregroundStyle(.secondary)
                                                }
                                            }
                                            
                                            Button("Create Virtual Environment & Install") {
                                                createVenvAndInstall(basePython: details.executablePath, packages: details.missingPackages)
                                            }
                                            .buttonStyle(.borderedProminent)
                                            .disabled(isInstalling)
                                            
                                            Text("This will create ~/ltx-venv and install packages there")
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    .padding(.top, 4)
                                }
                            }
                            
                            // Show install result
                            if let msg = installMessage {
                                Text(msg)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .padding(.top, 4)
                            }
                        }
                    }
                    
                    HStack {
                        Button("Validate Setup") {
                            validatePython()
                        }
                        .disabled(isValidating || pythonPath.isEmpty)
                        
                        if !detectedPaths.isEmpty {
                            Button("Show Detected (\(detectedPaths.count))") {
                                showPathPicker = true
                            }
                        }
                    }
                    
                    Text("Supports both Python executable (e.g., /opt/homebrew/bin/python3) and dylib paths. Auto Detect will search common locations including Homebrew, pyenv, conda, and virtualenvs.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Section("Model") {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "waveform.badge.checkmark")
                            .foregroundStyle(.green)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("LTX-2 Unified")
                                .font(.caption.bold())
                            Text("Generates video with synchronized audio (~42GB download). Model cached in ~/.cache/huggingface/")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                    
                    Toggle("Auto-load model on startup", isOn: $autoLoadModel)
                }
                
                Section("Storage") {
                    HStack {
                        TextField("Output Directory", text: $outputDirectory)
                            .textFieldStyle(.roundedBorder)
                        
                        Button("Browse...") {
                            selectOutputDirectory()
                        }
                        
                        Button("Open") {
                            openOutputDirectory()
                        }
                    }
                    
                    Text("Leave empty to use default location in Application Support")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .tabItem {
                Label("General", systemImage: "gear")
            }
            
            // Generation
            Form {
                Section("Queue") {
                    Toggle("Keep completed items in queue", isOn: $keepCompletedInQueue)
                }

                Section("Output") {
                    Toggle("Save audio track separately", isOn: $saveAudioTrackSeparately)
                        .help("When on, keeps a .wav file alongside each video. Default: off (audio only in mp4).")
                }

                Section("Prompt Enhancement") {
                    Toggle("Enable Prompt Enhancement", isOn: $enableGemmaPromptEnhancement)
                        .help("When on, Gemma rewrites your prompt with vivid details (lighting, camera, audio) before generation. Use Preview in the prompt view to see the enhanced prompt first.")
                    Text("Uses Gemma to rewrite prompts with vivid details for better video generation. First run downloads ~7GB. Requires mlx-video-with-audio.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Section("Defaults") {
                    Text("Default generation parameters can be set via Presets")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .formStyle(.grouped)
            .tabItem {
                Label("Generation", systemImage: "wand.and.stars")
            }
            
            // Audio
            Form {
                Section("ElevenLabs API") {
                    SecureField("API Key", text: $elevenLabsApiKey)
                        .textFieldStyle(.roundedBorder)
                    
                    HStack {
                        if isTestingElevenLabs {
                            ProgressView()
                                .scaleEffect(0.7)
                            Text("Testing connection...")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        } else if let result = elevenLabsTestResult {
                            Image(systemName: result.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .foregroundStyle(result.success ? .green : .red)
                            Text(result.message)
                                .font(.caption)
                        }
                        
                        Spacer()
                        
                        Button("Test Connection") {
                            testElevenLabsConnection()
                        }
                        .disabled(elevenLabsApiKey.isEmpty || isTestingElevenLabs)
                    }
                    
                    Text("Get your API key from [elevenlabs.io](https://elevenlabs.io)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Section("Default Audio Source") {
                    Picker("Default Source", selection: $defaultAudioSource) {
                        Text("ElevenLabs (Cloud)").tag("elevenlabs")
                        Text("MLX Audio (Local)").tag("mlx-audio")
                    }
                    .pickerStyle(.radioGroup)
                    
                    Text("ElevenLabs requires an API key but provides high-quality voices. MLX Audio runs locally on your Mac.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Section("MLX Audio") {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "info.circle.fill")
                            .foregroundStyle(.blue)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Local Text-to-Speech")
                                .font(.caption.bold())
                            Text("MLX Audio runs entirely on your Mac using Apple Silicon. No API key required, but requires the mlx-audio Python package to be installed.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .tabItem {
                Label("Audio", systemImage: "waveform")
            }
            
            // About
            VStack(spacing: 20) {
                Image(systemName: "film.stack")
                    .font(.system(size: 64))
                    .foregroundStyle(.blue.gradient)
                
                Text("LTX Video Generator")
                    .font(.title)
                    .bold()
                
                Text("Version 2.3.9")
                    .foregroundStyle(.secondary)
                
                Divider()
                    .frame(width: 200)
                
                VStack(spacing: 8) {
                    Text("Powered by LTX-2 from Lightricks")
                    Link("https://github.com/Lightricks/LTX-2",
                         destination: URL(string: "https://github.com/Lightricks/LTX-2")!)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                
                Spacer()
            }
            .padding(40)
            .tabItem {
                Label("About", systemImage: "info.circle")
            }
        }
        .frame(width: 550, height: 450)
        .sheet(isPresented: $showPathPicker) {
            DetectedPathsView(paths: detectedPaths, selectedPath: $pythonPath, isPresented: $showPathPicker)
        }
    }
    
    private func pathTypeIcon(_ type: PythonEnvironment.PythonPathType) -> String {
        switch type {
        case .executable: return "terminal"
        case .dylib: return "shippingbox"
        case .unknown: return "questionmark.circle"
        }
    }
    
    private func pathTypeColor(_ type: PythonEnvironment.PythonPathType) -> Color {
        switch type {
        case .executable: return .blue
        case .dylib: return .purple
        case .unknown: return .orange
        }
    }
    
    private func pathTypeDescription(_ type: PythonEnvironment.PythonPathType) -> String {
        switch type {
        case .executable: return "Python executable"
        case .dylib: return "Python dynamic library"
        case .unknown: return "Unknown path type - will attempt validation"
        }
    }
    
    private func selectPythonPath() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.message = "Select Python executable (python3) or library (libpython*.dylib)"
        panel.prompt = "Select"
        
        // Start in a reasonable location
        if let homeDir = FileManager.default.urls(for: .userDirectory, in: .localDomainMask).first {
            panel.directoryURL = homeDir
        }
        
        if panel.runModal() == .OK, let url = panel.url {
            pythonPath = url.path
            pythonStatus = nil
            pythonDetails = nil
        }
    }
    
    private func selectOutputDirectory() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        
        if panel.runModal() == .OK, let url = panel.url {
            outputDirectory = url.path
        }
    }
    
    private func openOutputDirectory() {
        let path = outputDirectory.isEmpty
            ? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                .appendingPathComponent("LTXVideoGenerator/Videos").path
            : outputDirectory
        
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: path)
    }
    
    private func detectPython() {
        isDetecting = true
        pythonStatus = nil
        pythonDetails = nil
        
        Task {
            // First, discover all Python installations
            let discovered = PythonEnvironment.shared.discoverPythonPaths()
            
            // Then try to auto-detect the best one
            let (bestPath, result) = await PythonEnvironment.shared.autoDetectPython()
            
            await MainActor.run {
                detectedPaths = discovered
                
                if let path = bestPath {
                    pythonPath = path
                }
                
                pythonStatus = (result.success, result.message)
                pythonDetails = result.details
                isDetecting = false
            }
        }
    }
    
    private func validatePython() {
        isValidating = true
        pythonStatus = nil
        pythonDetails = nil
        installMessage = nil
        
        Task {
            // Use safe subprocess validation - won't crash
            let result = await PythonEnvironment.shared.validateWithSubprocess(path: pythonPath)
            
            await MainActor.run {
                pythonStatus = (result.success, result.message)
                pythonDetails = result.details
                isValidating = false
                
                // If validation succeeded and we have details, configure for PythonKit
                if result.success, let details = result.details {
                    PythonEnvironment.shared.configureForPythonKit(details: details)
                }
            }
        }
    }
    
    private func installMissingPackages(pythonPath: String, packages: [String]) {
        isInstalling = true
        installMessage = nil
        
        Task {
            let result = await PythonEnvironment.shared.installPackages(pythonExecutable: pythonPath, packages: packages)
            
            await MainActor.run {
                isInstalling = false
                
                if result.success {
                    installMessage = "Installation successful! Re-validating..."
                    validatePython()
                } else {
                    installMessage = "Install failed: \(result.message)"
                }
            }
        }
    }
    
    private func createVenvAndInstall(basePython: String, packages: [String]) {
        isInstalling = true
        installMessage = "Creating virtual environment..."
        
        Task {
            let venvPath = PythonEnvironment.shared.getRecommendedVenvPath()
            let createResult = await PythonEnvironment.shared.createVirtualEnvironment(basePython: basePython, venvPath: venvPath)
            
            if createResult.success, let venvPython = createResult.pythonPath {
                await MainActor.run {
                    installMessage = "Venv created! Installing packages..."
                }
                
                // Install packages to the new venv
                let installResult = await PythonEnvironment.shared.installPackages(pythonExecutable: venvPython, packages: packages)
                
                await MainActor.run {
                    isInstalling = false
                    
                    if installResult.success {
                        // Update the Python path to use the new venv
                        pythonPath = venvPython
                        installMessage = "Virtual environment created and packages installed! Re-validating..."
                        validatePython()
                    } else {
                        installMessage = "Venv created but package install failed: \(installResult.message)"
                    }
                }
            } else {
                await MainActor.run {
                    isInstalling = false
                    installMessage = "Failed to create venv: \(createResult.message)"
                }
            }
        }
    }
    
    private func testElevenLabsConnection() {
        isTestingElevenLabs = true
        elevenLabsTestResult = nil
        
        Task {
            do {
                let url = URL(string: "https://api.elevenlabs.io/v1/user")!
                var request = URLRequest(url: url)
                request.setValue(elevenLabsApiKey, forHTTPHeaderField: "xi-api-key")
                
                let (data, response) = try await URLSession.shared.data(for: request)
                
                await MainActor.run {
                    isTestingElevenLabs = false
                    
                    if let httpResponse = response as? HTTPURLResponse {
                        if httpResponse.statusCode == 200 {
                            // Try to parse user info
                            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                               let subscription = json["subscription"] as? [String: Any],
                               let characterCount = subscription["character_count"] as? Int,
                               let characterLimit = subscription["character_limit"] as? Int {
                                elevenLabsTestResult = (true, "Connected! \(characterCount)/\(characterLimit) characters used")
                            } else {
                                elevenLabsTestResult = (true, "Connected successfully!")
                            }
                        } else if httpResponse.statusCode == 401 {
                            elevenLabsTestResult = (false, "Invalid API key")
                        } else {
                            elevenLabsTestResult = (false, "Error: HTTP \(httpResponse.statusCode)")
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    isTestingElevenLabs = false
                    elevenLabsTestResult = (false, "Connection failed: \(error.localizedDescription)")
                }
            }
        }
    }
}

// MARK: - Detected Paths Picker View

struct DetectedPathsView: View {
    let paths: [String]
    @Binding var selectedPath: String
    @Binding var isPresented: Bool
    
    @State private var validationResults: [String: (success: Bool, version: String?)] = [:]
    @State private var isValidating = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Detected Python Installations")
                    .font(.headline)
                Spacer()
                Button("Close") {
                    isPresented = false
                }
            }
            .padding()
            
            Divider()
            
            // List of paths
            if paths.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "magnifyingglass")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No Python installations found")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List(paths, id: \.self) { path in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(shortPath(path))
                                .font(.body)
                            Text(path)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                        
                        Spacer()
                        
                        if let result = validationResults[path] {
                            if result.success {
                                HStack(spacing: 4) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(.green)
                                    if let version = result.version {
                                        Text(version)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            } else {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.red)
                            }
                        } else if isValidating {
                            ProgressView()
                                .scaleEffect(0.6)
                        }
                        
                        Button("Select") {
                            selectedPath = path
                            isPresented = false
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .frame(width: 600, height: 400)
        .task {
            await validateAllPaths()
        }
    }
    
    private func shortPath(_ path: String) -> String {
        let components = path.components(separatedBy: "/")
        if components.count > 3 {
            // Show last 3 components
            return ".../" + components.suffix(3).joined(separator: "/")
        }
        return path
    }
    
    private func validateAllPaths() async {
        isValidating = true
        
        for path in paths {
            let result = await PythonEnvironment.shared.validateWithSubprocess(path: path)
            await MainActor.run {
                validationResults[path] = (result.success, result.details?.version)
            }
        }
        
        await MainActor.run {
            isValidating = false
        }
    }
}

#Preview {
    PreferencesView()
}
