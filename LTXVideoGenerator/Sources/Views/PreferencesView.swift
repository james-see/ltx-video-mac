import SwiftUI

struct PreferencesView: View {
    @AppStorage("pythonPath") private var pythonPath = ""
    @AppStorage("outputDirectory") private var outputDirectory = ""
    @AppStorage("autoLoadModel") private var autoLoadModel = false
    @AppStorage("keepCompletedInQueue") private var keepCompletedInQueue = false
    
    @State private var pythonStatus: (success: Bool, message: String)?
    @State private var pythonDetails: PythonDetails?
    @State private var isValidating = false
    @State private var isDetecting = false
    @State private var detectedPaths: [String] = []
    @State private var showPathPicker = false
    
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
                    if isValidating || isDetecting {
                        HStack {
                            ProgressView()
                                .scaleEffect(0.7)
                            Text(isDetecting ? "Searching for Python installations..." : "Validating Python setup...")
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
                                    if let dylib = details.dylibPath {
                                        Text("Library: \(dylib)")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                .padding(.leading, 20)
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
                    Toggle("Auto-load model on startup", isOn: $autoLoadModel)
                    
                    Text("The LTX-2 model will be downloaded on first use (~4GB)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
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
            
            // About
            VStack(spacing: 20) {
                Image(systemName: "film.stack")
                    .font(.system(size: 64))
                    .foregroundStyle(.blue.gradient)
                
                Text("LTX Video Generator")
                    .font(.title)
                    .bold()
                
                Text("Version 1.0.3")
                    .foregroundStyle(.secondary)
                
                Divider()
                    .frame(width: 200)
                
                VStack(spacing: 8) {
                    Text("Powered by LTX-2 from Lightricks")
                    Link("https://github.com/Lightricks/LTX-Video",
                         destination: URL(string: "https://github.com/Lightricks/LTX-Video")!)
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
