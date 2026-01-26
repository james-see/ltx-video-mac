import Foundation

/// Details about a validated Python installation
struct PythonDetails {
    let version: String
    let executablePath: String
    let dylibPath: String?
    let pythonHome: String
    let hasRequiredPackages: Bool
    let missingPackages: [String]
    let needsDiffusersGit: Bool  // True if LTX2Pipeline not available
    
    init(version: String, executablePath: String, dylibPath: String?, pythonHome: String, hasRequiredPackages: Bool, missingPackages: [String], needsDiffusersGit: Bool = false) {
        self.version = version
        self.executablePath = executablePath
        self.dylibPath = dylibPath
        self.pythonHome = pythonHome
        self.hasRequiredPackages = hasRequiredPackages
        self.missingPackages = missingPackages
        self.needsDiffusersGit = needsDiffusersGit
    }
}

/// Manages Python environment detection and validation
/// Uses subprocess-based validation to avoid PythonKit crashes
class PythonEnvironment {
    static let shared = PythonEnvironment()
    
    private(set) var isConfigured = false
    private(set) var lastValidationResult: PythonDetails?
    
    private init() {}
    
    // MARK: - Path Type Detection
    
    enum PythonPathType {
        case executable  // e.g., /usr/bin/python3, ~/venv/bin/python
        case dylib       // e.g., /opt/homebrew/.../libpython3.11.dylib
        case unknown
    }
    
    func detectPathType(_ path: String) -> PythonPathType {
        let lowercased = path.lowercased()
        if lowercased.contains("libpython") && lowercased.hasSuffix(".dylib") {
            return .dylib
        } else if FileManager.default.isExecutableFile(atPath: path) {
            return .executable
        } else if path.contains("/bin/python") {
            return .executable
        }
        return .unknown
    }
    
    // MARK: - Path Conversion
    
    /// Convert dylib path to executable path
    /// e.g., /opt/homebrew/opt/python@3.11/.../lib/libpython3.11.dylib -> /opt/homebrew/opt/python@3.11/bin/python3
    func dylibToExecutable(_ dylibPath: String) -> String? {
        // Pattern: .../lib/libpython3.X.dylib or .../lib/python3.X/lib-dynload/../libpython3.X.dylib
        guard let pythonHome = extractPythonHome(from: dylibPath) else {
            return nil
        }
        
        // Try common executable locations relative to python home
        let possibleExecutables = [
            "\(pythonHome)/bin/python3",
            "\(pythonHome)/bin/python",
            "\(pythonHome)/Python.framework/Versions/Current/bin/python3"
        ]
        
        for exec in possibleExecutables {
            if FileManager.default.isExecutableFile(atPath: exec) {
                return exec
            }
        }
        
        return nil
    }
    
    /// Convert executable path to dylib path
    /// e.g., /opt/homebrew/bin/python3 -> /opt/homebrew/.../libpython3.11.dylib
    func executableToDylib(_ execPath: String) -> String? {
        // First, resolve symlinks to get the real path
        let realPath = (try? FileManager.default.destinationOfSymbolicLink(atPath: execPath)) ?? execPath
        
        // Try to find dylib by running python to get its paths
        let script = """
        import sys
        import sysconfig
        print(sysconfig.get_config_var('LIBDIR') or '')
        print(sysconfig.get_config_var('LDLIBRARY') or '')
        print(sys.prefix)
        """
        
        if let output = runPythonSync(executable: execPath, script: script) {
            let lines = output.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
            if lines.count >= 3 {
                let libDir = lines[0]
                let ldLibrary = lines[1]
                let prefix = lines[2]
                
                // Try direct path first
                if !libDir.isEmpty && !ldLibrary.isEmpty {
                    let directPath = "\(libDir)/\(ldLibrary)"
                    if FileManager.default.fileExists(atPath: directPath) {
                        return directPath
                    }
                }
                
                // Try common patterns
                let version = extractPythonVersion(from: execPath) ?? "3.11"
                let searchPaths = [
                    "\(prefix)/lib/libpython\(version).dylib",
                    "\(prefix)/lib/python\(version)/lib-dynload/../libpython\(version).dylib",
                    "\(prefix)/Python.framework/Versions/\(version)/lib/libpython\(version).dylib"
                ]
                
                for path in searchPaths {
                    let resolved = (path as NSString).standardizingPath
                    if FileManager.default.fileExists(atPath: resolved) {
                        return resolved
                    }
                }
            }
        }
        
        return nil
    }
    
    /// Extract Python home directory from dylib path
    func extractPythonHome(from path: String) -> String? {
        // Handle various patterns:
        // /path/to/python/lib/libpython3.X.dylib -> /path/to/python
        // /path/to/Frameworks/Python.framework/Versions/3.X/lib/libpython3.X.dylib -> /path/to/Frameworks/Python.framework/Versions/3.X
        
        if let range = path.range(of: "/lib/libpython", options: .backwards) {
            return String(path[..<range.lowerBound])
        }
        
        if let range = path.range(of: "/Python.framework/") {
            // Find the version directory
            let afterFramework = path[range.upperBound...]
            if let versionEnd = afterFramework.range(of: "/lib/") {
                let versionPart = afterFramework[..<versionEnd.lowerBound]
                return String(path[..<range.upperBound]) + String(versionPart)
            }
        }
        
        return nil
    }
    
    /// Extract Python version from path (e.g., "3.11" from path containing python3.11)
    func extractPythonVersion(from path: String) -> String? {
        let pattern = try? NSRegularExpression(pattern: "python@?(\\d+\\.\\d+)")
        if let match = pattern?.firstMatch(in: path, range: NSRange(path.startIndex..., in: path)),
           let range = Range(match.range(at: 1), in: path) {
            return String(path[range])
        }
        
        // Try libpython pattern
        let libPattern = try? NSRegularExpression(pattern: "libpython(\\d+\\.\\d+)")
        if let match = libPattern?.firstMatch(in: path, range: NSRange(path.startIndex..., in: path)),
           let range = Range(match.range(at: 1), in: path) {
            return String(path[range])
        }
        
        return nil
    }
    
    // MARK: - Subprocess Validation (Safe - Won't Crash)
    
    /// Validate Python installation using subprocess - safe and won't crash the app
    func validateWithSubprocess(path: String) async -> (success: Bool, message: String, details: PythonDetails?) {
        // Step 1: Check file exists
        guard FileManager.default.fileExists(atPath: path) else {
            return (false, "File not found: \(path)", nil)
        }
        
        // Step 2: Determine path type and get executable
        let pathType = detectPathType(path)
        let executablePath: String
        let dylibPath: String?
        
        switch pathType {
        case .dylib:
            guard let exec = dylibToExecutable(path) else {
                return (false, "Could not find Python executable for dylib. Try providing the python3 executable path instead.", nil)
            }
            executablePath = exec
            dylibPath = path
        case .executable:
            executablePath = path
            dylibPath = executableToDylib(path)
        case .unknown:
            // Try treating as executable first
            if FileManager.default.isExecutableFile(atPath: path) {
                executablePath = path
                dylibPath = executableToDylib(path)
            } else {
                return (false, "Path is neither a valid Python executable nor a dylib: \(path)", nil)
            }
        }
        
        // Step 3: Verify executable works
        guard FileManager.default.isExecutableFile(atPath: executablePath) else {
            return (false, "Python executable not found or not executable: \(executablePath)", nil)
        }
        
        // Step 4: Get Python version via subprocess
        let versionScript = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        guard let versionOutput = runPythonSync(executable: executablePath, script: versionScript),
              !versionOutput.isEmpty else {
            return (false, "Failed to get Python version. The executable may be corrupted or incompatible.", nil)
        }
        
        let version = versionOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Step 5: Check Python version is 3.10+
        let versionParts = version.split(separator: ".").compactMap { Int($0) }
        if versionParts.count >= 2 {
            let major = versionParts[0]
            let minor = versionParts[1]
            if major < 3 || (major == 3 && minor < 10) {
                return (false, "Python \(version) is too old. LTX Video requires Python 3.10 or newer.", nil)
            }
        }
        
        // Step 6: Get Python home
        let homeScript = "import sys; print(sys.prefix)"
        let pythonHome = runPythonSync(executable: executablePath, script: homeScript)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        
        // Step 7: Check for required packages
        let requiredPackages = ["torch", "diffusers", "av"]
        var missingPackages: [String] = []
        
        for pkg in requiredPackages {
            let checkScript = "import \(pkg); print('OK')"
            let result = runPythonSync(executable: executablePath, script: checkScript)
            if result == nil || !result!.contains("OK") {
                missingPackages.append(pkg)
            }
        }
        
        // Step 8: Check specifically for LTX2Pipeline (requires diffusers from git)
        var needsDiffusersGit = false
        if missingPackages.isEmpty {
            let ltx2Check = "from diffusers import LTX2Pipeline; print('OK')"
            let ltx2Result = runPythonSync(executable: executablePath, script: ltx2Check)
            if ltx2Result == nil || !ltx2Result!.contains("OK") {
                needsDiffusersGit = true
            }
        }
        
        let hasRequired = missingPackages.isEmpty && !needsDiffusersGit
        
        let details = PythonDetails(
            version: version,
            executablePath: executablePath,
            dylibPath: dylibPath,
            pythonHome: pythonHome,
            hasRequiredPackages: hasRequired,
            missingPackages: missingPackages,
            needsDiffusersGit: needsDiffusersGit
        )
        
        if !missingPackages.isEmpty {
            let pipCommand = "pip install \(missingPackages.joined(separator: " "))"
            return (false, "Python \(version) found but missing packages: \(missingPackages.joined(separator: ", ")). Run: \(pipCommand)", details)
        }
        
        if needsDiffusersGit {
            return (false, "Python \(version) found but LTX-2 requires diffusers from git.", details)
        }
        
        return (true, "Python \(version) configured successfully with all required packages", details)
    }
    
    /// Run Python script synchronously and return output (safe - uses subprocess)
    private func runPythonSync(executable: String, script: String) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = ["-c", script]
        
        // Minimal clean environment
        var env: [String: String] = [:]
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"
        env["HOME"] = ProcessInfo.processInfo.environment["HOME"] ?? ""
        env["USER"] = ProcessInfo.processInfo.environment["USER"] ?? ""
        process.environment = env
        
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: outputData, encoding: .utf8)
            
            if process.terminationStatus == 0 {
                return output
            }
            return nil
        } catch {
            return nil
        }
    }
    
    // MARK: - Comprehensive Path Discovery
    
    /// Find Python installations on the system
    func discoverPythonPaths() -> [String] {
        var paths: [String] = []
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser.path
        
        // Homebrew Apple Silicon
        let homebrewArm = "/opt/homebrew"
        for version in ["3.13", "3.12", "3.11", "3.10"] {
            let exec = "\(homebrewArm)/opt/python@\(version)/bin/python\(version)"
            if fm.isExecutableFile(atPath: exec) {
                paths.append(exec)
            }
            // Also check Frameworks path for dylib
            let dylib = "\(homebrewArm)/opt/python@\(version)/Frameworks/Python.framework/Versions/\(version)/lib/libpython\(version).dylib"
            if fm.fileExists(atPath: dylib) && !paths.contains(where: { extractPythonVersion(from: $0) == version }) {
                paths.append(dylib)
            }
        }
        
        // Homebrew Intel
        let homebrewIntel = "/usr/local"
        for version in ["3.13", "3.12", "3.11", "3.10"] {
            let exec = "\(homebrewIntel)/opt/python@\(version)/bin/python\(version)"
            if fm.isExecutableFile(atPath: exec) {
                paths.append(exec)
            }
            let dylib = "\(homebrewIntel)/opt/python@\(version)/Frameworks/Python.framework/Versions/\(version)/lib/libpython\(version).dylib"
            if fm.fileExists(atPath: dylib) && !paths.contains(where: { extractPythonVersion(from: $0) == version }) {
                paths.append(dylib)
            }
        }
        
        // pyenv versions
        let pyenvRoot = "\(home)/.pyenv/versions"
        if let versions = try? fm.contentsOfDirectory(atPath: pyenvRoot) {
            for version in versions.sorted().reversed() {
                let exec = "\(pyenvRoot)/\(version)/bin/python3"
                if fm.isExecutableFile(atPath: exec) {
                    paths.append(exec)
                }
            }
        }
        
        // Conda/Miniconda/Anaconda
        let condaPaths = [
            "\(home)/miniconda3",
            "\(home)/anaconda3",
            "\(home)/miniforge3",
            "\(home)/mambaforge",
            "/opt/miniconda3",
            "/opt/anaconda3"
        ]
        for condaBase in condaPaths {
            let exec = "\(condaBase)/bin/python3"
            if fm.isExecutableFile(atPath: exec) {
                paths.append(exec)
            }
            // Also check envs
            let envsPath = "\(condaBase)/envs"
            if let envs = try? fm.contentsOfDirectory(atPath: envsPath) {
                for env in envs {
                    let envExec = "\(envsPath)/\(env)/bin/python3"
                    if fm.isExecutableFile(atPath: envExec) {
                        paths.append(envExec)
                    }
                }
            }
        }
        
        // Official Python.org installations
        let libraryFrameworks = "/Library/Frameworks/Python.framework/Versions"
        if let versions = try? fm.contentsOfDirectory(atPath: libraryFrameworks) {
            for version in versions.sorted().reversed() where version != "Current" {
                let exec = "\(libraryFrameworks)/\(version)/bin/python3"
                if fm.isExecutableFile(atPath: exec) {
                    paths.append(exec)
                }
            }
        }
        
        // Common virtualenv locations in home directory
        let commonVenvNames = ["venv", "env", ".venv", "ltx-venv", "ltx-env", "pytorch-env"]
        for venvName in commonVenvNames {
            let exec = "\(home)/\(venvName)/bin/python3"
            if fm.isExecutableFile(atPath: exec) {
                paths.append(exec)
            }
        }
        
        // System Python (usually not recommended but might work)
        if fm.isExecutableFile(atPath: "/usr/bin/python3") {
            paths.append("/usr/bin/python3")
        }
        
        // Generic homebrew python3
        for path in ["\(homebrewArm)/bin/python3", "\(homebrewIntel)/bin/python3"] {
            if fm.isExecutableFile(atPath: path) && !paths.contains(path) {
                paths.append(path)
            }
        }
        
        return paths
    }
    
    /// Auto-detect and validate the best Python installation
    func autoDetectPython() async -> (path: String?, result: (success: Bool, message: String, details: PythonDetails?)) {
        let candidates = discoverPythonPaths()
        
        if candidates.isEmpty {
            return (nil, (false, "No Python installations found. Please install Python 3.10+ with PyTorch and diffusers.", nil))
        }
        
        // Try each candidate, prefer ones with required packages
        var bestCandidate: (path: String, result: (success: Bool, message: String, details: PythonDetails?))? = nil
        
        for path in candidates {
            let result = await validateWithSubprocess(path: path)
            
            if result.success {
                // Found a fully working installation
                return (path, result)
            }
            
            // Keep track of best partial match (has Python, just missing packages)
            if result.details != nil && bestCandidate == nil {
                bestCandidate = (path, result)
            }
        }
        
        // Return best partial match if no fully working installation found
        if let best = bestCandidate {
            return (best.path, best.result)
        }
        
        return (nil, (false, "Found \(candidates.count) Python installation(s) but none are compatible. Please install Python 3.10+ with PyTorch and diffusers.", nil))
    }
    
    // MARK: - Package Installation
    
    /// Install diffusers from git for LTX-2 support
    /// Returns (success, output/error message)
    func installDiffusersFromGit(pythonExecutable: String) async -> (success: Bool, message: String) {
        // Find pip relative to python executable
        let pipPath = pythonExecutable.replacingOccurrences(of: "/python3", with: "/pip3")
            .replacingOccurrences(of: "/python", with: "/pip")
        
        // Try pip3 first, fall back to python -m pip
        let usePipModule = !FileManager.default.isExecutableFile(atPath: pipPath)
        
        let process = Process()
        if usePipModule {
            process.executableURL = URL(fileURLWithPath: pythonExecutable)
            process.arguments = ["-m", "pip", "install", "--upgrade", "git+https://github.com/huggingface/diffusers.git"]
        } else {
            process.executableURL = URL(fileURLWithPath: pipPath)
            process.arguments = ["install", "--upgrade", "git+https://github.com/huggingface/diffusers.git"]
        }
        
        // Set up environment
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:" + (env["PATH"] ?? "")
        process.environment = env
        
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
            let errorData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: outputData, encoding: .utf8) ?? ""
            let errorOutput = String(data: errorData, encoding: .utf8) ?? ""
            
            if process.terminationStatus == 0 {
                return (true, "Successfully installed diffusers from git.\n\(output)")
            } else {
                return (false, "Installation failed (exit \(process.terminationStatus)):\n\(errorOutput)\n\(output)")
            }
        } catch {
            return (false, "Failed to run pip: \(error.localizedDescription)")
        }
    }
    
    /// Install missing packages using pip
    func installPackages(pythonExecutable: String, packages: [String]) async -> (success: Bool, message: String) {
        let pipPath = pythonExecutable.replacingOccurrences(of: "/python3", with: "/pip3")
            .replacingOccurrences(of: "/python", with: "/pip")
        
        let usePipModule = !FileManager.default.isExecutableFile(atPath: pipPath)
        
        let process = Process()
        if usePipModule {
            process.executableURL = URL(fileURLWithPath: pythonExecutable)
            process.arguments = ["-m", "pip", "install"] + packages
        } else {
            process.executableURL = URL(fileURLWithPath: pipPath)
            process.arguments = ["install"] + packages
        }
        
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:" + (env["PATH"] ?? "")
        process.environment = env
        
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        
        do {
            try process.run()
            process.waitUntilExit()
            
            let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
            let errorData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: outputData, encoding: .utf8) ?? ""
            let errorOutput = String(data: errorData, encoding: .utf8) ?? ""
            
            if process.terminationStatus == 0 {
                return (true, "Successfully installed \(packages.joined(separator: ", ")).\n\(output)")
            } else {
                return (false, "Installation failed:\n\(errorOutput)\n\(output)")
            }
        } catch {
            return (false, "Failed to run pip: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Configuration (for PythonKit - only call after subprocess validation)
    
    /// Configure environment variables for PythonKit (only call after successful subprocess validation)
    func configureForPythonKit(details: PythonDetails) {
        guard let dylibPath = details.dylibPath else {
            print("Warning: No dylib path available for PythonKit configuration")
            return
        }
        
        setenv("PYTHON_LIBRARY", dylibPath, 1)
        setenv("PYTHONHOME", details.pythonHome, 1)
        
        // Detect Python version for paths
        let version = extractPythonVersion(from: dylibPath) ?? "3.11"
        let sitePackages = "\(details.pythonHome)/lib/python\(version)/site-packages"
        let libPath = "\(details.pythonHome)/lib/python\(version)"
        let pythonPathEnv = "\(sitePackages):\(libPath)"
        setenv("PYTHONPATH", pythonPathEnv, 1)
        
        isConfigured = true
        lastValidationResult = details
    }
    
    // MARK: - Legacy API (for backward compatibility)
    
    /// Legacy configure method - now does nothing on startup to prevent crashes
    func configure() {
        // Intentionally empty - we now defer configuration until after subprocess validation
        // This prevents crashes from PythonKit trying to load an invalid Python
    }
    
    /// Legacy reconfigure - use validateWithSubprocess instead
    func reconfigure(withPath path: String) {
        // For backward compatibility, but prefer using validateWithSubprocess + configureForPythonKit
        let pathType = detectPathType(path)
        
        var dylibPath = path
        var pythonHome = ""
        
        if pathType == .executable {
            if let dylib = executableToDylib(path) {
                dylibPath = dylib
            }
        }
        
        if let home = extractPythonHome(from: dylibPath) {
            pythonHome = home
        }
        
        if !dylibPath.isEmpty {
            setenv("PYTHON_LIBRARY", dylibPath, 1)
        }
        
        if !pythonHome.isEmpty {
            setenv("PYTHONHOME", pythonHome, 1)
            
            let version = extractPythonVersion(from: dylibPath) ?? "3.11"
            let sitePackages = "\(pythonHome)/lib/python\(version)/site-packages"
            let libPath = "\(pythonHome)/lib/python\(version)"
            setenv("PYTHONPATH", "\(sitePackages):\(libPath)", 1)
        }
    }
    
    /// Legacy validation - now uses safe subprocess validation
    func validatePythonSetup() -> (success: Bool, message: String) {
        // Get saved path
        guard let path = UserDefaults.standard.string(forKey: "pythonPath"), !path.isEmpty else {
            return (false, "No Python path configured. Please set a Python path in Preferences.")
        }
        
        // Use synchronous version for legacy compatibility
        let result = validateWithSubprocessSync(path: path)
        return (result.success, result.message)
    }
    
    /// Synchronous subprocess validation for legacy API
    private func validateWithSubprocessSync(path: String) -> (success: Bool, message: String, details: PythonDetails?) {
        // Run async validation on a background thread and wait
        var result: (success: Bool, message: String, details: PythonDetails?) = (false, "Validation timeout", nil)
        
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            result = await validateWithSubprocess(path: path)
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + 30)
        
        return result
    }
}
