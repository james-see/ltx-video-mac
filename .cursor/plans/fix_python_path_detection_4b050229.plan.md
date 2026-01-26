---
name: Fix Python Path Detection
overview: Fix crashes when Python cannot be detected by adding pre-validation, subprocess-based testing, and better path detection to avoid PythonKit fatal errors.
todos:
  - id: subprocess-validation
    content: Add subprocess-based Python validation in PythonEnvironment.swift that tests Python without using PythonKit
    status: completed
  - id: expand-detection
    content: Expand findPythonPath() with comprehensive path discovery (Homebrew, pyenv, conda, system, venvs)
    status: completed
  - id: path-type-support
    content: Add support for both dylib and executable paths with auto-conversion between them
    status: completed
  - id: update-prefs-ui
    content: Update PreferencesView to use safe subprocess validation and show clearer status messages
    status: completed
  - id: defer-init
    content: Remove eager PythonEnvironment.configure() from app init to prevent startup crashes
    status: completed
isProject: false
---

# Fix Python Path Detection and Validation Crashes

## Problem Analysis

The crash occurs because PythonKit's `Python.import()` calls `fatalError` internally when Python isn't properly configured, which bypasses Swift's `do/catch` error handling. The crash trace shows `swift_unexpectedError` being triggered in an async context.

**Key issues in current implementation:**

1. [PythonEnvironment.swift](LTXVideoGenerator/Sources/PythonEnvironment.swift) - Line 69-91: `validatePythonSetup()` uses `try Python.import("sys")` which can fatal crash before error handling
2. Hardcoded path `/Users/jc/.pyenv/versions/3.12.11/lib/libpython3.12.dylib` (line 52) doesn't exist on user machines
3. Limited auto-detection paths (only 4-6 locations checked)
4. Confusing UX: help text mentions dylib but some users try python executables

## Solution

### 1. Add Subprocess-Based Pre-Validation

Before calling any PythonKit functions, validate Python using a subprocess call which cannot crash the app:

```swift
// New method in PythonEnvironment.swift
func validateWithSubprocess(pythonPath: String) async -> (success: Bool, message: String, details: PythonDetails?) {
    // 1. Check file exists
    // 2. Determine if it's dylib or executable, convert as needed
    // 3. Run `python -c "import sys; print(sys.version)"` via Process
    // 4. Return validation result safely
}
```

### 2. Improve Path Detection

Expand [PythonEnvironment.swift](LTXVideoGenerator/Sources/PythonEnvironment.swift) `findPythonPath()` to check more common locations:

- Homebrew (Intel and Apple Silicon)
- pyenv installations
- Conda/Anaconda
- Official Python.org installations
- System Python
- Custom virtualenvs in home directory

### 3. Support Both Path Types

Accept both:

- **dylib paths**: `/opt/homebrew/opt/python@3.11/.../libpython3.11.dylib`
- **executable paths**: `/opt/homebrew/bin/python3`, `~/myenv/bin/python3`

Auto-detect type and convert between them as needed for PythonKit vs subprocess execution.

### 4. Update PreferencesView Validation

In [PreferencesView.swift](LTXVideoGenerator/Sources/Views/PreferencesView.swift):

- Use subprocess validation first (safe, won't crash)
- Only call PythonKit after subprocess confirms Python works
- Show clearer status messages during validation
- Add path type indicator (dylib vs executable)

### 5. Graceful App Startup

In [LTXVideoGeneratorApp.swift](LTXVideoGenerator/Sources/LTXVideoGeneratorApp.swift):

- Don't call `PythonEnvironment.shared.configure()` during init
- Defer Python initialization until actually needed
- Add `isPythonSafelyConfigured` check that uses subprocess validation

## Files to Modify

- **[PythonEnvironment.swift](LTXVideoGenerator/Sources/PythonEnvironment.swift)**: Complete rewrite of validation logic
- **[PreferencesView.swift](LTXVideoGenerator/Sources/Views/PreferencesView.swift)**: Update `detectPython()` and `validatePython()`
- **[LTXVideoGeneratorApp.swift](LTXVideoGenerator/Sources/LTXVideoGeneratorApp.swift)**: Remove eager Python init
- **[LTXBridge.swift](LTXVideoGenerator/Sources/Services/LTXBridge.swift)**: Minor updates to path handling

## Python Path Discovery Flow

```mermaid
flowchart TD
    Start[User clicks Detect] --> CheckSaved{Saved path exists?}
    CheckSaved -->|Yes| ValidateSaved[Validate saved path]
    CheckSaved -->|No| SearchPaths[Search common paths]
    
    SearchPaths --> Homebrew[Homebrew ARM/Intel]
    SearchPaths --> Pyenv[pyenv versions]
    SearchPaths --> Conda[Conda/Anaconda]
    SearchPaths --> System[System Python]
    SearchPaths --> Official[Python.org install]
    
    Homebrew --> CollectPaths[Collect found paths]
    Pyenv --> CollectPaths
    Conda --> CollectPaths
    System --> CollectPaths
    Official --> CollectPaths
    
    CollectPaths --> SubprocessTest[Test each via subprocess]
    SubprocessTest --> CheckPackages[Check torch/diffusers]
    CheckPackages --> ReturnBest[Return best match]
    ValidateSaved --> SubprocessTest
```

## Validation Flow

```mermaid
flowchart TD
    Validate[Validate Python Setup] --> CheckFile{File exists?}
    CheckFile -->|No| ErrorNotFound[Error: Path not found]
    CheckFile -->|Yes| DetectType{Detect path type}
    
    DetectType -->|dylib| ExtractExec[Extract executable path]
    DetectType -->|executable| ExtractDylib[Locate dylib]
    
    ExtractExec --> RunSubprocess[Run python subprocess]
    ExtractDylib --> RunSubprocess
    
    RunSubprocess -->|Fail| ErrorPython[Error: Python failed]
    RunSubprocess -->|Success| CheckVersion{Python 3.10+?}
    
    CheckVersion -->|No| ErrorVersion[Error: Version too old]
    CheckVersion -->|Yes| CheckPackages[Check packages via subprocess]
    
    CheckPackages --> MissingPkgs{Missing packages?}
    MissingPkgs -->|Yes| ShowInstall[Show pip install command]
    MissingPkgs -->|No| Success[Success: Ready to use]
```