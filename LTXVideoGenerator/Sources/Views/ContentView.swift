import SwiftUI

struct ContentView: View {
    @EnvironmentObject var generationService: GenerationService
    @EnvironmentObject var historyManager: HistoryManager
    @EnvironmentObject var presetManager: PresetManager

    // Persisted across app launches (issue #51).
    @AppStorage(SessionSettings.promptKey) private var prompt = ""
    @AppStorage(SessionSettings.negativePromptKey) private var negativePrompt = ""
    @AppStorage(SessionSettings.voiceoverTextKey) private var voiceoverText = ""
    @AppStorage(SessionSettings.selectedTabKey) private var selectedTab: Tab = .generate

    // GenerationParameters is a struct, so we persist it as JSON via UserDefaults
    // and bridge through a `@State` binding the children already expect.
    @State private var parameters: GenerationParameters = SessionSettings.loadParameters()
    @State private var showError = false

    /// Keep the error alert small enough that the OK button stays on screen.
    /// Generation failures can carry thousands of characters of Python traceback,
    /// which previously pushed the button below the bottom of the display. The
    /// full detail is always written to /tmp/ltx_generation.log.
    static func boundedAlertMessage(_ message: String) -> String {
        let limit = 500
        guard message.count > limit else { return message }
        let head = message.prefix(limit).trimmingCharacters(in: .whitespacesAndNewlines)
        return head + "…\n\n(Message truncated — full details in /tmp/ltx_generation.log)"
    }

    enum Tab: String, CaseIterable {
        case generate = "Generate"
        case history = "Video Archive"
    }
    
    var body: some View {
        NavigationSplitView {
            sidebarContent
        } detail: {
            detailContent
        }
        // Issue #52: relax the hard minimum so the window can shrink on small
        // displays (e.g. 13" MacBook Air with a non-default text size). Inner
        // panes are scrollable, so the Generate button stays reachable.
        .frame(
            minWidth: 900,
            idealWidth: 1280,
            minHeight: 480,
            idealHeight: 800,
            maxHeight: .infinity
        )
        .alert("Error", isPresented: $showError, presenting: generationService.error) { _ in
            Button("OK", role: .cancel) {
                generationService.clearError()
            }
        } message: { error in
            Text(Self.boundedAlertMessage(error.localizedDescription))
        }
        .onChange(of: generationService.error) { _, newError in
            showError = newError != nil
        }
        .onChange(of: parameters) { _, newValue in
            SessionSettings.saveParameters(newValue)
        }
    }
    
    private var sidebarContent: some View {
        // Issue #52: allow the sidebar to scroll vertically when the window is
        // very short, so QueueView and ModelStatusView are both still reachable.
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: 0) {
                tabSelector
                Divider()
                QueueView()
                    .frame(minHeight: 160, maxHeight: 300)
                Divider()
                ModelStatusView()
                    .padding()
            }
        }
        .frame(width: 320)
        .background(Color(nsColor: .windowBackgroundColor))
    }
    
    private var tabSelector: some View {
        VStack(spacing: 4) {
            ForEach(Tab.allCases, id: \.self) { tab in
                SidebarButton(
                    title: tab.rawValue,
                    icon: tab == .generate ? "wand.and.stars" : "film.stack",
                    isSelected: selectedTab == tab,
                    badge: tab == .generate ? generationService.queue.count : nil
                ) {
                    selectedTab = tab
                }
            }
        }
        .padding()
    }
    
    @ViewBuilder
    private var detailContent: some View {
        switch selectedTab {
        case .generate:
            GenerateView(
                prompt: $prompt,
                negativePrompt: $negativePrompt,
                voiceoverText: $voiceoverText,
                parameters: $parameters
            )
        case .history:
            HistoryView()
        }
    }
}

struct SidebarButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    var badge: Int?
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            buttonContent
        }
        .buttonStyle(.plain)
        .foregroundStyle(isSelected ? .primary : .secondary)
    }
    
    private var buttonContent: some View {
        HStack {
            Image(systemName: icon)
                .frame(width: 24)
            Text(title)
            Spacer()
            badgeView
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? Color.accentColor.opacity(0.15) : Color.clear)
        )
        .contentShape(Rectangle())
    }
    
    @ViewBuilder
    private var badgeView: some View {
        if let badge = badge, badge > 0 {
            Text("\(badge)")
                .font(.caption2.bold())
                .foregroundStyle(.white)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(Color.blue))
        }
    }
}

struct ModelStatusView: View {
    @EnvironmentObject var generationService: GenerationService
    @StateObject private var apiServer = APIServer.shared
    @AppStorage(LTXModelCatalog.selectedModelIDKey) private var selectedModelID = LTXModelCatalog.defaultModelID

    private var selectedModel: LTXModel {
        LTXModelCatalog.resolvedModel(id: selectedModelID)
    }
    
    var body: some View {
        VStack(spacing: 8) {
            // Model variant indicator
            HStack(spacing: 6) {
                Image(systemName: "cpu")
                    .foregroundStyle(.blue)
                Text(selectedModel.displayName)
                    .font(.caption.bold())
                Spacer()
                Text("MLX")
                    .font(.caption2.monospaced())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(Color.orange.opacity(0.2)))
                    .foregroundStyle(.orange)
            }
            
            // Model status
            HStack(spacing: 8) {
                Circle()
                    .fill(generationService.isModelLoaded ? Color.green : Color.gray)
                    .frame(width: 8, height: 8)
                Text(generationService.isModelLoaded ? "Environment Ready" : "Environment Not Ready")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if !generationService.isModelLoaded {
                    Button("Load") {
                        Task { await generationService.loadModel() }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                } else {
                    Button("Unload") {
                        Task { await generationService.unloadModel() }
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                }
            }

            if generationService.isModelLoaded {
                Text("Model files are downloaded on first generation if cache is missing.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            Divider()
            
            // API Server toggle
            HStack(spacing: 8) {
                Circle()
                    .fill(apiServer.isRunning ? Color.green : Color.gray)
                    .frame(width: 8, height: 8)
                Text("API Server")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if apiServer.isRunning {
                    Text(":\(apiServer.port)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.tertiary)
                }
                Toggle("", isOn: Binding(
                    get: { apiServer.isRunning },
                    set: { newValue in
                        if newValue {
                            apiServer.start(generationService: generationService)
                        } else {
                            apiServer.stop()
                        }
                    }
                ))
                .toggleStyle(.switch)
                .controlSize(.small)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
    }
}

/// Centralized keys + JSON helpers for the session-level settings that issue
/// #51 asked us to persist between launches. Plain values (Bool, String, raw
/// representable enums) live as `@AppStorage` directly in the views; complex
/// structs round-trip through JSON in `UserDefaults`.
enum SessionSettings {
    static let promptKey = "session.prompt"
    static let negativePromptKey = "session.negativePrompt"
    static let voiceoverTextKey = "session.voiceoverText"
    static let selectedTabKey = "session.selectedTab"
    static let parametersKey = "session.generationParameters"

    static let voiceoverSourceKey = "session.voiceoverSource"
    static let elevenLabsVoiceKey = "session.elevenLabsVoice"
    static let mlxVoiceKey = "session.mlxVoice"
    static let musicEnabledKey = "session.musicEnabled"
    static let musicGenreKey = "session.musicGenre"
    static let disableAudioKey = "session.disableAudio"
    static let gemmaRepetitionPenaltyKey = "session.gemmaRepetitionPenalty"
    static let gemmaTopPKey = "session.gemmaTopP"

    /// Keys that "Reset to defaults" wipes. Excludes app-level prefs that the
    /// user explicitly configured (Python path, ElevenLabs key, output dir).
    static let resettableKeys: [String] = [
        promptKey,
        negativePromptKey,
        voiceoverTextKey,
        selectedTabKey,
        parametersKey,
        voiceoverSourceKey,
        elevenLabsVoiceKey,
        mlxVoiceKey,
        musicEnabledKey,
        musicGenreKey,
        disableAudioKey,
        gemmaRepetitionPenaltyKey,
        gemmaTopPKey,
        "sourceImagePath",
        "enableGemmaPromptEnhancement",
        "saveAudioTrackSeparately",
        "keepCompletedInQueue",
        "autoLoadModel",
        "defaultAudioSource",
        LTXModelCatalog.selectedModelIDKey,
        LTXTextEncoderCatalog.selectedTextEncoderIDKey,
        LTXTextEncoderCatalog.customTextEncoderRepoKey,
    ]

    static func loadParameters() -> GenerationParameters {
        guard let data = UserDefaults.standard.data(forKey: parametersKey) else {
            return .default
        }
        if let decoded = try? JSONDecoder().decode(GenerationParameters.self, from: data) {
            return decoded
        }
        return .default
    }

    static func saveParameters(_ value: GenerationParameters) {
        if let data = try? JSONEncoder().encode(value) {
            UserDefaults.standard.set(data, forKey: parametersKey)
        }
    }

    static func resetAll() {
        let defaults = UserDefaults.standard
        for key in resettableKeys {
            defaults.removeObject(forKey: key)
        }
    }
}

struct GenerateView: View {
    @Binding var prompt: String
    @Binding var negativePrompt: String
    @Binding var voiceoverText: String
    @Binding var parameters: GenerationParameters
    
    var body: some View {
        HSplitView {
            promptArea
            parametersPanel
        }
    }
    
    private var promptArea: some View {
        // Issue #52: wrap the prompt + actions in a vertical ScrollView so the
        // Generate button stays reachable when the window is shorter than the
        // accumulated content (e.g. on a 13" MacBook Air with non-default text
        // size). The internal layout is unchanged.
        ScrollView(.vertical, showsIndicators: true) {
            VStack(spacing: 0) {
                PromptInputView(
                    prompt: $prompt,
                    negativePrompt: $negativePrompt,
                    voiceoverText: $voiceoverText,
                    parameters: $parameters
                )
                TipsView()
                    .padding()
            }
            .frame(maxWidth: .infinity)
        }
        .frame(minWidth: 400, idealWidth: 500)
    }
    
    private var parametersPanel: some View {
        ParametersView(parameters: $parameters)
            .frame(width: 300)
            .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct TipsView: View {
    let tips = [
        "Use detailed, descriptive prompts for better results",
        "Try different aspect ratios for cinematic or portrait videos",
        "Lower inference steps for quick previews, higher for quality",
        "Use the same seed to regenerate similar results",
        "Negative prompts help remove unwanted elements"
    ]
    
    @State private var currentTip = 0
    
    var body: some View {
        HStack {
            Image(systemName: "lightbulb.fill")
                .foregroundStyle(.yellow)
            Text(tips[currentTip])
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            nextButton
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.yellow.opacity(0.1))
        )
    }
    
    private var nextButton: some View {
        Button {
            withAnimation {
                currentTip = (currentTip + 1) % tips.count
            }
        } label: {
            Image(systemName: "arrow.right.circle")
        }
        .buttonStyle(.borderless)
    }
}
