import SwiftUI

struct ContentView: View {
    @EnvironmentObject var generationService: GenerationService
    @EnvironmentObject var historyManager: HistoryManager
    @EnvironmentObject var presetManager: PresetManager
    
    @State private var prompt = ""
    @State private var negativePrompt = ""
    @State private var parameters = GenerationParameters.default
    @State private var selectedTab: Tab = .generate
    @State private var showError = false
    
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
        .frame(minWidth: 1100, minHeight: 700)
        .alert("Error", isPresented: $showError, presenting: generationService.error) { _ in
            Button("OK", role: .cancel) {}
        } message: { error in
            Text(error.localizedDescription)
        }
        .onChange(of: generationService.error) { _, newError in
            showError = newError != nil
        }
    }
    
    private var sidebarContent: some View {
        VStack(spacing: 0) {
            tabSelector
            Divider()
            QueueView()
                .frame(maxHeight: 300)
            Spacer()
            ModelStatusView()
                .padding()
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
    @AppStorage("selectedModelVariant") private var selectedModelVariant = "distilled"
    
    private var currentModelVariant: LTXModelVariant {
        LTXModelVariant(rawValue: selectedModelVariant) ?? .distilled
    }
    
    var body: some View {
        VStack(spacing: 8) {
            // Model variant indicator
            HStack(spacing: 6) {
                Image(systemName: "cpu")
                    .foregroundStyle(.blue)
                Text(currentModelVariant.displayName)
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
                Text(generationService.isModelLoaded ? "Model Ready" : "Model Not Loaded")
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

struct GenerateView: View {
    @Binding var prompt: String
    @Binding var negativePrompt: String
    @Binding var parameters: GenerationParameters
    
    var body: some View {
        HSplitView {
            promptArea
            parametersPanel
        }
    }
    
    private var promptArea: some View {
        VStack {
            PromptInputView(
                prompt: $prompt,
                negativePrompt: $negativePrompt,
                parameters: $parameters
            )
            Spacer()
            TipsView()
                .padding()
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
