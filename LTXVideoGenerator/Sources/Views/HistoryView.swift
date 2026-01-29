import SwiftUI
import AVKit

struct HistoryView: View {
    @EnvironmentObject var historyManager: HistoryManager
    @EnvironmentObject var generationService: GenerationService
    
    @State private var selection = Set<GenerationResult>()
    @State private var searchText = ""
    @State private var sortOrder: SortOrder = .dateDescending
    @State private var addAudioResult: GenerationResult?
    
    enum SortOrder: String, CaseIterable {
        case dateDescending = "Newest First"
        case dateAscending = "Oldest First"
        case promptAZ = "Prompt A-Z"
    }
    
    var filteredResults: [GenerationResult] {
        var results = historyManager.results
        
        // Filter by search
        if !searchText.isEmpty {
            results = results.filter { result in
                result.prompt.localizedCaseInsensitiveContains(searchText)
            }
        }
        
        // Sort
        switch sortOrder {
        case .dateDescending:
            results.sort { $0.completedAt > $1.completedAt }
        case .dateAscending:
            results.sort { $0.completedAt < $1.completedAt }
        case .promptAZ:
            results.sort { $0.prompt < $1.prompt }
        }
        
        return results
    }
    
    var body: some View {
        HSplitView {
            // Grid view
            VStack(spacing: 0) {
                // Toolbar
                HStack {
                    Label("\(historyManager.results.count) videos", systemImage: "film")
                        .font(.headline)
                    
                    Spacer()
                    
                    Picker("Sort", selection: $sortOrder) {
                        ForEach(SortOrder.allCases, id: \.self) { order in
                            Text(order.rawValue).tag(order)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 140)
                    
                    if !selection.isEmpty {
                        Button(role: .destructive) {
                            historyManager.deleteResults(selection)
                            selection.removeAll()
                        } label: {
                            Label("Delete \(selection.count)", systemImage: "trash")
                        }
                    }
                }
                .padding()
                .background(Color(nsColor: .controlBackgroundColor))
                
                Divider()
                
                // Search
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search prompts...", text: $searchText)
                        .textFieldStyle(.plain)
                }
                .padding(8)
                .background(Color(nsColor: .controlBackgroundColor))
                
                Divider()
                
                if filteredResults.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "film.stack")
                            .font(.system(size: 48))
                            .foregroundStyle(.tertiary)
                        
                        Text(searchText.isEmpty ? "No videos generated yet" : "No results found")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                        
                        if searchText.isEmpty {
                            Text("Generated videos will appear here")
                                .font(.subheadline)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView {
                        LazyVGrid(columns: [
                            GridItem(.adaptive(minimum: 180, maximum: 250), spacing: 16)
                        ], spacing: 16) {
                            ForEach(filteredResults) { result in
                                HistoryThumbnailView(
                                    result: result,
                                    isSelected: selection.contains(result)
                                )
                                .onTapGesture {
                                    historyManager.selectedResult = result
                                }
                                .onTapGesture(count: 2) {
                                    historyManager.revealInFinder(result)
                                }
                                .contextMenu {
                                    contextMenu(for: result)
                                }
                            }
                        }
                        .padding()
                    }
                }
            }
            .frame(minWidth: 400)
            
            // Detail view
            if let selected = historyManager.selectedResult {
                HistoryDetailView(result: selected, onAddAudio: { result in
                    addAudioResult = result
                })
                    .frame(minWidth: 350, idealWidth: 400)
            } else {
                VStack {
                    Image(systemName: "sidebar.right")
                        .font(.largeTitle)
                        .foregroundStyle(.tertiary)
                    Text("Select a video to preview")
                        .foregroundStyle(.secondary)
                }
                .frame(minWidth: 350, idealWidth: 400, maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .sheet(item: $addAudioResult) { result in
            AddAudioView(
                result: result,
                onComplete: { updatedResult in
                    historyManager.updateResult(updatedResult)
                    addAudioResult = nil
                },
                onDismiss: {
                    addAudioResult = nil
                }
            )
        }
    }
    
    @ViewBuilder
    private func contextMenu(for result: GenerationResult) -> some View {
        Button {
            historyManager.selectedResult = result
        } label: {
            Label("Preview", systemImage: "eye")
        }
        
        Button {
            historyManager.revealInFinder(result)
        } label: {
            Label("Show in Finder", systemImage: "folder")
        }
        
        Divider()
        
        Button {
            addAudioResult = result
        } label: {
            Label(result.hasAudio ? "Replace Audio" : "Add Audio", systemImage: "waveform")
        }
        
        Divider()
        
        Button {
            reusePrompt(result)
        } label: {
            Label("Reuse Prompt", systemImage: "arrow.uturn.left")
        }
        
        Button {
            regenerate(result)
        } label: {
            Label("Regenerate with Same Seed", systemImage: "arrow.clockwise")
        }
        
        Divider()
        
        Button(role: .destructive) {
            historyManager.deleteResult(result)
        } label: {
            Label("Delete", systemImage: "trash")
        }
    }
    
    private func reusePrompt(_ result: GenerationResult) {
        // This would need to communicate back to the main view
        // For now, just copy to clipboard
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(result.prompt, forType: .string)
    }
    
    private func regenerate(_ result: GenerationResult) {
        var params = result.parameters
        params.seed = result.seed
        
        let request = GenerationRequest(
            prompt: result.prompt,
            negativePrompt: result.negativePrompt,
            parameters: params
        )
        generationService.addToQueue(request)
    }
}

struct HistoryThumbnailView: View {
    let result: GenerationResult
    let isSelected: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Thumbnail
            ZStack {
                if let thumbnailURL = result.thumbnailURL,
                   let image = NSImage(contentsOf: thumbnailURL) {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else {
                    Rectangle()
                        .fill(Color(nsColor: .controlBackgroundColor))
                    Image(systemName: "film")
                        .font(.title)
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isSelected ? Color.accentColor : .clear, lineWidth: 3)
            )
            
            // Info
            VStack(alignment: .leading, spacing: 2) {
                Text(result.prompt)
                    .font(.caption)
                    .lineLimit(2)
                    .foregroundStyle(.primary)
                
                Text(result.formattedDate)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(isSelected ? Color.accentColor.opacity(0.1) : .clear)
        )
    }
}

// AVPlayerView wrapper to avoid _AVKit_SwiftUI crash on macOS 26
struct VideoPlayerView: NSViewRepresentable {
    let player: AVPlayer
    
    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = true
        return view
    }
    
    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        nsView.player = player
    }
}

struct HistoryDetailView: View {
    @EnvironmentObject var historyManager: HistoryManager
    
    let result: GenerationResult
    let onAddAudio: (GenerationResult) -> Void
    @State private var player: AVPlayer?
    
    var body: some View {
        VStack(spacing: 0) {
            // Video player - using AVPlayerView wrapper to avoid SwiftUI crash
            if let player = player {
                VideoPlayerView(player: player)
                    .aspectRatio(
                        CGFloat(result.parameters.width) / CGFloat(result.parameters.height),
                        contentMode: .fit
                    )
                    .background(.black)
            } else {
                Rectangle()
                    .fill(.black)
                    .aspectRatio(16/9, contentMode: .fit)
                    .overlay {
                        ProgressView()
                    }
            }
            
            // Details
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Prompt
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Prompt")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(result.prompt)
                            .textSelection(.enabled)
                    }
                    
                    if !result.negativePrompt.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Negative Prompt")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(result.negativePrompt)
                                .textSelection(.enabled)
                        }
                    }
                    
                    Divider()
                    
                    // Parameters grid
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 12) {
                        DetailItem(label: "Resolution", value: "\(result.parameters.width)×\(result.parameters.height)")
                        DetailItem(label: "Frames", value: "\(result.parameters.numFrames)")
                        DetailItem(label: "FPS", value: "\(result.parameters.fps)")
                        DetailItem(label: "Steps", value: "\(result.parameters.numInferenceSteps)")
                        DetailItem(label: "Guidance", value: String(format: "%.1f", result.parameters.guidanceScale))
                        DetailItem(label: "Seed", value: "\(result.seed)")
                    }
                    
                    Divider()
                    
                    // Meta
                    HStack {
                        DetailItem(label: "Generated", value: result.formattedDate)
                        Spacer()
                        DetailItem(label: "Duration", value: result.formattedDuration)
                    }
                    
                    // Audio indicator
                    if result.hasAudio {
                        HStack(spacing: 4) {
                            Image(systemName: "waveform")
                                .foregroundStyle(.green)
                            Text("Has Audio")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                    
                    // Actions - use icon buttons with tooltips for compact layout
                    HStack(spacing: 8) {
                        Button {
                            historyManager.revealInFinder(result)
                        } label: {
                            Image(systemName: "folder")
                        }
                        .buttonStyle(.bordered)
                        .help("Show in Finder")
                        
                        ShareLink(item: result.videoURL) {
                            Image(systemName: "square.and.arrow.up")
                        }
                        .buttonStyle(.bordered)
                        .help("Share")
                        
                        Button {
                            onAddAudio(result)
                        } label: {
                            Image(systemName: "waveform")
                        }
                        .buttonStyle(.bordered)
                        .help(result.hasAudio ? "Replace Audio" : "Add Audio")
                        
                        Spacer()
                        
                        Button(role: .destructive) {
                            historyManager.deleteResult(result)
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.bordered)
                        .help("Delete")
                    }
                }
                .padding()
            }
        }
        .onAppear {
            loadPlayer()
        }
        .onChange(of: result.id) { _, _ in
            loadPlayer()
        }
    }
    
    private func loadPlayer() {
        player = AVPlayer(url: result.videoURL)
        player?.play()
        
        // Loop playback
        NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: player?.currentItem,
            queue: .main
        ) { _ in
            player?.seek(to: .zero)
            player?.play()
        }
    }
}

struct DetailItem: View {
    let label: String
    let value: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline)
        }
    }
}

#Preview {
    HistoryView()
        .environmentObject(HistoryManager())
        .environmentObject(GenerationService(historyManager: HistoryManager()))
        .frame(width: 900, height: 600)
}

#Preview("Detail View") {
    HistoryDetailView(result: .preview(), onAddAudio: { _ in })
        .environmentObject(HistoryManager())
        .frame(width: 400, height: 600)
}
