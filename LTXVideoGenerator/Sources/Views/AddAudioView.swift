import SwiftUI

enum AudioTab: String, CaseIterable, Identifiable {
    case voiceover = "Voiceover"
    case music = "Music"
    case both = "Both"
    
    var id: String { rawValue }
}

struct AddAudioView: View {
    @EnvironmentObject var historyManager: HistoryManager
    @StateObject private var audioService = AudioService.shared
    
    let result: GenerationResult
    let onComplete: (GenerationResult) -> Void
    let onDismiss: () -> Void
    
    @AppStorage("elevenLabsApiKey") private var elevenLabsApiKey = ""
    @AppStorage("defaultAudioSource") private var defaultAudioSource = "mlx-audio"
    
    // Tab selection
    @State private var selectedTab: AudioTab = .voiceover
    
    // Voiceover settings
    @State private var audioSource: AudioSource = .mlxAudio
    @State private var narrationText: String = ""
    @State private var selectedElevenLabsVoice: String = "21m00Tcm4TlvDq8ikWAM"
    @State private var selectedMLXVoice: String = "af_heart"
    
    // Music settings
    @State private var selectedMusicGenre: MusicGenre = .cinematicUplifting
    
    // Generation state
    @State private var isGenerating = false
    @State private var progress: Double = 0
    @State private var statusMessage = ""
    @State private var errorMessage: String?
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Add Audio to Video")
                    .font(.headline)
                Spacer()
                Button("Cancel") {
                    onDismiss()
                }
                .disabled(isGenerating)
            }
            .padding()
            
            Divider()
            
            // Tab Picker
            Picker("Audio Type", selection: $selectedTab) {
                ForEach(AudioTab.allCases) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.top, 12)
            
            // Content
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Voiceover section
                    if selectedTab == .voiceover || selectedTab == .both {
                        VoiceoverSection(
                            audioSource: $audioSource,
                            narrationText: $narrationText,
                            selectedElevenLabsVoice: $selectedElevenLabsVoice,
                            selectedMLXVoice: $selectedMLXVoice,
                            elevenLabsApiKey: elevenLabsApiKey,
                            prompt: result.prompt,
                            voiceoverText: result.voiceoverText
                        )
                    }
                    
                    // Divider between sections when showing both
                    if selectedTab == .both {
                        Divider()
                            .padding(.vertical, 8)
                    }
                    
                    // Music section
                    if selectedTab == .music || selectedTab == .both {
                        MusicSection(
                            selectedGenre: $selectedMusicGenre,
                            elevenLabsApiKey: elevenLabsApiKey
                        )
                    }
                    
                    // Error Message
                    if let error = errorMessage {
                        HStack(alignment: .top, spacing: 8) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.red)
                            Text(error)
                                .font(.caption)
                        }
                        .padding()
                        .background(Color.red.opacity(0.1))
                        .cornerRadius(8)
                    }
                    
                    // Progress
                    if isGenerating {
                        VStack(alignment: .leading, spacing: 8) {
                            ProgressView(value: progress)
                            Text(statusMessage)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding()
            }
            
            Divider()
            
            // Footer
            HStack {
                // Info about existing audio
                if result.hasAudio || result.hasMusic {
                    HStack(spacing: 4) {
                        Image(systemName: "info.circle")
                        Text(existingAudioMessage)
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                Button(buttonTitle) {
                    generateAudio()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isGenerateDisabled)
            }
            .padding()
        }
        .frame(width: 550, height: 650)
        .onAppear {
            // Pre-fill with voiceover text if available, otherwise use video prompt
            if !result.voiceoverText.isEmpty {
                narrationText = result.voiceoverText
            } else {
                narrationText = result.prompt
            }
            
            // Set default source from preferences
            audioSource = AudioSource(rawValue: defaultAudioSource) ?? .mlxAudio
            
            // If video already has voiceover but not music, default to music tab
            if result.hasAudio && !result.hasMusic {
                selectedTab = .music
            }
        }
    }
    
    private var existingAudioMessage: String {
        if result.hasAudio && result.hasMusic {
            return "Video has voiceover and music"
        } else if result.hasAudio {
            return "Video has voiceover"
        } else if result.hasMusic {
            return "Video has music"
        }
        return ""
    }
    
    private var buttonTitle: String {
        switch selectedTab {
        case .voiceover:
            return result.hasAudio ? "Replace Voiceover" : "Add Voiceover"
        case .music:
            return result.hasMusic ? "Replace Music" : "Add Music"
        case .both:
            return "Add Audio"
        }
    }
    
    private var isGenerateDisabled: Bool {
        if isGenerating { return true }
        
        switch selectedTab {
        case .voiceover:
            if narrationText.isEmpty { return true }
            if audioSource == .elevenLabs && elevenLabsApiKey.isEmpty { return true }
        case .music:
            if elevenLabsApiKey.isEmpty { return true }
        case .both:
            if narrationText.isEmpty { return true }
            if elevenLabsApiKey.isEmpty { return true } // Music always needs ElevenLabs
        }
        
        return false
    }
    
    private func generateAudio() {
        isGenerating = true
        progress = 0
        statusMessage = "Starting..."
        errorMessage = nil
        
        Task {
            do {
                var updatedResult = result
                
                switch selectedTab {
                case .voiceover:
                    updatedResult = try await generateVoiceover(for: updatedResult)
                case .music:
                    updatedResult = try await generateMusic(for: updatedResult)
                case .both:
                    // Generate voiceover first, then music, then merge
                    updatedResult = try await generateVoiceover(for: updatedResult)
                    updatedResult = try await generateMusic(for: updatedResult)
                }
                
                await MainActor.run {
                    isGenerating = false
                    onComplete(updatedResult)
                }
            } catch let error as AudioError {
                await MainActor.run {
                    isGenerating = false
                    errorMessage = error.localizedDescription
                }
            } catch {
                await MainActor.run {
                    isGenerating = false
                    errorMessage = error.localizedDescription
                }
            }
        }
    }
    
    private func generateVoiceover(for inputResult: GenerationResult) async throws -> GenerationResult {
        let voiceId = audioSource == .elevenLabs ? selectedElevenLabsVoice : selectedMLXVoice
        
        return try await audioService.addAudioToVideo(
            result: inputResult,
            text: narrationText,
            source: audioSource,
            voiceId: voiceId,
            historyManager: historyManager
        ) { pct, msg in
            DispatchQueue.main.async {
                self.progress = selectedTab == .both ? pct * 0.5 : pct
                self.statusMessage = msg
            }
        }
    }
    
    private func generateMusic(for inputResult: GenerationResult) async throws -> GenerationResult {
        // Calculate video duration in milliseconds
        let videoDurationMs = Int((Double(inputResult.parameters.numFrames) / Double(inputResult.parameters.fps)) * 1000)
        
        return try await audioService.addMusicToVideo(
            result: inputResult,
            genre: selectedMusicGenre,
            durationMs: videoDurationMs,
            historyManager: historyManager
        ) { pct, msg in
            DispatchQueue.main.async {
                self.progress = selectedTab == .both ? 0.5 + pct * 0.5 : pct
                self.statusMessage = msg
            }
        }
    }
}

// MARK: - Voiceover Section

struct VoiceoverSection: View {
    @Binding var audioSource: AudioSource
    @Binding var narrationText: String
    @Binding var selectedElevenLabsVoice: String
    @Binding var selectedMLXVoice: String
    let elevenLabsApiKey: String
    let prompt: String
    let voiceoverText: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Voiceover")
                .font(.headline)
            
            // Audio Source Picker
            VStack(alignment: .leading, spacing: 8) {
                Text("Source")
                    .font(.subheadline.bold())
                
                Picker("Source", selection: $audioSource) {
                    ForEach(AudioSource.allCases) { source in
                        Text(source.displayName).tag(source)
                    }
                }
                .pickerStyle(.segmented)
                
                Text(audioSource.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            
            // ElevenLabs API Key Warning
            if audioSource == .elevenLabs && elevenLabsApiKey.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("ElevenLabs API Key Required")
                            .font(.subheadline.bold())
                        Text("Add your API key in Preferences > Audio to use ElevenLabs.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
                .background(Color.orange.opacity(0.1))
                .cornerRadius(8)
            }
            
            // Voice Selection
            VStack(alignment: .leading, spacing: 8) {
                Text("Voice")
                    .font(.subheadline.bold())
                
                if audioSource == .elevenLabs {
                    Picker("Voice", selection: $selectedElevenLabsVoice) {
                        ForEach(ElevenLabsVoice.defaultVoices) { voice in
                            Text(voice.displayName).tag(voice.voice_id)
                        }
                    }
                    .pickerStyle(.menu)
                } else {
                    Picker("Voice", selection: $selectedMLXVoice) {
                        ForEach(MLXAudioVoice.defaultVoices) { voice in
                            Text(voice.name).tag(voice.id)
                        }
                    }
                    .pickerStyle(.menu)
                }
            }
            
            // Narration Text
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Narration Text")
                        .font(.subheadline.bold())
                    Spacer()
                    Button("Use Prompt") {
                        narrationText = prompt
                    }
                    .font(.caption)
                    .buttonStyle(.link)
                }
                
                TextEditor(text: $narrationText)
                    .font(.body)
                    .frame(minHeight: 100)
                    .padding(8)
                    .background(Color(nsColor: .textBackgroundColor))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
                    )
                
                HStack {
                    Text("\(narrationText.count) characters")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    Spacer()
                    
                    if audioSource == .elevenLabs {
                        Image(systemName: "questionmark.circle")
                            .foregroundStyle(.secondary)
                            .help(ElevenLabsFormattingHelp.tips)
                    }
                }
            }
        }
    }
}

// MARK: - Music Section

struct MusicSection: View {
    @Binding var selectedGenre: MusicGenre
    let elevenLabsApiKey: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Background Music")
                .font(.headline)
            
            // API Key Warning
            if elevenLabsApiKey.isEmpty {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("ElevenLabs API Key Required")
                            .font(.subheadline.bold())
                        Text("Music generation requires an ElevenLabs API key. Add your key in Preferences > Audio.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()
                .background(Color.orange.opacity(0.1))
                .cornerRadius(8)
            }
            
            // Genre Selection
            VStack(alignment: .leading, spacing: 8) {
                Text("Genre")
                    .font(.subheadline.bold())
                
                Picker("Genre", selection: $selectedGenre) {
                    ForEach(MusicGenre.groupedByCategory, id: \.category) { group in
                        Section(header: Text(group.category)) {
                            ForEach(group.genres) { genre in
                                Text(genre.displayName).tag(genre)
                            }
                        }
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 300)
            }
            
            // Genre Description
            VStack(alignment: .leading, spacing: 4) {
                Text("Preview:")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Text(selectedGenre.prompt)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }
            .padding()
            .background(Color(nsColor: .controlBackgroundColor))
            .cornerRadius(8)
            
            Text("Music will be generated to match your video length using ElevenLabs Music API.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

#Preview {
    AddAudioView(
        result: .preview(),
        onComplete: { _ in },
        onDismiss: {}
    )
    .environmentObject(HistoryManager())
}
