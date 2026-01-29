import SwiftUI

struct AddAudioView: View {
    @EnvironmentObject var historyManager: HistoryManager
    @StateObject private var audioService = AudioService.shared
    
    let result: GenerationResult
    let onComplete: (GenerationResult) -> Void
    let onDismiss: () -> Void
    
    @AppStorage("elevenLabsApiKey") private var elevenLabsApiKey = ""
    @AppStorage("defaultAudioSource") private var defaultAudioSource = "elevenlabs"
    
    @State private var audioSource: AudioSource = .elevenLabs
    @State private var narrationText: String = ""
    @State private var selectedElevenLabsVoice: String = "21m00Tcm4TlvDq8ikWAM"
    @State private var selectedMLXVoice: String = "af_heart"
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
            
            // Content
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Audio Source Picker
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Audio Source")
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
                                Button("Open Preferences") {
                                    NSApp.sendAction(Selector(("showPreferencesWindow:")), to: nil, from: nil)
                                }
                                .buttonStyle(.link)
                                .font(.caption)
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
                                    Text(voice.name).tag(voice.voice_id)
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
                            Button("Use Video Prompt") {
                                narrationText = result.prompt
                            }
                            .font(.caption)
                            .buttonStyle(.link)
                        }
                        
                        TextEditor(text: $narrationText)
                            .font(.body)
                            .frame(minHeight: 120)
                            .padding(8)
                            .background(Color(nsColor: .textBackgroundColor))
                            .cornerRadius(8)
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
                            )
                        
                        Text("\(narrationText.count) characters")
                            .font(.caption)
                            .foregroundStyle(.secondary)
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
                if result.hasAudio {
                    Label("This video already has audio", systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                Button("Generate Audio") {
                    generateAudio()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isGenerating || narrationText.isEmpty || (audioSource == .elevenLabs && elevenLabsApiKey.isEmpty))
            }
            .padding()
        }
        .frame(width: 500, height: 550)
        .onAppear {
            // Pre-fill with video prompt
            narrationText = result.prompt
            
            // Set default source from preferences
            audioSource = AudioSource(rawValue: defaultAudioSource) ?? .elevenLabs
        }
    }
    
    private func generateAudio() {
        isGenerating = true
        progress = 0
        statusMessage = "Starting..."
        errorMessage = nil
        
        let voiceId = audioSource == .elevenLabs ? selectedElevenLabsVoice : selectedMLXVoice
        
        Task {
            do {
                let updatedResult = try await audioService.addAudioToVideo(
                    result: result,
                    text: narrationText,
                    source: audioSource,
                    voiceId: voiceId,
                    historyManager: historyManager
                ) { pct, msg in
                    DispatchQueue.main.async {
                        self.progress = pct
                        self.statusMessage = msg
                    }
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
}

#Preview {
    AddAudioView(
        result: .preview(),
        onComplete: { _ in },
        onDismiss: {}
    )
    .environmentObject(HistoryManager())
}
