import SwiftUI
import UniformTypeIdentifiers

struct PromptInputView: View {
    @EnvironmentObject var generationService: GenerationService
    @EnvironmentObject var presetManager: PresetManager
    
    @Binding var prompt: String
    @Binding var negativePrompt: String
    @Binding var voiceoverText: String
    @Binding var parameters: GenerationParameters
    
    @State private var showNegativePrompt = false
    @State private var showVoiceover = false
    @State private var showMusic = false
    @State private var showImageToVideo = false
    @State private var sourceImagePath: String?
    @State private var sourceImageThumbnail: NSImage?
    @State private var showCompletedIndicator = false
    @FocusState private var isPromptFocused: Bool
    
    // Audio settings
    @AppStorage("elevenLabsApiKey") private var elevenLabsApiKey = ""
    @State private var voiceoverSource: AudioSource = .mlxAudio
    @State private var selectedElevenLabsVoice: String = "21m00Tcm4TlvDq8ikWAM"
    @State private var selectedMLXVoice: String = "af_heart"
    
    // Music settings
    @State private var musicEnabled = false
    @State private var selectedMusicGenre: MusicGenre = .cinematicUplifting
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Main prompt
            VStack(alignment: .leading, spacing: 8) {
                Label("Prompt", systemImage: "text.bubble.fill")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                
                TextEditor(text: $prompt)
                    .font(.body)
                    .frame(minHeight: 80, maxHeight: 120)
                    .scrollContentBackground(.hidden)
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(nsColor: .controlBackgroundColor))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(isPromptFocused ? Color.accentColor : Color.clear, lineWidth: 2)
                    )
                    .focused($isPromptFocused)
            }
            
            // Image-to-Video section
            DisclosureGroup(isExpanded: $showImageToVideo) {
                VStack(alignment: .leading, spacing: 12) {
                    if let imagePath = sourceImagePath, let thumbnail = sourceImageThumbnail {
                        // Show selected image
                        HStack(spacing: 12) {
                            Image(nsImage: thumbnail)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 80, height: 80)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.accentColor, lineWidth: 2)
                                )
                            
                            VStack(alignment: .leading, spacing: 4) {
                                Text(URL(fileURLWithPath: imagePath).lastPathComponent)
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .lineLimit(1)
                                
                                Text("\(Int(thumbnail.size.width))x\(Int(thumbnail.size.height))")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                
                                Button(role: .destructive) {
                                    clearSourceImage()
                                } label: {
                                    Label("Remove", systemImage: "xmark.circle.fill")
                                        .font(.caption)
                                }
                                .buttonStyle(.plain)
                                .foregroundStyle(.red)
                            }
                            
                            Spacer()
                        }
                        .padding(8)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color(nsColor: .controlBackgroundColor))
                        )
                    } else {
                        // Show picker button
                        Button {
                            selectSourceImage()
                        } label: {
                            HStack {
                                Image(systemName: "photo.badge.plus")
                                Text("Select Source Image...")
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                        }
                        .buttonStyle(.bordered)
                    }
                    
                    Text("Select an image to use as the first frame. Your prompt should describe the motion/action.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 8)
            } label: {
                HStack {
                    Label("Image to Video", systemImage: "photo.on.rectangle.angled")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    
                    if sourceImagePath != nil {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.caption)
                    }
                }
            }
            
            // Negative prompt toggle
            DisclosureGroup(isExpanded: $showNegativePrompt) {
                TextEditor(text: $negativePrompt)
                    .font(.body)
                    .frame(height: 60)
                    .scrollContentBackground(.hidden)
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(nsColor: .controlBackgroundColor))
                    )
            } label: {
                Label("Negative Prompt", systemImage: "minus.circle")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            // Voiceover narration toggle
            DisclosureGroup(isExpanded: $showVoiceover) {
                VStack(alignment: .leading, spacing: 12) {
                    // Source selection
                    HStack {
                        Text("Source:")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        
                        Picker("", selection: $voiceoverSource) {
                            ForEach(AudioSource.allCases) { source in
                                Text(source.displayName).tag(source)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 280)
                    }
                    
                    // ElevenLabs API key warning
                    if voiceoverSource == .elevenLabs && elevenLabsApiKey.isEmpty {
                        HStack(spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                                .font(.caption)
                            Text("ElevenLabs API key required. Set in Preferences > Audio.")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }
                        .padding(8)
                        .background(Color.orange.opacity(0.1))
                        .cornerRadius(6)
                    }
                    
                    // Voice selection
                    HStack {
                        Text("Voice:")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        
                        if voiceoverSource == .elevenLabs {
                            Picker("", selection: $selectedElevenLabsVoice) {
                                ForEach(ElevenLabsVoice.defaultVoices) { voice in
                                    Text(voice.displayName).tag(voice.voice_id)
                                }
                            }
                            .frame(maxWidth: 220)
                        } else {
                            Picker("", selection: $selectedMLXVoice) {
                                ForEach(MLXAudioVoice.defaultVoices) { voice in
                                    Text(voice.name).tag(voice.id)
                                }
                            }
                            .frame(maxWidth: 220)
                        }
                        
                        Spacer()
                    }
                    
                    // Narration text
                    TextEditor(text: $voiceoverText)
                        .font(.body)
                        .frame(height: 80)
                        .scrollContentBackground(.hidden)
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color(nsColor: .controlBackgroundColor))
                        )
                    
                    HStack {
                        Text("Optional narration text. Add audio later from History view.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        
                        Spacer()
                        
                        if voiceoverSource == .elevenLabs {
                            Image(systemName: "questionmark.circle")
                                .foregroundStyle(.blue)
                                .font(.caption)
                                .help(ElevenLabsFormattingHelp.tips)
                        }
                    }
                }
            } label: {
                HStack {
                    Label("Voiceover / Narration", systemImage: "waveform")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    
                    if !voiceoverText.isEmpty {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.caption)
                    }
                }
            }
            
            // Background Music toggle
            DisclosureGroup(isExpanded: $showMusic) {
                VStack(alignment: .leading, spacing: 12) {
                    // Enable toggle
                    Toggle("Generate background music", isOn: $musicEnabled)
                        .font(.subheadline)
                    
                    if musicEnabled {
                        // ElevenLabs API key warning
                        if elevenLabsApiKey.isEmpty {
                            HStack(spacing: 6) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.orange)
                                    .font(.caption)
                                Text("ElevenLabs API key required for music. Set in Preferences > Audio.")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                            }
                            .padding(8)
                            .background(Color.orange.opacity(0.1))
                            .cornerRadius(6)
                        }
                        
                        // Genre selection with categories
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Genre:")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            
                            Picker("", selection: $selectedMusicGenre) {
                                ForEach(MusicGenre.groupedByCategory, id: \.category) { group in
                                    Section(header: Text(group.category)) {
                                        ForEach(group.genres) { genre in
                                            Text(genre.displayName).tag(genre)
                                        }
                                    }
                                }
                            }
                            .frame(maxWidth: 300)
                        }
                        
                        Text("Music will be generated using ElevenLabs Music API and mixed with your video.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            } label: {
                HStack {
                    Label("Background Music", systemImage: "music.note")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    
                    if musicEnabled {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.caption)
                    }
                }
            }
            
            // Quick actions
            HStack(spacing: 12) {
                // Generate button - changes appearance based on state
                if showCompletedIndicator {
                    // Completion state - green button
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                        Text("Complete!")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(.green)
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                } else if generationService.currentRequest != nil {
                    // Processing state - shows spinner (when there's an active generation)
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                            .tint(.white)
                        Text("Generating...")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(Color.accentColor.opacity(0.7))
                    .foregroundStyle(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    // Normal state - generate button
                    Button {
                        generateVideo()
                    } label: {
                        Label("Generate", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(prompt.isEmpty || generationService.isProcessing)
                }
                
                // Track completion - only when currentRequest goes away (actual generation done)
                Color.clear
                    .frame(width: 0, height: 0)
                    .onChange(of: generationService.currentRequest) { oldRequest, newRequest in
                        // Generation completed when we had a request and now we don't
                        if oldRequest != nil && newRequest == nil && generationService.error == nil {
                            showCompletedIndicator = true
                            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                                withAnimation {
                                    showCompletedIndicator = false
                                }
                            }
                        }
                    }
                
                // Add to queue button
                Button {
                    addToQueue()
                } label: {
                    Label("Add to Queue", systemImage: "plus.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .disabled(prompt.isEmpty)
                
                // Batch button
                Menu {
                    Button("Generate 3 variations") {
                        generateBatch(count: 3)
                    }
                    Button("Generate 5 variations") {
                        generateBatch(count: 5)
                    }
                    Divider()
                    Button("Generate with random seeds...") {
                        // Could show a dialog for count
                        generateBatch(count: 3)
                    }
                } label: {
                    Image(systemName: "square.stack.3d.up")
                }
                .menuStyle(.borderlessButton)
                .frame(width: 44)
            }
            
            // Status
            if generationService.isProcessing {
                HStack(spacing: 12) {
                    ProgressView(value: generationService.progress)
                        .progressViewStyle(.linear)
                    
                    Text(generationService.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
        .padding()
    }
    
    private func generateVideo() {
        let request = GenerationRequest(
            prompt: prompt,
            negativePrompt: negativePrompt,
            voiceoverText: voiceoverText,
            voiceoverSource: voiceoverSource.rawValue,
            voiceoverVoice: voiceoverSource == .elevenLabs ? selectedElevenLabsVoice : selectedMLXVoice,
            sourceImagePath: sourceImagePath,
            musicEnabled: musicEnabled,
            musicGenre: musicEnabled ? selectedMusicGenre.rawValue : nil,
            parameters: parameters
        )
        generationService.addToQueue(request)
    }
    
    private func addToQueue() {
        let request = GenerationRequest(
            prompt: prompt,
            negativePrompt: negativePrompt,
            voiceoverText: voiceoverText,
            voiceoverSource: voiceoverSource.rawValue,
            voiceoverVoice: voiceoverSource == .elevenLabs ? selectedElevenLabsVoice : selectedMLXVoice,
            sourceImagePath: sourceImagePath,
            musicEnabled: musicEnabled,
            musicGenre: musicEnabled ? selectedMusicGenre.rawValue : nil,
            parameters: parameters
        )
        generationService.addToQueue(request)
    }
    
    private func generateBatch(count: Int) {
        let requests = (0..<count).map { _ in
            GenerationRequest(
                prompt: prompt,
                negativePrompt: negativePrompt,
                voiceoverText: voiceoverText,
                voiceoverSource: voiceoverSource.rawValue,
                voiceoverVoice: voiceoverSource == .elevenLabs ? selectedElevenLabsVoice : selectedMLXVoice,
                sourceImagePath: sourceImagePath,
                musicEnabled: musicEnabled,
                musicGenre: musicEnabled ? selectedMusicGenre.rawValue : nil,
                parameters: GenerationParameters(
                    numInferenceSteps: parameters.numInferenceSteps,
                    guidanceScale: parameters.guidanceScale,
                    width: parameters.width,
                    height: parameters.height,
                    numFrames: parameters.numFrames,
                    fps: parameters.fps,
                    seed: Int.random(in: 0..<Int(Int32.max))
                )
            )
        }
        generationService.addBatch(requests)
    }
    
    // MARK: - Image Selection
    
    private func selectSourceImage() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.image, .png, .jpeg, .webP]
        panel.message = "Select source image for image-to-video generation"
        panel.prompt = "Select"
        
        if panel.runModal() == .OK, let url = panel.url {
            sourceImagePath = url.path
            loadThumbnail(from: url)
            
            // Auto-expand the section when image is selected
            showImageToVideo = true
        }
    }
    
    private func loadThumbnail(from url: URL) {
        if let image = NSImage(contentsOf: url) {
            // Create a smaller thumbnail for display
            let maxSize: CGFloat = 160
            let aspectRatio = image.size.width / image.size.height
            
            let thumbnailSize: NSSize
            if aspectRatio > 1 {
                thumbnailSize = NSSize(width: maxSize, height: maxSize / aspectRatio)
            } else {
                thumbnailSize = NSSize(width: maxSize * aspectRatio, height: maxSize)
            }
            
            let thumbnail = NSImage(size: thumbnailSize)
            thumbnail.lockFocus()
            image.draw(in: NSRect(origin: .zero, size: thumbnailSize),
                      from: NSRect(origin: .zero, size: image.size),
                      operation: .copy,
                      fraction: 1.0)
            thumbnail.unlockFocus()
            
            sourceImageThumbnail = thumbnail
        }
    }
    
    private func clearSourceImage() {
        sourceImagePath = nil
        sourceImageThumbnail = nil
    }
}

#Preview {
    PromptInputView(
        prompt: .constant("A majestic eagle soaring through mountains"),
        negativePrompt: .constant(""),
        voiceoverText: .constant(""),
        parameters: .constant(.default)
    )
    .environmentObject(GenerationService(historyManager: HistoryManager()))
    .environmentObject(PresetManager())
    .frame(width: 500)
}
