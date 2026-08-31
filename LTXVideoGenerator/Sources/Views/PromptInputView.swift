import SwiftUI
import UniformTypeIdentifiers

struct PromptInputView: View {
    @EnvironmentObject var generationService: GenerationService
    @EnvironmentObject var presetManager: PresetManager
    @EnvironmentObject var characterProfileManager: CharacterProfileManager
    
    @Binding var prompt: String
    @Binding var negativePrompt: String
    @Binding var voiceoverText: String
    @Binding var parameters: GenerationParameters
    
    @State private var showNegativePrompt = false
    @State private var showVoiceover = false
    @State private var showMusic = false
    @State private var showImageToVideo = false
    @AppStorage("sourceImagePath") private var storedImagePath = ""
    @State private var sourceImageThumbnail: NSImage?
    @State private var showCompletedIndicator = false
    @FocusState private var isPromptFocused: Bool
    
    // Audio settings
    @AppStorage("elevenLabsApiKey") private var elevenLabsApiKey = ""
    @AppStorage("enableGemmaPromptEnhancement") private var enableGemmaPromptEnhancement = false
    @AppStorage(LTXModelCatalog.selectedModelIDKey) private var selectedModelID = LTXModelCatalog.defaultModelID
    @AppStorage(LTXTextEncoderCatalog.selectedTextEncoderIDKey) private var selectedTextEncoderID = LTXTextEncoderCatalog.defaultTextEncoderID

    // Issue #51: persist these between launches and across tab switches.
    @AppStorage(SessionSettings.voiceoverSourceKey) private var voiceoverSource: AudioSource = .mlxAudio
    @AppStorage(SessionSettings.elevenLabsVoiceKey) private var selectedElevenLabsVoice: String = "21m00Tcm4TlvDq8ikWAM"
    @AppStorage(SessionSettings.mlxVoiceKey) private var selectedMLXVoice: String = "af_heart"

    // Music settings (persisted)
    @AppStorage(SessionSettings.musicEnabledKey) private var musicEnabled = false
    @AppStorage(SessionSettings.musicGenreKey) private var selectedMusicGenre: MusicGenre = .cinematicUplifting

    // Audio disable for unified model — persisted (issue #51 explicitly called
    // this out: "Generate Audio is forced to true on launch").
    @AppStorage(SessionSettings.disableAudioKey) private var disableAudio = false

    // Gemma prompt enhancement (persisted)
    @State private var showPromptEnhancement = false
    @AppStorage(SessionSettings.gemmaRepetitionPenaltyKey) private var gemmaRepetitionPenalty: Double = 1.2
    @AppStorage(SessionSettings.gemmaTopPKey) private var gemmaTopP: Double = 0.9
    @State private var showEnhancedPreview = false
    @State private var enhancedPreview: String?
    @State private var isPreviewing = false
    @State private var previewError: String?
    @State private var previewStatusMessage = ""
    @State private var showMemoryRiskAlert = false
    @State private var dismissedHeavyEncoderComboHint = false
    @State private var pendingQueueAction: PendingQueueAction?
    @State private var showSaveCharacterProfile = false
    @State private var newCharacterProfileName = ""

    private var sourceImagePath: String? {
        storedImagePath.isEmpty ? nil : storedImagePath
    }

    private var selectedModel: LTXModel {
        LTXModelCatalog.resolvedModel(id: selectedModelID)
    }

    private var estimatedMemoryGB: Int {
        parameters.estimatedVRAM
    }

    private var machineMemoryGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / 1_000_000_000)
    }

    private var warningThresholdGB: Int {
        // Keep a sensible floor for smaller machines, but scale up for high-memory Macs.
        max(36, Int(Double(machineMemoryGB) * 0.7))
    }

    private var isHighMemoryRisk: Bool {
        estimatedMemoryGB >= warningThresholdGB
            || (parameters.width * parameters.height >= 768 * 512 && parameters.numFrames >= 97 && machineMemoryGB <= 64)
    }

    private var memoryRiskGuidance: String {
        let tilingHint = parameters.vaeTilingMode == "aggressive"
            ? ""
            : " Switch tiling to aggressive for lower peak memory."
        return "This request may hit Metal memory limits (estimated ~\(estimatedMemoryGB)GB on a ~\(machineMemoryGB)GB machine). Recommended retry settings: 512x320 resolution, 25/33/49 frames, 24 FPS.\(tilingHint)"
    }

    /// Non-blocking preflight: very large bf16 stacks on small unified-memory Macs (#53).
    private var heavyEncoderCombinationWarning: String? {
        let physGB = Int(MacOSSystemMemory.physicalMemoryBytes / 1_073_741_824)
        let avail = MacOSSystemMemory.approximateAvailableMemoryGBFormatted()
        let model = selectedModel
        let isLargeBf16Unified = model.id == "ltx2_unified" || model.id == "ltx23_unified"
        let is12bBf16Encoder = selectedTextEncoderID == "gemma3_12b_bf16"
        guard isLargeBf16Unified, is12bBf16Encoder, physGB < 32 else { return nil }
        return "This pairing (large unified LTX bf16 model + Gemma 12B bf16 text encoder) often needs on the order of 50 GB unified memory during TEXT_ENCODER. Your Mac has about \(physGB) GB physical memory (~\(avail) GB available, approximate). Prefer Gemma 4B bf16 or 4-bit (Preferences → General → Text Encoder), enable aggressive VAE tiling, or fewer pixels/frames. You can still generate."
    }

    private var selectedVoiceId: String {
        voiceoverSource == .elevenLabs ? selectedElevenLabsVoice : selectedMLXVoice
    }

    private var generationActionsDisabled: Bool {
        prompt.isEmpty || generationService.isProcessing || generationService.currentRequest != nil
    }

    /// Enqueueing more clips is allowed even while one is generating — the queue
    /// processes sequentially, so only an empty prompt should block it.
    private var addToQueueDisabled: Bool {
        prompt.isEmpty
    }
    
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

            // Character consistency profiles
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 12) {
                    if characterProfileManager.profiles.isEmpty {
                        Text("Save a character profile to reuse the same visual prompt, source image, voice, music, model, and generation settings.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        HStack {
                            Picker("", selection: $characterProfileManager.selectedProfile) {
                                ForEach(characterProfileManager.profiles) { profile in
                                    Text(profile.name).tag(profile as CharacterProfile?)
                                }
                            }
                            .labelsHidden()

                            Button("Apply") {
                                if let profile = characterProfileManager.selectedProfile {
                                    applyCharacterProfile(profile)
                                }
                            }
                            .disabled(characterProfileManager.selectedProfile == nil)

                            if let profile = characterProfileManager.selectedProfile {
                                Button(role: .destructive) {
                                    characterProfileManager.deleteProfile(profile)
                                } label: {
                                    Image(systemName: "trash")
                                }
                                .buttonStyle(.borderless)
                                .help("Delete character profile")
                            }
                        }
                    }

                    Button {
                        showSaveCharacterProfile = true
                    } label: {
                        Label("Save Current Character Profile", systemImage: "person.crop.square")
                    }
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Text("Profiles do not train LoRAs or clone voices; they bundle the consistent visual and audio settings already supported by the app.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.top, 8)
            } label: {
                Label("Character Profile", systemImage: "person.crop.rectangle.stack")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            // Gemma Prompt Enhancement
            DisclosureGroup(isExpanded: $showPromptEnhancement) {
                VStack(alignment: .leading, spacing: 12) {
                    if !enableGemmaPromptEnhancement {
                        Text("Turn on in Settings to use prompt rewriting.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    ParameterSlider(
                        title: "Repetition Penalty",
                        value: $gemmaRepetitionPenalty,
                        range: 1.0...2.0,
                        step: 0.05,
                        icon: "arrow.triangle.2.circlepath",
                        format: "%.2f"
                    )
                    .disabled(!enableGemmaPromptEnhancement)
                    
                    ParameterSlider(
                        title: "Top-P",
                        value: $gemmaTopP,
                        range: 0.0...1.0,
                        step: 0.05,
                        icon: "chart.bar.fill",
                        format: "%.2f"
                    )
                    .disabled(!enableGemmaPromptEnhancement)
                    
                    if enableGemmaPromptEnhancement {
                        Text("Controls prompt rewriting. Higher repetition penalty reduces repeated phrases. Lower top-p makes output more focused.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Button {
                                Task { await runPreview() }
                            } label: {
                                if isPreviewing {
                                    ProgressView()
                                        .scaleEffect(0.7)
                                    Text("Enhancing...")
                                } else {
                                    Label("Preview enhanced prompt", systemImage: "eye")
                                }
                            }
                            .disabled(prompt.trimmingCharacters(in: .whitespaces).isEmpty || isPreviewing)
                        if isPreviewing, !previewStatusMessage.isEmpty {
                            Text(previewStatusMessage)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.top, 8)
                .opacity(enableGemmaPromptEnhancement ? 1 : 0.6)
            } label: {
                Label("Prompt Enhancement", systemImage: "sparkles")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
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
                    
                    Text("Select an image to anchor the first or last frame. Your prompt should describe the motion/action.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    
                    if sourceImagePath != nil {
                        Divider()

                        Picker("Use image as", selection: $parameters.imageFramePosition) {
                            Text("First frame").tag("first")
                            Text("Last frame").tag("last")
                        }
                        .pickerStyle(.segmented)

                        Text(parameters.imageFramePosition == "last"
                            ? "The image becomes the final frame. Your prompt should describe the motion/action leading up to it."
                            : "The image becomes the first frame. Your prompt should describe the motion/action.")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Divider()

                        ParameterSlider(
                            title: "Image Strength",
                            value: $parameters.imageStrength,
                            range: 0.0...1.0,
                            step: 0.05,
                            icon: "photo.fill",
                            format: "%.2f"
                        )

                        Text("How strongly the source image influences generation. 1.0 = exact match, lower = more creative freedom.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Divider()

                    HStack {
                        Label("Additional Keyframes", systemImage: "rectangle.stack.badge.plus")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button {
                            addKeyframe()
                        } label: {
                            Label("Add", systemImage: "plus")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }

                    if parameters.keyframes.isEmpty {
                        Text("Pin extra images anywhere on the timeline (e.g. a mid-point pose). Each image is held at its position — keep them visually consistent or the motion will jump.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach($parameters.keyframes) { $kf in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Image(systemName: "photo")
                                        .foregroundStyle(.secondary)
                                    Text(URL(fileURLWithPath: kf.imagePath).lastPathComponent)
                                        .font(.caption)
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                    Spacer()
                                    Button(role: .destructive) {
                                        parameters.keyframes.removeAll { $0.id == kf.id }
                                    } label: {
                                        Image(systemName: "xmark.circle.fill")
                                    }
                                    .buttonStyle(.plain)
                                    .foregroundStyle(.red)
                                }

                                HStack {
                                    Text("Position")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .frame(width: 58, alignment: .leading)
                                    Slider(value: $kf.position, in: 0...1, step: 0.05)
                                    Text("\(Int((kf.position * 100).rounded()))%")
                                        .font(.caption2.monospacedDigit())
                                        .frame(width: 42, alignment: .trailing)
                                }

                                HStack {
                                    Text("Strength")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .frame(width: 58, alignment: .leading)
                                    Slider(value: $kf.strength, in: 0...1, step: 0.05)
                                    Text(String(format: "%.2f", kf.strength))
                                        .font(.caption2.monospacedDigit())
                                        .frame(width: 42, alignment: .trailing)
                                }
                            }
                            .padding(8)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color(nsColor: .controlBackgroundColor))
                            )
                        }

                        Text("Position: 0% = first frame, 100% = last frame. There are only ~\(max(1, 1 + (parameters.numFrames - 1) / 8)) distinct latent slots, so nearby positions may land on the same frame.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
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
            
            // Audio included banner for unified model
            if selectedModel.supportsBuiltInAudio {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: disableAudio ? "waveform.badge.minus" : "waveform.badge.checkmark")
                        .foregroundColor(disableAudio ? .secondary : .green)
                        .font(.title3)
                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("Generate Audio", isOn: Binding(
                            get: { !disableAudio },
                            set: { disableAudio = !$0 }
                        ))
                        .font(.caption.bold())
                        
                        Text(disableAudio
                            ? "Audio generation is disabled. Video will be silent (faster)."
                            : "Synchronized audio will be generated automatically.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(disableAudio ? Color.secondary.opacity(0.05) : Color.green.opacity(0.1))
                .cornerRadius(8)
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
            
            if let heavyHint = heavyEncoderCombinationWarning, !dismissedHeavyEncoderComboHint {
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(heavyHint)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 8)
                    Button("Dismiss") {
                        dismissedHeavyEncoderComboHint = true
                    }
                    .buttonStyle(.borderless)
                    .controlSize(.small)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.orange.opacity(0.08))
                )
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
                        requestSingleGeneration()
                    } label: {
                        Label("Generate", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(generationActionsDisabled)
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
                    requestSingleGeneration()
                } label: {
                    Label("Add to Queue", systemImage: "plus.circle")
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .disabled(addToQueueDisabled)
                
                // Batch button
                Menu {
                    Button("Generate 3 variations") {
                        requestBatchGeneration(count: 3)
                    }
                    Button("Generate 5 variations") {
                        requestBatchGeneration(count: 5)
                    }
                    Divider()
                    Button("Generate with random seeds...") {
                        // Could show a dialog for count
                        requestBatchGeneration(count: 3)
                    }
                } label: {
                    Image(systemName: "square.stack.3d.up")
                }
                .menuStyle(.borderlessButton)
                .frame(width: 44)
                .disabled(addToQueueDisabled)
            }

            if !disableAudio && parameters.fps != 24 {
                HStack(spacing: 6) {
                    Image(systemName: "info.circle.fill")
                        .foregroundStyle(.orange)
                    Text("Sync speech works best at 24 fps. Current setting: \(parameters.fps) fps.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
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
        .sheet(isPresented: $showEnhancedPreview) {
            EnhancedPreviewSheet(
                enhancedPrompt: enhancedPreview ?? "",
                originalPrompt: prompt,
                error: previewError,
                onDismiss: {
                    showEnhancedPreview = false
                    enhancedPreview = nil
                    previewError = nil
                }
            )
        }
        .sheet(isPresented: $showSaveCharacterProfile) {
            SaveCharacterProfileSheet(
                profileName: $newCharacterProfileName,
                isPresented: $showSaveCharacterProfile,
                onSave: saveCurrentCharacterProfile
            )
        }
        .onAppear {
            if !storedImagePath.isEmpty && sourceImageThumbnail == nil {
                let url = URL(fileURLWithPath: storedImagePath)
                if FileManager.default.fileExists(atPath: storedImagePath) {
                    loadThumbnail(from: url)
                    showImageToVideo = true
                } else {
                    storedImagePath = ""
                }
            }
        }
        .onChange(of: selectedModelID) { _, _ in
            if !selectedModel.supportsBuiltInAudio {
                disableAudio = false
            }
            dismissedHeavyEncoderComboHint = false
        }
        .onChange(of: selectedTextEncoderID) { _, _ in
            dismissedHeavyEncoderComboHint = false
        }
        .alert("High Memory Risk", isPresented: $showMemoryRiskAlert) {
            Button("Continue Anyway") {
                executePendingQueueAction()
            }
            Button("Cancel", role: .cancel) {
                pendingQueueAction = nil
            }
        } message: {
            Text(memoryRiskGuidance)
        }
    }

    private func runPreview() async {
        guard !prompt.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        isPreviewing = true
        previewError = nil
        enhancedPreview = nil
        do {
            let enhanced = try await LTXBridge.shared.previewEnhancedPrompt(
                prompt: prompt,
                modelRepo: selectedModel.repo,
                temperature: gemmaTopP,
                sourceImagePath: sourceImagePath
            ) { status in
                DispatchQueue.main.async { previewStatusMessage = status }
            }
            await MainActor.run {
                enhancedPreview = enhanced
                showEnhancedPreview = true
            }
        } catch {
            await MainActor.run {
                previewError = error.localizedDescription
                showEnhancedPreview = true
            }
        }
        await MainActor.run { isPreviewing = false }
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
            disableAudio: disableAudio,
            gemmaRepetitionPenalty: gemmaRepetitionPenalty,
            gemmaTopP: gemmaTopP,
            modelId: selectedModelID,
            textEncoderId: selectedTextEncoderID,
            parameters: parameters
        )
        generationService.addToQueue(request)
    }

    private func saveCurrentCharacterProfile(name: String) {
        let profile = CharacterProfile(
            name: name,
            prompt: prompt,
            negativePrompt: negativePrompt,
            sourceImagePath: sourceImagePath,
            voiceoverText: voiceoverText,
            voiceoverSource: voiceoverSource.rawValue,
            voiceoverVoice: selectedVoiceId,
            musicEnabled: musicEnabled,
            musicGenre: musicEnabled ? selectedMusicGenre.rawValue : nil,
            disableAudio: disableAudio,
            modelId: selectedModelID,
            parameters: parameters
        )
        characterProfileManager.addProfile(profile)
        newCharacterProfileName = ""
    }

    private func applyCharacterProfile(_ profile: CharacterProfile) {
        prompt = profile.prompt
        negativePrompt = profile.negativePrompt
        voiceoverText = profile.voiceoverText
        voiceoverSource = AudioSource(rawValue: profile.voiceoverSource) ?? .mlxAudio
        if voiceoverSource == .elevenLabs {
            selectedElevenLabsVoice = profile.voiceoverVoice
        } else {
            selectedMLXVoice = profile.voiceoverVoice
        }
        musicEnabled = profile.musicEnabled
        if let musicGenre = profile.musicGenre, let genre = MusicGenre(rawValue: musicGenre) {
            selectedMusicGenre = genre
        }
        disableAudio = profile.disableAudio
        selectedModelID = profile.modelId
        parameters = profile.parameters

        storedImagePath = profile.sourceImagePath ?? ""
        if let imagePath = profile.sourceImagePath, !imagePath.isEmpty {
            loadThumbnail(from: URL(fileURLWithPath: imagePath))
            showImageToVideo = true
        } else {
            sourceImageThumbnail = nil
        }

        showNegativePrompt = !negativePrompt.isEmpty
        showVoiceover = !voiceoverText.isEmpty
        showMusic = musicEnabled
    }

    private func requestSingleGeneration() {
        if isHighMemoryRisk {
            pendingQueueAction = .single
            showMemoryRiskAlert = true
            return
        }
        generateVideo()
    }
    
    private func generateBatch(count: Int) {
        let requests = (0..<count).map { _ -> GenerationRequest in
            // Copy the full parameter set (preserving frame position, keyframes,
            // tiling, etc.) and only randomize the seed per batch item.
            var perItemParameters = parameters
            perItemParameters.seed = Int.random(in: 0..<Int(Int32.max))
            return GenerationRequest(
                prompt: prompt,
                negativePrompt: negativePrompt,
                voiceoverText: voiceoverText,
                voiceoverSource: voiceoverSource.rawValue,
                voiceoverVoice: voiceoverSource == .elevenLabs ? selectedElevenLabsVoice : selectedMLXVoice,
                sourceImagePath: sourceImagePath,
                musicEnabled: musicEnabled,
                musicGenre: musicEnabled ? selectedMusicGenre.rawValue : nil,
                disableAudio: disableAudio,
                gemmaRepetitionPenalty: gemmaRepetitionPenalty,
                gemmaTopP: gemmaTopP,
                modelId: selectedModelID,
                textEncoderId: selectedTextEncoderID,
                parameters: perItemParameters
            )
        }
        generationService.addBatch(requests)
    }

    private func requestBatchGeneration(count: Int) {
        if isHighMemoryRisk {
            pendingQueueAction = .batch(count)
            showMemoryRiskAlert = true
            return
        }
        generateBatch(count: count)
    }

    private func executePendingQueueAction() {
        guard let action = pendingQueueAction else { return }
        pendingQueueAction = nil
        switch action {
        case .single:
            generateVideo()
        case .batch(let count):
            generateBatch(count: count)
        }
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
            storedImagePath = url.path
            loadThumbnail(from: url)
            
            // Auto-expand the section when image is selected
            showImageToVideo = true
        }
    }
    
    private func addKeyframe() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.image, .png, .jpeg, .webP]
        panel.message = "Select keyframe image(s) to pin along the timeline"
        panel.prompt = "Add"

        if panel.runModal() == .OK {
            for url in panel.urls {
                parameters.keyframes.append(
                    Keyframe(imagePath: url.path, position: 1.0, strength: 1.0)
                )
            }
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
        storedImagePath = ""
        sourceImageThumbnail = nil
    }
}

private enum PendingQueueAction {
    case single
    case batch(Int)
}

private struct SaveCharacterProfileSheet: View {
    @Binding var profileName: String
    @Binding var isPresented: Bool
    let onSave: (String) -> Void

    var body: some View {
        VStack(spacing: 20) {
            Text("Save Character Profile")
                .font(.headline)

            TextField("Profile Name", text: $profileName)
                .textFieldStyle(.roundedBorder)

            HStack {
                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Save") {
                    onSave(profileName)
                    isPresented = false
                }
                .keyboardShortcut(.defaultAction)
                .disabled(profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding()
        .frame(width: 320)
    }
}

private struct EnhancedPreviewSheet: View {
    let enhancedPrompt: String
    let originalPrompt: String
    let error: String?
    let onDismiss: () -> Void

    private var displayText: String {
        if !enhancedPrompt.isEmpty { return enhancedPrompt }
        return originalPrompt
    }

    private var isEmptyNote: String? {
        enhancedPrompt.isEmpty && !originalPrompt.isEmpty ? "Enhancement produced no output (possibly filtered). Showing your original prompt:" : nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("Enhanced Prompt Preview", systemImage: "sparkles")
                    .font(.headline)
                Spacer()
                Button("Done") { onDismiss() }
                    .keyboardShortcut(.cancelAction)
            }
            if let err = error {
                Text(err)
                    .foregroundStyle(.red)
                    .font(.caption)
            }
            if let note = isEmptyNote {
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if !displayText.isEmpty {
                ScrollView {
                    Text(displayText)
                        .font(.body)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .padding(8)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            Spacer()
        }
        .padding(24)
        .frame(minWidth: 400, minHeight: 300)
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
    .environmentObject(CharacterProfileManager())
    .frame(width: 500)
}
