import SwiftUI
import Metal

struct ParametersView: View {
    @EnvironmentObject var presetManager: PresetManager
    
    @Binding var parameters: GenerationParameters
    @State private var showSavePreset = false
    @State private var newPresetName = ""
    @State private var availableVRAM = getAvailableVRAM()
    
    let vramTimer = Timer.publish(every: 5, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Preset picker
            VStack(alignment: .leading, spacing: 8) {
                Label("Preset", systemImage: "slider.horizontal.3")
                    .font(.headline)
                    .foregroundStyle(.secondary)
                
                HStack {
                    Picker("", selection: $presetManager.selectedPreset) {
                        ForEach(presetManager.presets) { preset in
                            Text(preset.name).tag(preset as Preset?)
                        }
                    }
                    .labelsHidden()
                    .onChange(of: presetManager.selectedPreset) { _, newValue in
                        if let preset = newValue {
                            parameters = preset.parameters
                        }
                    }
                    
                    Button {
                        showSavePreset = true
                    } label: {
                        Image(systemName: "plus.circle")
                    }
                    .buttonStyle(.borderless)
                    .help("Save current settings as preset")
                }
            }
            
            Divider()
            
            // Parameters
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Inference steps
                    ParameterSlider(
                        title: "Inference Steps",
                        value: Binding(
                            get: { Double(parameters.numInferenceSteps) },
                            set: { parameters.numInferenceSteps = Int($0) }
                        ),
                        range: 10...100,
                        step: 5,
                        icon: "arrow.triangle.2.circlepath"
                    )
                    
                    // Guidance scale
                    ParameterSlider(
                        title: "Guidance Scale",
                        value: $parameters.guidanceScale,
                        range: 1...15,
                        step: 0.5,
                        icon: "dial.medium",
                        format: "%.1f"
                    )
                    
                    Divider()
                    
                    // Resolution
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Resolution", systemImage: "rectangle.dashed")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        
                        HStack(spacing: 16) {
                            VStack(alignment: .leading) {
                                Text("Width")
                                    .font(.caption)
                                    .foregroundStyle(.tertiary)
                                Picker("", selection: $parameters.width) {
                                    Text("320").tag(320)
                                    Text("512").tag(512)
                                    Text("640").tag(640)
                                    Text("768").tag(768)
                                    Text("896").tag(896)
                                    Text("1024").tag(1024)
                                }
                                .labelsHidden()
                                .frame(width: 80)
                            }
                            
                            VStack(alignment: .leading) {
                                Text("Height")
                                    .font(.caption)
                                    .foregroundStyle(.tertiary)
                                Picker("", selection: $parameters.height) {
                                    Text("320").tag(320)
                                    Text("384").tag(384)
                                    Text("512").tag(512)
                                    Text("576").tag(576)
                                    Text("768").tag(768)
                                }
                                .labelsHidden()
                                .frame(width: 80)
                            }
                            
                            Spacer()
                            
                            Text("\(parameters.width)×\(parameters.height)")
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(
                                    RoundedRectangle(cornerRadius: 6)
                                        .fill(Color(nsColor: .controlBackgroundColor))
                                )
                        }
                    }
                    
                    Divider()
                    
                    // Frame count
                    ParameterSlider(
                        title: "Frames",
                        value: Binding(
                            get: { Double(parameters.numFrames) },
                            set: { parameters.numFrames = Int($0) }
                        ),
                        range: 25...1000,
                        step: 25,
                        icon: "film.stack"
                    )
                    
                    if parameters.numFrames > 500 {
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                            Text("High frame count may exceed GPU memory")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    
                    // FPS
                    VStack(alignment: .leading, spacing: 8) {
                        Label("FPS", systemImage: "speedometer")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        
                        Picker("", selection: $parameters.fps) {
                            Text("12 fps").tag(12)
                            Text("20 fps").tag(20)
                            Text("24 fps").tag(24)
                            Text("30 fps").tag(30)
                        }
                        .pickerStyle(.segmented)
                    }
                    
                    // Video length estimate
                    HStack {
                        Image(systemName: "film")
                        Text("Video Length: \(parameters.videoLength)")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    
                    Divider()
                    
                    // Seed
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Seed", systemImage: "dice")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        
                        HStack {
                            TextField("Random", value: $parameters.seed, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 120)
                            
                            Button {
                                parameters.seed = Int.random(in: 0..<Int(Int32.max))
                            } label: {
                                Image(systemName: "dice.fill")
                            }
                            .buttonStyle(.borderless)
                            .help("Generate random seed")
                            
                            if parameters.seed != nil {
                                Button {
                                    parameters.seed = nil
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                }
                                .buttonStyle(.borderless)
                                .help("Clear seed (use random)")
                            }
                        }
                    }
                    
                    Divider()
                    
                    // VAE Tiling Mode
                    VStack(alignment: .leading, spacing: 8) {
                        Label("VAE Tiling", systemImage: "square.grid.3x3")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        
                        Picker("", selection: $parameters.vaeTilingMode) {
                            Text("Auto").tag("auto")
                            Text("None").tag("none")
                            Text("Default").tag("default")
                            Text("Aggressive").tag("aggressive")
                            Text("Conservative").tag("conservative")
                            Text("Spatial Only").tag("spatial")
                            Text("Temporal Only").tag("temporal")
                        }
                        .labelsHidden()
                        
                        Text("Controls memory vs speed tradeoff during decoding. Aggressive uses less memory; Conservative is faster but needs more memory.")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    
                    // Estimated time
                    HStack {
                        Image(systemName: "clock")
                        Text("Estimated: \(parameters.estimatedDuration)")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 8)
                    
                    Spacer()
                    
                    // VRAM info
                    HStack {
                        Spacer()
                        HStack(spacing: 4) {
                            Image(systemName: "memorychip")
                            Text("\(availableVRAM) available")
                        }
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .onReceive(vramTimer) { _ in
                            availableVRAM = getAvailableVRAM()
                        }
                    }
                }
                .padding(.horizontal, 4)
            }
        }
        .padding()
        .sheet(isPresented: $showSavePreset) {
            SavePresetSheet(
                presetName: $newPresetName,
                parameters: parameters,
                isPresented: $showSavePreset
            )
            .environmentObject(presetManager)
        }
    }
}

struct ParameterSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let icon: String
    var format: String = "%.0f"
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(title, systemImage: icon)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                
                Spacer()
                
                Text(String(format: format, value))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color(nsColor: .controlBackgroundColor))
                    )
            }
            
            Slider(value: $value, in: range, step: step)
        }
    }
}

struct SavePresetSheet: View {
    @EnvironmentObject var presetManager: PresetManager
    @Binding var presetName: String
    let parameters: GenerationParameters
    @Binding var isPresented: Bool
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Save Preset")
                .font(.headline)
            
            TextField("Preset Name", text: $presetName)
                .textFieldStyle(.roundedBorder)
            
            HStack {
                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.cancelAction)
                
                Spacer()
                
                Button("Save") {
                    _ = presetManager.saveCurrentAsPreset(name: presetName, parameters: parameters)
                    presetName = ""
                    isPresented = false
                }
                .keyboardShortcut(.defaultAction)
                .disabled(presetName.isEmpty)
            }
        }
        .padding()
        .frame(width: 300)
    }
}

private func getAvailableVRAM() -> String {
    guard let device = MTLCreateSystemDefaultDevice() else {
        return "N/A"
    }
    
    // On Apple Silicon, recommendedMaxWorkingSetSize gives us usable memory
    let bytes = device.recommendedMaxWorkingSetSize
    let gb = Double(bytes) / 1_073_741_824.0
    return String(format: "%.0fGB", gb)
}

#Preview {
    ParametersView(parameters: .constant(.default))
        .environmentObject(PresetManager())
        .frame(width: 300, height: 600)
}
