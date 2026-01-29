import Foundation
import AVFoundation

enum AudioError: LocalizedError, Equatable {
    case elevenLabsKeyNotSet
    case elevenLabsApiFailed(String)
    case mlxAudioFailed(String)
    case ffmpegFailed(String)
    case pythonNotConfigured
    
    var errorDescription: String? {
        switch self {
        case .elevenLabsKeyNotSet:
            return "ElevenLabs API key not configured. Please add your API key in Preferences > Audio."
        case .elevenLabsApiFailed(let msg):
            return "ElevenLabs API error: \(msg)"
        case .mlxAudioFailed(let msg):
            return "MLX Audio generation failed: \(msg)"
        case .ffmpegFailed(let msg):
            return "FFmpeg merge failed: \(msg)"
        case .pythonNotConfigured:
            return "Python environment not configured."
        }
    }
}

enum AudioSource: String, CaseIterable, Identifiable {
    case elevenLabs = "elevenlabs"
    case mlxAudio = "mlx-audio"
    
    var id: String { rawValue }
    
    var displayName: String {
        switch self {
        case .elevenLabs: return "ElevenLabs (Cloud)"
        case .mlxAudio: return "MLX Audio (Local)"
        }
    }
    
    var description: String {
        switch self {
        case .elevenLabs: return "High-quality cloud TTS, requires API key"
        case .mlxAudio: return "On-device TTS, runs locally on Apple Silicon"
        }
    }
}

struct ElevenLabsVoice: Identifiable, Codable {
    let voice_id: String
    let name: String
    let accent: String
    
    var id: String { voice_id }
    
    var displayName: String {
        "\(name) (\(accent))"
    }
    
    static let defaultVoices: [ElevenLabsVoice] = [
        // American voices
        ElevenLabsVoice(voice_id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel", accent: "US Female"),
        ElevenLabsVoice(voice_id: "EXAVITQu4vr4xnSDxMaL", name: "Sarah", accent: "US Female"),
        ElevenLabsVoice(voice_id: "cgSgspJ2msm6clMCkdW9", name: "Jessica", accent: "US Female"),
        ElevenLabsVoice(voice_id: "9BWtsMINqrJLrRacOk9x", name: "Aria", accent: "US Female"),
        ElevenLabsVoice(voice_id: "pNInz6obpgDQGcFmaJgB", name: "Adam", accent: "US Male"),
        ElevenLabsVoice(voice_id: "TxGEqnHWrfWFTfGW9XjX", name: "Josh", accent: "US Male"),
        ElevenLabsVoice(voice_id: "nPczCjzI2devNBz1zQrb", name: "Brian", accent: "US Male"),
        ElevenLabsVoice(voice_id: "cjVigY5qzO86Huf0OWal", name: "Eric", accent: "US Male"),
        // British voices
        ElevenLabsVoice(voice_id: "Xb7hH8MSUJpSbSDYk0k2", name: "Alice", accent: "UK Female"),
        ElevenLabsVoice(voice_id: "pFZP5JQG7iQjIQuC4Bku", name: "Lily", accent: "UK Female"),
        ElevenLabsVoice(voice_id: "XB0fDUnXU5powFXDhCwa", name: "Charlotte", accent: "UK Female"),
        ElevenLabsVoice(voice_id: "onwK4e9ZLuTAKqWW03F9", name: "Daniel", accent: "UK Male"),
        ElevenLabsVoice(voice_id: "JBFqnCBsd6RMkjVDRZzb", name: "George", accent: "UK Male"),
        ElevenLabsVoice(voice_id: "SOYHLrjzK2X1ezoPC6cr", name: "Harry", accent: "UK Male"),
        // Australian voices
        ElevenLabsVoice(voice_id: "IKne3meq5aSn9XLyUdCD", name: "Charlie", accent: "AU Male"),
        ElevenLabsVoice(voice_id: "XrExE9yKIg1WjnnlVkGX", name: "Matilda", accent: "AU Female")
    ]
}

/// Help text for ElevenLabs voice formatting
enum ElevenLabsFormattingHelp {
    static let tips = """
    Formatting Tips:
    • <break time="1s" /> - Add a pause (up to 3s)
    • — or -- - Dash for short pause
    • ... - Ellipsis adds hesitation
    • CAPS - Adds emphasis
    • "quotes" - Slightly different delivery
    
    Example:
    "Welcome to the show. <break time="1s" /> Today... we explore the unknown."
    """
    
    static let shortTip = "Use <break time=\"1s\" /> for pauses, ... for hesitation, CAPS for emphasis"
}

struct MLXAudioVoice: Identifiable {
    let id: String
    let name: String
    
    // Kokoro model voice presets
    static let defaultVoices: [MLXAudioVoice] = [
        // American English Female
        MLXAudioVoice(id: "af_heart", name: "Heart (US Female)"),
        MLXAudioVoice(id: "af_bella", name: "Bella (US Female)"),
        MLXAudioVoice(id: "af_nova", name: "Nova (US Female)"),
        MLXAudioVoice(id: "af_sky", name: "Sky (US Female)"),
        // American English Male
        MLXAudioVoice(id: "am_adam", name: "Adam (US Male)"),
        MLXAudioVoice(id: "am_echo", name: "Echo (US Male)"),
        // British English Female
        MLXAudioVoice(id: "bf_alice", name: "Alice (UK Female)"),
        MLXAudioVoice(id: "bf_emma", name: "Emma (UK Female)"),
        // British English Male
        MLXAudioVoice(id: "bm_daniel", name: "Daniel (UK Male)"),
        MLXAudioVoice(id: "bm_george", name: "George (UK Male)")
    ]
}

// MARK: - Music Genre Presets

enum MusicGenre: String, CaseIterable, Identifiable {
    // Electronic
    case electronicEDM = "electronic_edm"
    case electronicHouse = "electronic_house"
    case electronicTechno = "electronic_techno"
    case electronicAmbient = "electronic_ambient"
    case electronicChillwave = "electronic_chillwave"
    case electronicSynthwave = "electronic_synthwave"
    case electronicDrumAndBass = "electronic_dnb"
    case electronicTrance = "electronic_trance"
    
    // Hip-Hop/R&B
    case hiphopTrap = "hiphop_trap"
    case hiphopLofi = "hiphop_lofi"
    case hiphopBoom = "hiphop_boom"
    case rnbSlow = "rnb_slow"
    case rnbModern = "rnb_modern"
    case rnbSoul = "rnb_soul"
    
    // Rock
    case rockClassic = "rock_classic"
    case rockAlternative = "rock_alternative"
    case rockIndie = "rock_indie"
    case rockMetal = "rock_metal"
    case rockPunk = "rock_punk"
    case rockAcoustic = "rock_acoustic"
    
    // Pop
    case popModern = "pop_modern"
    case popIndie = "pop_indie"
    case popDance = "pop_dance"
    case popAcoustic = "pop_acoustic"
    
    // Jazz/Blues
    case jazzSmooth = "jazz_smooth"
    case jazzBebop = "jazz_bebop"
    case jazzLounge = "jazz_lounge"
    case bluesElectric = "blues_electric"
    case bluesAcoustic = "blues_acoustic"
    
    // Classical/Orchestral
    case classicalOrchestral = "classical_orchestral"
    case classicalPiano = "classical_piano"
    case classicalChamber = "classical_chamber"
    case cinematicEpic = "cinematic_epic"
    case cinematicTense = "cinematic_tense"
    case cinematicUplifting = "cinematic_uplifting"
    
    // World
    case worldLatin = "world_latin"
    case worldReggae = "world_reggae"
    case worldAfrobeat = "world_afrobeat"
    case worldMiddleEastern = "world_middle_eastern"
    case worldAsian = "world_asian"
    
    // Country/Folk
    case countryModern = "country_modern"
    case countryClassic = "country_classic"
    case folkAcoustic = "folk_acoustic"
    case folkIndie = "folk_indie"
    
    // Functional/Background
    case functionalCorporate = "functional_corporate"
    case functionalMotivational = "functional_motivational"
    case functionalRelaxing = "functional_relaxing"
    case functionalSuspense = "functional_suspense"
    case functionalAction = "functional_action"
    case functionalRomantic = "functional_romantic"
    case functionalHappy = "functional_happy"
    case functionalSad = "functional_sad"
    case functionalDramatic = "functional_dramatic"
    case functionalMystery = "functional_mystery"
    
    var id: String { rawValue }
    
    var displayName: String {
        switch self {
        // Electronic
        case .electronicEDM: return "EDM / Dance"
        case .electronicHouse: return "House"
        case .electronicTechno: return "Techno"
        case .electronicAmbient: return "Ambient Electronic"
        case .electronicChillwave: return "Chillwave"
        case .electronicSynthwave: return "Synthwave / Retro"
        case .electronicDrumAndBass: return "Drum & Bass"
        case .electronicTrance: return "Trance"
        // Hip-Hop/R&B
        case .hiphopTrap: return "Trap"
        case .hiphopLofi: return "Lo-Fi Hip Hop"
        case .hiphopBoom: return "Boom Bap"
        case .rnbSlow: return "Slow R&B"
        case .rnbModern: return "Modern R&B"
        case .rnbSoul: return "Soul"
        // Rock
        case .rockClassic: return "Classic Rock"
        case .rockAlternative: return "Alternative Rock"
        case .rockIndie: return "Indie Rock"
        case .rockMetal: return "Metal"
        case .rockPunk: return "Punk Rock"
        case .rockAcoustic: return "Acoustic Rock"
        // Pop
        case .popModern: return "Modern Pop"
        case .popIndie: return "Indie Pop"
        case .popDance: return "Dance Pop"
        case .popAcoustic: return "Acoustic Pop"
        // Jazz/Blues
        case .jazzSmooth: return "Smooth Jazz"
        case .jazzBebop: return "Bebop Jazz"
        case .jazzLounge: return "Lounge Jazz"
        case .bluesElectric: return "Electric Blues"
        case .bluesAcoustic: return "Acoustic Blues"
        // Classical/Orchestral
        case .classicalOrchestral: return "Orchestral"
        case .classicalPiano: return "Piano Classical"
        case .classicalChamber: return "Chamber Music"
        case .cinematicEpic: return "Cinematic Epic"
        case .cinematicTense: return "Cinematic Tension"
        case .cinematicUplifting: return "Cinematic Uplifting"
        // World
        case .worldLatin: return "Latin"
        case .worldReggae: return "Reggae"
        case .worldAfrobeat: return "Afrobeat"
        case .worldMiddleEastern: return "Middle Eastern"
        case .worldAsian: return "Asian / Eastern"
        // Country/Folk
        case .countryModern: return "Modern Country"
        case .countryClassic: return "Classic Country"
        case .folkAcoustic: return "Acoustic Folk"
        case .folkIndie: return "Indie Folk"
        // Functional
        case .functionalCorporate: return "Corporate / Business"
        case .functionalMotivational: return "Motivational / Inspiring"
        case .functionalRelaxing: return "Relaxing / Calm"
        case .functionalSuspense: return "Suspenseful"
        case .functionalAction: return "Action / Intense"
        case .functionalRomantic: return "Romantic"
        case .functionalHappy: return "Happy / Upbeat"
        case .functionalSad: return "Sad / Melancholic"
        case .functionalDramatic: return "Dramatic"
        case .functionalMystery: return "Mysterious"
        }
    }
    
    var category: String {
        switch self {
        case .electronicEDM, .electronicHouse, .electronicTechno, .electronicAmbient,
             .electronicChillwave, .electronicSynthwave, .electronicDrumAndBass, .electronicTrance:
            return "Electronic"
        case .hiphopTrap, .hiphopLofi, .hiphopBoom, .rnbSlow, .rnbModern, .rnbSoul:
            return "Hip-Hop / R&B"
        case .rockClassic, .rockAlternative, .rockIndie, .rockMetal, .rockPunk, .rockAcoustic:
            return "Rock"
        case .popModern, .popIndie, .popDance, .popAcoustic:
            return "Pop"
        case .jazzSmooth, .jazzBebop, .jazzLounge, .bluesElectric, .bluesAcoustic:
            return "Jazz / Blues"
        case .classicalOrchestral, .classicalPiano, .classicalChamber,
             .cinematicEpic, .cinematicTense, .cinematicUplifting:
            return "Classical / Cinematic"
        case .worldLatin, .worldReggae, .worldAfrobeat, .worldMiddleEastern, .worldAsian:
            return "World"
        case .countryModern, .countryClassic, .folkAcoustic, .folkIndie:
            return "Country / Folk"
        case .functionalCorporate, .functionalMotivational, .functionalRelaxing,
             .functionalSuspense, .functionalAction, .functionalRomantic,
             .functionalHappy, .functionalSad, .functionalDramatic, .functionalMystery:
            return "Functional / Mood"
        }
    }
    
    /// ElevenLabs Music API prompt for this genre
    var prompt: String {
        switch self {
        // Electronic
        case .electronicEDM:
            return "High-energy EDM track with driving beats, euphoric synth leads, and a powerful drop. Festival-ready electronic dance music with pulsing bass and dynamic build-ups."
        case .electronicHouse:
            return "Groovy house music with a four-on-the-floor beat, warm bassline, and soulful elements. Deep and melodic with infectious rhythm."
        case .electronicTechno:
            return "Dark, hypnotic techno with industrial textures, relentless rhythms, and minimalist approach. Underground club sound with driving percussion."
        case .electronicAmbient:
            return "Atmospheric ambient electronic with lush pads, subtle textures, and spacious soundscapes. Dreamy and ethereal with gentle movement."
        case .electronicChillwave:
            return "Nostalgic chillwave with hazy synths, mellow beats, and a dreamy lo-fi aesthetic. Relaxed and wistful with vintage vibes."
        case .electronicSynthwave:
            return "Retro synthwave with 80s-inspired synths, pulsing arpeggios, and neon-soaked atmosphere. Nostalgic and cinematic with driving energy."
        case .electronicDrumAndBass:
            return "Fast-paced drum and bass with rolling breakbeats, heavy sub-bass, and intricate percussion. High-energy jungle rhythms with liquid melodies."
        case .electronicTrance:
            return "Euphoric trance music with soaring melodies, uplifting progressions, and energetic beats. Emotional and transcendent with powerful builds."
        // Hip-Hop/R&B
        case .hiphopTrap:
            return "Modern trap beat with hard-hitting 808s, crisp hi-hats, and dark atmospheric pads. Heavy bass drops and aggressive energy."
        case .hiphopLofi:
            return "Chill lo-fi hip hop with dusty vinyl crackle, mellow jazz samples, and laid-back boom bap drums. Relaxed study vibes."
        case .hiphopBoom:
            return "Classic boom bap hip hop with punchy drums, soulful samples, and head-nodding groove. Old school New York vibes."
        case .rnbSlow:
            return "Smooth slow R&B with silky melodies, gentle rhythms, and romantic atmosphere. Sensual and intimate with lush harmonies."
        case .rnbModern:
            return "Contemporary R&B with sleek production, subtle trap influences, and moody atmosphere. Smooth vocals and innovative beats."
        case .rnbSoul:
            return "Soulful R&B with rich harmonies, gospel influences, and emotional depth. Warm and heartfelt with classic soul elements."
        // Rock
        case .rockClassic:
            return "Classic rock with powerful guitar riffs, driving drums, and anthemic energy. Timeless rock sound with blues influences."
        case .rockAlternative:
            return "Alternative rock with distorted guitars, dynamic shifts, and raw emotional intensity. 90s-inspired grunge and indie influences."
        case .rockIndie:
            return "Indie rock with jangly guitars, creative arrangements, and authentic DIY spirit. Catchy melodies with artistic sensibility."
        case .rockMetal:
            return "Heavy metal with crushing guitar riffs, aggressive drums, and intense energy. Powerful and relentless with technical precision."
        case .rockPunk:
            return "Fast punk rock with aggressive guitar chords, driving tempo, and rebellious energy. Raw and energetic with attitude."
        case .rockAcoustic:
            return "Acoustic rock with warm guitar tones, natural instrumentation, and heartfelt emotion. Stripped-back and intimate."
        // Pop
        case .popModern:
            return "Contemporary pop with polished production, catchy hooks, and radio-ready sound. Upbeat and accessible with modern synths."
        case .popIndie:
            return "Indie pop with quirky melodies, organic textures, and artistic flair. Bright and playful with unique character."
        case .popDance:
            return "Danceable pop with electronic beats, infectious rhythms, and euphoric energy. Club-ready with catchy melodies."
        case .popAcoustic:
            return "Acoustic pop with gentle guitar, warm vocals, and intimate feel. Simple and heartfelt with natural beauty."
        // Jazz/Blues
        case .jazzSmooth:
            return "Smooth jazz with silky saxophone melodies, gentle rhythms, and sophisticated harmonies. Relaxed and elegant."
        case .jazzBebop:
            return "Bebop jazz with complex improvisations, fast tempos, and intricate melodies. Virtuosic and energetic."
        case .jazzLounge:
            return "Lounge jazz with sultry atmosphere, cocktail piano, and relaxed sophistication. Elegant and smooth."
        case .bluesElectric:
            return "Electric blues with soulful guitar licks, driving rhythm section, and raw emotion. Gritty and expressive."
        case .bluesAcoustic:
            return "Acoustic blues with fingerpicked guitar, rootsy feel, and authentic emotion. Delta-inspired and soulful."
        // Classical/Orchestral
        case .classicalOrchestral:
            return "Full orchestral piece with sweeping strings, majestic brass, and dynamic timpani. Cinematic and powerful."
        case .classicalPiano:
            return "Solo piano classical with elegant melodies, expressive dynamics, and emotional depth. Refined and beautiful."
        case .classicalChamber:
            return "Chamber music with intimate string ensemble, delicate interplay, and refined elegance. Sophisticated and nuanced."
        case .cinematicEpic:
            return "Epic cinematic score with massive orchestral swells, thundering percussion, and heroic themes. Blockbuster movie intensity."
        case .cinematicTense:
            return "Tense cinematic underscore with suspenseful strings, ominous brass, and building dread. Thriller atmosphere."
        case .cinematicUplifting:
            return "Uplifting cinematic music with soaring strings, triumphant brass, and inspirational themes. Emotional and hopeful."
        // World
        case .worldLatin:
            return "Latin music with infectious rhythms, warm percussion, and passionate energy. Salsa and tropical vibes."
        case .worldReggae:
            return "Reggae with laid-back offbeat rhythms, warm bass, and island vibes. Positive and relaxed."
        case .worldAfrobeat:
            return "Afrobeat with polyrhythmic percussion, horn sections, and infectious grooves. High-energy African fusion."
        case .worldMiddleEastern:
            return "Middle Eastern music with exotic scales, traditional instruments, and mystical atmosphere. Evocative and hypnotic."
        case .worldAsian:
            return "Asian-influenced music with traditional instruments, pentatonic melodies, and serene atmosphere. Peaceful and meditative."
        // Country/Folk
        case .countryModern:
            return "Modern country with polished production, twangy guitars, and heartfelt lyrics. Contemporary Nashville sound."
        case .countryClassic:
            return "Classic country with steel guitar, fiddle, and traditional instrumentation. Honky-tonk authenticity."
        case .folkAcoustic:
            return "Acoustic folk with fingerpicked guitar, natural instrumentation, and storytelling tradition. Warm and organic."
        case .folkIndie:
            return "Indie folk with creative arrangements, layered harmonies, and artistic expression. Modern folk sensibility."
        // Functional
        case .functionalCorporate:
            return "Professional corporate music with clean production, positive energy, and business-appropriate tone. Inspiring and modern, suitable for presentations and advertisements."
        case .functionalMotivational:
            return "Motivational music with building energy, triumphant themes, and inspiring crescendos. Empowering and uplifting for achievement moments."
        case .functionalRelaxing:
            return "Calm relaxation music with gentle melodies, soft textures, and peaceful atmosphere. Soothing and tranquil for meditation or background."
        case .functionalSuspense:
            return "Suspenseful music with tension-building elements, dramatic pauses, and ominous undertones. Perfect for thrillers and mystery content."
        case .functionalAction:
            return "High-energy action music with driving rhythms, intense percussion, and adrenaline-pumping dynamics. Perfect for sports or action sequences."
        case .functionalRomantic:
            return "Romantic music with tender melodies, gentle strings, and heartfelt emotion. Warm and intimate for love scenes."
        case .functionalHappy:
            return "Happy upbeat music with cheerful melodies, bright instrumentation, and positive energy. Feel-good vibes for joyful content."
        case .functionalSad:
            return "Melancholic music with minor keys, gentle piano, and emotional depth. Touching and bittersweet for poignant moments."
        case .functionalDramatic:
            return "Dramatic music with powerful dynamics, intense orchestration, and emotional weight. Impactful and moving."
        case .functionalMystery:
            return "Mysterious music with enigmatic melodies, subtle tension, and intriguing atmosphere. Curious and thought-provoking."
        }
    }
    
    /// Grouped genres by category for picker UI
    static var groupedByCategory: [(category: String, genres: [MusicGenre])] {
        let categories = ["Electronic", "Hip-Hop / R&B", "Rock", "Pop", "Jazz / Blues", 
                         "Classical / Cinematic", "World", "Country / Folk", "Functional / Mood"]
        return categories.map { cat in
            (category: cat, genres: allCases.filter { $0.category == cat })
        }
    }
}

@MainActor
class AudioService: ObservableObject {
    static let shared = AudioService()
    
    @Published var isGenerating = false
    @Published var progress: Double = 0
    @Published var statusMessage: String = ""
    @Published var error: AudioError?
    
    private init() {}
    
    // MARK: - ElevenLabs
    
    var elevenLabsApiKey: String {
        UserDefaults.standard.string(forKey: "elevenLabsApiKey") ?? ""
    }
    
    var isElevenLabsConfigured: Bool {
        !elevenLabsApiKey.isEmpty
    }
    
    func generateWithElevenLabs(
        text: String,
        voiceId: String,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> URL {
        guard isElevenLabsConfigured else {
            throw AudioError.elevenLabsKeyNotSet
        }
        
        progressHandler(0.1, "Connecting to ElevenLabs...")
        
        let url = URL(string: "https://api.elevenlabs.io/v1/text-to-speech/\(voiceId)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(elevenLabsApiKey, forHTTPHeaderField: "xi-api-key")
        
        let body: [String: Any] = [
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": [
                "stability": 0.5,
                "similarity_boost": 0.75
            ]
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        progressHandler(0.3, "Generating audio...")
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AudioError.elevenLabsApiFailed("Invalid response")
        }
        
        if httpResponse.statusCode != 200 {
            let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw AudioError.elevenLabsApiFailed("Status \(httpResponse.statusCode): \(errorMessage)")
        }
        
        progressHandler(0.8, "Saving audio file...")
        
        let outputURL = URL(fileURLWithPath: outputPath)
        try data.write(to: outputURL)
        
        progressHandler(1.0, "Audio generated")
        
        return outputURL
    }
    
    // MARK: - ElevenLabs Music
    
    func generateMusic(
        genre: MusicGenre,
        durationMs: Int,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> URL {
        guard isElevenLabsConfigured else {
            throw AudioError.elevenLabsKeyNotSet
        }
        
        progressHandler(0.05, "Connecting to ElevenLabs Music...")
        
        let url = URL(string: "https://api.elevenlabs.io/v1/music")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(elevenLabsApiKey, forHTTPHeaderField: "xi-api-key")
        
        // Build the music prompt with instrumental-only instruction
        let musicPrompt = "\(genre.prompt) Instrumental only, no vocals."
        
        let body: [String: Any] = [
            "prompt": musicPrompt,
            "music_length_ms": durationMs
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        progressHandler(0.1, "Generating \(genre.displayName) music...")
        
        // Music generation can take longer, so we use a longer timeout
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300 // 5 minutes
        config.timeoutIntervalForResource = 600 // 10 minutes
        let session = URLSession(configuration: config)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw AudioError.elevenLabsApiFailed("Invalid response from Music API")
        }
        
        if httpResponse.statusCode != 200 {
            let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw AudioError.elevenLabsApiFailed("Music API status \(httpResponse.statusCode): \(errorMessage)")
        }
        
        progressHandler(0.9, "Saving music file...")
        
        let outputURL = URL(fileURLWithPath: outputPath)
        try data.write(to: outputURL)
        
        progressHandler(1.0, "Music generated")
        
        return outputURL
    }
    
    // MARK: - MLX Audio
    
    func generateWithMLXAudio(
        text: String,
        voice: String,
        outputPath: String,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> URL {
        guard let pythonPath = UserDefaults.standard.string(forKey: "pythonPath"),
              !pythonPath.isEmpty else {
            throw AudioError.pythonNotConfigured
        }
        
        progressHandler(0.1, "Starting MLX Audio generation...")
        
        let resourcesPath = Bundle.main.bundlePath + "/Contents/Resources"
        let generatorScript = resourcesPath + "/audio_generator.py"
        
        // Escape text for Python
        let escapedText = text
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
        
        let script = """
        import sys
        import json
        
        # Add Resources directory to path
        sys.path.insert(0, "\(resourcesPath)")
        
        try:
            from audio_generator import generate_audio
            
            result = generate_audio(
                text="\(escapedText)",
                voice="\(voice)",
                output_path="\(outputPath)"
            )
            
            print(json.dumps({"success": True, "audio_path": "\(outputPath)"}))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}), file=sys.stderr)
            sys.exit(1)
        """
        
        let output = try await runPython(
            executable: pythonPath,
            script: script,
            timeout: 300
        ) { stderr in
            DispatchQueue.main.async {
                if stderr.hasPrefix("PROGRESS:") {
                    let parts = stderr.dropFirst(9).split(separator: ":")
                    if parts.count >= 2,
                       let pct = Double(parts[0]) {
                        let msg = String(parts[1...].joined(separator: ":"))
                        progressHandler(pct / 100.0, msg)
                    }
                } else if stderr.hasPrefix("STATUS:") {
                    let msg = String(stderr.dropFirst(7))
                    progressHandler(0.5, msg)
                }
            }
        }
        
        // Parse result
        if let data = output.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let success = json["success"] as? Bool, success {
            progressHandler(1.0, "Audio generated")
            return URL(fileURLWithPath: outputPath)
        }
        
        throw AudioError.mlxAudioFailed("Failed to parse output: \(output)")
    }
    
    // MARK: - FFmpeg Merge
    
    func mergeAudioWithVideo(
        videoURL: URL,
        audioURL: URL,
        outputURL: URL,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws {
        progressHandler(0.1, "Merging audio with video...")
        
        // Find ffmpeg
        let ffmpegPaths = [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg"
        ]
        
        guard let ffmpegPath = ffmpegPaths.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw AudioError.ffmpegFailed("FFmpeg not found. Install with: brew install ffmpeg")
        }
        
        // FFmpeg command: combine video and audio
        // -y: overwrite output
        // -i: input files
        // -c:v copy: copy video stream without re-encoding
        // -c:a aac: encode audio as AAC
        // Note: Do NOT use -shortest as it truncates video to audio length
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpegPath)
        process.arguments = [
            "-y",
            "-i", videoURL.path,
            "-i", audioURL.path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            outputURL.path
        ]
        
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        process.standardOutput = Pipe()
        
        try process.run()
        
        progressHandler(0.5, "Processing...")
        
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            let errorData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let errorOutput = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw AudioError.ffmpegFailed(errorOutput)
        }
        
        progressHandler(1.0, "Merge complete")
    }
    
    // MARK: - Full Pipeline
    
    func addAudioToVideo(
        result: GenerationResult,
        text: String,
        source: AudioSource,
        voiceId: String,
        historyManager: HistoryManager,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> GenerationResult {
        isGenerating = true
        progress = 0
        statusMessage = "Starting..."
        error = nil
        
        defer {
            isGenerating = false
        }
        
        // Generate audio file path - use same directory as the video
        let videoDirectory = result.videoURL.deletingLastPathComponent()
        let audioFileName = "\(result.id.uuidString)_audio"
        let audioExtension = source == .elevenLabs ? "mp3" : "wav"
        let audioPath = videoDirectory
            .appendingPathComponent("\(audioFileName).\(audioExtension)").path
        
        // Generate audio
        progressHandler(0.1, "Generating audio...")
        
        let audioURL: URL
        switch source {
        case .elevenLabs:
            audioURL = try await generateWithElevenLabs(
                text: text,
                voiceId: voiceId,
                outputPath: audioPath
            ) { pct, msg in
                progressHandler(0.1 + pct * 0.4, msg)
            }
        case .mlxAudio:
            audioURL = try await generateWithMLXAudio(
                text: text,
                voice: voiceId,
                outputPath: audioPath
            ) { pct, msg in
                progressHandler(0.1 + pct * 0.4, msg)
            }
        }
        
        // Merge with video
        progressHandler(0.5, "Merging audio with video...")
        
        let outputFileName = "\(result.id.uuidString)_with_audio.mp4"
        let outputPath = videoDirectory.appendingPathComponent(outputFileName)
        
        try await mergeAudioWithVideo(
            videoURL: result.videoURL,
            audioURL: audioURL,
            outputURL: outputPath
        ) { pct, msg in
            progressHandler(0.5 + pct * 0.4, msg)
        }
        
        // Update result with new video path (with audio)
        progressHandler(0.95, "Updating history...")
        
        let updatedResult = GenerationResult(
            id: result.id,
            requestId: result.requestId,
            prompt: result.prompt,
            negativePrompt: result.negativePrompt,
            voiceoverText: result.voiceoverText,
            voiceoverSource: result.voiceoverSource,
            voiceoverVoice: result.voiceoverVoice,
            parameters: result.parameters,
            videoPath: outputPath.path,
            thumbnailPath: result.thumbnailPath,
            audioPath: audioPath,
            musicPath: result.musicPath,
            musicGenre: result.musicGenre,
            createdAt: result.createdAt,
            completedAt: result.completedAt,
            duration: result.duration,
            seed: result.seed
        )
        
        // Clean up original video file (optional - keep backup)
        // try? FileManager.default.removeItem(at: result.videoURL)
        
        progressHandler(1.0, "Complete!")
        
        return updatedResult
    }
    
    // MARK: - Video Duration Helper
    
    /// Get the actual duration of a video file in milliseconds
    func getVideoDurationMs(url: URL) async -> Int? {
        let asset = AVAsset(url: url)
        do {
            let duration = try await asset.load(.duration)
            let seconds = CMTimeGetSeconds(duration)
            if seconds.isFinite && seconds > 0 {
                return Int(seconds * 1000)
            }
        } catch {
            print("Failed to get video duration: \(error)")
        }
        return nil
    }
    
    // MARK: - Add Music to Video
    
    func addMusicToVideo(
        result: GenerationResult,
        genre: MusicGenre,
        historyManager: HistoryManager,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws -> GenerationResult {
        isGenerating = true
        progress = 0
        statusMessage = "Starting music generation..."
        error = nil
        
        defer {
            isGenerating = false
        }
        
        // Get actual video duration from file
        progressHandler(0.05, "Analyzing video duration...")
        let fallbackDurationMs = Int((Double(result.parameters.numFrames) / Double(result.parameters.fps)) * 1000)
        let durationMs: Int
        if let actualDuration = await getVideoDurationMs(url: result.videoURL) {
            durationMs = actualDuration
            print("Using actual video duration: \(durationMs)ms")
        } else {
            durationMs = fallbackDurationMs
            print("Could not get video duration from file, using calculated: \(durationMs)ms")
        }
        
        // Generate music file path - use same directory as the video
        let videoDirectory = result.videoURL.deletingLastPathComponent()
        let musicFileName = "\(result.id.uuidString)_music.mp3"
        let musicPath = videoDirectory.appendingPathComponent(musicFileName).path
        
        // Generate music
        progressHandler(0.1, "Generating \(genre.displayName) music (\(durationMs / 1000)s)...")
        
        let musicURL = try await generateMusic(
            genre: genre,
            durationMs: durationMs,
            outputPath: musicPath
        ) { pct, msg in
            progressHandler(0.1 + pct * 0.4, msg)
        }
        
        // Merge music with video
        progressHandler(0.5, "Merging music with video...")
        
        let outputFileName = "\(result.id.uuidString)_with_music.mp4"
        let outputPath = videoDirectory.appendingPathComponent(outputFileName)
        
        // Use different merge based on whether video has voiceover
        if result.hasAudio, let audioURL = result.audioURL {
            // Merge video + voiceover + music (with music ducked)
            try await mergeVideoWithVoiceoverAndMusic(
                videoURL: result.videoURL,
                voiceoverURL: audioURL,
                musicURL: musicURL,
                outputURL: outputPath
            ) { pct, msg in
                progressHandler(0.5 + pct * 0.4, msg)
            }
        } else {
            // Just merge video + music
            try await mergeMusicWithVideo(
                videoURL: result.videoURL,
                musicURL: musicURL,
                outputURL: outputPath
            ) { pct, msg in
                progressHandler(0.5 + pct * 0.4, msg)
            }
        }
        
        // Update result with new video path and music info
        progressHandler(0.95, "Updating history...")
        
        let updatedResult = GenerationResult(
            id: result.id,
            requestId: result.requestId,
            prompt: result.prompt,
            negativePrompt: result.negativePrompt,
            voiceoverText: result.voiceoverText,
            voiceoverSource: result.voiceoverSource,
            voiceoverVoice: result.voiceoverVoice,
            parameters: result.parameters,
            videoPath: outputPath.path,
            thumbnailPath: result.thumbnailPath,
            audioPath: result.audioPath,
            musicPath: musicPath,
            musicGenre: genre.rawValue,
            createdAt: result.createdAt,
            completedAt: result.completedAt,
            duration: result.duration,
            seed: result.seed
        )
        
        progressHandler(1.0, "Complete!")
        
        return updatedResult
    }
    
    // MARK: - Music Merge (video + music only)
    
    func mergeMusicWithVideo(
        videoURL: URL,
        musicURL: URL,
        outputURL: URL,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws {
        progressHandler(0.1, "Merging music with video...")
        
        // Find ffmpeg
        let ffmpegPaths = [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg"
        ]
        
        guard let ffmpegPath = ffmpegPaths.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw AudioError.ffmpegFailed("FFmpeg not found. Install with: brew install ffmpeg")
        }
        
        // FFmpeg command: combine video and music
        // -y: overwrite output
        // -stream_loop -1: loop music if needed
        // -c:v copy: copy video stream without re-encoding
        // -c:a aac: encode audio as AAC
        // -shortest: stop when video ends
        // -af volume=0.3: reduce music volume to 30% for background
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpegPath)
        process.arguments = [
            "-y",
            "-i", videoURL.path,
            "-stream_loop", "-1",
            "-i", musicURL.path,
            "-c:v", "copy",
            "-filter:a", "volume=0.3",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-map", "0:v:0",
            "-map", "1:a:0",
            outputURL.path
        ]
        
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        process.standardOutput = Pipe()
        
        try process.run()
        
        progressHandler(0.5, "Processing...")
        
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            let errorData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let errorOutput = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw AudioError.ffmpegFailed(errorOutput)
        }
        
        progressHandler(1.0, "Music merge complete")
    }
    
    // MARK: - Full Audio Merge (video + voiceover + music)
    
    func mergeVideoWithVoiceoverAndMusic(
        videoURL: URL,
        voiceoverURL: URL,
        musicURL: URL,
        outputURL: URL,
        progressHandler: @escaping (Double, String) -> Void
    ) async throws {
        progressHandler(0.1, "Merging voiceover and music with video...")
        
        // Find ffmpeg
        let ffmpegPaths = [
            "/opt/homebrew/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/usr/bin/ffmpeg"
        ]
        
        guard let ffmpegPath = ffmpegPaths.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw AudioError.ffmpegFailed("FFmpeg not found. Install with: brew install ffmpeg")
        }
        
        // FFmpeg command: combine video, voiceover, and music
        // Music is ducked (lowered volume) and voiceover is kept at full volume
        // Using amix filter to blend audio tracks
        // Music loops via -stream_loop -1, -t matches video length
        // duration=longest ensures music continues after voiceover ends
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpegPath)
        process.arguments = [
            "-y",
            "-i", videoURL.path,
            "-i", voiceoverURL.path,
            "-stream_loop", "-1",
            "-i", musicURL.path,
            "-c:v", "copy",
            "-filter_complex", "[1:a]apad[voicepad];[voicepad]volume=1.0[voice];[2:a]volume=0.2[music];[voice][music]amix=inputs=2:duration=longest:dropout_transition=2[aout]",
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            outputURL.path
        ]
        
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        process.standardOutput = Pipe()
        
        try process.run()
        
        progressHandler(0.5, "Processing audio mix...")
        
        process.waitUntilExit()
        
        if process.terminationStatus != 0 {
            let errorData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let errorOutput = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw AudioError.ffmpegFailed(errorOutput)
        }
        
        progressHandler(1.0, "Audio merge complete")
    }
    
    // MARK: - Python Runner
    
    private func runPython(
        executable: String,
        script: String,
        timeout: TimeInterval,
        stderrHandler: ((String) -> Void)? = nil
    ) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: executable)
                process.arguments = ["-c", script]
                
                var env: [String: String] = [:]
                let pythonBin = URL(fileURLWithPath: executable).deletingLastPathComponent().path
                env["PATH"] = "\(pythonBin):/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin"
                env["HOME"] = ProcessInfo.processInfo.environment["HOME"] ?? ""
                env["USER"] = ProcessInfo.processInfo.environment["USER"] ?? ""
                env["TMPDIR"] = ProcessInfo.processInfo.environment["TMPDIR"] ?? "/tmp"
                process.environment = env
                
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe
                
                stderrPipe.fileHandleForReading.readabilityHandler = { handle in
                    let data = handle.availableData
                    if !data.isEmpty, let str = String(data: data, encoding: .utf8) {
                        stderrHandler?(str)
                    }
                }
                
                do {
                    try process.run()
                    process.waitUntilExit()
                    
                    stderrPipe.fileHandleForReading.readabilityHandler = nil
                    
                    let outputData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let output = String(data: outputData, encoding: .utf8) ?? ""
                    
                    if process.terminationStatus != 0 {
                        continuation.resume(throwing: AudioError.mlxAudioFailed("Process exited with code \(process.terminationStatus)"))
                    } else {
                        continuation.resume(returning: output.trimmingCharacters(in: .whitespacesAndNewlines))
                    }
                } catch {
                    continuation.resume(throwing: AudioError.mlxAudioFailed(error.localizedDescription))
                }
            }
        }
    }
}
