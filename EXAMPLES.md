# Example Prompts

Copy-paste prompts for LTX Video Generator. Style follows **official LTX-2.5** prompting ([ltx.io guide](https://ltx.io/blog/ltx-2-5-prompt-guide), [docs.ltx.io](https://docs.ltx.io/api-documentation/implementation-guides/prompting-guide)) — not older shot-list templates.

**Mac note:** LTX-2.5 Distilled / Dev run via `ltx-2-mlx`. Prefer a **single continuous take** for dialogue + face reliability. Official prose multishot works on CUDA LTX-2.5; native multishot / Prompt Relay / DFR on Mac is still incomplete.

---

## Quick rules (2.5)

| Do | Don’t |
|---|---|
| One flowing present-tense paragraph (~4–8 sentences for a single take) | Tag soup written for another video model |
| Cover: shot · scene/lighting · action · character (physical cues) · camera · audio | Abstract emotion labels (“sad”, “tense”) without body language |
| Spoken lines **only** in `"quotes"`; name language / accent / delivery | Bare stage tokens that can be spoken (`Beat.`, `CUT TO:`) |
| Pauses as action: *he pauses*, *holds the silence*, *a beat of silence* | Shot lists, `START FRAME (0–2.5s)`, numbered JUMP CUTs |
| Multishot (when you want cuts): name the edit in prose — *A hard cut transitions to…* — re-ID subjects + audio continuity | Sluglines / numbered beats **unless** the cut is also named in prose |

Optional: screenplay-style headers + `Character: "line"` for longer dialogue scenes (official samples do this). Leave **Prompt Enhancement** off when the prompt is already this dense.

---

## 1. Diner — "Maybe It Is a Spirit" (single take)

**Genre:** Cinematic drama, dialogue-driven  
**Tone:** Warm exhaustion, the hush after something impossible

```
Medium two-shot, about 35mm, slight dolly in. A quiet roadside diner at dusk; wood-paneled walls, amber pendant light over a black vinyl booth, slatted blinds casting horizontal amber bars. Soft refrigeration hum, faint wind chimes. Two coffee cups on a wooden table. On the left, an old man in his seventies — white hair, denim jacket, deep lined face, small tired smile — sits with elbows on the table, eyes down on his coffee. Across from him, his son in his early forties — flannel over a t-shirt, warm curious face — leans forward, staring at his father. The old man speaks quietly in deep American English, "I've seen one before. A long time ago." The son answers low and measured, "When?" The old man softens, nostalgic, "The night your mother passed." The son’s eyes wet; he whispers, barely audible. The old man looks up with a warm gentle smile, "Maybe it is a spirit." Camera holds the two-shot; only their hands and faces move.
```

---

## 2. Desert highway — "Watch for the Tall Beings" (prose multishot)

Official 2.5 multishot: **one chronological paragraph**, cuts named in prose. Prefer example 3 for Mac dialogue reliability.

```
Medium close-up, about 50mm, from inside a car looking out the open driver-side window onto a late-afternoon Nevada desert highway — bleached sky, tan dust, faded asphalt, no other cars. Cinematic realism, desaturated warm tones. A police officer leans into frame — mirrored aviators, thick brown moustache, beige uniform — his cruiser idling behind with the light bar off; heat shimmer ripples off the road. Soft engine idle and dry silence. He speaks steady and matter-of-fact in American English, "License looks fine. But I need you to keep your speed down through here." He shifts weight, one hand on the door frame. A hard cut transitions to a tight close-up, about 85mm, of the officer’s face from just below the aviators to the moustache; a bead of sweat traces his temple, his jaw tightens. The idle continues across the cut. He drops half a register, quieter, "And I'm gonna tell you something else. Watch for the tall beings out in these parts." His eyes flick once toward the open desert, then back. A hard cut jumps to a reverse close-up through the window frame of the driver — mid-thirties, slightly sunburned, plain t-shirt — expression shifting from mild confusion to a cautious nod as he swallows; ambient engine noise continues. He speaks measured and respectful, "Ok sir, thank you," and nods once, hands gripping the wheel tighter. A hard cut to a wider shot from behind the car: the officer steps back, pats the roof, and walks toward his cruiser; the empty two-lane stretches to a vanishing point. Dialogue drops; only faint idle and dry silence remain. In the far distance, heat distortion makes a vague vertical shape shimmer and dissolve. Camera holds static.
```

---

## 3. Street corner — "They're coming" (single CU, Mac-friendly)

Best pattern for LTX-2.5 Distilled / Dev on this app: one framing, quoted dialogue, pauses as action.

```
Handheld medium close-up, about 85mm, cinematic realism. Cool grey-blue overcast light on a Colorado Springs street corner; the Front Range soft behind low rooftops. A busker in his late 40s fills the frame — weathered face, deep lines, grey stubble, old canvas jacket over a hoodie, fingerless gloves, battered acoustic guitar on a strap. He stops mid-riff; his fingers freeze on the strings. He slowly lifts his eyes past the camera toward the mountains; his face goes still. Soft traffic hum and wind; the rough bluesy guitar fades as he stops. He speaks in a low urgent American-English whisper, "They're coming." He pauses and holds the silence, eyes wet, unblinking. He continues, quieter and heavier, "The tall ones are back." Static handheld with a slight tremor; he stays planted, only his head and eyes move. Crisp intimate audio, faint outdoor room tone, no music.
```

---

## 4. News hit — oil geyser (screenplay-style, official sample shape)

```
EXT. TOWN STREET – MORNING – LIVE NEWS BROADCAST

The shot opens on a news reporter standing in front of a row of cordoned-off cars, yellow caution tape fluttering behind him. Warm early sun catches the lens. A faint hum of chatter and distant drilling. The reporter, composed but smiling nervously, looks into the camera, microphone in hand.
Reporter: "Thank you, Sylvia. And yes — this is a sentence I never thought I'd say on live television — but this morning, here in the quiet town of New Castle, Vermont… black gold has been found!"

He gestures toward the field behind him. "If my cameraman can pan over, you'll see what all the excitement's about." The camera pans right, slowly revealing a construction site and workers in hard hats. A beat of silence — then a geyser of oil erupts upward in a violent plume.
Workers cheer and scramble. Reporter (off-screen, shouting): "There it is, folks — a moment New Castle will never forget!" The camera catches sunlight on the oil mist, then pulls back to the whole scene: a small town against the fountain of oil.
```
