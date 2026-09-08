# Agent notes — ltx-video-mac

Native SwiftUI macOS app (14+). Generation is a local subprocess — not in-process MLX. You are talking to a 20+ year full-stack owner: be concise, no tutorial filler.

Three backends via `GenerationBackend` on `LTXModel`:

| Backend | When | Process |
|---|---|---|
| `mlxVideoWithAudio` | LTX-2 / 2.3 `notapalindrome` packs | `python -m mlx_video.generate_av` |
| `ltx2Mlx` | LTX-2.5 + LTX-2.3 12GB | `ltx-2-mlx generate` (or adapter wrapper) |
| `h3c` | MiniMax H3 | native `./h3` (C/Metal) |

## Repos

| Repo | Typical path | Role |
|---|---|---|
| This app | `/Users/jc/p/ltx-video-mac` (or clone anywhere) | SwiftUI shell, queue, UI, local REST API |
| Library | `/Users/jc/p/mlx-video-with-audio` | LTX-2 / 2.3 I2V/T2V + audio (PyPI `mlx-video-with-audio`) |
| LTX-2.5 runtime | `~/projects/ltx-2-mlx` (Preferences local toggle) | `dgrauet/ltx-2-mlx` — do **not** port 2.5 into mlx-video-with-audio |
| H3 engine | `~/projects/h3.c` or App Support clone | `antirez/h3.c` (BF16/Turbo); int8 fork for `minimax_h3_int8` |

**Preference hardcodes:** local overrides look at `~/projects/mlx-video-with-audio` and `~/projects/ltx-2-mlx` regardless of where this git checkout lives. Symlink or clone there when using the toggles.

`LTXBridge` prefers pip `mlx-video-with-audio` unless `~/projects/mlx-video-with-audio` is newer, Preferences “Use local mlx-video-with-audio repo” is on, or `LTX_FORCE_LOCAL_MLX_VIDEO=1`.

If a PR changes the Python CLI (`--keyframe`, kwargs on `generate_video_with_audio`, etc.), land and **publish the library first**. Shipping the app against an unreleased flag breaks I2V for everyone on PyPI.

## LTX-2.5 reference

Upstream: [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (CUDA `ltx-pipelines` / Diffusers / Comfy). Mac path: MLX conversion [mlx-community/ltx-2.5-mlx](https://huggingface.co/mlx-community/ltx-2.5-mlx) + runtime [dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) `@v0.15.2`. Plan: `plans/ltx-2.5-support.md`.

### Catalog ↔ official model family

| App `model_id` | Pack | CLI shape | Maps to official |
|---|---|---|---|
| `ltx25_distilled` | `mlx-community/ltx-2.5-mlx` (~100GB) | `--distilled` | Distilled DiT, fixed **8 steps, CFG=1** |
| `ltx25_dev` | same + overlay LoRA ~8.3GB | `--two-stage` + `transformer-dev` + fuse `ltx-2.5-22b-distilled-lora-450-bf16` | Dev two-stage (half-res + spatial upscale + distilled-LoRA refine); stage-1 steps/CFG from UI (defaults 30 / 3), stage-2 = 3 |
| `ltx25_distilled_ditq8` | same + `mlx-community/ltx-2.5-mlx-ditq8` overlay | `--distilled` on shadowed pack | Distilled + Q8 DiT (0.15.2 has no `--dit`) |

Gemma 4 is **bundled** in `gemma4-12b-ltx-v1/`. Do not use the Gemma 3 picker for 2.5. `mlx-community/ltx-2.5-mlx-q8` is the **text encoder**, not a DiT quant — never catalog it as DiT.

App loads the community pack through bundled `LTXVideoGenerator/Resources/ltx25_community_adapter.py` (patches Gemma 4 path + mixed-precision quant + DurationHead skip). Keep that file in the Xcode Resources target.

### Implemented vs official “what's new”

| Official feature | Status here |
|---|---|
| Distilled 8-step / CFG=1 | Yes |
| Dev + distilled LoRA two-stage | Yes (`ltx25_dev`; LoRA from `dgrauet/ltx-2.5-mlx` only — gated) |
| Gemma 4 12B TE | Yes (pack + adapter) |
| Prompt enhancer | Yes → `--enhance-prompt` (not Gemma 3 preview path) |
| Audio VAE + vocoder | Yes (in pack) |
| DiffVAE + spatial/temporal upscalers | In pack; two-stage uses spatial upscaler |
| I2V first-frame | Yes → `--image` |
| `num_frames % 8 == 1`; W/H ÷32 | Defaults OK; app snaps W/H to **64** (stricter) |
| Extra timeline keyframes | Yes — repeatable `--image PATH FRAME_IDX STRENGTH` (pixel idx; mlx-video uses latent `--keyframe`) |
| Duration predictor (omit frames) | Load fixed in adapter for community split q/k/v; app still passes `-f` from slider ([upstream #125](https://github.com/dgrauet/ltx-2-mlx/issues/125)) |
| Native multishot / DFR | **Not wired** — no CLI in 0.15.2 ([upstream #127](https://github.com/dgrauet/ltx-2-mlx/issues/127)). `--segment` Prompt Relay not in UI yet |
| Disable audio | **Ignored** on 0.15.2 ([upstream #126](https://github.com/dgrauet/ltx-2-mlx/issues/126)) |

Parity epic: https://github.com/james-see/ltx-video-mac/issues/85

**Prompting:** official LTX-2.5 style only — [ltx.io](https://ltx.io/blog/ltx-2-5-prompt-guide) / [docs.ltx.io](https://docs.ltx.io/api-documentation/implementation-guides/prompting-guide). App copy: `EXAMPLES.md`, `docs/usage.md`, README tips. Flowing present-tense paragraphs, quoted dialogue, prose hard cuts. No `START FRAME` / JUMP CUT lists. Prefer single continuous take on Mac Distilled/Dev until multishot/DFR is wired.

Do **not** port 2.5 into `mlx-video-with-audio`. Pin install: `Ltx2MlxInstall` in `PythonEnvironment.swift` (`v0.15.2` core + pipelines) + `mlx-lm>=0.31.2`. Only required when a 2.5 / 12GB model is selected.

## Dev / build

### Day-to-day (Swift)

```bash
cd /Users/jc/p/ltx-video-mac   # or your clone
open LTXVideoGenerator/LTXVideoGenerator.xcodeproj
# Scheme: LTXVideoGenerator · Debug · My Mac (Apple Silicon)
```

Xcode project is the source of truth for shipping (Resources, signing). `LTXVideoGenerator/Package.swift` exists for SPM/PythonKit experiments — do not assume SPM alone ships the app bundle (`ltx25_community_adapter.py`, icons, etc.).

Before push of compile-affecting changes: build succeeds in Xcode or:

```bash
xcodebuild -project LTXVideoGenerator/LTXVideoGenerator.xcodeproj \
  -scheme LTXVideoGenerator -configuration Debug build
```

### Local signed / notarized DMG

```bash
./scripts/build-local.sh          # archive + Developer ID + notarytool + DMG → build/
./scripts/build-release.sh        # CI-oriented (used by .github/workflows/release.yml)
./scripts/setup-secrets.sh        # notarytool keychain profile (once)
```

### Python env the app uses

Point Preferences at a venv (or Auto Detect). Install base deps:

```bash
pip install -r LTXVideoGenerator/requirements.txt
```

LTX-2.5 / 12GB (opt-in; app can auto-install on model select):

```bash
pip install \
  "git+https://github.com/dgrauet/ltx-2-mlx.git@v0.15.2#subdirectory=packages/ltx-core-mlx" \
  "git+https://github.com/dgrauet/ltx-2-mlx.git@v0.15.2#subdirectory=packages/ltx-pipelines-mlx" \
  "mlx-lm>=0.31.2"
```

Or: clone `dgrauet/ltx-2-mlx` → `~/projects/ltx-2-mlx`, `uv sync --all-extras`, enable **Use local ltx-2-mlx repo**.

Library local: clone → `~/projects/mlx-video-with-audio` (or symlink from `/Users/jc/p/mlx-video-with-audio`), enable **Use local mlx-video-with-audio repo** when iterating CLI flags.

H3: Xcode CLT + first Generate clones/builds `h3`, or set **h3 binary path** / **H3 model directory**. Gated HF weights need `hf auth login`.

### Layout agents touch most

| Path | Notes |
|---|---|
| `LTXVideoGenerator/Sources/Services/LTXBridge.swift` | Backend dispatch, embedded Python scripts |
| `LTXVideoGenerator/Sources/PythonEnvironment.swift` | Pins, validate, optional git installs |
| `LTXVideoGenerator/Sources/Models/GenerationRequest.swift` | Catalog (`LTXModel`, backends) |
| `LTXVideoGenerator/Resources/ltx25_community_adapter.py` | Must stay in app Resources |
| `LTXVideoGenerator/requirements.txt` | Pip pins (keep in sync with `mlxVideoMinVersion`) |

Python: format with `black`. New Swift files → add to the Xcode target or release archives omit them.

## Dependencies

### Host

- macOS 14+, Apple Silicon
- Xcode (dev) / Xcode CLT (H3 `make`)
- Python **3.10+** (Preferences path; prefer 3.12 Homebrew/pyenv)
- Optional: `uv` for local `ltx-2-mlx`

### Pip (always — `requirements.txt`)

`mlx`, `mlx-vlm`, `mlx-lm>=0.31.2`, `mlx-video-with-audio>=0.1.37`, `transformers`, `safetensors`, `huggingface_hub`, `numpy`, `Pillow`, `opencv-python`, `tqdm`, `mlx-audio`

Pin source of truth: `mlxVideoMinVersion` in `PythonEnvironment.swift` **and** `requirements.txt`.

### Opt-in

| Need | Dep |
|---|---|
| LTX-2.5 / 12GB | `ltx-core-mlx` + `ltx-pipelines-mlx` @ `v0.15.2` (git; not PyPI) |
| MiniMax H3 | local `h3` binary + HF weights (~92–144GB) |
| Voiceover / music | ElevenLabs keys and/or `mlx-audio` |

### Weights (HF cache via `HuggingFaceCacheConfiguration`)

Default `~/.cache/huggingface/`; override with Preferences Model Cache Directory (`HF_HOME` + `HF_HUB_CACHE` on every download Process). Fail closed if configured folder missing/unwritable.

## Library release (mlx-video-with-audio)

Do **not** run twine. Tag push publishes.

1. Merge the library PR to `main`
2. Bump `mlx_video/version.py` (commit style: `v0.1.37: short why`)
3. `git tag v<version> && git push origin main --tags`
4. Wait for `.github/workflows/publish.yml`, then `pip index versions mlx-video-with-audio`

Then pin the app: `mlxVideoMinVersion` in `LTXVideoGenerator/Sources/PythonEnvironment.swift` **and** `LTXVideoGenerator/requirements.txt`. Leave old `LTXBridge` error-hint versions alone unless the hint itself is wrong.

## App release (this repo)

Unreleased user-facing work on `main` needs a version tag. Current marketing version lives in `project.pbxproj` (`MARKETING_VERSION`; last shipped pattern `2.3.x`).

1. Move `CHANGELOG.md` `## Unreleased` into `## [X.Y.Z] - YYYY-MM-DD` (today is 2026+)
2. Bump both `MARKETING_VERSION` entries in `LTXVideoGenerator/LTXVideoGenerator.xcodeproj/project.pbxproj`
3. Commit: `Bump version to X.Y.Z, update CHANGELOG`
4. `git tag vX.Y.Z && git push origin main --tags`
5. `.github/workflows/release.yml` notarizes and attaches the DMG (~2 min)

Do not bump `CURRENT_PROJECT_VERSION` unless asked. Merge commits (`gh pr merge --merge`), not squash — that is the existing history.

## Pull requests

Default: review, thank the author by name, then merge or close with a concrete reason. No empty “LGTM”.

- Checkout the PR (`gh pr checkout N`) and read `gh pr diff`, not `git diff origin/main` (stale branches look huge)
- Approve with what was good and that it is merging (or why it is blocked)
- Independent app PRs can merge in any order
- Shared files: `LTXBridge.swift`, `PromptInputView.swift`, `README.md`. If GitHub conflicts after earlier merges, resolve on a maintainer branch and credit the author — do not force-push their fork
- Queue is already single-flight (`processNextIfNeeded`). “Add to Queue while generating” is correct; do not re-disable it to “fix” concurrency

## Code

- SwiftUI only. Keep Xcode project membership in sync when adding files or release archives miss them
- No ORMs. Direct code, explicit subprocess env
- Python: format with `black`
- Hugging Face cache: `HuggingFaceCacheConfiguration` (`HF_HOME` + `HF_HUB_CACHE`). Apply it on every Process env that can download weights (generation, Gemma preview, MLX Audio). Fail closed if the configured folder is missing/unwritable
- REST API (`APIServer.swift`) binds `127.0.0.1` because `/generate` accepts local file paths. Do not bind `0.0.0.0` again
- Generation log: `/tmp/ltx_generation.log`. User-facing alerts stay short
- Cancel must kill the Python process group (or H3 process), not just the Swift `Task`

## Changelog and docs

Keep a Changelog + semver. User-facing behavior goes in `CHANGELOG.md` under Unreleased until the version tag. Mention Settings paths and min package versions when they change. README / `docs/installation.md` for storage and API examples. Architecture: `docs/architecture.md`.

## Commit style

Subject only. One line. Imperative. Why, not a file list. No body unless the user asks. No trailers, no `Co-authored-by`.

```
Add AGENTS.md with release, PR, and library-pin workflow
Require mlx-video-with-audio 0.1.37 for keyframe I2V
Fix #75: pause/cleanup video player on view disappear
Add higher resolution options + warning (#68)
```

Fixed phrases — do not invent variants:

| Case | Message |
|---|---|
| App version tag | `Bump version to X.Y.Z, update CHANGELOG` |
| Library version | `v0.1.37: short why` |
| `gh pr merge --merge` | leave GitHub’s `Merge pull request #N from …` |

Optional `feat:` / `fix:` / `docs:` / `chore:` is fine on contributor PRs; maintainer commits on `main` prefer the sentence form above. Pass the message via HEREDOC. Never `--amend` unless the user asked and the last commit is yours and unpushed.

Do not commit unless the user asks, except when they already asked to cut a release or land the version pin as part of that task.

## Git / `gh` / permissions

`gh` and `git` are allowlisted. Run them without `required_permissions` or smart-mode prompts unless a command actually fails. Never `git config`, never `--no-verify`, never force-push `main`. No commit trailers / Co-authored-by.

## Debug first

```bash
cat /tmp/ltx_generation.log
pip show mlx-video-with-audio
pip show ltx-pipelines-mlx ltx-core-mlx 2>/dev/null || true
which ltx-2-mlx; ltx-2-mlx --help 2>/dev/null | head
```
