# Agent notes — ltx-video-mac

Native SwiftUI macOS app. Generation is a Python subprocess (`mlx-video-with-audio` on MLX). You are talking to a 20+ year full-stack owner: be concise, no tutorial filler.

## Repos

| Repo | Path | Role |
|---|---|---|
| This app | `/Users/jc/projects/ltx-video-mac` | SwiftUI shell, queue, UI, local REST API |
| Library | `/Users/jc/projects/mlx-video-with-audio` | Actual I2V/T2V + audio. Owner: same person |

`LTXBridge` prefers pip unless `~/projects/mlx-video-with-audio` is newer, Preferences “Use local mlx-video-with-audio repo” is on, or `LTX_FORCE_LOCAL_MLX_VIDEO=1`.

If a PR changes the Python CLI (`--keyframe`, kwargs on `generate_video_with_audio`, etc.), land and **publish the library first**. Shipping the app against an unreleased flag breaks I2V for everyone on PyPI.

## Library release (mlx-video-with-audio)

Do **not** run twine. Tag push publishes.

1. Merge the library PR to `main`
2. Bump `mlx_video/version.py` (commit style: `v0.1.37: short why`)
3. `git tag v<version> && git push origin main --tags`
4. Wait for `.github/workflows/publish.yml`, then `pip index versions mlx-video-with-audio`

Then pin the app: `mlxVideoMinVersion` in `LTXVideoGenerator/Sources/PythonEnvironment.swift` **and** `LTXVideoGenerator/requirements.txt`. Leave old `LTXBridge` error-hint versions alone unless the hint itself is wrong.

## App release (this repo)

Unreleased user-facing work on `main` needs a version tag. Last pattern: `2.3.67`.

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
- Cancel must kill the Python process group, not just the Swift `Task`

## Changelog and docs

Keep a Changelog + semver. User-facing behavior goes in `CHANGELOG.md` under Unreleased until the version tag. Mention Settings paths and min package versions when they change. README / `docs/installation.md` for storage and API examples.

## Git / `gh` / permissions

`gh` and `git` are allowlisted. Run them without `required_permissions` or smart-mode prompts unless a command actually fails. Never `git config`, never `--no-verify`, never force-push `main`. No commit trailers / Co-authored-by.

Do not commit unless the user asks, except when they already asked to cut a release or land the version pin as part of that task.

## Debug first

```bash
cat /tmp/ltx_generation.log
pip show mlx-video-with-audio
```
