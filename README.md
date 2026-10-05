# PromptToVideo Studio

A local-first React/Vite and Express studio that turns a prompt into a narrated, captioned video. The API persists jobs on disk, streams stage progress over SSE, and starts Python workers for the generation pipeline. No paid API key is required; cloud providers are not part of the default pipeline.

## What runs locally

- **Script:** Ollama with `qwen2.5:7b` by default. Strict JSON, language, scene-count, placeholder, narration-length, and English image-prompt checks run on every response. Invalid output is retried twice with feedback; persistent errors fail the job.
- **Images:** SDXL-Turbo through Diffusers, FP16, CPU offload, attention slicing, and VAE slicing/tiling. Models load from cache only. Image failures, blank frames, and unresolved near-duplicates fail QA; no gradient/placeholder image is substituted.
- **Voice:** Piper with separate local male/female voices for English and Hindi. Voice audio is synthesized before images and drives scene timing.
- **Captions:** faster-whisper runs locally from cache only. Caption timing, language, line count, word count, safe margins, overlap, and Hindi font/glyph coverage are validated; missing Whisper is an error.
- **Music and SFX:** locally generated mood beds and transition sounds. No track is downloaded or required.
- **Render:** FFmpeg with scene motion, transitions, one-pass bottom-center captions, narration/SFX/music mix, AAC audio, H.264, and ffprobe output validation. NVENC is attempted first; libx264 is the fallback.

The UI includes Shorts and landscape modes, prompt/style/language/narrator controls, render quality, music mood, effects level, SSE progress, live scene previews, cancellation, retry, re-render, and deletion. Jobs, logs, cache, and output files are stored locally.

## Requirements

- Windows 10/11 or Linux
- Node.js 20+ and npm
- Python 3.10+ (Python 3.12 is used by the included setup scripts)
- NVIDIA GPU with 8 GB VRAM recommended for SDXL; CPU-only use is not the intended performance target
- FFmpeg and ffprobe. `imageio-ffmpeg` supplies FFmpeg; install FFmpeg separately if `ffprobe` is not already available.
- Several GB of disk space for Python packages, model weights, cached assets, and rendered videos
- Ollama, configured model weights, the selected Piper voice pair, FFmpeg/ffprobe, SDXL/Whisper caches, at least 5 GiB free disk, and 6 GiB free RAM are required before a job starts. Preflight fails explicitly rather than generating substitute content.

## Windows setup

From PowerShell in the project directory:

```powershell
npm install
.\scripts\setup.ps1
```

Install CUDA-enabled PyTorch and pull the local language model separately:

```powershell
.\scripts\setup.ps1 -InstallML -Models
ollama pull qwen2.5:7b
```

Install FFmpeg if needed and ensure `ffprobe.exe` is on `PATH`. The setup script installs Python dependencies and downloads the configured Piper voices. `-Models` caches the faster-whisper model; SDXL-Turbo weights are downloaded and cached the first time an image job runs.

## Linux setup

Install Node.js, Python, FFmpeg/ffprobe, and Ollama using your distribution's supported packages first. Then run:

```bash
npm install
INSTALL_ML=1 DOWNLOAD_MODELS=1 bash scripts/setup.sh
ollama pull qwen2.5:7b
```

`INSTALL_ML=1` installs the CUDA 12.4 PyTorch wheels. Adjust that for the CUDA/driver combination supported by your machine. The included install command targets NVIDIA CUDA; CPU-only machines can omit it but cannot use SDXL as configured.

## Doctor and run

Run the local prerequisite checklist:

```powershell
npm run doctor
```

Start the Express API and Vite frontend together:

```powershell
npm run dev
```

Open `http://localhost:5173`. The API listens on `http://127.0.0.1:8000`. The home page displays the doctor checks for CUDA, Ollama/model, SDXL, Piper/voices, Whisper, fonts, FFmpeg, and ffprobe. Each queued job also runs a language/voice-specific offline preflight. Job folders contain `qa_report.json`, `language_voice.json`, `script.json` (only after valid Ollama output), and image-prompt debug artifacts as those stages complete. QA failures never produce a Ready state; post-render issues appear as Needs review with the reason.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend\tests -v
npm run test:api
npm run build
node --check backend\server.js
```

The Python tests cover worker-level behavior and existing backend units. They do not download or exercise the multi-gigabyte SDXL model. `logs/job-<id>.log` contains structured per-job stage/provider details and process stderr; `jobs/<id>/job.json` stores resumable job state; `outputs/<id>/` stores the MP4, thumbnail, metadata, previews, and scene clips.

## API

- `GET /api/health` — cached local doctor report
- `GET /api/jobs` and `GET /api/jobs/:id`
- `POST /api/jobs` — validate and enqueue a local job
- `GET /api/jobs/:id/events` — SSE state/stage updates
- `POST /api/jobs/:id/cancel`
- `POST /api/jobs/:id/retry`
- `POST /api/jobs/:id/scenes/:scene/retry`
- `POST /api/jobs/:id/rerender`
- `GET /api/jobs/:id/download`
- `DELETE /api/jobs/:id`

## Performance and limitations

Render time depends on model initialization, CPU offload, resolution, scene count, disk speed, and VRAM pressure; no duration is guaranteed. Full model-backed render checks require sufficient free resources and preinstalled local weights, and the tests do not download or exercise multi-gigabyte models. Music/SFX are procedural rather than a bundled royalty-free library. Dynamic quality/effects options adjust inference steps and render filters but are not a substitute for a dedicated high-end generation model. The app is a local development tool and does not include accounts or public-server security controls.
