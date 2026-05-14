# Sunbird GenAI Pipeline (Internship Part 2)

A small **Gradio** web app that runs text or audio through Sunbird AI: optional **speech-to-text**, **summarisation**, **translation** into a Ugandan language, and **text-to-speech**. Intermediate results (transcript, summary, translation, audio) are shown in the UI.

**Live demo (Hugging Face Space):** [huggingface.co/spaces/mariakawa/sunbird](https://huggingface.co/spaces/mariakawa/sunbird)

## Architecture

| Step | Sunbird endpoint | Role |
|------|------------------|------|
| Optional STT | `POST /tasks/stt` | Audio → transcript |
| Summarise + translate | `POST /tasks/sunflower_simple` | One combined LLM call when possible; fallback to two calls if parsing fails |
| TTS | `POST /tasks/tts` | Translated text → audio URL |
| Download | `GET` (audio URL) | Bytes for the Gradio player |

Flow: **Input** → (STT if audio) → **summary** → **translation** → **TTS** → **outputs**.

Code layout:

```text
GEN_AI/
├── app.py                 # Gradio UI, layout, run handler
├── backend/
│   ├── sunbird_client.py  # HTTP client: timeouts, no response caching
│   └── pipeline.py        # Orchestration + stage timing logs
├── requirements.txt
└── .env.example
```

## Requirements

- Python **3.10+**
- A **Sunbird API** token from [api.sunbird.ai](https://api.sunbird.ai/) (API Key after login)

## Local setup

```powershell
cd "path\to\Sunbird-Internship\GEN_AI"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env: set SUNBIRD_API_TOKEN (one line, no "Bearer " prefix, no trailing newline)
python app.py
```

On macOS/Linux:

```bash
cd GEN_AI && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
# edit .env then:
python app.py
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUNBIRD_API_TOKEN` | Yes | Bearer token value only (the app adds `Authorization: Bearer …`). |
| `LOG_LEVEL` | No | Default `INFO`. Use `DEBUG` for verbose logs. |
| `GENAI_MAX_SOURCE_CHARS` | No | Cap on characters sent to the model (default `16000`). |
| `SUNBIRD_CONNECT_TIMEOUT` | No | Connect timeout seconds (default `10`). |
| `SUNBIRD_READ_TIMEOUT_LLM` | No | LLM read timeout (default `60`). |
| `SUNBIRD_READ_TIMEOUT_STT` | No | STT read timeout (default `120`). |
| `SUNBIRD_READ_TIMEOUT_TTS` | No | TTS read timeout (default `60`). |
| `SUNBIRD_READ_TIMEOUT_DOWNLOAD` | No | Audio download read timeout (default `90`). |

See `.env.example` for copy-paste hints.

## Hugging Face Space

1. Create a Space (Gradio SDK, `app.py` entry).
2. Under **Settings → Variables and secrets**, add secret **`SUNBIRD_API_TOKEN`** (same rules as `.env`).
3. Push this folder’s contents to the Space repo (or sync from your monorepo).

Do **not** commit `.env` or real tokens to git.

## Usage (short)

1. Choose **Text** or **Audio** mode, pick **Language**, optionally disable **Generate speech** for faster text-only runs.
2. Enter text or upload audio (max ~**5 minutes** when duration can be checked for WAV).
3. Click **Run pipeline** and wait for Sunbird responses (latency is mostly server-side).

## Known limitations

- **Latency:** Total time depends on Sunbird STT/LLM/TTS; the UI cannot guarantee sub‑10s runs.
- **Audio length:** ~5 minute cap when WAV duration is known; large unknown-duration files are rejected as a safety measure.
- **Combined LLM parse:** If the model omits delimiter markers, the app falls back to two separate LLM calls (slower but same outputs).

## Security

Never commit `.env` or tokens. `.gitignore` excludes `.env` and virtual environments.

## Related repository

Part 1 exercises and the original assessment brief live in **[internship-assessment](https://github.com/kAWAMIRIA23/internship-assessment)**. This `GEN_AI` app satisfies **Part 2** of that assessment.
