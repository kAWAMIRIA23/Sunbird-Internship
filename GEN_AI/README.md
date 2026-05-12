cd 'C:\Users\LENOVO\Desktop\rrr\New folder\GEN_AI'
.\venv\Scripts\Activate.ps1
python app.py# Sunbird GenAI Pipeline App

Small Generative AI web app powered by Sunbird AI APIs and the Sunflower LLM.

## Features

- Text or audio input mode
- Audio transcription via Sunbird Speech-to-Text (`/tasks/stt`)
- Summarization via Sunflower Simple Inference (`/tasks/sunflower_simple`)
- Translation to Ugandan languages (Luganda, Runyankole, Ateso, Lugbara, Acholi) via Sunflower
- Speech synthesis via Sunbird Text-to-Speech (`/tasks/tts`)
- Visible intermediate outputs:
  - Original text
  - Transcript (when audio input is used)
  - Summary
  - Translated summary
  - Playable generated audio
- Audio input validation:
  - Rejects files above 5 minutes when duration can be verified (WAV)
  - Rejects very large unknown-duration files as a safety fallback
- User-friendly API error surfacing

## Project Structure

```text
.
├── app.py
├── backend/
│   ├── __init__.py
│   ├── sunbird_client.py
│   └── pipeline.py
├── requirements.txt
└── .env.example
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file from `.env.example` and set your token:

```bash
SUNBIRD_API_TOKEN=your_real_token_here
```

## Run

```bash
python app.py
```

The app will open in your browser and let you process either text or audio.

## Notes

- Do not commit `.env` files or real API tokens.
- Generated TTS audio URL from Sunbird is temporary. This app downloads the bytes immediately and streams them in the UI.
