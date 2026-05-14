"""
Gradio UI for the Sunbird GenAI pipeline (UI + wiring only).

Orchestration lives in backend.pipeline; HTTP in backend.sunbird_client.
"""

from __future__ import annotations

import logging
import os
import uuid
import wave
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from backend.pipeline import GenAIPipeline, TARGET_LANGUAGES
from backend.sunbird_client import SunbirdApiError, SunbirdClient

logger = logging.getLogger(__name__)

# Scoped styles for the input column (Gradio 6: pass this to launch(css=...), not Blocks(css=...)).
INPUT_PANEL_CSS = """
#input-panel .gr-form { gap: 0.75rem; }
#input-panel .gr-box { border: none !important; box-shadow: none !important; }
#input-panel .gr-dropdown .secondary-wrap,
#input-panel .gr-dropdown button.secondary-wrap {
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    box-shadow: none !important;
}
#input-panel .gr-textbox .wrap {
    border-color: rgba(255, 255, 255, 0.12) !important;
    box-shadow: none !important;
}
#input-panel .gr-radio .wrap,
#input-panel .gr-radio .form {
    border: none !important;
    box-shadow: none !important;
}
"""


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


MAX_AUDIO_SECONDS = 5 * 60
MAX_UNKNOWN_DURATION_FILE_BYTES = 50 * 1024 * 1024


def estimate_audio_duration_seconds(audio_path: str) -> Optional[float]:
    """Estimate duration for WAV files only."""
    if not audio_path.lower().endswith(".wav"):
        return None
    with wave.open(audio_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        framerate = wav_file.getframerate()
        if framerate == 0:
            return None
        return frames / float(framerate)


def _normalize_mode(mode: str | int | None) -> str:
    """Gradio Radio usually returns 'Text'|'Audio'; normalize edge cases."""
    if mode is None:
        return "Text"
    s = str(mode).strip().lower()
    if s in ("text", "0", "t"):
        return "Text"
    return "Audio"


def run_pipeline(
    input_mode: str,
    text_input: str,
    audio_file: str,
    target_language: str,
    generate_speech: bool,
    progress: gr.Progress = gr.Progress(),
):
    """
    Gradio handler — runs exactly once per button click.

    Parameter order MUST match run_button.click(inputs=[...]):
    input_mode, text_input, audio_input, target_language, generate_speech
    """
    run_id = uuid.uuid4().hex[:12]
    mode = _normalize_mode(input_mode)

    if not os.getenv("SUNBIRD_API_TOKEN"):
        raise gr.Error("Missing SUNBIRD_API_TOKEN in environment. Add it to your .env file.")

    # Fresh client per run: no shared process-global state, no session reuse across runs.
    client = SunbirdClient()
    pipeline = GenAIPipeline(client=client, run_id=run_id)

    preview = (text_input or "")[:200].replace("\n", " ")
    logger.info(
        "run_id=%s handler=start mode=%s (raw=%r) lang=%s generate_speech=%s text_preview=%r audio_path=%r",
        run_id,
        mode,
        input_mode,
        target_language,
        generate_speech,
        preview,
        audio_file,
    )

    progress(0.02, desc="Starting…")

    try:
        if mode == "Text":
            if not text_input or not text_input.strip():
                raise gr.Error("Please enter text before running the pipeline.")
            progress(0.15, desc="Summarising / translating…")
            result = pipeline.run_with_text(
                input_text=text_input,
                target_language=target_language,
                generate_speech=generate_speech,
            )
        else:
            if not audio_file:
                raise gr.Error("Please upload an audio file before running the pipeline.")

            progress(0.1, desc="Transcribing…")
            duration_seconds = estimate_audio_duration_seconds(audio_file)
            if duration_seconds is not None and duration_seconds > MAX_AUDIO_SECONDS:
                raise gr.Error("Audio is longer than 5 minutes. Please upload a shorter file.")

            if duration_seconds is None:
                file_size = os.path.getsize(audio_file)
                if file_size > MAX_UNKNOWN_DURATION_FILE_BYTES:
                    raise gr.Error(
                        "Could not verify duration for this format and file is too large. "
                        "Please upload a file likely under 5 minutes."
                    )

            progress(0.2, desc="Summarising / translating…")
            result = pipeline.run_with_audio(
                audio_file_path=audio_file,
                target_language=target_language,
                generate_speech=generate_speech,
            )

        progress(0.95, desc="Preparing results…")
        transcript_display = result.transcript if result.transcript is not None else "N/A (text input mode)"
        logger.info(
            "run_id=%s handler=done orig_chars=%s summary_chars=%s trans_chars=%s audio_bytes=%s",
            run_id,
            len(result.original_text or ""),
            len(result.summary or ""),
            len(result.translated_summary or ""),
            len(result.tts_audio_bytes or b""),
        )
        audio_value = result.tts_audio_bytes if result.tts_audio_bytes else None
        return (
            result.original_text,
            transcript_display,
            result.summary,
            result.translated_summary,
            audio_value,
        )
    except SunbirdApiError as api_err:
        logger.exception("run_id=%s Sunbird API error", run_id)
        raise gr.Error(f"Sunbird API request failed: {api_err}") from api_err
    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("run_id=%s Unexpected pipeline error", run_id)
        raise gr.Error(f"Unexpected error: {exc}") from exc


def toggle_input_panels(mode: str):
    """Show only the panel for the selected mode. Uses Radio value directly on Run (no stale gr.State)."""
    use_text = mode == "Text"
    return gr.update(visible=use_text), gr.update(visible=not use_text)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Sunbird GenAI Pipeline") as app:
        gr.Markdown(
            """
            # Sunbird GenAI: STT → Summarise → Translate → TTS
            Choose **Text** or **Audio**, set the **target language**, then **Run pipeline** in the card below.
            """
        )

        with gr.Column(elem_id="input-panel"):
            gr.Markdown("**Input**")
            with gr.Row():
                input_mode = gr.Radio(
                    choices=["Text", "Audio"],
                    value="Text",
                    label="Mode",
                    scale=2,
                )
                target_language = gr.Dropdown(
                    choices=list(TARGET_LANGUAGES.keys()),
                    value="Luganda",
                    label="Language",
                    info="Ugandan language for translation and speech",
                    scale=1,
                )

            text_input = gr.Textbox(
                label="Text to summarise",
                lines=8,
                visible=True,
            )
            audio_input = gr.Audio(
                label="Upload audio (max ~5 minutes)",
                type="filepath",
                sources=["upload"],
                visible=False,
            )

            input_mode.change(
                fn=toggle_input_panels,
                inputs=[input_mode],
                outputs=[text_input, audio_input],
                queue=False,
                show_progress="hidden",
            )

            generate_speech = gr.Checkbox(
                value=True,
                label="Generate speech (TTS) — turn off for faster text-only results",
            )
            run_button = gr.Button("Run pipeline", variant="primary")

        gr.Markdown(
            "_Most wait time is Sunbird’s servers (network + models). "
            "Disable “Generate speech” if you only need text._"
        )

        gr.Markdown("## Pipeline outputs")
        original_text_output = gr.Textbox(label="Original text", lines=6)
        transcript_output = gr.Textbox(label="Transcript (audio input)", lines=6)
        summary_output = gr.Textbox(label="Summary", lines=6)
        translated_output = gr.Textbox(label="Translated summary", lines=6)
        audio_output = gr.Audio(label="Generated speech")

        run_button.click(
            fn=run_pipeline,
            inputs=[input_mode, text_input, audio_input, target_language, generate_speech],
            outputs=[
                original_text_output,
                transcript_output,
                summary_output,
                translated_output,
                audio_output,
            ],
            show_progress="minimal",
            api_name="run_pipeline",
        )

    return app


if __name__ == "__main__":
    _configure_logging()
    app = create_app()
    app.queue(max_size=8, default_concurrency_limit=1)
    app.launch(css=INPUT_PANEL_CSS)
