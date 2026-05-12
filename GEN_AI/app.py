import os
import tempfile
import wave
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

from backend.pipeline import GenAIPipeline, TARGET_LANGUAGES
from backend.sunbird_client import SunbirdApiError, SunbirdClient


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


def run_pipeline(input_mode: str, text_input: str, audio_file: str, target_language: str):
    if not os.getenv("SUNBIRD_API_TOKEN"):
        raise gr.Error("Missing SUNBIRD_API_TOKEN in environment. Add it to your .env file.")

    client = SunbirdClient()
    pipeline = GenAIPipeline(client=client)

    try:
        if input_mode == "Text":
            if not text_input or not text_input.strip():
                raise gr.Error("Please enter text before running the pipeline.")
            result = pipeline.run_with_text(input_text=text_input, target_language=target_language)
        else:
            if not audio_file:
                raise gr.Error("Please upload an audio file before running the pipeline.")

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

            result = pipeline.run_with_audio(audio_file_path=audio_file, target_language=target_language)

        transcript = result.transcript if result.transcript is not None else "N/A (text input mode)"
        return (
            result.original_text,
            transcript,
            result.summary,
            result.translated_summary,
            result.tts_audio_bytes,
        )
    except SunbirdApiError as api_err:
        raise gr.Error(f"Sunbird API request failed: {api_err}") from api_err
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Unexpected error: {exc}") from exc


def toggle_inputs(input_mode: str):
    return (
        gr.update(visible=input_mode == "Text"),
        gr.update(visible=input_mode == "Audio"),
    )


def create_app() -> gr.Blocks:
    load_dotenv()

    with gr.Blocks(title="Sunbird GenAI Pipeline") as app:
        gr.Markdown(
            """
            # Sunbird GenAI: STT -> Summarise -> Translate -> TTS
            Provide text or upload audio, then process with Sunbird APIs to generate
            a translated spoken summary.
            """
        )

        with gr.Row():
            input_mode = gr.Radio(
                choices=["Text", "Audio"],
                value="Text",
                label="Input type",
            )
            target_language = gr.Dropdown(
                choices=list(TARGET_LANGUAGES.keys()),
                value="Luganda",
                label="Target Ugandan language",
            )

        text_input = gr.Textbox(
            label="Enter text to summarise",
            lines=8,
            visible=True,
        )
        audio_input = gr.Audio(
            label="Upload audio file (max 5 minutes)",
            type="filepath",
            sources=["upload"],
            visible=False,
        )

        run_button = gr.Button("Run pipeline", variant="primary")

        gr.Markdown("## Pipeline outputs")
        original_text_output = gr.Textbox(label="Original text", lines=6)
        transcript_output = gr.Textbox(label="Transcript (audio input)", lines=6)
        summary_output = gr.Textbox(label="Summary", lines=6)
        translated_output = gr.Textbox(label="Translated summary", lines=6)
        audio_output = gr.Audio(label="Generated speech")

        input_mode.change(
            fn=toggle_inputs,
            inputs=[input_mode],
            outputs=[text_input, audio_input],
        )

        run_button.click(
            fn=run_pipeline,
            inputs=[input_mode, text_input, audio_input, target_language],
            outputs=[
                original_text_output,
                transcript_output,
                summary_output,
                translated_output,
                audio_output,
            ],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
