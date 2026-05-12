from dataclasses import dataclass
import time
from functools import wraps
from typing import Optional

from backend.sunbird_client import SunbirdClient


TARGET_LANGUAGES = {
    "Luganda": {"code": "lug", "speaker_id": 248},
    "Runyankole": {"code": "nyn", "speaker_id": 243},
    "Ateso": {"code": "teo", "speaker_id": 242},
    "Lugbara": {"code": "lgg", "speaker_id": 245},
    "Acholi": {"code": "ach", "speaker_id": 241},
}


@dataclass
class PipelineResult:
    original_text: str
    transcript: Optional[str]
    summary: str
    translated_summary: str
    tts_audio_bytes: bytes
    tts_audio_mime: str = "audio/wav"


class GenAIPipeline:
    def __init__(self, client: SunbirdClient) -> None:
        self.client = client

    def _timed(self, name: str):
        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start
                print(f"[timing] {name}: {duration:.3f}s")
                return result

            return wrapped

        return decorator

    def summarize(self, text: str) -> str:
        instruction = (
            "Summarize the following text clearly in 3-5 sentences. "
            "Keep key points and avoid adding new information.\n\n"
            f"Text:\n{text}"
        )
        @self._timed("summarize")
        def _call():
            return self.client.sunflower_simple(instruction=instruction, temperature=0.2)

        response = _call()
        return self._extract_model_text(response)

    def translate(self, text: str, target_language: str) -> str:
        instruction = (
            "Translate the following summary into "
            f"{target_language}. Keep the meaning accurate and natural.\n\n"
            f"Summary:\n{text}"
        )
        @self._timed("translate")
        def _call():
            return self.client.sunflower_simple(instruction=instruction, temperature=0.2)

        response = _call()
        return self._extract_model_text(response)

    def run_with_text(self, input_text: str, target_language: str) -> PipelineResult:
        source_text = input_text.strip()
        summary = self.summarize(source_text)
        translated_summary = self.translate(summary, target_language)
        tts_audio_bytes = self._synthesize(translated_summary, target_language)
        return PipelineResult(
            original_text=source_text,
            transcript=None,
            summary=summary,
            translated_summary=translated_summary,
            tts_audio_bytes=tts_audio_bytes,
        )

    def run_with_audio(
        self, audio_file_path: str, target_language: str, fallback_original_text: str = ""
    ) -> PipelineResult:
        stt_output = self.client.speech_to_text(audio_file_path=audio_file_path)
        transcript = (stt_output.get("text") or "").strip()
        source_text = transcript or fallback_original_text.strip()
        summary = self.summarize(source_text)
        translated_summary = self.translate(summary, target_language)
        tts_audio_bytes = self._synthesize(translated_summary, target_language)
        return PipelineResult(
            original_text=source_text,
            transcript=transcript,
            summary=summary,
            translated_summary=translated_summary,
            tts_audio_bytes=tts_audio_bytes,
        )

    def _synthesize(self, text: str, target_language: str) -> bytes:
        speaker_id = TARGET_LANGUAGES[target_language]["speaker_id"]
        @self._timed("tts")
        def _call_tts_and_download():
            tts_output = self.client.text_to_speech(text=text, speaker_id=speaker_id)
            audio_url = tts_output.get("audio_url") or tts_output.get("audioUrl")
            return self.client.download_audio(audio_url)

        return _call_tts_and_download()

    @staticmethod
    def _extract_model_text(response_payload: dict) -> str:
        # Sunflower simple currently returns text in top-level "response".
        root_response = response_payload.get("response")
        if isinstance(root_response, str) and root_response.strip():
            return root_response.strip()

        output = response_payload.get("output")
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, dict):
            for key in ("content", "text", "response"):
                value = output.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return str(output).strip()
