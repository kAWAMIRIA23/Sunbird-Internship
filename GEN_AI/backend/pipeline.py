"""
Pipeline orchestration: optional STT -> summary -> translation -> TTS.

Each stage uses distinct variables (no reuse of the same mutable string for multiple roles).
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from backend.sunbird_client import SunbirdApiError, SunbirdClient

logger = logging.getLogger(__name__)

TARGET_LANGUAGES = {
    "Luganda": {"code": "lug", "speaker_id": 248},
    "Runyankole": {"code": "nyn", "speaker_id": 243},
    "Ateso": {"code": "teo", "speaker_id": 242},
    "Lugbara": {"code": "lgg", "speaker_id": 245},
    "Acholi": {"code": "ach", "speaker_id": 241},
}

_MAX_SOURCE_CHARS = int(os.getenv("GENAI_MAX_SOURCE_CHARS", "16000"))

_COMBINED_BLOCK = re.compile(
    r"<<<SUMMARY>>>\s*(.*?)\s*<<<TRANSLATION>>>\s*(.*?)\s*(?:<<<END>>>|$)",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class PipelineResult:
    original_text: str
    transcript: Optional[str]
    summary: str
    translated_summary: str
    tts_audio_bytes: Optional[bytes]
    tts_audio_mime: str = "audio/wav"


class GenAIPipeline:
    def __init__(self, client: SunbirdClient, *, run_id: str | None = None) -> None:
        self.client = client
        self.run_id = run_id or uuid.uuid4().hex[:12]

    def _log_stage(self, name: str, started: float) -> None:
        duration = time.perf_counter() - started
        logger.info("run_id=%s pipeline_stage=%s duration_sec=%.3f", self.run_id, name, duration)

    def _run_stage(self, name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            self._log_stage(name, t0)

    def _clip_for_model(self, text: str) -> str:
        if len(text) <= _MAX_SOURCE_CHARS:
            return text
        logger.warning(
            "run_id=%s source truncated from %s to %s chars (GENAI_MAX_SOURCE_CHARS)",
            self.run_id,
            len(text),
            _MAX_SOURCE_CHARS,
        )
        return text[:_MAX_SOURCE_CHARS]

    def summarize(self, text: str) -> str:
        """One LLM call: English summary only (legacy / tests)."""
        body = self._clip_for_model(text.strip())
        instruction = (
            "Summarize in 2-4 short English sentences. Keep facts; add nothing new.\n\n"
            f"Text:\n{body}"
        )

        def _call():
            return self.client.sunflower_simple(instruction=instruction, temperature=0.2)

        response = self._run_stage("summarize", _call)
        return self._extract_model_text(response)

    def translate(self, text: str, target_language: str) -> str:
        """One LLM call: translate given text (legacy / tests)."""
        body = self._clip_for_model(text.strip())
        instruction = (
            f"Translate the text into {target_language}. Natural, accurate, same meaning.\n\n"
            f"Text:\n{body}"
        )

        def _call():
            return self.client.sunflower_simple(instruction=instruction, temperature=0.2)

        response = self._run_stage("translate", _call)
        return self._extract_model_text(response)

    def _build_combined_instruction(self, text: str, target_language: str) -> str:
        body = self._clip_for_model(text.strip())
        return (
            "Do two steps in one reply. Use EXACTLY these markers and order:\n"
            "<<<SUMMARY>>>\n"
            "2-4 short English sentences. Keep facts; add nothing new.\n"
            "<<<TRANSLATION>>>\n"
            f"Same meaning in {target_language}. Natural phrasing.\n"
            "<<<END>>>\n\n"
            f"Source text:\n{body}"
        )

    def _parse_combined(self, raw: str) -> tuple[Optional[str], Optional[str]]:
        m = _COMBINED_BLOCK.search(raw.strip())
        if not m:
            return None, None
        summary_text, translation_text = m.group(1).strip(), m.group(2).strip()
        if not summary_text or not translation_text:
            return None, None
        return summary_text, translation_text

    def summarize_and_translate(self, source_text: str, target_language: str) -> tuple[str, str]:
        """
        Prefer one LLM round-trip. If delimiters are missing, fall back to two calls
        (extra latency, same logical mapping: source -> summary -> translation).
        """
        instruction = self._build_combined_instruction(source_text, target_language)

        def _combined():
            return self.client.sunflower_simple(instruction=instruction, temperature=0.2)

        response = self._run_stage("summarize_translate_combined", _combined)
        raw_llm = self._extract_model_text(response)
        summary_text, translation_text = self._parse_combined(raw_llm)
        if summary_text is not None and translation_text is not None:
            logger.info("run_id=%s combined_parse=ok", self.run_id)
            return summary_text, translation_text

        logger.warning(
            "run_id=%s combined_parse=fail prefix=%s",
            self.run_id,
            raw_llm[:200].replace("\n", " "),
        )

        summary_fb = self._run_stage(
            "summarize_fallback", lambda: self._summarize_only_inner(source_text)
        )
        translation_fb = self._run_stage(
            "translate_fallback",
            lambda: self._translate_only_inner(summary_fb, target_language),
        )
        return summary_fb, translation_fb

    def _summarize_only_inner(self, source_text: str) -> str:
        body = self._clip_for_model(source_text.strip())
        instruction = (
            "Summarize in 2-4 short English sentences. Keep facts; add nothing new.\n\n"
            f"Text:\n{body}"
        )
        response = self.client.sunflower_simple(instruction=instruction, temperature=0.2)
        return self._extract_model_text(response)

    def _translate_only_inner(self, summary_text: str, target_language: str) -> str:
        body = self._clip_for_model(summary_text.strip())
        instruction = (
            f"Translate the text into {target_language}. Natural, accurate, same meaning.\n\n"
            f"Text:\n{body}"
        )
        response = self.client.sunflower_simple(instruction=instruction, temperature=0.2)
        return self._extract_model_text(response)

    def run_with_text(
        self, input_text: str, target_language: str, *, generate_speech: bool = True
    ) -> PipelineResult:
        source_text = input_text.strip()
        source_text = self._clip_for_model(source_text)
        logger.info(
            "run_id=%s path=text source_chars=%s lang=%s generate_speech=%s",
            self.run_id,
            len(source_text),
            target_language,
            generate_speech,
        )
        summary_text, translated_text = self.summarize_and_translate(source_text, target_language)
        if generate_speech:
            audio_bytes: Optional[bytes] = self._synthesize(translated_text, target_language)
        else:
            logger.info("run_id=%s tts=skipped (text-only mode)", self.run_id)
            audio_bytes = None
        return PipelineResult(
            original_text=source_text,
            transcript=None,
            summary=summary_text,
            translated_summary=translated_text,
            tts_audio_bytes=audio_bytes,
        )

    def run_with_audio(
        self,
        audio_file_path: str,
        target_language: str,
        fallback_original_text: str = "",
        *,
        generate_speech: bool = True,
    ) -> PipelineResult:
        logger.info(
            "run_id=%s path=audio lang=%s generate_speech=%s file=%s",
            self.run_id,
            target_language,
            generate_speech,
            audio_file_path,
        )

        def _stt():
            return self.client.speech_to_text(audio_file_path=audio_file_path)

        stt_payload = self._run_stage("stt", _stt)
        transcript_text = (stt_payload.get("text") or "").strip()
        source_text = transcript_text if transcript_text else fallback_original_text.strip()
        source_text = self._clip_for_model(source_text)
        logger.info(
            "run_id=%s after_stt source_chars=%s transcript_chars=%s",
            self.run_id,
            len(source_text),
            len(transcript_text),
        )
        summary_text, translated_text = self.summarize_and_translate(source_text, target_language)
        if generate_speech:
            audio_bytes = self._synthesize(translated_text, target_language)
        else:
            logger.info("run_id=%s tts=skipped (text-only mode)", self.run_id)
            audio_bytes = None
        return PipelineResult(
            original_text=source_text,
            transcript=transcript_text,
            summary=summary_text,
            translated_summary=translated_text,
            tts_audio_bytes=audio_bytes,
        )

    def _synthesize(self, translated_text: str, target_language: str) -> bytes:
        speaker_id = TARGET_LANGUAGES[target_language]["speaker_id"]

        def _tts():
            return self.client.text_to_speech(text=translated_text, speaker_id=speaker_id)

        tts_payload = self._run_stage("tts", _tts)
        audio_url = tts_payload.get("audio_url") or tts_payload.get("audioUrl")
        if not audio_url:
            raise SunbirdApiError(f"TTS response missing audio URL: {tts_payload!r}")

        def _download():
            return self.client.download_audio(audio_url)

        return self._run_stage("audio_download", _download)

    @staticmethod
    def _extract_model_text(response_payload: dict) -> str:
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
