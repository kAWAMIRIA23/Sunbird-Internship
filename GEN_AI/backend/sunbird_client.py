"""
HTTP client for Sunbird AI APIs (STT, Sunflower, TTS).

- Every HTTP call uses an explicit (connect, read) timeout.
- No urllib3 automatic retries on POST (avoids retry storms).
- No in-memory response caching (avoids stale / cross-request mix-ups on shared runtimes).
"""

from __future__ import annotations

import logging
import mimetypes
import os
import time
from typing import Any, Dict, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class SunbirdApiError(Exception):
    """Raised when a Sunbird API request fails."""


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


class SunbirdClient:
    """Thin wrapper around Sunbird AI APIs used in the app."""

    BASE_URL = "https://api.sunbird.ai"

    def __init__(
        self,
        token: str | None = None,
        connect_timeout: float | None = None,
        read_timeout_llm: float | None = None,
        read_timeout_stt: float | None = None,
        read_timeout_tts: float | None = None,
        read_timeout_download: float | None = None,
    ) -> None:
        raw = token or os.getenv("SUNBIRD_API_TOKEN")
        self.token = raw.strip() if raw else None
        if not self.token:
            raise ValueError("Missing SUNBIRD_API_TOKEN. Add it to your environment or .env file.")

        # Defaults tuned to fail fast; raise via env if your tenant needs longer reads.
        self.connect_timeout = connect_timeout if connect_timeout is not None else _env_float(
            "SUNBIRD_CONNECT_TIMEOUT", 10.0
        )
        self.read_timeout_llm = read_timeout_llm if read_timeout_llm is not None else _env_float(
            "SUNBIRD_READ_TIMEOUT_LLM", 60.0
        )
        self.read_timeout_stt = read_timeout_stt if read_timeout_stt is not None else _env_float(
            "SUNBIRD_READ_TIMEOUT_STT", 120.0
        )
        self.read_timeout_tts = read_timeout_tts if read_timeout_tts is not None else _env_float(
            "SUNBIRD_READ_TIMEOUT_TTS", 60.0
        )
        self.read_timeout_download = (
            read_timeout_download
            if read_timeout_download is not None
            else _env_float("SUNBIRD_READ_TIMEOUT_DOWNLOAD", 90.0)
        )

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

        no_retries = Retry(total=0, connect=0, read=False, redirect=False)
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=no_retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _timeout(self, read_seconds: float) -> Tuple[float, float]:
        return (self.connect_timeout, read_seconds)

    def _request_post(
        self,
        url: str,
        *,
        action: str,
        read_timeout: float,
        **kwargs: Any,
    ) -> requests.Response:
        timeout = self._timeout(read_timeout)
        t0 = time.perf_counter()
        logger.debug("%s POST %s timeout=%s", action, url, timeout)
        try:
            response = self.session.post(url, timeout=timeout, **kwargs)
        except requests.exceptions.Timeout as exc:
            raise SunbirdApiError(
                f"{action} timed out after connect={self.connect_timeout}s read={read_timeout}s: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise SunbirdApiError(f"{action} request failed: {exc}") from exc
        elapsed = time.perf_counter() - t0
        logger.info("%s finished status=%s duration_sec=%.3f", action, response.status_code, elapsed)
        return response

    def _request_get(
        self,
        url: str,
        *,
        action: str,
        read_timeout: float,
        **kwargs: Any,
    ) -> requests.Response:
        timeout = self._timeout(read_timeout)
        t0 = time.perf_counter()
        logger.debug("%s GET %s timeout=%s", action, url, timeout)
        try:
            response = self.session.get(url, timeout=timeout, **kwargs)
        except requests.exceptions.Timeout as exc:
            raise SunbirdApiError(
                f"{action} timed out after connect={self.connect_timeout}s read={read_timeout}s: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise SunbirdApiError(f"{action} request failed: {exc}") from exc
        elapsed = time.perf_counter() - t0
        logger.info("%s finished status=%s duration_sec=%.3f", action, response.status_code, elapsed)
        return response

    def _raise_for_error(self, response: requests.Response, action: str) -> None:
        if response.ok:
            return
        details = response.text[:2000] if response.text else ""
        raise SunbirdApiError(
            f"{action} failed with status {response.status_code}. Response (truncated): {details}"
        )

    def speech_to_text(self, audio_file_path: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/tasks/stt"
        file_name = os.path.basename(audio_file_path)
        mime_type, _ = mimetypes.guess_type(file_name)
        if not mime_type:
            mime_type = "application/octet-stream"
        with open(audio_file_path, "rb") as audio_file:
            files = {"audio": (file_name, audio_file, mime_type)}
            response = self._request_post(
                url,
                action="STT",
                read_timeout=self.read_timeout_stt,
                files=files,
            )
        self._raise_for_error(response, "Speech-to-text")
        payload = response.json()

        output = payload.get("output")
        if isinstance(output, dict) and isinstance(output.get("text"), str) and output["text"].strip():
            return output

        transcript = payload.get("audio_transcription")
        if isinstance(transcript, str) and transcript.strip():
            return {
                "text": transcript.strip(),
                "language": payload.get("language"),
                "audio_transcription_id": payload.get("audio_transcription_id"),
            }

        raise SunbirdApiError(f"Speech-to-text returned no transcript. Payload: {payload}")

    def sunflower_simple(
        self, instruction: str, model_type: str = "qwen", temperature: float = 0.2
    ) -> Dict[str, Any]:
        """One POST per call; no client-side response cache."""
        url = f"{self.BASE_URL}/tasks/sunflower_simple"
        response = self._request_post(
            url,
            action="Sunflower",
            read_timeout=self.read_timeout_llm,
            data={
                "instruction": instruction,
                "model_type": model_type,
                "temperature": temperature,
            },
        )
        self._raise_for_error(response, "Sunflower simple inference")
        payload = response.json()
        if not isinstance(payload, dict):
            raise SunbirdApiError(f"Sunflower returned unexpected payload type: {type(payload)}")
        if "response" not in payload and "output" not in payload:
            raise SunbirdApiError(
                "Sunflower response missing both 'response' and 'output'. "
                f"Payload: {payload}"
            )
        return payload

    def text_to_speech(self, text: str, speaker_id: int) -> Dict[str, Any]:
        """Single TTS request (no retry loop — avoids duplicate synthesis and long hangs)."""
        url = f"{self.BASE_URL}/tasks/tts"
        response = self._request_post(
            url,
            action="TTS",
            read_timeout=self.read_timeout_tts,
            headers={"Content-Type": "application/json"},
            json={"text": text, "speaker_id": speaker_id},
        )
        self._raise_for_error(response, "Text-to-speech")
        payload = response.json()
        output = payload.get("output", {})
        audio_url = output.get("audio_url") or output.get("audioUrl")
        if not audio_url:
            raise SunbirdApiError(f"Text-to-speech returned no audio URL. Payload: {payload}")
        return output

    def download_audio(self, audio_url: str) -> bytes:
        """Stream download; no cross-run audio cache."""
        response = self._request_get(
            audio_url,
            action="Audio download",
            read_timeout=self.read_timeout_download,
            stream=True,
        )
        self._raise_for_error(response, "Audio download")
        chunks = []
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
        return b"".join(chunks)
