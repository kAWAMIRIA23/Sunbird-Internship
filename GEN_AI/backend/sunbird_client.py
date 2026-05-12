import os
import mimetypes
import time
from typing import Any, Dict
from collections import OrderedDict
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SunbirdApiError(Exception):
    """Raised when a Sunbird API request fails."""


class SunbirdClient:
    """Thin wrapper around Sunbird AI APIs used in the app."""

    BASE_URL = "https://api.sunbird.ai"

    def __init__(self, token: str | None = None, timeout_seconds: int = 120) -> None:
        self.token = token or os.getenv("SUNBIRD_API_TOKEN")
        if not self.token:
            raise ValueError("Missing SUNBIRD_API_TOKEN. Add it to your environment or .env file.")
        # Use a shorter default timeout to fail fast and reduce perceived slowness.
        self.timeout_seconds = timeout_seconds

        # Use a persistent Session to reuse TCP connections and enable connection pooling.
        self.session = requests.Session()
        # Set auth header on the session to avoid re-sending it per-request.
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

        # Configure a HTTPAdapter with a small pool and retry strategy for transient errors.
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        )
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Simple in-memory caches to avoid repeated identical requests.
        # Cache for sunflower_simple responses (instruction -> payload)
        self._sunflower_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._sunflower_cache_lock = threading.Lock()
        self._sunflower_cache_max = 128

        # Cache for downloaded audio (audio_url -> bytes)
        self._audio_cache: OrderedDict[str, bytes] = OrderedDict()
        self._audio_cache_lock = threading.Lock()
        self._audio_cache_max = 64

    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _raise_for_error(self, response: requests.Response, action: str) -> None:
        if response.ok:
            return
        details = response.text
        raise SunbirdApiError(
            f"{action} failed with status {response.status_code}. Response: {details}"
        )

    def speech_to_text(self, audio_file_path: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/tasks/stt"
        file_name = os.path.basename(audio_file_path)
        mime_type, _ = mimetypes.guess_type(file_name)
        if not mime_type:
            mime_type = "application/octet-stream"
        with open(audio_file_path, "rb") as audio_file:
            files = {"audio": (file_name, audio_file, mime_type)}
            response = self.session.post(
                url,
                files=files,
                timeout=self.timeout_seconds,
            )
        self._raise_for_error(response, "Speech-to-text")
        payload = response.json()

        # Sunbird STT can return either nested output.text or top-level audio_transcription.
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
        url = f"{self.BASE_URL}/tasks/sunflower_simple"
        # Check cache first (use a simple key derived from instruction+params).
        key = f"{instruction}|{model_type}|{temperature}"
        with self._sunflower_cache_lock:
            cached = self._sunflower_cache.get(key)
            if cached is not None:
                # Move to end to mark as recently used
                self._sunflower_cache.move_to_end(key)
                return cached

        response = self.session.post(
            url,
            data={
                "instruction": instruction,
                "model_type": model_type,
                "temperature": temperature,
            },
            timeout=self.timeout_seconds,
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

        # Store in cache (payload can be large, but we cap entries)
        with self._sunflower_cache_lock:
            self._sunflower_cache[key] = payload
            if len(self._sunflower_cache) > self._sunflower_cache_max:
                self._sunflower_cache.popitem(last=False)
        return payload

    def text_to_speech(self, text: str, speaker_id: int) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/tasks/tts"
        last_error_message = ""

        for attempt in range(3):
            try:
                response = self.session.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={"text": text, "speaker_id": speaker_id},
                    timeout=self.timeout_seconds,
                )
                self._raise_for_error(response, "Text-to-speech")
                payload = response.json()
                output = payload.get("output", {})
                audio_url = output.get("audio_url") or output.get("audioUrl")
                if audio_url:
                    return output

                service_error = output.get("Error") if isinstance(output, dict) else None
                if service_error:
                    last_error_message = str(service_error)
                else:
                    last_error_message = f"Text-to-speech returned no audio URL. Payload: {payload}"
            except requests.exceptions.RequestException as exc:
                last_error_message = str(exc)

            # Retry on transient/network/provider-side failures.
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

        raise SunbirdApiError(
            "Text-to-speech failed after retries. "
            f"Last error: {last_error_message}"
        )

    def download_audio(self, audio_url: str) -> bytes:
        # Stream the download to avoid blocking the session while large files transfer.
        # Check audio cache first
        with self._audio_cache_lock:
            cached_audio = self._audio_cache.get(audio_url)
            if cached_audio is not None:
                self._audio_cache.move_to_end(audio_url)
                return cached_audio

        response = self.session.get(audio_url, timeout=self.timeout_seconds, stream=True)
        self._raise_for_error(response, "Audio download")
        chunks = []
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
        data = b"".join(chunks)

        with self._audio_cache_lock:
            self._audio_cache[audio_url] = data
            if len(self._audio_cache) > self._audio_cache_max:
                self._audio_cache.popitem(last=False)

        return data
