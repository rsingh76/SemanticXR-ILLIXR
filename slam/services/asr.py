"""Speech-to-text for the visualization server.

Encapsulates backend selection (faster-whisper / openai-whisper / openai-api),
warmup, WAV writing from streamed PCM chunks, and transcription. Configured via
`config.model.visualization.asr` — see config/settings.py::ASRConfig.
"""

import os
import struct
import time
from typing import Iterable

import numpy as np


# WAV header constants — matches what the XR client streams.
_NUM_CHANNELS = 1
_COMPRESSION_CODE = 1  # PCM
_SAMPLE_RATE = 12000
_BITS_PER_SAMPLE = 16


def _write_wav_header(outfile):
    bytes_per_sample = _BITS_PER_SAMPLE // 8
    byte_rate = _SAMPLE_RATE * _NUM_CHANNELS * bytes_per_sample
    block_align = _NUM_CHANNELS * bytes_per_sample
    outfile.write(b"RIFF")
    outfile.write(struct.pack("<I", 36))  # placeholder; patched in _patch_wav_sizes
    outfile.write(b"WAVE")
    outfile.write(b"fmt ")
    outfile.write(struct.pack("<I", 16))  # PCM fmt chunk size
    outfile.write(struct.pack("<H", _COMPRESSION_CODE))
    outfile.write(struct.pack("<H", _NUM_CHANNELS))
    outfile.write(struct.pack("<I", _SAMPLE_RATE))
    outfile.write(struct.pack("<I", byte_rate))
    outfile.write(struct.pack("<H", block_align))
    outfile.write(struct.pack("<H", _BITS_PER_SAMPLE))
    outfile.write(b"data")
    outfile.write(struct.pack("<I", 0))  # placeholder; patched in _patch_wav_sizes


def _patch_wav_sizes(outfile, data_size: int):
    outfile.seek(4)
    outfile.write(struct.pack("<I", 36 + data_size))
    outfile.seek(40)
    outfile.write(struct.pack("<I", data_size))


class SpeechRecognizer:
    """Configurable ASR wrapper.

    Lifecycle: one instance is built at server start; `transcribe_chunks` is
    called once per gRPC `clientTextQuery`. Not thread-safe — the underlying
    WAV file path is shared across calls.
    """

    def __init__(self, asr_config, wav_path: str):
        self._wav_path = wav_path
        self.enabled = asr_config.enabled
        self.backend = asr_config.backend
        self.model_name = None

        self._openai_client = None
        self._local_model = None

        if not self.enabled:
            print("[ASR] disabled — client is expected to send text queries directly.")
            return

        if self.backend == "openai-api":
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "whisper-1"
            print(f"[ASR] backend=openai-api model={self.model_name}")
        elif self.backend == "faster-whisper":
            from faster_whisper import WhisperModel
            # faster-whisper wants device="cuda" + device_index=0, not "cuda:0"
            device_type, _, device_index = asr_config.device.partition(":")
            self._local_model = WhisperModel(
                asr_config.model,
                device=device_type,
                device_index=int(device_index) if device_index else 0,
                compute_type=asr_config.compute_type,
            )
            self.model_name = asr_config.model
            print(f"[ASR] backend=faster-whisper model={self.model_name} compute={asr_config.compute_type}")
        elif self.backend == "openai-whisper":
            import whisper
            self._local_model = whisper.load_model(asr_config.model, device=asr_config.device)
            self.model_name = asr_config.model
            print(f"[ASR] backend=openai-whisper model={self.model_name}")
        else:
            raise ValueError(
                f"Unknown asr.backend: {self.backend!r} "
                "(expected 'faster-whisper', 'openai-whisper', or 'openai-api')"
            )

        if asr_config.warmup and self._local_model is not None:
            self._warmup()

    def _warmup(self):
        silence = np.zeros(16000, dtype=np.float32)  # 1s silence at 16 kHz
        start = time.perf_counter_ns()
        if self.backend == "faster-whisper":
            segs, _ = self._local_model.transcribe(silence)
            list(segs)  # drain generator
        elif self.backend == "openai-whisper":
            self._local_model.transcribe(silence, fp16=False)
        print(f"[ASR] warmup done in {(time.perf_counter_ns() - start) / 1e6:.1f} ms")

    def transcribe_chunks(self, chunks: Iterable[bytes]) -> tuple[str, float]:
        """Assemble streamed PCM chunks into a WAV file on disk, then transcribe.

        Returns (transcript, transcription_ms) where transcription_ms measures
        only the model call, not WAV assembly.
        """
        combined = b"".join(chunks)
        with open(self._wav_path, "wb") as outfile:
            _write_wav_header(outfile)
            outfile.write(combined)
            _patch_wav_sizes(outfile, len(combined))
        start = time.perf_counter_ns()
        text = self._transcribe_wav()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        return text, elapsed_ms

    def _transcribe_wav(self) -> str:
        print(f"[ASR] transcribing via {self.backend} ({self.model_name})...")
        if self.backend == "openai-api":
            with open(self._wav_path, "rb") as audio_file:
                transcript = self._openai_client.audio.transcriptions.create(
                    model=self.model_name,
                    file=audio_file,
                )
            return transcript.text
        if self.backend == "faster-whisper":
            segments, _ = self._local_model.transcribe(self._wav_path)
            return " ".join(seg.text for seg in segments).strip()
        if self.backend == "openai-whisper":
            result = self._local_model.transcribe(self._wav_path)
            return result["text"].strip()
        raise RuntimeError(f"transcribe called with unknown backend {self.backend!r}")
