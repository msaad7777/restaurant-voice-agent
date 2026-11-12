"""Speech in and out. faster-whisper runs in-process; Piper runs as its own HTTP server.

Piper (piper-tts 1.8) is GPL-3.0, so this project only talks to it over HTTP and does not
import or ship it. Start it with `scripts/piper.sh`.
"""
import io
import json
import os
import time
import urllib.request
import wave

PIPER_URL = os.environ.get("PIPER_URL", "http://127.0.0.1:5100")
AGENT_VOICE = "en_US-joe-medium"
STT_MODEL = os.environ.get("STT_MODEL", "base.en")

# Words the caller is likely to say; biases Whisper towards the right spelling.
HOTWORDS = "Juniper and Rye, burrata, calamari, linguine, pesto, risotto, agnolotti, sorbet, torte, gluten, sesame"

_stt = {}


def stt_model(name: str = STT_MODEL):
    if name not in _stt:
        from faster_whisper import WhisperModel
        _stt[name] = WhisperModel(name, device="cpu", compute_type="int8")
    return _stt[name]


def transcribe(audio, model: str = STT_MODEL, hotwords: bool = True) -> tuple[str, int]:
    """audio: path, file-like or bytes of any format ffmpeg/av can read. Returns (text, ms)."""
    if isinstance(audio, (bytes, bytearray)):
        audio = io.BytesIO(audio)
    t = time.perf_counter()
    segs, _ = stt_model(model).transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                          hotwords=HOTWORDS if hotwords else None,
                                          condition_on_previous_text=False)
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, round((time.perf_counter() - t) * 1000)


def synthesize(text: str, voice: str = AGENT_VOICE, speaker_id: int | None = None, length_scale: float = 1.0) -> tuple[bytes, int]:
    """Returns (wav bytes, ms). Raises URLError if Piper isn't running."""
    body = {"text": text, "voice": voice, "length_scale": length_scale}
    if speaker_id is not None:
        body["speaker_id"] = speaker_id
    req = urllib.request.Request(f"{PIPER_URL}/synthesize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        wav = r.read()
    return wav, round((time.perf_counter() - t) * 1000)


def wav_seconds(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / w.getframerate()


def truncate_wav(wav: bytes, fraction: float) -> bytes:
    """Keep the first `fraction` of a WAV: a caller cut off mid-sentence."""
    with wave.open(io.BytesIO(wav)) as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    keep = int(len(frames) * fraction) // (params.sampwidth * params.nchannels) * (params.sampwidth * params.nchannels)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(frames[:keep])
    return out.getvalue()
