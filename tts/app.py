import base64
import time
from os.path import basename
from pathlib import Path

from piper import PiperVoice

MODEL_PATH = Path(__file__).resolve().parent / "models" / "de_DE-karlsson-low.onnx"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

voice = PiperVoice.load(str(MODEL_PATH))

async def infer_tts(data: dict):
    start = time.perf_counter()

    req_id = data["request_id"]
    text = data["input"]["text"]
    audio, info = synthesize(text)

    end = time.perf_counter()
    latency_ms = int((end - start) * 1000)
    result = {
        "request_id": req_id,
        "output": {
            "data_base64": audio.decode("ascii"),
            "sample_rate": info.sample_rate,
            "channels": info.num_speakers,
            "encoding": "pcm_s16le"
        },
        "usage": {
            "latency_ms": latency_ms,
            "model": basename(MODEL_PATH),
            "version": info.piper_version,
        },
        "error": None
    }
    return result


def synthesize(text: str):
    global voice
    if text == "" or text is None:
        return b"", voice.config
    if voice is None:
        raise RuntimeError("Voice model is not loaded.")

    audio = voice.synthesize(text)
    raw_bytes = b"".join([chunk.audio_int16_bytes for chunk in audio])
    return base64.b64encode(raw_bytes), voice.config
