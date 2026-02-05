import base64
import time
import numpy as np
from faster_whisper import WhisperModel

MODEL = "small"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
whisper = WhisperModel(MODEL, device="cpu", compute_type=COMPUTE_TYPE)

async def infer_stt(data: dict):
    start = time.perf_counter()

    req_id = data["request_id"]
    sample_rate = data["input"]["sample_rate"] # currently unused
    base64_audio = data["input"]["data_base64"]
    encoding = data["input"]["encoding"] # currently unused
    channels = data["input"]["channels"] # currently unused
    text, language, confidence = transcribe(base64_audio)

    end = time.perf_counter()
    latency_ms = int((end - start) * 1000)
    result = {
        "request_id": req_id,
        "output": {
            "text": text,
            "language": language,
            "confidence": round(confidence * 100),
        },
        "usage": {
            "latency_ms": latency_ms,
            "model": "whisper-" + MODEL,
            "compute_type":  COMPUTE_TYPE,
            "beam_size": BEAM_SIZE
        },
        "error": None
    }
    return result


def transcribe(base64_audio: str):
    global whisper
    audio_bytes = base64.b64decode(base64_audio)
    audio_array = np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0
    segments, info = whisper.transcribe(audio_array, beam_size=BEAM_SIZE)
    text = ""
    for segment in segments:
        text += segment.text + " "
    return text.strip(), info.language, info.language_probability
