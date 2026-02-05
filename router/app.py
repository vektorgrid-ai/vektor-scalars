import time
import ollama
from pydantic import BaseModel

MODEL = "llama3.2"
ollama.pull(MODEL)

class RoutingDecision(BaseModel):
    speciality: str
    confidence: float
    reason: str

async def infer_router(data: dict):
    start = time.perf_counter()

    req_id = data["request_id"]
    text = data["input"]["text"]
    specialities = data["config"]["allowed_specialities"]
    context = data["context"] if "context" in data else ""
    decision = route_intent(text, specialities, context)

    end = time.perf_counter()
    latency_ms = int((end - start) * 1000)
    result = {
        "request_id": req_id,
        "output": {
            "specialty": decision.speciality,
            "confidence": decision.confidence,
            "reason": decision.reason,
        },
        "usage": {
            "latency_ms": latency_ms,
            "model": MODEL
        },
        "error": None
    }
    return result


def route_intent(text, specialities, context):
    prompt = f"Given the following context: {context}, and the allowed specialities: {specialities}, determine the most appropriate speciality for the input text: '{text}'."
    print(prompt)
    response = ollama.generate(model=MODEL, prompt=prompt, system="You are an expert router that assigns specialities based on input text."
                                                                  "Use the general speciality only as fallback if no other speciality fits well.",
                               format=RoutingDecision.model_json_schema())
    decision = RoutingDecision.model_validate_json(response.response)
    return decision
