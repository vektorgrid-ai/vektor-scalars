import os
import time
import requests
from typing import Optional, Dict, Any

CORE_URL = os.getenv("CORE_URL", "http://core")
worker_id: Optional[str] = None

def _attempt_register(payload: Dict[str, Any]) -> str:
    res = requests.post(f"{CORE_URL}/worker/register", json=payload, timeout=5)
    res.raise_for_status()
    data = res.json()
    if not data.get("accepted", False):
        raise RuntimeError("Worker registration was not accepted")
    return data["worker_id"]


def register_with_core(worker_type: str) -> None:
    worker_endpoint = os.getenv("WORKER_ENDPOINT")
    if not worker_endpoint:
        worker_endpoint = "http://localhost:" + os.getenv("PORT", "8000")
    else:
        # normalize by removing trailing slash
        worker_endpoint = worker_endpoint.rstrip("/")

    payload = {
        "type": worker_type,
        "endpoint": worker_endpoint
    }

    while True:
        try:
            wid = _attempt_register(payload)

            global worker_id
            worker_id = wid
            print(f"Registered worker with ID: {worker_id}")
            print(f"Advertising worker endpoint: {worker_endpoint}")
            break
        except Exception as e:
            print(f"Failed to register worker: {e}")
            print("retrying")
            time.sleep(2)


def start_heartbeat() -> None:
    while True:
        if worker_id is None:
            time.sleep(1)
            continue
        try:
            requests.post(f"{CORE_URL}/worker/heartbeat", json={"worker_id": worker_id}, timeout=5)
        except Exception:
            pass
        time.sleep(30)
