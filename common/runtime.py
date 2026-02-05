import os
from fastapi import FastAPI
import asyncio
from common.registration import register_with_core, start_heartbeat
from common.health import add_health_endpoint
import uvicorn

class WorkerRuntime:
    def __init__(
        self,
        worker_type: str,
        infer_handler,
        speciality: str | None = None,
    ):
        self.worker_type = worker_type
        self.speciality = speciality
        self.infer_handler = infer_handler
        self.app = FastAPI()

        add_health_endpoint(self.app)
        self._add_infer_endpoint()

    def _add_infer_endpoint(self):
        @self.app.post("/infer")
        async def infer(request: dict):
            return await self.infer_handler(request)

    def _start_hb_thread(self):
        import threading, asyncio

        def _run():
            asyncio.run(start_heartbeat())

        threading.Thread(target=_run, daemon=True).start()

    def run(self):
        port = os.getenv("PORT", "8000")
        register = "NO_REGISTER" not in os.environ
        if register:
            register_with_core(self.worker_type)
            # start heartbeat in background thread so it won't block the server
            self._start_hb_thread()

        uvicorn.run(self.app, host="0.0.0.0", port=int(port), log_level="info")
