from common.runtime import WorkerRuntime
from tts.app import infer_tts

runtime = WorkerRuntime(
    worker_type="tts",
    infer_handler=infer_tts,
)

runtime.run()
