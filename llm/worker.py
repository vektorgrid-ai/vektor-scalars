from common.runtime import WorkerRuntime
from llm.app import infer_llm

runtime = WorkerRuntime(
    worker_type="llm",
    infer_handler=infer_llm,
)

runtime.run()
