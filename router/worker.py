from common.runtime import WorkerRuntime
from router.app import infer_router

runtime = WorkerRuntime(
    worker_type="router",
    infer_handler=infer_router,
)

runtime.run()
