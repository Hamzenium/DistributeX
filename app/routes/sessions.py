from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from bson import ObjectId
from multiprocessing import Process, Queue

from app.database import users_collection
from app.utils.security import get_current_user_id
from app.workers.peer_worker import peer_worker
from app.workers.registry import workers

router = APIRouter(prefix="/sessions", tags=["sessions"])

# -----------------------------------
# Fixed worker identity
# -----------------------------------
HEARTBEAT_WORKER = "heartbeat-worker"


def user_command_queue(user_id: str) -> str:
    # USER-scoped command queue
    return f"peer.{user_id}.command"


@router.post("/join")
async def join_session(user_id: str = Depends(get_current_user_id)):

    # Validate user
    try:
        mongo_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    if not users_collection.find_one({"_id": mongo_id}):
        raise HTTPException(status_code=404, detail="User not found")

    if HEARTBEAT_WORKER in workers:
        return {"message": "Heartbeat worker already running"}

    # ✅ USER-based command queue
    command_queue = user_command_queue(user_id)

    control_queue = Queue()
    heartbeat_consumer_queue = Queue()  # placeholder

    process = Process(
        target=peer_worker,
        args=(HEARTBEAT_WORKER, command_queue, control_queue),
        daemon=True,
    )
    process.start()

    workers[HEARTBEAT_WORKER] = {
        "process": process,
        "control_queue": control_queue,
        "heartbeat_consumer_queue": heartbeat_consumer_queue,
        "command_queue": command_queue,
        "user_id": user_id,
        "started_at": datetime.utcnow(),
    }

    return {
        "message": "Heartbeat worker started",
        "worker": HEARTBEAT_WORKER,
        "command_queue": command_queue,
        "pid": process.pid,
    }


@router.post("/command")
async def send_command(
    command: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Sends a command to the Heartbeat Worker.
    """
    worker = workers.get(HEARTBEAT_WORKER)
    if not worker:
        raise HTTPException(status_code=400, detail="Heartbeat worker not running")

    worker["control_queue"].put(command)

    return {
        "worker": HEARTBEAT_WORKER,
        "command": command,
        "status": "sent",
    }


@router.post("/leave")
async def leave_session(user_id: str = Depends(get_current_user_id)):
    """
    Gracefully stops the Heartbeat Worker.
    Non-blocking.
    """
    worker = workers.get(HEARTBEAT_WORKER)
    if not worker:
        return {"message": "Heartbeat worker not running"}

    # Signal shutdown
    worker["control_queue"].put("STOP")

    # Remove from registry immediately (do NOT join here)
    del workers[HEARTBEAT_WORKER]

    return {
        "message": "Heartbeat worker shutdown initiated",
        "worker": HEARTBEAT_WORKER,
    }
