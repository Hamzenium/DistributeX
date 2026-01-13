from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from bson import ObjectId
from multiprocessing import Process, Queue

from app.database import users_collection
from app.utils.security import get_current_user_id
from app.workers.peer_worker import peer_worker
from app.workers.registry import workers

router = APIRouter(prefix="/sessions", tags=["sessions"])

# ------------------------------------------------------------------
# Fixed PROCESS identity (local, technical)
# {
 #"type": "ENABLE_HEARTBEAT",
#"queue": "heartbeat.consumer.node-42"
#}
# ------------------------------------------------------------------
HEARTBEAT_WORKER = "heartbeat-worker"


def user_command_queue(session_uid: str) -> str:
    """
    Session-scoped RabbitMQ command queue.

    Commands sent to this queue control the worker representing
    this session.
    """
    return f"peer.{session_uid}.command"


@router.post("/join")
async def join_session(user_id: str = Depends(get_current_user_id)):
    """
    Join a session and start the heartbeat worker.

    IMPORTANT:
    - user_id / session_uid is the LOGICAL identity
    - HEARTBEAT_WORKER is the PROCESS identity
    """

    # ---- Validate user ----
    try:
        mongo_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    if not users_collection.find_one({"_id": mongo_id}):
        raise HTTPException(status_code=404, detail="User not found")

    # ---- Prevent duplicate worker ----
    if HEARTBEAT_WORKER in workers:
        return {"message": "Heartbeat worker already running"}

    # SESSION UID (logical identity used for heartbeat)
    session_uid = user_id

    # RabbitMQ command queue scoped to session
    command_queue_name = user_command_queue(session_uid)

    # Local control queue (FastAPI → worker)
    control_queue = Queue()

    # Spawn worker process
    process = Process(
        target=peer_worker,
        args=(
            HEARTBEAT_WORKER,       # process_uid
            session_uid,            # session_uid (heartbeat payload)
            command_queue_name,
            control_queue,
        ),
        daemon=True,
    )
    process.start()

    # Register worker IMMEDIATELY
    workers[HEARTBEAT_WORKER] = {
        "process": process,
        "control_queue": control_queue,
        "command_queue": command_queue_name,
        "session_uid": session_uid,
        "started_at": datetime.utcnow(),
    }

    return {
        "message": "Heartbeat worker started",
        "process_uid": HEARTBEAT_WORKER,
        "session_uid": session_uid,
        "command_queue": command_queue_name,
        "pid": process.pid,
    }


@router.post("/command")
async def send_command(
    command: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Send a control command to the heartbeat worker.
    """

    worker = workers.get(HEARTBEAT_WORKER)
    if not worker:
        raise HTTPException(status_code=400, detail="Heartbeat worker not running")

    worker["control_queue"].put(command)

    return {
        "process_uid": HEARTBEAT_WORKER,
        "command": command,
        "status": "sent",
    }


@router.post("/leave")
async def leave_session(user_id: str = Depends(get_current_user_id)):
    """
    Gracefully stop the heartbeat worker.

    IMPORTANT:
    - Non-blocking
    - Registry is updated immediately
    """

    worker = workers.get(HEARTBEAT_WORKER)
    if not worker:
        return {"message": "Heartbeat worker not running"}

    # Signal shutdown
    worker["control_queue"].put("STOP")

    # Remove from registry immediately (DO NOT join)
    del workers[HEARTBEAT_WORKER]

    return {
        "message": "Heartbeat worker shutdown initiated",
        "process_uid": HEARTBEAT_WORKER,
    }
