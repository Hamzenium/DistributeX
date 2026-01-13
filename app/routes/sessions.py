from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Form,
)
from datetime import datetime
from bson import ObjectId
from multiprocessing import Process, Queue
import json
import os

from app.database import users_collection, sessions_collection
from app.utils.security import get_current_user_id
from app.workers.peer_worker import peer_worker
from app.workers.registry import workers
from app.storage.s3 import s3


router = APIRouter(prefix="/sessions", tags=["sessions"])

# ============================================================================
# Constants
# ============================================================================

HEARTBEAT_WORKER = "heartbeat-worker"
S3_BUCKET = os.getenv("S3_BUCKET_NAME")


# ============================================================================
# Helpers
# ============================================================================

def user_command_queue(session_uid: str) -> str:
    """
    Session-scoped RabbitMQ command queue.
    """
    return f"peer.{session_uid}.command"


# ============================================================================
# Create Session (CSV + hyperparameters + ownership link)
# ============================================================================

@router.post("/create")
async def create_session(
    num_peers: int = Form(...),
    hyperparameters: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Create a new distributed training session.

    This endpoint:
    1. Validates the user
    2. Validates hyperparameters
    3. Uploads CSV to Stackhero S3
    4. Creates a session document
    5. Links session to user (owned_sessions)
    """

    # --------------------------
    # Validate user
    # --------------------------
    try:
        owner_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    if not users_collection.find_one({"_id": owner_id}):
        raise HTTPException(status_code=404, detail="User not found")

    # --------------------------
    # Parse hyperparameters
    # --------------------------
    try:
        hyperparams_list = json.loads(hyperparameters)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Hyperparameters must be valid JSON",
        )

    if not isinstance(hyperparams_list, list):
        raise HTTPException(
            status_code=400,
            detail="Hyperparameters must be a list of objects",
        )

    if len(hyperparams_list) != num_peers:
        raise HTTPException(
            status_code=400,
            detail="Number of hyperparameter objects must equal num_peers",
        )

    # --------------------------
    # Create session ID
    # --------------------------
    session_id = ObjectId()

    # --------------------------
    # Upload CSV to S3
    # --------------------------
    if not S3_BUCKET:
        raise HTTPException(
            status_code=500,
            detail="S3_BUCKET_NAME not configured",
        )

    s3_key = f"dataset.csv"

    try:
        s3.upload_fileobj(
            file.file,
            S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": "text/csv"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file to S3: {str(e)}",
        )

    # --------------------------
    # Create session document
    # --------------------------
    session_doc = {
        "_id": session_id,
        "owner_user_id": owner_id,
        "num_peers": num_peers,
        "dataset": {
            "s3_bucket": S3_BUCKET,
            "s3_key": s3_key,
            "original_filename": file.filename,
        },
        "hyperparameters": hyperparams_list,
        "status": "CREATED",
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
    }

    sessions_collection.insert_one(session_doc)

    # --------------------------
    # Link session to user
    # --------------------------
    users_collection.update_one(
        {"_id": owner_id},
        {
            "$addToSet": {
                "owned_sessions": session_id
            }
        }
    )

    return {
        "message": "Session created successfully",
        "session_uid": str(session_id),
        "num_peers": num_peers,
        "dataset_s3_key": s3_key,
    }


# ============================================================================
# Join Session (start worker)
# ============================================================================

@router.post("/join")
async def join_session(user_id: str = Depends(get_current_user_id)):
    """
    Start the heartbeat worker.

    NOTE:
    For now, session_uid == user_id.
    Later, this will be replaced with a real session_uid.
    """

    try:
        mongo_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    if not users_collection.find_one({"_id": mongo_id}):
        raise HTTPException(status_code=404, detail="User not found")

    if HEARTBEAT_WORKER in workers:
        return {"message": "Heartbeat worker already running"}

    session_uid = user_id
    command_queue_name = user_command_queue(session_uid)
    control_queue = Queue()

    process = Process(
        target=peer_worker,
        args=(
            HEARTBEAT_WORKER,   # process_uid
            session_uid,        # session_uid
            command_queue_name,
            control_queue,
        ),
        daemon=True,
    )
    process.start()

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


# ============================================================================
# Send command to worker
# ============================================================================

@router.post("/command")
async def send_command(
    command: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Send a control command to the worker.
    """

    worker = workers.get(HEARTBEAT_WORKER)
    if not worker:
        raise HTTPException(
            status_code=400,
            detail="Heartbeat worker not running",
        )

    worker["control_queue"].put(command)

    return {
        "process_uid": HEARTBEAT_WORKER,
        "command": command,
        "status": "sent",
    }


# ============================================================================
# Leave Session (stop worker)
# ============================================================================

@router.post("/leave")
async def leave_session(user_id: str = Depends(get_current_user_id)):
    """
    Gracefully stop the worker (non-blocking).
    """

    worker = workers.get(HEARTBEAT_WORKER)
    if not worker:
        return {"message": "Heartbeat worker not running"}

    worker["control_queue"].put("STOP")
    del workers[HEARTBEAT_WORKER]

    return {
        "message": "Heartbeat worker shutdown initiated",
        "process_uid": HEARTBEAT_WORKER,
    }
