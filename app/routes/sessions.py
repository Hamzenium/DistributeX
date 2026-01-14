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
from app.coordinator.coordinator import run_coordinator


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

@router.post("/start")
async def start_session(
    num_peers: int = Form(...),
    hyperparameters: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Create a session and immediately start computation.

    ATOMIC guarantees:
    - If peers are unavailable → nothing is created
    - If this returns 200 → session is RUNNING and coordinator is live
    """

    # --------------------------------------------------
    # Validate owner
    # --------------------------------------------------
    try:
        owner_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    owner = users_collection.find_one({"_id": owner_id})
    if not owner:
        raise HTTPException(status_code=404, detail="User not found")

    # --------------------------------------------------
    # Parse & validate hyperparameters
    # --------------------------------------------------
    try:
        hyperparams_list = json.loads(hyperparameters)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Hyperparameters must be valid JSON")

    if not isinstance(hyperparams_list, list):
        raise HTTPException(status_code=400, detail="Hyperparameters must be a list")

    if len(hyperparams_list) != num_peers:
        raise HTTPException(
            status_code=400,
            detail="Hyperparameters length must equal num_peers",
        )

    # --------------------------------------------------
    # Find available peers FIRST (fail fast)
    # --------------------------------------------------
    available_users = list(
        users_collection.find(
            {
                "status": "ONLINE",
                "_id": {"$ne": owner_id},
            }
        ).limit(num_peers)
    )

    if len(available_users) < num_peers:
        raise HTTPException(
            status_code=400,
            detail="Not enough online peers available",
        )

    # --------------------------------------------------
    # Create session ID
    # --------------------------------------------------
    session_id = ObjectId()

    # --------------------------------------------------
    # Upload dataset to S3
    # --------------------------------------------------
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3_BUCKET_NAME not configured")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"{timestamp}.dataset.csv"

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

    # --------------------------------------------------
    # Build peers + lock users
    # --------------------------------------------------
    peers = []
    initial_peer_results = {}  # top-level results dict

    for i, user in enumerate(available_users):
        peer_uid = str(user["_id"])
        queue_name = f"peer.{peer_uid}.command"

        peers.append(
            {
                "uid": peer_uid,
                "peer_queue": queue_name,
                "results": [],  # list of epochs for this peer
                "status": "TRAINING",  # initial status
                "hyperparameters": hyperparams_list[i],  # attach hyperparams per peer
            }
        )

        # Initialize top-level results
        initial_peer_results[peer_uid] = []

        # Lock peer immediately
        users_collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {"status": "TRAINING"},
                "$addToSet": {"joined_sessions": session_id},
            },
        )

    # --------------------------------------------------
    # Create session document (RUNNING)
    # --------------------------------------------------
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
        "status": "RUNNING",
        "created_at": datetime.utcnow(),
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "peers": peers,
        "results": initial_peer_results,
    }

    sessions_collection.insert_one(session_doc)

    # Link session to owner
    users_collection.update_one(
        {"_id": owner_id},
        {"$addToSet": {"owned_sessions": session_id}},
    )

    # --------------------------------------------------
    # Spawn coordinator (BACKGROUND PROCESS)
    # --------------------------------------------------
    Process(
        target=run_coordinator,
        args=(str(session_id),),
        daemon=True,
    ).start()

    return {
        "message": "Session started",
        "session_uid": str(session_id),
        "assigned_peers": peers,
    }

# ============================================================================
# Join Session (start worker)
# ============================================================================

@router.post("/join")
async def join_session(user_id: str = Depends(get_current_user_id)):
    """
    Start the heartbeat worker AND mark user as ONLINE.
    """

    try:
        mongo_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    user = users_collection.find_one({"_id": mongo_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if HEARTBEAT_WORKER in workers:
        return {"message": "Heartbeat worker already running"}

    session_uid = user_id
    command_queue_name = user_command_queue(session_uid)
    control_queue = Queue()

    process = Process(
        target=peer_worker,
        args=(
            HEARTBEAT_WORKER,
            session_uid,
            command_queue_name,
            control_queue,
        ),
        daemon=True,
    )
    process.start()

    users_collection.update_one(
        {"_id": mongo_id},
        {
            "$set": {
                "status": "ONLINE",
                "last_online_at": datetime.utcnow(),
            }
        }
    )

    workers[HEARTBEAT_WORKER] = {
        "process": process,
        "control_queue": control_queue,
        "command_queue": command_queue_name,
        "session_uid": session_uid,
        "started_at": datetime.utcnow(),
    }

    return {
        "message": "Peer is ONLINE",
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

# ============================================================================
# Check Training Status (User-scoped)
# ============================================================================

@router.get("/training-status")
async def get_training_status(user_id: str = Depends(get_current_user_id)):
    """
    Returns whether the latest session the user joined is still running.

    - If session is RUNNING → "training is going on"
    - Otherwise → "training is completed"
    """
    try:
        user_oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    user = users_collection.find_one({"_id": user_oid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    joined_sessions = user.get("joined_sessions", [])
    if not joined_sessions:
        return {"message": "Waiting to Join Session"}

    # Get the latest joined session
    latest_session_id = joined_sessions[-1]
    session = sessions_collection.find_one({"_id": latest_session_id})
    if not session:
        return {"message": "Session not found"}

    # Check session status
    status = session.get("status", "UNKNOWN")
    if status == "RUNNING":
        return {"status": "training is going on"}
    else:
        return {"status": "training is completed"}