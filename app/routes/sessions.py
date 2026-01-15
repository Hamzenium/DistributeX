from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from datetime import datetime
from bson import ObjectId
from multiprocessing import Process, Queue
import json
import os
import tempfile

from app.database import users_collection, sessions_collection
from app.utils.security import get_current_user_id
from app.storage.s3 import s3
from app.coordinator.coordinator import run_coordinator
from app.workers.peer_worker import peer_worker
from app.workers.registry import workers

router = APIRouter(prefix="/sessions", tags=["sessions"])

S3_BUCKET = os.getenv("S3_BUCKET_NAME")

# Name of the heartbeat worker process
HEARTBEAT_WORKER = "heartbeat-worker"

# ------------------------------------------------------------------
# Helper: generate command queue name
# ------------------------------------------------------------------
def user_command_queue(session_uid: str) -> str:
    """
    Generates a command queue name for a given session/user.
    """
    return f"peer.{session_uid}.command"

# ------------------------------------------------------------------
# Background process: upload CSV → update session → start coordinator
# ------------------------------------------------------------------
def upload_and_start_coordinator(
    temp_file_path: str,
    original_filename: str,
    session_id: str,
    coordinator_uid: str,
):
    """
    Runs in a separate OS process.

    1. Upload CSV to S3
    2. Update session with dataset metadata
    3. Start coordinator (blocking inside its own process)
    """

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    s3_key = f"{timestamp}.dataset.csv"

    # 1. Upload CSV
    s3.upload_file(
        temp_file_path,
        S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": "text/csv"},
    )

    # 2. Update session dataset metadata
    sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {
            "$set": {
                "dataset": {
                    "s3_bucket": S3_BUCKET,
                    "s3_key": s3_key,
                    "original_filename": original_filename,
                }
            }
        },
    )

    # 3. Start coordinator (long-running)
    run_coordinator(session_id, coordinator_uid)

    # 4. Cleanup temp file
    try:
        os.remove(temp_file_path)
    except Exception:
        pass

# ------------------------------------------------------------------
# Start session endpoint
# ------------------------------------------------------------------
@router.post("/start")
async def start_session(
    num_peers: int = Form(...),
    hyperparameters: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Creates a session and starts training asynchronously.
    """

    # Validate owner
    try:
        owner_oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    owner = users_collection.find_one({"_id": owner_oid})
    if not owner:
        raise HTTPException(status_code=404, detail="User not found")

    # Parse hyperparameters
    try:
        hyperparams_list = json.loads(hyperparameters)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Hyperparameters must be valid JSON")

    if not isinstance(hyperparams_list, list) or len(hyperparams_list) != num_peers:
        raise HTTPException(
            status_code=400,
            detail="Hyperparameters list must match num_peers",
        )

    # Find available peers
    available_users = list(
        users_collection.find(
            {"status": "ONLINE", "_id": {"$ne": owner_oid}}
        ).limit(num_peers)
    )

    if len(available_users) < num_peers:
        raise HTTPException(
            status_code=400,
            detail="Not enough online peers available",
        )

    # Create session
    session_id = ObjectId()
    peers = {}
    results = {}

    # --------------------------------------------------
    # Upload dataset to S3
    # --------------------------------------------------
    if not S3_BUCKET:
        raise HTTPException(
            status_code=500, detail="S3_BUCKET_NAME not configured")

    s3_key = f"sessions/{session_id}/dataset.csv"

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

    for user in available_users:
    for i, user in enumerate(available_users):
        peer_uid = str(user["_id"])
        peers[peer_uid] = {
            "uid": peer_uid,
            "peer_queue": f"peer.{peer_uid}.command",
            "status": "TRAINING",
            "results": [],
            "hyperparameters": hyperparams_list[i],
        }
        results[peer_uid] = []

        # Lock peer
        users_collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {"status": "TRAINING"},
                "$addToSet": {"joined_sessions": session_id},
            },
        )

    session_doc = {
        "_id": session_id,
        "owner_user_id": str(owner_oid),
        "num_peers": num_peers,
        "dataset": {},
        "hyperparameters": hyperparams_list,
        "status": "RUNNING",
        "created_at": datetime.utcnow(),
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "peers": list(peers.values()),
        "results": results,
    }

    sessions_collection.insert_one(session_doc)

    # Link session to owner
    users_collection.update_one(
        {"_id": owner_oid},
        {"$addToSet": {"owned_sessions": session_id}},
    )

    # Save uploaded file locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        temp_file_path = tmp.name
        tmp.write(await file.read())

    # Spawn background process
    Process(
        target=upload_and_start_coordinator,
        args=(
            temp_file_path,
            file.filename,
            str(session_id),
            str(owner_oid),
        ),
        daemon=True,
    ).start()

    return {
        "message": "Session started. Uploading dataset & starting coordinator.",
        "session_uid": str(session_id),
        "assigned_peers": list(peers.values()),
    }

# ============================================================================
# Join Session (start worker)
# ============================================================================


@router.get("")
async def get_sessions(user_id: str = Depends(get_current_user_id)):
    """
    Get all sessions for the current user (owned or joined).
    """
    try:
        mongo_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    user = users_collection.find_one({"_id": mongo_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get sessions where user is owner or participant
    owned_session_ids = user.get("owned_sessions", [])
    joined_session_ids = user.get("joined_sessions", [])

    all_session_ids = list(set(owned_session_ids + joined_session_ids))

    sessions = list(sessions_collection.find(
        {"_id": {"$in": all_session_ids}}))

    # Convert ObjectId to string for JSON serialization
    for session in sessions:
        session["_id"] = str(session["_id"])
        session["owner_user_id"] = str(session["owner_user_id"])
        for peer in session.get("peers", []):
            if "_id" in peer:
                peer["_id"] = str(peer["_id"])

    return {"sessions": sessions}


# ------------------------------------------------------------------
# Join session endpoint
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Send command to worker
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Leave session (stop worker)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Check training status (user-scoped)
# ------------------------------------------------------------------
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

    latest_session_id = joined_sessions[-1]
    session = sessions_collection.find_one({"_id": latest_session_id})
    if not session:
        return {"message": "Session not found"}

    status = session.get("status", "UNKNOWN")
    if status == "RUNNING":
        return {"status": "training is going on"}
    else:
        return {"status": "training is completed"}
