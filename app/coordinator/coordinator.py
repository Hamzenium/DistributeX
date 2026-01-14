"""
Coordinator Process

This process orchestrates a single training session.

Responsibilities:
1. Enable peers for training.
2. Send TRAIN commands with hyperparameters and dataset info.
3. Collect per-epoch training results and heartbeats.
4. Track when each peer completes.
5. Send STOP commands to all peers after training.
6. Update MongoDB: peer statuses, user statuses, and top-level session results.
"""

import asyncio
import json
import os
import ssl
from datetime import datetime
from bson import ObjectId
import urllib.parse

import aio_pika
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables (MongoDB & RabbitMQ credentials)
# ---------------------------------------------------------------------------
load_dotenv()

MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
MONGO_DB = os.getenv("MONGO_DB", "hypertuneai")

# Validate env variables
if not all([MONGO_USERNAME, MONGO_PASSWORD, MONGO_CLUSTER]):
    raise RuntimeError("MongoDB env vars missing in coordinator")

# URL-encode username/password for MongoDB connection
username = urllib.parse.quote_plus(MONGO_USERNAME)
password = urllib.parse.quote_plus(MONGO_PASSWORD)

# Construct MongoDB URI
mongo_uri = (
    f"mongodb+srv://{username}:{password}@{MONGO_CLUSTER}/"
    f"{MONGO_DB}?retryWrites=true&w=majority"
)

# Connect to MongoDB
mongo_client = MongoClient(
    mongo_uri,
    tlsCAFile=certifi.where(),  # ensures proper TLS
    serverSelectionTimeoutMS=5000,  # fail fast if DB not reachable
)

# Collections
db = mongo_client[MONGO_DB]
sessions_collection = db["sessions"]
users_collection = db["users"]

# ---------------------------------------------------------------------------
# RabbitMQ helper functions
# ---------------------------------------------------------------------------
async def connect_rabbitmq():
    """
    Establish a robust RabbitMQ connection and return a channel.
    """
    rabbit_url = os.getenv("RABBITMQ_URL")
    if not rabbit_url:
        raise RuntimeError("RABBITMQ_URL not configured")

    # SSL context for secure connection
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.load_verify_locations("isrgrootx1.pem")  # root certificate

    # Connect using aio_pika
    connection = await aio_pika.connect_robust(
        rabbit_url,
        ssl_context=ssl_ctx,
    )

    # Open a channel and set QoS
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)  # process 1 message at a time
    return connection, channel


async def publish_command(channel, queue_name: str, payload):
    """
    Send a command to a peer via its RabbitMQ queue.
    payload can be a dictionary or raw string.
    """
    if isinstance(payload, dict):
        body = json.dumps(payload).encode()
    else:
        body = str(payload).encode()

    await channel.default_exchange.publish(
        aio_pika.Message(body=body),
        routing_key=queue_name,
    )

# ---------------------------------------------------------------------------
# Event consumer for coordinator
# ---------------------------------------------------------------------------
async def start_event_consumer(channel, queue_name: str, event_queue: asyncio.Queue):
    """
    Long-lived consumer to receive messages from peers.
    Messages can be:
    - Raw heartbeat strings (peer is alive)
    - JSON payloads containing epoch results and completion status
    """
    queue = await channel.declare_queue(queue_name, durable=True)

    async def on_message(message: aio_pika.IncomingMessage):
        async with message.process():
            raw = message.body.decode()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = raw  # fallback for raw heartbeat string
            event_queue.put_nowait(payload)

    await queue.consume(on_message)
    print(f"[coordinator] consuming {queue_name}")

# ---------------------------------------------------------------------------
# Main coordinator logic
# ---------------------------------------------------------------------------
async def coordinator_main(session_uid: str):
    """
    Orchestrates one session:
    - Enable peers
    - Send TRAIN commands
    - Collect results per epoch
    - Detect when all peers finish
    - Stop all peers and update statuses
    """
    print(f"[coordinator] starting for session {session_uid}")

    session_oid = ObjectId(session_uid)
    session = sessions_collection.find_one({"_id": session_oid})
    if not session:
        raise RuntimeError("Session not found")

    peers = session.get("peers", [])
    if not peers:
        raise RuntimeError("Session has no peers")

    # --------------------------
    # Setup RabbitMQ connection
    # --------------------------
    connection, channel = await connect_rabbitmq()
    coordinator_queue = f"coordinator.{session_uid}.events"
    event_queue = asyncio.Queue()
    await start_event_consumer(channel, coordinator_queue, event_queue)

    # --------------------------
    # Enable peers
    # --------------------------
    for peer in peers:
        await publish_command(
            channel,
            peer["peer_queue"],
            {"type": "ENABLE", "queue": coordinator_queue},
        )
    print("[coordinator] peers enabled")
    await asyncio.sleep(5)  # safety delay to ensure peers are ready

    # --------------------------
    # Build dynamic TRAIN payload
    # --------------------------
    dataset_info = session.get("dataset", {})
    csv_link = f"s3://{dataset_info.get('s3_bucket')}/{dataset_info.get('s3_key')}"

    # Use hyperparameters from session
    hyperparams = session.get("hyperparameters", [{}])[0]

    # Dynamic x_labels: all columns except "label"
    x_labels = [
        col for col in session.get("dataset", {}).get("columns", []) if col != "label"
    ]
    y_label = "label"

    train_payload = {
        "type": "TRAIN",
        "csv_link": csv_link,
        "x_labels": x_labels,
        "y_label": y_label,
        "batch_size": hyperparams.get("batch_size"),
        "epochs": hyperparams.get("epochs"),
        "learning_rate": hyperparams.get("lr"),
    }

    # --------------------------
    # Send TRAIN to all peers
    # --------------------------
    for peer in peers:
        await publish_command(channel, peer["peer_queue"], train_payload)
    print(f"[coordinator] training started with payload: {train_payload}")

    # --------------------------
    # Track progress and completion
    # --------------------------
    completed_peers = set()
    total_peers = len(peers)

    while True:
        event = await event_queue.get()

        # Ignore raw heartbeat strings
        if isinstance(event, str):
            print(f"[coordinator] raw heartbeat: {event}")
            continue

        peer_uid = event.get("peer_uid")
        epochs = event.get("epochs")  # list of per-epoch results
        status = event.get("status")  # e.g., "completed"

        if not peer_uid:
            continue

        # --------------------------
        # Save per-peer epoch results
        # --------------------------
        if epochs:
            # Update peer object
            sessions_collection.update_one(
                {"_id": session_oid, "peers.uid": peer_uid},
                {"$push": {"peers.$.results": {"$each": epochs}}},
            )
            # Update top-level session results
            sessions_collection.update_one(
                {"_id": session_oid},
                {"$push": {f"results.{peer_uid}": {"$each": epochs}}},
            )
            print(f"[coordinator] {peer_uid} → {len(epochs)} epochs saved")

        # --------------------------
        # Mark peer as completed
        # --------------------------
        if status == "completed":
            completed_peers.add(peer_uid)

            # Update peer status in session
            sessions_collection.update_one(
                {"_id": session_oid, "peers.uid": peer_uid},
                {"$set": {"peers.$.status": "OFFLINE"}},
            )

            # Update peer user status in users collection
            users_collection.update_one(
                {"_id": ObjectId(peer_uid)},
                {"$set": {"status": "OFFLINE"}},
            )

            print(
                f"[coordinator] {peer_uid} completed ({len(completed_peers)}/{total_peers})"
            )

        # --------------------------
        # Stop all peers if all completed
        # --------------------------
        if len(completed_peers) == total_peers:
            print("[coordinator] all peers completed, sending STOP...")

            for peer in peers:
                await publish_command(channel, peer["peer_queue"], {"type": "STOP"})

            # Update session status
            sessions_collection.update_one(
                {"_id": session_oid},
                {"$set": {"status": "COMPLETED", "completed_at": datetime.utcnow()}},
            )

            print("[coordinator] session COMPLETED")
            break

# ---------------------------------------------------------------------------
# Multiprocessing entrypoint
# ---------------------------------------------------------------------------
def run_coordinator(session_uid: str):
    """
    Entry point for background process. Runs coordinator_main in asyncio loop.
    """
    asyncio.run(coordinator_main(session_uid))
