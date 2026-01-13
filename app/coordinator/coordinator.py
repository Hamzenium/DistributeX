"""
Coordinator Process

Orchestrates ONE training session.

- Enables peers
- Starts training
- Collects heartbeats + loss
- Acts as authoritative execution controller
"""

import asyncio
import json
import os
import ssl
from datetime import datetime
from bson import ObjectId

import aio_pika
from pymongo import MongoClient
import certifi


# ============================================================================
# MongoDB setup (EXPLICIT DB SELECTION)
# ============================================================================

from dotenv import load_dotenv
import urllib.parse

load_dotenv()

MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
MONGO_DB = os.getenv("MONGO_DB", "hypertuneai")

if not all([MONGO_USERNAME, MONGO_PASSWORD, MONGO_CLUSTER]):
    raise RuntimeError("MongoDB env vars missing in coordinator")

username = urllib.parse.quote_plus(MONGO_USERNAME)
password = urllib.parse.quote_plus(MONGO_PASSWORD)

mongo_uri = (
    f"mongodb+srv://{username}:{password}@{MONGO_CLUSTER}/"
    f"{MONGO_DB}?retryWrites=true&w=majority"
)

mongo_client = MongoClient(
    mongo_uri,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000,
)

db = mongo_client[MONGO_DB]
sessions_collection = db["sessions"]


# ============================================================================
# RabbitMQ helpers
# ============================================================================

async def connect_rabbitmq():
    rabbit_url = os.getenv("RABBITMQ_URL")
    if not rabbit_url:
        raise RuntimeError("RABBITMQ_URL not configured")

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.load_verify_locations("isrgrootx1.pem")

    connection = await aio_pika.connect_robust(
        rabbit_url,
        ssl_context=ssl_ctx,
    )

    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    return connection, channel


async def publish_command(channel, queue_name: str, payload):
    """
    Send a control command to a peer.
    """
    if isinstance(payload, dict):
        body = json.dumps(payload).encode()
    else:
        body = str(payload).encode()

    await channel.default_exchange.publish(
        aio_pika.Message(body=body),
        routing_key=queue_name,
    )


# ============================================================================
# Coordinator consumer
# ============================================================================

async def start_event_consumer(channel, queue_name: str, event_queue: asyncio.Queue):
    """
    Long-lived consumer for peer → coordinator events.
    Handles BOTH raw and JSON payloads.
    """

    queue = await channel.declare_queue(queue_name, durable=True)

    async def on_message(message: aio_pika.IncomingMessage):
        async with message.process():
            raw = message.body.decode()

            # Try JSON first
            try:
                payload = json.loads(raw)
            except Exception:
                payload = raw  # raw heartbeat (session_uid)

            event_queue.put_nowait(payload)

    await queue.consume(on_message)
    print(f"[coordinator] consuming {queue_name}")


# ============================================================================
# Coordinator main logic
# ============================================================================

async def coordinator_main(session_uid: str):
    print(f"[coordinator] starting for session {session_uid}")

    session_oid = ObjectId(session_uid)

    # --------------------------------------------------
    # Load session
    # --------------------------------------------------
    session = sessions_collection.find_one({"_id": session_oid})
    if not session:
        raise RuntimeError("Session not found")

    peers = session.get("peers", [])
    if not peers:
        raise RuntimeError("Session has no peers")

    # --------------------------------------------------
    # Setup RabbitMQ
    # --------------------------------------------------
    connection, channel = await connect_rabbitmq()

    coordinator_queue = f"coordinator.{session_uid}.events"
    event_queue = asyncio.Queue()

    await start_event_consumer(channel, coordinator_queue, event_queue)

    # --------------------------------------------------
    # ENABLE peers
    # --------------------------------------------------
    for peer in peers:
        await publish_command(
            channel,
            peer["peer_queue"],
            {
                "type": "ENABLE",
                "queue": coordinator_queue,
            },
        )

    print("[coordinator] peers enabled")

    # --------------------------------------------------
    # Safety delay
    # --------------------------------------------------
    await asyncio.sleep(5)

    # --------------------------------------------------
    # TRAIN peers
    # --------------------------------------------------
    for peer in peers:
        await publish_command(
            channel,
            peer["peer_queue"],
            "TRAIN",
        )

    print("[coordinator] training started")

    # --------------------------------------------------
    # Collect heartbeats + loss
    # --------------------------------------------------
    peer_results = {}

    while True:
        event = await event_queue.get()

        # ------------------------------
        # Case 1: raw heartbeat (string)
        # ------------------------------
        if isinstance(event, str):
            # This is a liveness heartbeat (session_uid)
            print(f"[coordinator] heartbeat received for session {event}")
            continue

        # ------------------------------
        # Case 2: training heartbeat
        # ------------------------------
        peer_uid = event.get("peer_uid")
        loss = event.get("loss")

        if not peer_uid:
            continue

        peer_results.setdefault(peer_uid, []).append(
            {
                "loss": loss,
                "ts": datetime.utcnow(),
            }
        )

        print(f"[coordinator] {peer_uid} → loss={loss}")

        # Optional: persist progress
        sessions_collection.update_one(
            {"_id": session_oid, "peers.uid": peer_uid},
            {
                "$set": {
                    "updated_at": datetime.utcnow(),
                }
            },
        )


# ============================================================================
# Multiprocessing entrypoint
# ============================================================================

def run_coordinator(session_uid: str):
    asyncio.run(coordinator_main(session_uid))
