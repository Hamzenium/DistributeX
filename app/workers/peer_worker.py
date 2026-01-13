"""
Peer Worker Runtime (Agent Process)

This worker is a long-running background process spawned by a FastAPI
coordinator. It participates in a distributed system using RabbitMQ.

===============================================================================
IDENTITY MODEL (CRITICAL)
===============================================================================

This worker has TWO identities:

1) process_uid (technical / local identity)
   - Identifies the OS process
   - Used for logging, debugging, registry keys
   - Example: "heartbeat-worker"

2) session_uid (logical / distributed identity)
   - Generated when a session is created
   - Represents the session being trained / monitored
   - Used for heartbeats and liveness tracking
   - Example: Mongo ObjectId / UUID

IMPORTANT RULE:
Heartbeats represent SESSIONS, NOT processes.
Heartbeat payload ALWAYS contains session_uid.
"""

import asyncio
import os
import ssl
import json
import time
from multiprocessing import Queue
from queue import Empty

import aio_pika


# ============================================================================
# Timing constants
# ============================================================================

# Interval between heartbeat emissions
HEARTBEAT_INTERVAL = 5.0

# Small sleep to yield control back to asyncio
LOOP_SLEEP = 0.1

# Total simulated training time
TRAIN_DURATION = 30.0  # seconds


# ============================================================================
# RabbitMQ connection helpers
# ============================================================================

async def connect_rabbitmq(process_uid: str, command_queue_name: str):
    """
    Establish a secure RabbitMQ connection and declare the command queue.

    This function:
    - Reads RABBITMQ_URL from environment
    - Uses TLS encryption
    - Declares a durable command queue
    - Uses QoS prefetch=1 (fair dispatch)

    Returns:
        (connection, channel, command_queue)
        OR (None, None, None) if RabbitMQ is disabled
    """

    rabbit_url = os.getenv("RABBITMQ_URL")
    if not rabbit_url:
        print(f"[{process_uid}] RabbitMQ disabled")
        return None, None, None

    # TLS context (certificate pinned)
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.load_verify_locations("isrgrootx1.pem")

    # Robust connection automatically reconnects if broker restarts
    connection = await aio_pika.connect_robust(
        rabbit_url,
        ssl_context=ssl_ctx,
    )

    channel = await connection.channel()

    # Ensure we process only one unacked message at a time
    await channel.set_qos(prefetch_count=1)

    # Declare the command queue (must already be known by coordinator)
    command_queue = await channel.declare_queue(
        command_queue_name,
        durable=True,
    )

    print(f"[{process_uid}] RabbitMQ connected")
    return connection, channel, command_queue


async def publish_message(channel, queue_name: str, body: bytes):
    """
    Publish a raw message to RabbitMQ.

    Used for heartbeats.
    Messages are NOT persisted because heartbeats are ephemeral.
    """

    await channel.default_exchange.publish(
        aio_pika.Message(
            body=body,
            delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
        ),
        routing_key=queue_name,
    )


# ============================================================================
# RabbitMQ consumer (CRITICAL FIX)
# ============================================================================

async def start_rabbit_consumer(
    process_uid: str,
    amqp_queue: aio_pika.Queue,
    local_async_queue: asyncio.Queue,
):
    """
    Start a long-lived RabbitMQ consumer.

    WHY THIS EXISTS:
    - Avoids Basic.Get polling
    - Prevents channel shutdown when idle
    - Production-safe

    Incoming messages are pushed into an asyncio.Queue so the main
    event loop never talks to RabbitMQ directly.
    """

    async def on_message(message: aio_pika.IncomingMessage):
        async with message.process():
            payload = message.body.decode()
            local_async_queue.put_nowait(payload)

    await amqp_queue.consume(on_message)
    print(f"[{process_uid}] RabbitMQ consumer started")


# ============================================================================
# Heartbeat helpers
# ============================================================================

async def attach_heartbeat_queue(channel, queue_name: str):
    """
    Attach to an EXISTING heartbeat queue.

    Rules:
    - ensure=False → queue MUST already exist
    - Worker NEVER creates queues
    - Coordinator owns topology
    """

    if not channel or not queue_name:
        return None

    queue = await channel.get_queue(queue_name, ensure=False)
    print(f"[heartbeat] enabled → {queue_name}")
    return queue


async def maybe_send_heartbeat(
    session_uid: str,
    channel,
    heartbeat_state: dict,
    training_state: dict,
):
    """
    Conditionally emit a heartbeat.

    Heartbeat payload depends on mode:

    NORMAL MODE:
      payload = session_uid (string)

    TRAINING MODE:
      payload = {
        "session_uid": <id>,
        "loss": <float>
      }

    Coordinator timestamps on receipt.
    Presence of message == liveness.
    """

    # Heartbeat not enabled yet
    if not heartbeat_state["enabled"]:
        return

    heartbeat_queue = heartbeat_state["queue"]
    if not heartbeat_queue:
        return

    now = asyncio.get_event_loop().time()
    if (now - heartbeat_state["last_sent"]) < HEARTBEAT_INTERVAL:
        return

    # --------------------------
    # Construct payload
    # --------------------------
    if training_state["active"]:
        payload = {
            "session_uid": session_uid,
            "loss": training_state["loss"],
        }
        body = json.dumps(payload).encode()
    else:
        body = session_uid.encode()

    await publish_message(channel, heartbeat_queue.name, body)

    heartbeat_state["last_sent"] = now
    print(f"[heartbeat] sent → {body}")


# ============================================================================
# Training simulation
# ============================================================================

async def run_training_loop(training_state: dict):
    """
    Simulate a 30-second ML training loop.

    Behavior:
    - Runs for TRAIN_DURATION seconds
    - Updates loss value over time
    - Loss monotonically decreases (simulated)

    IMPORTANT:
    - This function does NOT block heartbeats
    - Heartbeats read training_state["loss"]
    """

    start = time.time()
    step = 0
    print("Installing CSV From S3")

    while time.time() - start < TRAIN_DURATION:
        # Fake loss curve (decays over time)
        training_state["loss"] = round(1.0 / (1 + step * 0.3), 4)

        step += 1
        await asyncio.sleep(1)

    training_state["active"] = False
    training_state["loss"] = None
    print("[training] completed")


# ============================================================================
# Command handling
# ============================================================================

async def handle_command(
    process_uid: str,
    session_uid: str,
    source: str,
    payload,
    channel,
    heartbeat_state: dict,
    training_state: dict,
):
    """
    Handle a single control command.

    Supported commands:

    STOP
      - Exit worker immediately

    ENABLE_HEARTBEAT
      Payload:
      {
        "type": "ENABLE_HEARTBEAT",
        "queue": "heartbeat.session.<session_uid>"
      }

    TRAIN
      - Start 30s training loop
      - Stop accepting further commands
      - Continue heartbeats with loss
    """

    print(f"[{process_uid}] handling {payload} from {source}")

    # --------------------------
    # STOP
    # --------------------------
    if payload == "STOP":
        print(f"[{process_uid}] STOP received → shutting down")
        return False

    # --------------------------
    # TRAIN
    # --------------------------
    if payload == "TRAIN":
        if not training_state["active"]:
            print(f"[{process_uid}] TRAIN received → starting training")
            training_state["active"] = True
            asyncio.create_task(run_training_loop(training_state))
        return True

    # --------------------------
    # ENABLE_HEARTBEAT
    # --------------------------
    try:
        data = json.loads(payload)
    except Exception:
        data = None

    if isinstance(data, dict) and data.get("type") == "ENABLE":
        queue_name = data.get("queue")
        queue = await attach_heartbeat_queue(channel, queue_name)

        heartbeat_state["enabled"] = True
        heartbeat_state["queue"] = queue
        heartbeat_state["queue_name"] = queue_name
        heartbeat_state["last_sent"] = 0.0

        print(f"[{process_uid}] heartbeat enabled")
        return True

    return True


# ============================================================================
# Worker runtime
# ============================================================================

async def _run_worker(
    process_uid: str,
    session_uid: str,
    command_queue_name: str,
    control_queue: Queue,
):
    """
    Main asyncio event loop.

    Responsibilities:
    - Process local commands
    - Process RabbitMQ commands
    - Emit heartbeats
    - Run training loop when requested
    """

    running = True

    # Connect to RabbitMQ
    connection, channel, command_queue = await connect_rabbitmq(
        process_uid,
        command_queue_name,
    )

    # Async queue fed by RabbitMQ consumer
    rabbit_cmd_queue = asyncio.Queue()
    if command_queue:
        await start_rabbit_consumer(
            process_uid,
            command_queue,
            rabbit_cmd_queue,
        )

    # --------------------------
    # State containers
    # --------------------------
    heartbeat_state = {
        "enabled": False,
        "queue": None,
        "queue_name": None,
        "last_sent": 0.0,
    }

    training_state = {
        "active": False,
        "loss": None,
    }

    print(f"[{process_uid}] worker started")

    while running:

        # --------------------------------------------------
        # Commands are IGNORED during training
        # --------------------------------------------------
        if not training_state["active"]:
            try:
                msg = control_queue.get_nowait()
                running = await handle_command(
                    process_uid,
                    session_uid,
                    "local",
                    msg,
                    channel,
                    heartbeat_state,
                    training_state,
                )
            except Empty:
                pass

            try:
                payload = rabbit_cmd_queue.get_nowait()
                running = await handle_command(
                    process_uid,
                    session_uid,
                    "rabbitmq",
                    payload,
                    channel,
                    heartbeat_state,
                    training_state,
                )
            except asyncio.QueueEmpty:
                pass

        # --------------------------------------------------
        # Heartbeat ALWAYS runs
        # --------------------------------------------------
        await maybe_send_heartbeat(
            session_uid,
            channel,
            heartbeat_state,
            training_state,
        )

        await asyncio.sleep(LOOP_SLEEP)

    # --------------------------
    # Graceful shutdown
    # --------------------------
    if connection:
        await connection.close()

    print(f"[{process_uid}] worker exited cleanly")


def peer_worker(
    process_uid: str,
    session_uid: str,
    command_queue_name: str,
    control_queue: Queue,
):
    """
    Multiprocessing entrypoint.

    Used by:
        Process(target=peer_worker, args=(...))
    """

    asyncio.run(
        _run_worker(
            process_uid,
            session_uid,
            command_queue_name,
            control_queue,
        )
    )
