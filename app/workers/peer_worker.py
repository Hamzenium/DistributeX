"""
Peer Worker Runtime (Agent Process) with DNN

This module represents a single "peer" in a distributed system.
Responsibilities:
- Listen for commands (TRAIN, ENABLE, STOP) from either a local multiprocessing.Queue or RabbitMQ.
- Download datasets from S3, train a small fully-connected DNN model.
- Send periodic heartbeats:
    - During training: includes epoch, loss, accuracy.
    - Idle: reports status.
    - After training: sends a final "DONE" message.
- Clean up temporary dataset files after use.
"""

import asyncio
import os
import ssl
import json
from multiprocessing import Queue
from queue import Empty

import aio_pika  # Asynchronous RabbitMQ client
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.storage.s3 import s3  # boto3 S3 client


# ============================================================================
# Timing constants
# ============================================================================
HEARTBEAT_INTERVAL = 5.0  # seconds between idle heartbeats
LOOP_SLEEP = 0.1          # sleep in main loop to avoid busy-waiting


# ============================================================================
# RabbitMQ helper functions
# ============================================================================
async def connect_rabbitmq(process_uid: str, command_queue_name: str):
    """
    Connect to RabbitMQ over SSL and declare the command queue.

    Args:
        process_uid: Unique identifier of this peer (for logging).
        command_queue_name: Name of the command queue to listen on.

    Returns:
        connection, channel, command_queue objects.
        If RabbitMQ is disabled, returns (None, None, None)
    """
    rabbit_url = os.getenv("RABBITMQ_URL")
    if not rabbit_url:
        print(f"[{process_uid}] RabbitMQ disabled")
        return None, None, None

    # Create SSL context for secure RabbitMQ connection
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.load_verify_locations("isrgrootx1.pem")

    # Establish connection and channel
    connection = await aio_pika.connect_robust(rabbit_url, ssl_context=ssl_ctx)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)  # limit unacknowledged messages

    # Declare durable queue to receive commands
    command_queue = await channel.declare_queue(command_queue_name, durable=True)
    print(f"[{process_uid}] RabbitMQ connected")
    return connection, channel, command_queue


async def publish_message(channel, queue_name: str, body: bytes):
    """
    Publish a message to a RabbitMQ queue.

    Args:
        channel: RabbitMQ channel
        queue_name: Target queue
        body: Message bytes (JSON encoded)
    """
    await channel.default_exchange.publish(
        aio_pika.Message(body=body, delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT),
        routing_key=queue_name,
    )


async def start_rabbit_consumer(process_uid: str, amqp_queue: aio_pika.Queue, local_async_queue: asyncio.Queue):
    """
    Consume messages from RabbitMQ and put them into a local asyncio.Queue.
    This ensures uniform processing whether messages come from RabbitMQ or local queue.

    Args:
        process_uid: For logging
        amqp_queue: RabbitMQ queue
        local_async_queue: Async queue for worker loop
    """
    async def on_message(message: aio_pika.IncomingMessage):
        async with message.process():  # auto-acknowledge
            local_async_queue.put_nowait(message.body.decode())

    await amqp_queue.consume(on_message)
    print(f"[{process_uid}] RabbitMQ consumer started")


# ============================================================================
# Heartbeat helpers
# ============================================================================
async def attach_heartbeat_queue(channel, queue_name: str):
    """
    Attach a queue for sending heartbeat messages.

    Returns the queue object, or None if not possible.
    """
    if not channel or not queue_name:
        return None
    queue = await channel.get_queue(queue_name, ensure=False)
    print(f"[heartbeat] enabled → {queue_name}")
    return queue


async def send_heartbeat(session_uid: str, channel, heartbeat_state: dict, training_state: dict,
                         epoch=None, loss=None, accuracy=None):
    """
    Send heartbeat message:
    - During training: include epoch, loss, accuracy
    - Idle: include peer_uid and status
    """
    if not heartbeat_state.get("enabled") or not channel or not heartbeat_state.get("queue"):
        return

    queue = heartbeat_state["queue"]
    if training_state.get("active"):
        payload = {
            "peer_uid": session_uid,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy
        }
    else:
        payload = {
            "peer_uid": session_uid,
            "status": training_state.get("status", "idle")
        }

    body = json.dumps(payload).encode()
    await publish_message(channel, queue.name, body)
    heartbeat_state["last_sent"] = asyncio.get_event_loop().time()
    print(f"[heartbeat] sent → {payload}")


async def maybe_send_heartbeat(session_uid: str, channel, heartbeat_state: dict, training_state: dict):
    """
    Send idle heartbeat respecting HEARTBEAT_INTERVAL to avoid spamming messages.
    """
    if training_state.get("active"):
        return
    now = asyncio.get_event_loop().time()
    if (now - heartbeat_state["last_sent"]) < HEARTBEAT_INTERVAL:
        return
    await send_heartbeat(session_uid, channel, heartbeat_state, training_state)


# ============================================================================
# Dataset download
# ============================================================================
def download_dataset_from_s3(s3_url: str):
    """
    Download a CSV dataset from S3 to a local temp file.
    
    Returns:
        Local file path
    """
    if not s3_url.startswith("s3://"):
        raise ValueError("Invalid S3 URL")

    _, path = s3_url.split("s3://", 1)
    bucket_name, key = path.split("/", 1)
    local_file = f"/tmp/{os.path.basename(key)}"

    s3.download_file(bucket_name, key, local_file)
    print(f"[dataset] downloaded {s3_url} → {local_file}")
    return local_file


# ============================================================================
# Simple fully connected DNN
# ============================================================================
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim=10):
        """
        Fully connected DNN:
        input_dim -> 128 -> 64 -> output_dim
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# DNN training
# ============================================================================
async def train_dnn(session_uid, channel, heartbeat_state, dataset_path, x_labels, y_label,
                    batch_size=64, epochs=10, learning_rate=0.001):
    """
    Train a SimpleDNN on a CSV dataset.

    Steps:
    1. Load CSV into PyTorch tensors
    2. Initialize model, loss, optimizer
    3. Train for 'epochs' epochs
    4. Send heartbeat after each epoch
    5. Send final "DONE" heartbeat
    6. Delete dataset file
    """
    import pandas as pd
    df = pd.read_csv(dataset_path)

    X = df[x_labels].values.astype("float32")
    y = df[y_label].values.astype("int64")
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleDNN(input_dim=X.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_state = {"active": True, "loss": None, "accuracy": None, "status": "training"}

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (pred.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

        avg_loss = epoch_loss / len(loader)
        accuracy = correct / total
        training_state["loss"] = avg_loss
        training_state["accuracy"] = accuracy

        # Send heartbeat per epoch
        await send_heartbeat(session_uid, channel, heartbeat_state, training_state,
                             epoch=epoch+1, loss=avg_loss, accuracy=accuracy)
        print(f"[training] Epoch {epoch+1}/{epochs} → loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

    # ----------------------------
    # Training complete
    # ----------------------------
    training_state["active"] = False
    training_state["status"] = "idle"
    print("[training] complete")

    # Send final DONE heartbeat
    done_payload = {"peer_uid": session_uid, "type": "DONE", "status": "completed"}
    if heartbeat_state.get("enabled") and heartbeat_state.get("queue") and channel:
        await publish_message(channel, heartbeat_state["queue"].name, json.dumps(done_payload).encode())
        print(f"[heartbeat] sent final DONE → {done_payload}")

    # Clean up local dataset
    try:
        os.remove(dataset_path)
        print(f"[dataset] deleted {dataset_path}")
    except Exception as e:
        print(f"[dataset] could not delete {dataset_path}: {e}")

    return model


# ============================================================================
# Command handling
# ============================================================================
async def handle_command(process_uid, session_uid, source, payload, channel, heartbeat_state):
    """
    Process a single command from local or RabbitMQ queue:
    - STOP: terminate worker
    - ENABLE: enable heartbeat queue
    - TRAIN: download dataset, train model
    """
    print(f"[{process_uid}] handling {payload} from {source}")

    if payload == "STOP":
        print(f"[{process_uid}] STOP received → shutting down")
        return False

    try:
        data = json.loads(payload)
    except Exception:
        data = None

    if isinstance(data, dict):
        cmd_type = data.get("type")

        if cmd_type == "ENABLE":
            queue_name = data.get("queue")
            queue = await attach_heartbeat_queue(channel, queue_name)
            heartbeat_state.update({
                "enabled": True,
                "queue": queue,
                "queue_name": queue_name,
                "last_sent": 0.0
            })
            return True

        elif cmd_type == "TRAIN":
            csv_link = data.get("csv_link")
            x_labels = data.get("x_labels")
            y_label = data.get("y_label")
            batch_size = data.get("batch_size", 64)
            epochs = data.get("epochs", 10)
            learning_rate = data.get("learning_rate", 0.001)

            dataset_path = download_dataset_from_s3(csv_link)
            await train_dnn(session_uid, channel, heartbeat_state, dataset_path,
                            x_labels, y_label, batch_size, epochs, learning_rate)
            return True

    return True


# ============================================================================
# Main async worker loop
# ============================================================================
async def _run_worker(process_uid, session_uid, command_queue_name, control_queue: Queue):
    """
    Main loop:
    - Listens for messages from RabbitMQ and local queue
    - Handles commands
    - Sends idle heartbeats when not training
    """
    running = True
    connection, channel, command_queue = await connect_rabbitmq(process_uid, command_queue_name)
    rabbit_cmd_queue = asyncio.Queue()

    if command_queue:
        await start_rabbit_consumer(process_uid, command_queue, rabbit_cmd_queue)

    heartbeat_state = {"enabled": False, "queue": None, "queue_name": None, "last_sent": 0.0}
    print(f"[{process_uid}] worker started")

    while running:
        # Handle messages from local queue
        try:
            msg = control_queue.get_nowait()
            running = await handle_command(process_uid, session_uid, "local", msg, channel, heartbeat_state)
        except Empty:
            pass

        # Handle messages from RabbitMQ queue
        try:
            payload = rabbit_cmd_queue.get_nowait()
            running = await handle_command(process_uid, session_uid, "rabbitmq", payload, channel, heartbeat_state)
        except asyncio.QueueEmpty:
            pass

        # Send idle heartbeat if not training
        await maybe_send_heartbeat(session_uid, channel, heartbeat_state, {"active": False, "status": "idle"})
        await asyncio.sleep(LOOP_SLEEP)

    if connection:
        await connection.close()
    print(f"[{process_uid}] worker exited cleanly")


# ============================================================================
# Entry point for multiprocessing.Process
# ============================================================================
def peer_worker(process_uid, session_uid, command_queue_name, control_queue: Queue):
    """
    Entry point for a peer process.
    Runs the async worker loop and ensures queues are cleaned up on exit.
    """
    try:
        asyncio.run(_run_worker(process_uid, session_uid, command_queue_name, control_queue))
    finally:
        if control_queue:
            control_queue.close()
            control_queue.join_thread()