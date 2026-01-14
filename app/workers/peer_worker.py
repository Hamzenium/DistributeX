"""
Peer Worker Runtime (Agent Process) with DNN
This module represents a "peer" in a distributed system that:
- Listens for commands (TRAIN, ENABLE, STOP) via a queue (RabbitMQ or local).
- Downloads datasets from S3, trains a simple DNN model.
- Sends heartbeats periodically or after every epoch during training.
- Cleans up temporary files after use.
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

from app.storage.s3 import s3  # boto3 client for S3 operations

# ============================================================================
# Timing constants
# ============================================================================
HEARTBEAT_INTERVAL = 5.0  # seconds between idle heartbeats
LOOP_SLEEP = 0.1  # sleep in main loop to avoid busy-waiting

# ============================================================================
# RabbitMQ helpers
# ============================================================================
async def connect_rabbitmq(process_uid: str, command_queue_name: str):
    """
    Connect to RabbitMQ using SSL and return connection, channel, and queue.
    - process_uid: unique identifier of this peer
    - command_queue_name: the name of the queue to listen for commands
    """
    rabbit_url = os.getenv("RABBITMQ_URL")
    if not rabbit_url:
        print(f"[{process_uid}] RabbitMQ disabled")
        return None, None, None

    # Create SSL context for secure connection
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.load_verify_locations("isrgrootx1.pem")

    # Connect to RabbitMQ
    connection = await aio_pika.connect_robust(rabbit_url, ssl_context=ssl_ctx)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)  # control message flow
    command_queue = await channel.declare_queue(command_queue_name, durable=True)
    print(f"[{process_uid}] RabbitMQ connected")
    return connection, channel, command_queue


async def publish_message(channel, queue_name: str, body: bytes):
    """
    Publish a message to a given RabbitMQ queue.
    - body must be bytes, typically json-encoded
    """
    await channel.default_exchange.publish(
        aio_pika.Message(body=body, delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT),
        routing_key=queue_name,
    )


async def start_rabbit_consumer(process_uid: str, amqp_queue: aio_pika.Queue, local_async_queue: asyncio.Queue):
    """
    Consume messages from RabbitMQ and put them into a local asyncio queue.
    This allows our async worker loop to process messages uniformly.
    """
    async def on_message(message: aio_pika.IncomingMessage):
        async with message.process():  # automatically acknowledge message
            local_async_queue.put_nowait(message.body.decode())

    await amqp_queue.consume(on_message)
    print(f"[{process_uid}] RabbitMQ consumer started")


# ============================================================================
# Heartbeat helpers
# ============================================================================
async def attach_heartbeat_queue(channel, queue_name: str):
    """
    Attach a heartbeat queue for sending status updates.
    Returns the queue object or None.
    """
    if not channel or not queue_name:
        return None
    queue = await channel.get_queue(queue_name, ensure=False)
    print(f"[heartbeat] enabled → {queue_name}")
    return queue


async def send_heartbeat(session_uid: str, channel, heartbeat_state: dict, training_state: dict,
                         epoch=None, loss=None, accuracy=None):
    """
    Send a heartbeat to the configured heartbeat queue.
    - During training: include peer_uid, epoch, loss, accuracy.
    - When idle: include peer_uid and status.
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
    Send idle heartbeat respecting HEARTBEAT_INTERVAL
    This ensures we don't spam messages when not training.
    """
    if training_state.get("active"):
        return  # training sends its own heartbeats

    now = asyncio.get_event_loop().time()
    if (now - heartbeat_state["last_sent"]) < HEARTBEAT_INTERVAL:
        return
    await send_heartbeat(session_uid, channel, heartbeat_state, training_state)


# ============================================================================
# Dataset download from S3
# ============================================================================
def download_dataset_from_s3(s3_url: str):
    """
    Download dataset from S3 to a temporary local path.
    Returns the local file path.
    """
    if not s3_url.startswith("s3://"):
        raise ValueError("Invalid S3 URL")

    # Parse bucket and key
    _, path = s3_url.split("s3://", 1)
    bucket_name, key = path.split("/", 1)
    local_file = f"/tmp/{os.path.basename(key)}"

    s3.download_file(bucket_name, key, local_file)
    print(f"[dataset] downloaded {s3_url} → {local_file}")
    return local_file


# ============================================================================
# DNN model
# ============================================================================
class SimpleDNN(nn.Module):
    """
    A small fully-connected neural network
    - Input: flattened image or vector (input_dim)
    - Output: number of classes (default 10)
    """
    def __init__(self, input_dim, output_dim=10):
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


async def train_dnn(session_uid, channel, heartbeat_state, dataset_path, x_labels, y_label,
                    batch_size=64, epochs=10, learning_rate=0.001):
    """
    Train a SimpleDNN on a CSV dataset.
    - Loads CSV into PyTorch TensorDataset
    - Trains with Adam optimizer and CrossEntropyLoss
    - Sends heartbeat after each epoch
    - Deletes the dataset file after training
    """
    import pandas as pd
    df = pd.read_csv(dataset_path)

    # Prepare input features and labels
    X = df[x_labels].values.astype("float32")
    y = df[y_label].values.astype("int64")
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and move to GPU if available
    model = SimpleDNN(input_dim=X.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track training state
    training_state = {"active": True, "loss": None, "accuracy": None, "status": "training"}

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            # Accumulate stats for heartbeat
            epoch_loss += loss.item()
            correct += (pred.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

        # Compute metrics
        avg_loss = epoch_loss / len(loader)
        accuracy = correct / total
        training_state["loss"] = avg_loss
        training_state["accuracy"] = accuracy

        # Send heartbeat after each epoch
        await send_heartbeat(session_uid, channel, heartbeat_state, training_state,
                             epoch=epoch+1, loss=avg_loss, accuracy=accuracy)
        print(f"[training] Epoch {epoch+1}/{epochs} → loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")

    # Mark training complete
    training_state["active"] = False
    training_state["status"] = "idle"
    print("[training] complete")

    # Delete dataset file to free disk
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
    Handle a single command:
    - STOP: terminate the worker
    - ENABLE: enable heartbeat queue
    - TRAIN: download dataset and train model
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
            heartbeat_state.update({"enabled": True, "queue": queue, "queue_name": queue_name, "last_sent": 0.0})
            return True
        elif cmd_type == "TRAIN":
            # Extract training parameters from command
            csv_link = data.get("csv_link")
            x_labels = data.get("x_labels")
            y_label = data.get("y_label")
            batch_size = data.get("batch_size", 64)
            epochs = data.get("epochs", 10)
            learning_rate = data.get("learning_rate", 0.001)

            # Download dataset and start training
            dataset_path = download_dataset_from_s3(csv_link)
            await train_dnn(session_uid, channel, heartbeat_state, dataset_path, x_labels, y_label,
                            batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
            return True

    return True


# ============================================================================
# Worker runtime
# ============================================================================
async def _run_worker(process_uid, session_uid, command_queue_name, control_queue: Queue):
    """
    Main async worker loop:
    - Handles local commands from multiprocessing.Queue
    - Handles RabbitMQ commands
    - Sends idle heartbeats
    """
    running = True
    connection, channel, command_queue = await connect_rabbitmq(process_uid, command_queue_name)
    rabbit_cmd_queue = asyncio.Queue()
    if command_queue:
        await start_rabbit_consumer(process_uid, command_queue, rabbit_cmd_queue)

    heartbeat_state = {"enabled": False, "queue": None, "queue_name": None, "last_sent": 0.0}

    print(f"[{process_uid}] worker started")

    while running:
        # Handle local queue messages
        try:
            msg = control_queue.get_nowait()
            running = await handle_command(process_uid, session_uid, "local", msg, channel, heartbeat_state)
        except Empty:
            pass

        # Handle RabbitMQ messages
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


def peer_worker(process_uid, session_uid, command_queue_name, control_queue: Queue):
    """
    Entry point for a multiprocessing.Process
    - Runs the async worker loop
    - Cleans up queues after exit
    """
    try:
        asyncio.run(_run_worker(process_uid, session_uid, command_queue_name, control_queue))
    finally:
        if control_queue:
            control_queue.close()
            control_queue.join_thread()




{
  "type": "TRAIN",
  "csv_link": "20260114_064724.mnist.csv",
  "x_labels": [
    "1x1","1x2","1x3","1x4","1x5","1x6","1x7","1x8","1x9","1x10","1x11","1x12","1x13","1x14","1x15","1x16","1x17","1x18","1x19","1x20","1x21","1x22","1x23","1x24","1x25","1x26","1x27","1x28",
    "2x1","2x2","2x3","2x4","2x5","2x6","2x7","2x8","2x9","2x10","2x11","2x12","2x13","2x14","2x15","2x16","2x17","2x18","2x19","2x20","2x21","2x22","2x23","2x24","2x25","2x26","2x27","2x28",
    "3x1","3x2","3x3","3x4","3x5","3x6","3x7","3x8","3x9","3x10","3x11","3x12","3x13","3x14","3x15","3x16","3x17","3x18","3x19","3x20","3x21","3x22","3x23","3x24","3x25","3x26","3x27","3x28",
    "4x1","4x2","4x3","4x4","4x5","4x6","4x7","4x8","4x9","4x10","4x11","4x12","4x13","4x14","4x15","4x16","4x17","4x18","4x19","4x20","4x21","4x22","4x23","4x24","4x25","4x26","4x27","4x28",
    "5x1","5x2","5x3","5x4","5x5","5x6","5x7","5x8","5x9","5x10","5x11","5x12","5x13","5x14","5x15","5x16","5x17","5x18","5x19","5x20","5x21","5x22","5x23","5x24","5x25","5x26","5x27","5x28",
    "6x1","6x2","6x3","6x4","6x5","6x6","6x7","6x8","6x9","6x10","6x11","6x12","6x13","6x14","6x15","6x16","6x17","6x18","6x19","6x20","6x21","6x22","6x23","6x24","6x25","6x26","6x27","6x28",
    "7x1","7x2","7x3","7x4","7x5","7x6","7x7","7x8","7x9","7x10","7x11","7x12","7x13","7x14","7x15","7x16","7x17","7x18","7x19","7x20","7x21","7x22","7x23","7x24","7x25","7x26","7x27","7x28",
    "8x1","8x2","8x3","8x4","8x5","8x6","8x7","8x8","8x9","8x10","8x11","8x12","8x13","8x14","8x15","8x16","8x17","8x18","8x19","8x20","8x21","8x22","8x23","8x24","8x25","8x26","8x27","8x28",
    "9x1","9x2","9x3","9x4","9x5","9x6","9x7","9x8","9x9","9x10","9x11","9x12","9x13","9x14","9x15","9x16","9x17","9x18","9x19","9x20","9x21","9x22","9x23","9x24","9x25","9x26","9x27","9x28",
    "10x1","10x2","10x3","10x4","10x5","10x6","10x7","10x8","10x9","10x10","10x11","10x12","10x13","10x14","10x15","10x16","10x17","10x18","10x19","10x20","10x21","10x22","10x23","10x24","10x25","10x26","10x27","10x28",
    "11x1","11x2","11x3","11x4","11x5","11x6","11x7","11x8","11x9","11x10","11x11","11x12","11x13","11x14","11x15","11x16","11x17","11x18","11x19","11x20","11x21","11x22","11x23","11x24","11x25","11x26","11x27","11x28",
    "12x1","12x2","12x3","12x4","12x5","12x6","12x7","12x8","12x9","12x10","12x11","12x12","12x13","12x14","12x15","12x16","12x17","12x18","12x19","12x20","12x21","12x22","12x23","12x24","12x25","12x26","12x27","12x28",
    "13x1","13x2","13x3","13x4","13x5","13x6","13x7","13x8","13x9","13x10","13x11","13x12","13x13","13x14","13x15","13x16","13x17","13x18","13x19","13x20","13x21","13x22","13x23","13x24","13x25","13x26","13x27","13x28",
    "14x1","14x2","14x3","14x4","14x5","14x6","14x7","14x8","14x9","14x10","14x11","14x12","14x13","14x14","14x15","14x16","14x17","14x18","14x19","14x20","14x21","14x22","14x23","14x24","14x25","14x26","14x27","14x28",
    "15x1","15x2","15x3","15x4","15x5","15x6","15x7","15x8","15x9","15x10","15x11","15x12","15x13","15x14","15x15","15x16","15x17","15x18","15x19","15x20","15x21","15x22","15x23","15x24","15x25","15x26","15x27","15x28",
    "16x1","16x2","16x3","16x4","16x5","16x6","16x7","16x8","16x9","16x10","16x11","16x12","16x13","16x14","16x15","16x16","16x17","16x18","16x19","16x20","16x21","16x22","16x23","16x24","16x25","16x26","16x27","16x28",
    "17x1","17x2","17x3","17x4","17x5","17x6","17x7","17x8","17x9","17x10","17x11","17x12","17x13","17x14","17x15","17x16","17x17","17x18","17x19","17x20","17x21","17x22","17x23","17x24","17x25","17x26","17x27","17x28",
    "18x1","18x2","18x3","18x4","18x5","18x6","18x7","18x8","18x9","18x10","18x11","18x12","18x13","18x14","18x15","18x16","18x17","18x18","18x19","18x20","18x21","18x22","18x23","18x24","18x25","18x26","18x27","18x28",
    "19x1","19x2","19x3","19x4","19x5","19x6","19x7","19x8","19x9","19x10","19x11","19x12","19x13","19x14","19x15","19x16","19x17","19x18","19x19","19x20","19x21","19x22","19x23","19x24","19x25","19x26","19x27","19x28",
    "20x1","20x2","20x3","20x4","20x5","20x6","20x7","20x8","20x9","20x10","20x11","20x12","20x13","20x14","20x15","20x16","20x17","20x18","20x19","20x20","20x21","20x22","20x23","20x24","20x25","20x26","20x27","20x28",
    "21x1","21x2","21x3","21x4","21x5","21x6","21x7","21x8","21x9","21x10","21x11","21x12","21x13","21x14","21x15","21x16","21x17","21x18","21x19","21x20","21x21","21x22","21x23","21x24","21x25","21x26","21x27","21x28",
    "22x1","22x2","22x3","22x4","22x5","22x6","22x7","22x8","22x9","22x10","22x11","22x12","22x13","22x14","22x15","22x16","22x17","22x18","22x19","22x20","22x21","22x22","22x23","22x24","22x25","22x26","22x27","22x28",
    "23x1","23x2","23x3","23x4","23x5","23x6","23x7","23x8","23x9","23x10","23x11","23x12","23x13","23x14","23x15","23x16","23x17","23x18","23x19","23x20","23x21","23x22","23x23","23x24","23x25","23x26","23x27","23x28",
    "24x1","24x2","24x3","24x4","24x5","24x6","24x7","24x8","24x9","24x10","24x11","24x12","24x13","24x14","24x15","24x16","24x17","24x18","24x19","24x20","24x21","24x22","24x23","24x24","24x25","24x26","24x27","24x28",
    "25x1","25x2","25x3","25x4","25x5","25x6","25x7","25x8","25x9","25x10","25x11","25x12","25x13","25x14","25x15","25x16","25x17","25x18","25x19","25x20","25x21","25x22","25x23","25x24","25x25","25x26","25x27","25x28",
    "26x1","26x2","26x3","26x4","26x5","26x6","26x7","26x8","26x9","26x10","26x11","26x12","26x13","26x14","26x15","26x16","26x17","26x18","26x19","26x20","26x21","26x22","26x23","26x24","26x25","26x26","26x27","26x28",
    "27x1","27x2","27x3","27x4","27x5","27x6","27x7","27x8","27x9","27x10","27x11","27x12","27x13","27x14","27x15","27x16","27x17","27x18","27x19","27x20","27x21","27x22","27x23","27x24","27x25","27x26","27x27","27x28",
    "28x1","28x2","28x3","28x4","28x5","28x6","28x7","28x8","28x9","28x10","28x11","28x12","28x13","28x14","28x15","28x16","28x17","28x18","28x19","28x20","28x21","28x22","28x23","28x24","28x25","28x26","28x27","28x28"
  ],
  "y_label": "label",
  "batch_size": 64,
  "epochs": 10,
  "learning_rate": 0.001
}
