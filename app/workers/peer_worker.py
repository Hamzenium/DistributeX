import asyncio
import os
import ssl
import aio_pika
from multiprocessing import Queue
from queue import Empty
from datetime import datetime
import json


HEARTBEAT_INTERVAL = 5.0   # seconds
LOOP_SLEEP = 0.1           # rest every 0.1s


def peer_worker(
    worker_name: str,
    rabbit_queue_name: str,
    control_queue: Queue,
):
    asyncio.run(
        _run_worker(worker_name, rabbit_queue_name, control_queue)
    )


async def _run_worker(
    worker_name: str,
    rabbit_queue_name: str,
    control_queue: Queue,
):
    running = True

    rabbit_url = os.getenv("RABBITMQ_URL")
    connection = None
    channel = None
    command_queue = None

    # Heartbeat state (disabled initially)
    heartbeat_queue = None
    heartbeat_queue_name = None
    heartbeat_enabled = False
    last_heartbeat = 0.0

    if rabbit_url:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.load_verify_locations("isrgrootx1.pem")

        connection = await aio_pika.connect_robust(
            rabbit_url,
            ssl_context=ssl_ctx,
        )

        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        # Command queue (this worker listens here)
        command_queue = await channel.declare_queue(
            rabbit_queue_name,
            durable=True,
        )

        print(f"[{worker_name}] RabbitMQ connected")

    else:
        print(f"[{worker_name}] RabbitMQ disabled")

    print(f"[{worker_name}] worker started (heartbeat disabled)")

    # -------------------------------
    # Main loop
    # -------------------------------
    while running:

        # Local control queue
        try:
            msg = control_queue.get_nowait()
            print(f"[{worker_name}] local cmd: {msg}")

            if msg in ("STOP", False):
                running = False
                break

            # Enable heartbeat with PRE-EXISTING queue
            if isinstance(msg, dict) and msg.get("type") == "ENABLE_HEARTBEAT":
                heartbeat_queue_name = msg.get("queue")

                if heartbeat_queue_name and channel:
                    heartbeat_queue = await channel.get_queue(
                        heartbeat_queue_name,
                        ensure=False,
                    )
                    heartbeat_enabled = True
                    print(
                        f"[{worker_name}] heartbeat enabled → {heartbeat_queue_name}"
                    )
                else:
                    print(f"[{worker_name}] invalid heartbeat command")

            else:
                await handle_command(worker_name, "local", msg)

        except Empty:
            pass

        await asyncio.sleep(LOOP_SLEEP)

        # RabbitMQ command queue
        if command_queue:
            try:
                message = await command_queue.get(
                    timeout=0.1,
                    fail=False,
                )

                if message:
                    async with message.process():
                        payload = message.body.decode()
                        print(f"[{worker_name}] rabbit cmd: {payload}")

                        try:
                            data = json.loads(payload)
                        except Exception:
                            data = None

                        if payload in ("STOP", False):
                            running = False
                            break

                        if (
                            isinstance(data, dict)
                            and data.get("type") == "ENABLE_HEARTBEAT"
                        ):
                            heartbeat_queue_name = data.get("queue")

                            if heartbeat_queue_name and channel:
                                heartbeat_queue = await channel.get_queue(
                                    heartbeat_queue_name,
                                    ensure=False,  # DO NOT DECLARE
                                )
                                heartbeat_enabled = True
                                print(
                                    f"[{worker_name}] heartbeat enabled (rabbitmq) → "
                                    f"{heartbeat_queue_name}"
                                )
                        else:
                            await handle_command(
                                worker_name, "rabbitmq", payload
                            )

            except asyncio.TimeoutError:
                pass

        await asyncio.sleep(LOOP_SLEEP)

        # Heartbeat sender (every 5 seconds)
        if heartbeat_enabled and heartbeat_queue:
            now = asyncio.get_event_loop().time()

            if (now - last_heartbeat) >= HEARTBEAT_INTERVAL:
                heartbeat = {
                    "worker": worker_name,
                    "queue": heartbeat_queue_name,
                    "ts": datetime.utcnow().isoformat(),
                    "status": "alive",
                }

                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(heartbeat).encode(),
                        delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
                    ),
                    routing_key=heartbeat_queue.name,
                )

                last_heartbeat = now
                print(
                    f"[{worker_name}] heartbeat sent → {heartbeat_queue_name}"
                )

        await asyncio.sleep(LOOP_SLEEP)

    if connection:
        await connection.close()

    print(f"[{worker_name}] worker exited cleanly")


async def handle_command(worker_name: str, source: str, payload: str):
    print(f"[{worker_name}] handling {payload} from {source}")
    await asyncio.sleep(1)
