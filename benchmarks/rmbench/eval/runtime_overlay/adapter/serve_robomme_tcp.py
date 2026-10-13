"""Raw-TCP server for the official RMBench policy client.

Unlike RMBench's generic policy_model_server, this module deliberately avoids
importing environment packages such as SAPIEN. It serves the same 4-byte
big-endian JSON framing expected by script/eval_policy_client.py.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import socket
import threading
import time
import traceback
from typing import Any

import numpy as np
import yaml


class NumpyEncoder(json.JSONEncoder):
    def default(self, value):
        if isinstance(value, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(value.tobytes()).decode("ascii"),
                "dtype": str(value.dtype),
                "shape": value.shape,
            }
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return super().default(value)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(data: str) -> Any:
    def object_hook(value):
        if value.get("__numpy_array__"):
            raw = base64.b64decode(value["data"])
            return np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"])
        return value

    return json.loads(data, object_hook=object_hook)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed before a complete message arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class TCPPolicyServer:
    def __init__(self, model, *, host: str, port: int):
        self.model = model
        self.host = host
        self.port = port
        self.ready = threading.Event()
        self._running = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        profile_path = os.environ.get("RMBENCH_TCP_PROFILE_PATH")
        self._profile_path = profile_path if profile_path else None
        self._profile_lock = threading.Lock()

    def start_background(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self._socket = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            self.port = server.getsockname()[1]
            server.listen()
            server.settimeout(0.5)
            self._running.set()
            self.ready.set()
            while self._running.is_set():
                try:
                    client, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(
                    target=self._handle_client, args=(client,), daemon=True
                ).start()

    def stop(self) -> None:
        self._running.clear()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @staticmethod
    def _send(sock: socket.socket, payload: dict) -> tuple[float, float, int]:
        started = time.perf_counter()
        raw = numpy_to_json(payload).encode("utf-8")
        encode_seconds = time.perf_counter() - started
        started = time.perf_counter()
        sock.sendall(len(raw).to_bytes(4, "big") + raw)
        return encode_seconds, time.perf_counter() - started, len(raw)

    def _record_profile(self, row: dict[str, Any]) -> None:
        if self._profile_path is None:
            return
        with self._profile_lock:
            with open(self._profile_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            while self._running.is_set():
                command = "<unparsed>"
                try:
                    size = int.from_bytes(recv_exact(client, 4), "big")
                    started = time.perf_counter()
                    raw_request = recv_exact(client, size)
                    receive_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    request = json_to_numpy(raw_request.decode("utf-8"))
                    decode_seconds = time.perf_counter() - started
                    command = request["cmd"]
                    observation = request.get("obs")
                    method = getattr(self.model, command, None)
                    if not callable(method):
                        raise AttributeError(f"no model method named {command!r}")
                    started = time.perf_counter()
                    result = method(observation) if observation is not None else method()
                    method_seconds = time.perf_counter() - started
                    encode_seconds, send_seconds, response_size = self._send(
                        client, {"res": result}
                    )
                    self._record_profile(
                        {
                            "command": command,
                            "request_bytes": size,
                            "request_receive_ms": receive_seconds * 1_000.0,
                            "request_decode_ms": decode_seconds * 1_000.0,
                            "elapsed_ms": method_seconds * 1_000.0,
                            "response_bytes": response_size,
                            "response_encode_ms": encode_seconds * 1_000.0,
                            "response_send_ms": send_seconds * 1_000.0,
                        }
                    )
                except (ConnectionError, BrokenPipeError, ConnectionResetError):
                    return
                except Exception as error:
                    print(
                        f"RMBENCH_POLICY_ERROR command={command!r}: {error}",
                        flush=True,
                    )
                    traceback.print_exc()
                    try:
                        self._send(
                            client,
                            {
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                    except OSError:
                        pass
                    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    adapter_module = config.pop("adapter_module", "robomme_tcp_policy")
    get_model = importlib.import_module(adapter_module).get_model

    server = TCPPolicyServer(get_model(config), host=args.host, port=args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
