"""SAPIEN-free raw-TCP server for the official RMBench policy client."""

from __future__ import annotations

import argparse
import base64
import json
import socket
import threading
import traceback
from collections.abc import Mapping
from typing import Any

import numpy as np

from serve_rmbench_molmoact2 import get_model


class _NumpyEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
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
    return json.dumps(data, cls=_NumpyEncoder)


def json_to_numpy(data: str) -> Any:
    def object_hook(value: dict[str, Any]) -> Any:
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
    def __init__(self, model: object, *, host: str, port: int) -> None:
        self.model = model
        self.host = host
        self.port = port
        self._running = threading.Event()
        self._socket: socket.socket | None = None

    @staticmethod
    def _send(sock: socket.socket, payload: dict[str, Any]) -> None:
        raw = numpy_to_json(payload).encode("utf-8")
        sock.sendall(len(raw).to_bytes(4, "big") + raw)

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            while self._running.is_set():
                try:
                    size = int.from_bytes(recv_exact(client, 4), "big")
                    request = json_to_numpy(recv_exact(client, size).decode("utf-8"))
                    if not isinstance(request, Mapping):
                        raise TypeError("RMBench TCP request must be a JSON object")
                    command = request["cmd"]
                    if not isinstance(command, str):
                        raise TypeError("RMBench TCP command must be a string")
                    method = getattr(self.model, command, None)
                    if not callable(method):
                        raise AttributeError(f"no policy method named {command!r}")
                    observation = request.get("obs")
                    result = method(observation) if observation is not None else method()
                    self._send(client, {"res": result})
                except (ConnectionError, BrokenPipeError, ConnectionResetError):
                    return
                except Exception as error:
                    try:
                        self._send(
                            client,
                            {"error": str(error), "traceback": traceback.format_exc()},
                        )
                    except OSError:
                        pass
                    return

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self._socket = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            self._running.set()
            try:
                while self._running.is_set():
                    client, _ = server.accept()
                    threading.Thread(
                        target=self._handle_client,
                        args=(client,),
                        daemon=True,
                    ).start()
            finally:
                self._running.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, Mapping):
        raise TypeError("MolmoAct2 RMBench server config must be a mapping")
    TCPPolicyServer(get_model(config), host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
