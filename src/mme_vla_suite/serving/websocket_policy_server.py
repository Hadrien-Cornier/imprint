import asyncio
import os
import http
import logging
import time
import traceback

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from mme_vla_suite.policies.policy import MME_VLA_Policy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: MME_VLA_Policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._timing_log = os.environ.get("MME_TIMING_LOG", "") or None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if obs.get("reset", False):
                    tstart = time.monotonic()
                    self._policy.reset(
                        prompt=obs.get("prompt", None),
                        lang_emb=obs.get("lang_emb", None),
                    )
                    tend = time.monotonic() - tstart
                    await websocket.send(packer.pack(
                        {"reset_finished": True, "reset_time_ms": tend * 1000}))
                elif obs.get("add_buffer", False):
                    tstart = time.monotonic()
                    self._policy.add_buffer(obs)
                    tend = time.monotonic() - tstart
                    await websocket.send(packer.pack(
                        {"add_buffer_finished": True, "add_buffer_time_ms": tend * 1000}))
                else:
                    _ti = time.monotonic()
                    outputs = self._policy.infer(obs)
                    _infer_ms = (time.monotonic() - _ti) * 1000.0
                    if self._timing_log is not None:
                        try:
                            _mt = outputs.get("infer_time_ms", -1.0)
                            with open(self._timing_log, "a") as _f:
                                _f.write(f"infer {_infer_ms:.3f} model_ms={_mt:.3f}\n")
                        except Exception:
                            pass
                    await websocket.send(packer.pack(outputs))

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
