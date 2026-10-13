import dataclasses
import logging
import pathlib

import tyro

from mme_vla_suite.serving.websocket_policy_server import WebsocketPolicyServer
import rmbench_policy


@dataclasses.dataclass
class Args:
    ckpt_dir: str
    config: str = "mme_wan_stage2_base"
    history_config: str = rmbench_policy.DEFAULT_HISTORY_CONFIG
    prediction_horizon: int = 50
    host: str = "0.0.0.0"
    port: int = 8000
    seed: int = 42


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_dir = pathlib.Path(args.ckpt_dir).resolve()
    policy = rmbench_policy.create_policy(
        checkpoint_dir,
        config_name=args.config,
        history_config=args.history_config,
        prediction_horizon=args.prediction_horizon,
        seed=args.seed,
    )
    metadata = rmbench_policy.server_metadata(
        config_name=args.config,
        history_config=args.history_config,
        prediction_horizon=args.prediction_horizon,
    )
    logging.info("Serving RM-Bench policy on %s:%d: %s", args.host, args.port, metadata)
    WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
