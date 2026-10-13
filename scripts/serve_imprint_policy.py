#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from mme_vla_suite.policies import policy_config
from mme_vla_suite.serving.websocket_policy_server import WebsocketPolicyServer
from mme_vla_suite.training import config as training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--history-config")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = training_config.get_config(args.config)
    if args.history_config:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                history_config=args.history_config,
                use_history=True,
            ),
        )
    policy = policy_config.create_trained_policy(config, args.checkpoint.resolve(), seed=args.seed)
    WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "config": args.config,
            "history_config": config.model.history_config,
            "checkpoint": str(args.checkpoint.resolve()),
        },
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
