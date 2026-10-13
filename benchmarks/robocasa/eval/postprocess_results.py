#!/usr/bin/env python3
"""Produce paper-compatible RoboCasa metrics from raw evaluator statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "imprint.robocasa.paper_metrics.v1"
SWEETEN_TASK_GROUP = "sweeten_hot_chocolate"
SWEETEN_RULE = "stable_placement_count == goal_count"
RAW_RULE = "raw evaluator success"


def _required_int(mapping: dict[str, Any], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}.{key} must be an integer")
    return value


def _required_bool(mapping: dict[str, Any], key: str, *, context: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a boolean")
    return value


def _paper_success(
    task_group: str,
    episode: dict[str, Any],
    *,
    context: str,
) -> tuple[bool, dict[str, int]]:
    if task_group != SWEETEN_TASK_GROUP:
        return _required_bool(episode, "success", context=context), {}

    validation = episode.get("sweeten_validation")
    if not isinstance(validation, dict):
        raise ValueError(f"{context}.sweeten_validation must be an object")
    goal_count = _required_int(
        validation,
        "goal_count",
        context=f"{context}.sweeten_validation",
    )
    stable_count = _required_int(
        validation,
        "stable_placement_count",
        context=f"{context}.sweeten_validation",
    )
    return stable_count == goal_count, {
        "goal_count": goal_count,
        "stable_placement_count": stable_count,
    }


def paper_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert one variant or task-group stats object into paper metrics."""
    task_group = payload.get("task_group")
    if not isinstance(task_group, str) or not task_group:
        raise ValueError("task_group must be a nonempty string")

    variants = payload.get("variants")
    if variants is None:
        variants = [payload]
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants must be a nonempty list")

    episode_rows = []
    for variant_index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"variants[{variant_index}] must be an object")
        env_name = variant.get("env_name")
        if not isinstance(env_name, str) or not env_name:
            raise ValueError(f"variants[{variant_index}].env_name must be a nonempty string")
        episodes = variant.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"variants[{variant_index}].episodes must be a list")
        declared_episodes = _required_int(
            variant,
            "num_episodes",
            context=f"variants[{variant_index}]",
        )
        if declared_episodes != len(episodes):
            raise ValueError(
                f"variants[{variant_index}] declares {declared_episodes} episodes "
                f"but contains {len(episodes)}"
            )

        for episode_index, episode in enumerate(episodes):
            context = f"variants[{variant_index}].episodes[{episode_index}]"
            if not isinstance(episode, dict):
                raise ValueError(f"{context} must be an object")
            raw_success = _required_bool(episode, "success", context=context)
            paper_success, details = _paper_success(
                task_group,
                episode,
                context=context,
            )
            episode_number = _required_int(episode, "episode", context=context)
            episode_rows.append(
                {
                    "env_name": env_name,
                    "episode": episode_number,
                    "seed": episode.get("seed"),
                    "raw_success": raw_success,
                    "paper_success": paper_success,
                    **details,
                }
            )

    total = len(episode_rows)
    if total == 0:
        raise ValueError("cannot report metrics for zero episodes")
    raw_successes = sum(row["raw_success"] for row in episode_rows)
    paper_successes = sum(row["paper_success"] for row in episode_rows)
    rule = SWEETEN_RULE if task_group == SWEETEN_TASK_GROUP else RAW_RULE
    return {
        "schema": SCHEMA,
        "task_group": task_group,
        "paper_success_rule": rule,
        "num_episodes": total,
        "raw_num_successes": raw_successes,
        "raw_success_rate": raw_successes / total,
        "paper_num_successes": paper_successes,
        "paper_success_rate": paper_successes / total,
        "episodes": episode_rows,
    }


def write_paper_metrics(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    with source.open() as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("RoboCasa stats root must be an object")
    metrics = paper_metrics(payload)
    metrics["source_stats"] = str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="stats.json or group_stats.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination; defaults to paper_metrics.json beside the source file.",
    )
    args = parser.parse_args()
    destination = args.output or args.source.with_name("paper_metrics.json")
    metrics = write_paper_metrics(args.source, destination)
    print(
        f"{metrics['task_group']}: "
        f"paper={metrics['paper_num_successes']}/{metrics['num_episodes']} "
        f"raw={metrics['raw_num_successes']}/{metrics['num_episodes']} "
        f"rule={metrics['paper_success_rule']}"
    )


if __name__ == "__main__":
    main()
