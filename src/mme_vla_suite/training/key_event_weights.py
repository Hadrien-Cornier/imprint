from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class KeyEventWeightLookup:
    weight: float
    weighted_steps: dict[int, frozenset[int]]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_store_path: str,
        expected_episode_count: int,
        expected_radius: int,
        expected_weight: float,
    ) -> "KeyEventWeightLookup":
        payload = json.loads(Path(path).read_text())
        expected = {
            "schema_version": 1,
            "training_store_path": expected_store_path,
            "episode_count": expected_episode_count,
            "window_radius": expected_radius,
            "action_loss_weight": float(expected_weight),
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise ValueError(
                    f"key-event manifest {field}={payload.get(field)!r}, "
                    f"expected {value!r}"
                )

        episodes = payload.get("episodes")
        expected_ids = {str(index) for index in range(expected_episode_count)}
        if not isinstance(episodes, dict) or set(episodes) != expected_ids:
            raise ValueError("key-event manifest episode IDs are incomplete")

        weighted_steps: dict[int, frozenset[int]] = {}
        for episode_id, episode in episodes.items():
            steps = episode.get("weighted_steps")
            if not isinstance(steps, list) or steps != sorted(set(steps)):
                raise ValueError(
                    f"episode {episode_id} weighted_steps must be sorted and unique"
                )
            weighted_steps[int(episode_id)] = frozenset(int(step) for step in steps)
        return cls(weight=float(expected_weight), weighted_steps=weighted_steps)

    def weight_for(self, episode_index: int, step_index: int) -> float:
        if episode_index not in self.weighted_steps:
            raise KeyError(
                f"episode {episode_index} is absent from key-event manifest"
            )
        if step_index in self.weighted_steps[episode_index]:
            return self.weight
        return 1.0
