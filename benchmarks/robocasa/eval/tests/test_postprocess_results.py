from __future__ import annotations

import json
from pathlib import Path
import sys

EVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_ROOT))

from postprocess_results import paper_metrics  # noqa: E402
from postprocess_results import write_paper_metrics  # noqa: E402


def _episode(index: int, success: bool, stable: int, goal: int) -> dict:
    return {
        "episode": index,
        "seed": 100000 + index,
        "success": success,
        "sweeten_validation": {
            "stable_placement_count": stable,
            "goal_count": goal,
        },
    }


def test_sweeten_paper_metric_matches_stable_placement_completion() -> None:
    metrics = paper_metrics(
        {
            "task_group": "sweeten_hot_chocolate",
            "variants": [
                {
                    "env_name": "SweetenHotChocolate",
                    "num_episodes": 3,
                    "episodes": [
                        _episode(0, False, 1, 1),
                        _episode(1, False, 1, 2),
                        _episode(2, True, 2, 2),
                    ],
                }
            ],
        }
    )

    assert metrics["paper_success_rule"] == "stable_placement_count == goal_count"
    assert metrics["raw_num_successes"] == 1
    assert metrics["paper_num_successes"] == 2
    assert metrics["paper_success_rate"] == 2 / 3


def test_other_tasks_preserve_raw_success() -> None:
    metrics = paper_metrics(
        {
            "task_group": "organize_condiments",
            "env_name": "OrganizeCondiments",
            "num_episodes": 2,
            "episodes": [
                {"episode": 0, "seed": 1, "success": True},
                {"episode": 1, "seed": 2, "success": False},
            ],
        }
    )

    assert metrics["paper_success_rule"] == "raw evaluator success"
    assert metrics["raw_num_successes"] == 1
    assert metrics["paper_num_successes"] == 1


def test_writer_creates_auditable_output(tmp_path: Path) -> None:
    source = tmp_path / "stats.json"
    destination = tmp_path / "paper_metrics.json"
    source.write_text(
        json.dumps(
            {
                "task_group": "sweeten_hot_chocolate",
                "env_name": "SweetenHotChocolate",
                "num_episodes": 1,
                "episodes": [_episode(0, False, 1, 1)],
            }
        )
    )

    written = write_paper_metrics(source, destination)
    reloaded = json.loads(destination.read_text())

    assert written["paper_num_successes"] == 1
    assert reloaded["paper_num_successes"] == 1
    assert reloaded["raw_num_successes"] == 0
    assert reloaded["source_stats"] == str(source.resolve())
