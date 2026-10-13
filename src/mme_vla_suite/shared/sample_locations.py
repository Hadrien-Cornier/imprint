"""Normalize supported Stage-1 sample metadata schemas."""


def sample_location(sample):
    """Return ``(episode_index, step_index)`` from a sample-list entry."""
    if isinstance(sample, dict):
        return int(sample["episode_index"]), int(sample["step_index"])
    return int(sample[0]), int(sample[1])
