"""Static memory-stream capacities shared by training and policy evaluation."""


def max_recent_tokens(config, default: int = 1024) -> int:
    """Return the configured recent-token capacity, preserving legacy configs."""
    value = int(getattr(config, "max_recent_tokens", 0) or 0)
    return value if value > 0 else default


def max_step_tokens(config, default: int = 800) -> int:
    """Return the configured action/proprio capacity, preserving legacy configs."""
    value = int(getattr(config, "max_step_tokens", 0) or 0)
    return value if value > 0 else default
