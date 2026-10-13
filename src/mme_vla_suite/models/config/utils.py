import os
from omegaconf import DictConfig


def get_history_config(history_config: str | DictConfig):
    if history_config in ["None", "none"]:
        return None
    if isinstance(history_config, str):
        import omegaconf
        # Legacy CWD-relative location (training runs from the openpi_rc root).
        cwd_path = os.path.join("src/mme_vla_suite/models/config/robomme", history_config)
        # Module-relative location: robust from any CWD (tests, serving, etc.).
        module_path = os.path.join(os.path.dirname(__file__), "robomme", history_config)
        path = cwd_path if os.path.exists(cwd_path) else module_path
        history_config = omegaconf.OmegaConf.load(path)
        return history_config
    elif isinstance(history_config, DictConfig):
        return history_config
    elif history_config is None:
        return None
    else:
        raise ValueError(f"Invalid history config: {history_config}")
