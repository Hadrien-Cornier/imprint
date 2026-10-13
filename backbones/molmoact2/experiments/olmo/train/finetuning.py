"""Fine-tuning policy helpers shared by training launchers."""

from __future__ import annotations


def resolve_vlm_finetuning_flags(
    *,
    ft_vlm: bool,
    ft_llm: bool | None,
    ft_connector: bool | None,
    ft_vit: bool | None,
) -> tuple[bool, bool, bool]:
    """Resolve component overrides against the backwards-compatible VLM flag."""

    return (
        bool(ft_vlm) if ft_llm is None else bool(ft_llm),
        bool(ft_vlm) if ft_connector is None else bool(ft_connector),
        bool(ft_vlm) if ft_vit is None else bool(ft_vit),
    )
