import pytest
import torch

from olmo.train.optim import (
    BoltOnWarmupScheduler,
    LinearWithWarmup,
    OptimizerConfig,
    SchedulerConfig,
    SchedulerType,
)


def test_linear_with_warmup_scheduler():
    initial_lr = 1.0
    max_steps = 10_000
    scheduler = LinearWithWarmup(
        grad_clip_warmup_steps=None, grad_clip_warmup_factor=None, warmup_steps=2000, warmup_min_lr=None
    )
    assert scheduler.get_lr(initial_lr, 0, max_steps) == 0.1
    assert scheduler.get_lr(initial_lr, 2000, max_steps) == 1.0
    assert scheduler.get_lr(initial_lr, 10_000, max_steps) == 0.1
    assert scheduler.get_lr(initial_lr, 3_000, max_steps) > scheduler.get_lr(initial_lr, 5_000, max_steps)


def test_bolt_on_warmup_scheduler():
    initial_lr = 1.0
    max_steps = 11_000
    alpha_f = 0.1
    scheduler = LinearWithWarmup(
        grad_clip_warmup_steps=None,
        grad_clip_warmup_factor=None,
        warmup_steps=1000,
        alpha_f=alpha_f,
        warmup_min_lr=None,
    )
    scheduler2 = BoltOnWarmupScheduler.wrap(scheduler, 5000, 6000)
    assert scheduler.get_lr(initial_lr, 100, max_steps) > 0.0
    assert scheduler2.get_lr(initial_lr, 100, max_steps) == 0.0
    assert scheduler2.get_lr(initial_lr, 5000, max_steps) == 0.0
    assert scheduler2.get_lr(initial_lr, 5500, max_steps) == pytest.approx(0.25 * (1 + alpha_f))
    assert scheduler2.get_lr(initial_lr, 6000, max_steps) == pytest.approx(0.5 * (1 + alpha_f))
    assert scheduler2.get_lr(initial_lr, 7000, max_steps) == scheduler.get_lr(initial_lr, 7000, max_steps)


class _MemoryOptimizerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action = torch.nn.Linear(2, 2)
        self.memory_encoder = torch.nn.Linear(2, 2)
        self.memory_film = torch.nn.Linear(2, 4)

    def get_llm_parameters(self):
        return []

    def get_connector_parameters(self):
        return []

    def get_vit_parameters(self):
        return []

    def get_frame_selection_parameters(self):
        return []

    def get_temporal_token_scorer_parameters(self):
        return []

    def get_action_expert_parameters(self):
        return self.parameters()

    def get_memory_encoder_parameters(self):
        return self.memory_encoder.parameters()

    def get_memory_film_parameters(self):
        return self.memory_film.parameters()

    def get_non_weight_decay_params(self):
        return []


def test_memory_optimizer_groups_are_disjoint():
    model = _MemoryOptimizerModel()
    config = OptimizerConfig(
        action_expert_learning_rate=2.5e-5,
        memory_encoder_learning_rate=2.5e-6,
        memory_film_learning_rate=5e-5,
        action_expert_weight_decay=0.0,
        memory_encoder_weight_decay=0.0,
        memory_film_weight_decay=0.0,
    )
    groups = config.get_param_groups(None, None, model)
    by_name = {group["group_name"]: group for group in groups}

    assert set(by_name) == {"action_expert", "memory_encoder", "memory_film"}
    assert by_name["action_expert"]["lr"] == 2.5e-5
    assert by_name["memory_encoder"]["lr"] == 2.5e-6
    assert by_name["memory_film"]["lr"] == 5e-5
    all_params = [parameter for group in groups for parameter in group["params"]]
    assert len(all_params) == len(set(all_params)) == len(list(model.parameters()))


def test_memory_optimizer_groups_have_independent_warmup():
    scheduler = SchedulerConfig(
        name=SchedulerType.multimodal,
        action_expert_t_warmup=500,
        memory_encoder_t_warmup=500,
        memory_film_t_warmup=2000,
        warmup_min_lr=0.0,
    ).build()

    film_lr = scheduler.get_lr(5e-5, 500, 100_000, "memory_film")
    expert_lr = scheduler.get_lr(2.5e-5, 500, 100_000, "action_expert")
    assert film_lr < 5e-5
    assert expert_lr == pytest.approx(2.5e-5, rel=1e-3)
