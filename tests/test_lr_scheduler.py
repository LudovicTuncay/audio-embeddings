from src.utils.lr_schedulers import LinearWarmupCosineDecay


def test_lr_scheduler_warmup_increases_linearly() -> None:
    scheduler = LinearWarmupCosineDecay(
        warmup_steps=4,
        total_steps=20,
        final_lr_ratio=0.1,
    )

    values = [scheduler(step) for step in range(4)]

    assert values[0] == 0.0
    assert values[1] > values[0]
    assert values[2] > values[1]
    assert values[3] > values[2]


def test_lr_scheduler_decay_stays_within_bounds() -> None:
    final_lr_ratio = 0.2
    scheduler = LinearWarmupCosineDecay(
        warmup_steps=2,
        total_steps=10,
        final_lr_ratio=final_lr_ratio,
    )

    values = [scheduler(step) for step in range(0, 16)]

    assert all(final_lr_ratio <= value <= 1.0 for value in values[2:])
    assert abs(scheduler(10) - final_lr_ratio) < 1e-8
    assert abs(scheduler(15) - final_lr_ratio) < 1e-8
