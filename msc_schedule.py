"""Resume-safe update-count schedules for MSC pretraining."""


def polynomial_lr(initial_lr: float, step: int, total_steps: int, power: float) -> float:
    if initial_lr <= 0 or total_steps <= 0 or power <= 0:
        raise ValueError("initial_lr, total_steps, and power must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")
    fraction = min(step, total_steps) / total_steps
    return initial_lr * (1.0 - fraction) ** power
