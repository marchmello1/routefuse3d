import pytest

from msc_schedule import polynomial_lr


def test_polynomial_lr_boundaries_and_resume_equivalence():
    assert polynomial_lr(0.01, 0, 100, 0.9) == pytest.approx(0.01)
    assert polynomial_lr(0.01, 50, 100, 0.9) == pytest.approx(0.01 * 0.5**0.9)
    assert polynomial_lr(0.01, 100, 100, 0.9) == 0.0
    assert polynomial_lr(0.01, 120, 100, 0.9) == 0.0
    # The schedule is a function of global step, not the wall-clock launch.
    resumed_step = 50 + 1
    assert polynomial_lr(0.01, resumed_step, 100, 0.9) == pytest.approx(
        0.01 * (1.0 - 51 / 100) ** 0.9
    )


@pytest.mark.parametrize("lr,step,total,power", [(0, 0, 10, 0.9), (0.01, -1, 10, 0.9), (0.01, 1, 0, 0.9), (0.01, 1, 10, 0)])
def test_polynomial_lr_rejects_invalid_configuration(lr, step, total, power):
    with pytest.raises(ValueError):
        polynomial_lr(lr, step, total, power)
