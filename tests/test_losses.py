"""Loss building blocks: local losses (P, D) and their aggregation over samples (P,)."""

import pytest
import torch

from cnnsim.losses import aggregate, image_loss, mae, mean_output, mse, msen, pixel_error_rate


def _pair():
    # P=2 candidates, D=3 samples, 1×2×2 images; ref is the zero image except a +1 pixel
    ref = torch.zeros(3, 1, 2, 2)
    ref[:, 0, 0, 0] = 1.0
    y = torch.zeros(2, 3, 1, 2, 2)
    y[1] = -1.0  # candidate 1: all pixels -1
    return y, ref


def test_local_losses_values_and_shapes():
    y, ref = _pair()
    for f in (mse, msen, mae, pixel_error_rate):
        assert f(y, ref).shape == (2, 3)
    # candidate 0: one pixel off by 1 -> mse = mae = 1/4; candidate 1: (4 + 1 + 1 + 1)/4, (2+1+1+1)/4
    torch.testing.assert_close(mse(y, ref)[:, 0], torch.tensor([0.25, 1.75]))
    torch.testing.assert_close(msen(y, ref), mse(y, ref) / 4)
    torch.testing.assert_close(mae(y, ref)[:, 0], torch.tensor([0.25, 1.25]))
    # candidate 0: the +1 pixel has y=0, not > 0 -> 1 wrong; candidate 1: same pixel wrong
    torch.testing.assert_close(pixel_error_rate(y, ref)[:, 0], torch.tensor([0.25, 0.25]))


def test_msen_is_in_unit_interval_for_bounded_outputs():
    ref = torch.ones(1, 1, 3, 3)
    assert msen(-torch.ones(1, 1, 1, 3, 3), ref).item() == 1.0
    assert msen(torch.ones(1, 1, 1, 3, 3), ref).item() == 0.0


def test_mean_output():
    y = torch.arange(8.0).reshape(1, 2, 1, 2, 2)
    torch.testing.assert_close(mean_output(y), torch.tensor([[1.5, 5.5]]))


def test_aggregations():
    local = torch.tensor([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0]])
    torch.testing.assert_close(aggregate(local, "mean"), torch.tensor([7 / 3, 3.0]))
    torch.testing.assert_close(aggregate(local, "max"), torch.tensor([4.0, 3.0]))
    torch.testing.assert_close(aggregate(local, "rms"), torch.tensor([(21 / 3) ** 0.5, 3.0]))
    torch.testing.assert_close(aggregate(local, "harmonic"), torch.tensor([3 / 1.75, 3.0]))
    w = torch.tensor([1.0, 1.0, 2.0])
    torch.testing.assert_close(aggregate(local, "mean", weights=w), torch.tensor([11 / 4, 3.0]))


def test_harmonic_is_finite_with_a_zero_loss():
    assert torch.isfinite(aggregate(torch.tensor([[0.0, 1.0]]), "harmonic")).all()


def test_aggregation_errors():
    local = torch.ones(1, 2)
    with pytest.raises(ValueError, match="weights"):
        aggregate(local, "max", weights=torch.ones(2))
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate(local, "median")


def test_image_loss_reads_the_given_key():
    y, ref = _pair()
    loss = image_loss("mae", "max", key="ref")
    torch.testing.assert_close(loss(y, {"ref": ref}), aggregate(mae(y, ref), "max"))
    with pytest.raises(KeyError):
        image_loss("huber")
