import numpy as np
import pytest

from sse_transfer.evaluation.n15_deployment_shift import (
    distance_distribution_summary,
    family_novelty_fraction,
    wasserstein_distance_1d,
)


def test_distance_distribution_summary_basic() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = distance_distribution_summary(d)
    assert out["n"] == 5
    assert out["median"] == pytest.approx(3.0)
    assert out["q25"] <= out["median"] <= out["q75"]
    assert out["q90"] >= out["q75"]
    assert out["max"] == pytest.approx(5.0)


def test_distance_distribution_summary_rejects_empty() -> None:
    with pytest.raises(ValueError):
        distance_distribution_summary(np.array([]))


def test_wasserstein_distance_identical_distributions_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert wasserstein_distance_1d(a, a) == pytest.approx(0.0)


def test_wasserstein_distance_shifted_distributions() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([5.0, 5.0, 5.0])
    assert wasserstein_distance_1d(a, b) == pytest.approx(5.0)


def test_family_novelty_fraction_all_novel() -> None:
    held = ["Garnet", "Garnet", "Garnet"]
    train = ["Perovskite", "NASICON"]
    out = family_novelty_fraction(held, train)
    assert out["n"] == 3
    assert out["n_novel"] == 3
    assert out["fraction_novel"] == pytest.approx(1.0)


def test_family_novelty_fraction_none_novel() -> None:
    held = ["Garnet", "Perovskite"]
    train = ["Garnet", "Perovskite", "NASICON"]
    out = family_novelty_fraction(held, train)
    assert out["fraction_novel"] == pytest.approx(0.0)


def test_family_novelty_fraction_mixed() -> None:
    held = ["Garnet", "NewFamily"]
    train = ["Garnet", "Perovskite"]
    out = family_novelty_fraction(held, train)
    assert out["n_novel"] == 1
    assert out["fraction_novel"] == pytest.approx(0.5)
