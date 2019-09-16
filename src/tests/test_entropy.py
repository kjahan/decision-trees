import math

from src.models.entropy import Entropy


def test_compute_list_entropy():
    values = [0, 1, 0, 1]
    expected_entropy = -1 * (0.5*math.log(0.5, 2) + 0.5*math.log(0.5, 2))
    assert Entropy.compute_list_entropy(values) == expected_entropy
