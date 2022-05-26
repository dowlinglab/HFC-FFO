import pytest
import numpy as np
import matplotlib.pyplot as plt


from fffit.pareto import (
    is_pareto_efficient_simple,
    is_pareto_efficient,
    find_pareto_set,
    plt_pareto_2D
)
from fffit.tests.base_test import BaseTest


class TestPareto(BaseTest):
    def test_pareto_simple_known(self):
        costs = np.asarray([[0.2, 0.2], [0.1, 0.1],])
        result, pareto_points, dominated_points = find_pareto_set(
            costs, is_pareto_efficient_simple
        )
        assert np.allclose(result, [False, True])
        assert np.allclose(pareto_points, [[0.1, 0.1]])
        assert np.allclose(dominated_points, [[0.2, 0.2]])

    def test_paret_simple_max(self):
        costs = np.asarray([[0.2, 0.2], [0.1, 0.1],])
        result, pareto_points, dominated_points = find_pareto_set(
            costs, is_pareto_efficient_simple, max_front=True
        )
        assert np.allclose(result, [True, False])
        assert np.allclose(pareto_points, [[0.2, 0.2]])
        assert np.allclose(dominated_points, [[0.1, 0.1]])

    def test_pareto_efficient_known(self):
        costs = np.asarray([[0.2, 0.2], [0.1, 0.1],])
        result, pareto_points, dominated_points = find_pareto_set(
            costs, is_pareto_efficient
        )
        assert np.allclose(result, [False, True])
        assert np.allclose(pareto_points, [[0.1, 0.1]])
        assert np.allclose(dominated_points, [[0.2, 0.2]])

    def test_paret_efficient_max(self):
        costs = np.asarray([[0.2, 0.2], [0.1, 0.1],])
        result, pareto_points, dominated_points = find_pareto_set(
            costs, is_pareto_efficient, max_front=True
        )
        assert np.allclose(result, [True, False])
        assert np.allclose(pareto_points, [[0.2, 0.2]])
        assert np.allclose(dominated_points, [[0.1, 0.1]])

    def test_pareto_simple_3d(self):
        costs = np.asarray(
            [[0.2, 0.2, 0.1], [0.1, 0.1, 0.2], [0.1, 0.1, 0.1],]
        )
        result, pareto_points, dominated_points = find_pareto_set(
            costs, is_pareto_efficient_simple
        )
        assert np.allclose(result, [False, False, True])
        assert np.allclose(pareto_points, [[0.1, 0.1, 0.1]])
        assert np.allclose(
            dominated_points, [[0.2, 0.2, 0.1], [0.1, 0.1, 0.2]]
        )

        costs = np.asarray(
            [
                [0.2, 0.2, 0.1],
                [0.1, 0.1, 0.2],
                [0.05, 0.3, 0.4],
                [0.1, 0.15, 0.3],
            ]
        )
        result, pareto_points, dominated_points = find_pareto_set(
            costs, is_pareto_efficient_simple
        )
        assert np.allclose(result, [True, True, True, False])
        assert np.allclose(
            pareto_points, [[0.2, 0.2, 0.1], [0.1, 0.1, 0.2], [0.05, 0.3, 0.4]]
        )
        assert np.allclose(dominated_points, [[0.1, 0.15, 0.3]])

    def test_compare_pareto(self):
        np.random.seed(5)
        costs = np.random.random(size=(1000, 10))
        result1, pareto_points1, dominated_points1 = find_pareto_set(
            costs, is_pareto_efficient_simple
        )
        result2, pareto_points2, dominated_points2 = find_pareto_set(
            costs, is_pareto_efficient
        )
        assert np.allclose(result1, result2)
        assert np.allclose(pareto_points1, pareto_points2)
        assert np.allclose(dominated_points1, dominated_points2)

    def test_compare_pareto_max(self):
        np.random.seed(5)
        costs = np.random.random(size=(1000, 10))
        result1, pareto_points1, dominated_points1 = find_pareto_set(
            costs, is_pareto_efficient_simple, max_front=True
        )
        result2, pareto_points2, dominated_points2 = find_pareto_set(
            costs, is_pareto_efficient, max_front=True
        )
        assert np.allclose(result1, result2)
        assert np.allclose(pareto_points1, pareto_points2)
        assert np.allclose(dominated_points1, dominated_points2)

