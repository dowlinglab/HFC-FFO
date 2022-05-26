import pytest
import numpy as np


from fffit.utils import _clean_bounds_values
from fffit.utils import values_real_to_scaled
from fffit.utils import values_scaled_to_real
from fffit.utils import variances_scaled_to_real
from fffit.tests.base_test import BaseTest


class TestUtils(BaseTest):
    def test_values_real_to_scaled_single(self):
        bounds = [2.0, 4.0]
        value = 2.0
        scaled_value = values_real_to_scaled(value, bounds)
        assert np.isclose(scaled_value, 0.0)

        value = 4.0
        scaled_value = values_real_to_scaled(value, bounds)
        assert np.isclose(scaled_value, 1.0)

        value = 3.0
        scaled_value = values_real_to_scaled(value, bounds)
        assert np.isclose(scaled_value, 0.5)

        value = [1.0]
        scaled_value = values_real_to_scaled(value, bounds)
        assert np.isclose(scaled_value, -0.5)

        bounds = [-5.0, -4.0]
        value = -4.5
        scaled_value = values_real_to_scaled(value, bounds)
        assert np.isclose(scaled_value, 0.5)

    def test_values_real_to_scaled_multiple_values(self):
        bounds = [2.0, 4.0]
        values = [2.0, 4.0]
        scaled_values = values_real_to_scaled(values, bounds)
        assert np.isclose(scaled_values, [[0.0], [1.0]]).all()

        values = [2.0, 4.0, 5.0]
        scaled_values = values_real_to_scaled(values, bounds)
        assert np.isclose(scaled_values, [[0.0], [1.0], [1.5]]).all()

    def test_values_real_to_scaled_multiple_bounds(self):
        bounds = [[2.0, 4.0], [2.0, 3.0]]
        values = [[2.0, 3.0]]
        scaled_values = values_real_to_scaled(values, bounds)
        assert np.isclose(scaled_values, [[0.0, 1.0]]).all()

        values = [[2.0, 3.0], [3.0, 2.0]]
        scaled_values = values_real_to_scaled(values, bounds)
        assert np.isclose(scaled_values, [[0.0, 1.0], [0.5, 0.0]]).all()

    def test_values_scaled_to_real_single(self):
        bounds = [2.0, 4.0]
        scaled_value = 0.0
        value = values_scaled_to_real(scaled_value, bounds)
        assert np.isclose(value, 2.0)

        scaled_value = 1.0
        value = values_scaled_to_real(scaled_value, bounds)
        assert np.isclose(value, 4.0)

        scaled_value = 0.5
        value = values_scaled_to_real(scaled_value, bounds)
        assert np.isclose(value, 3.0)

        scaled_value = [-0.5]
        value = values_scaled_to_real(scaled_value, bounds)
        assert np.isclose(value, 1.0)

        bounds = [-5.0, -4.0]
        scaled_value = 0.5
        value = values_scaled_to_real(scaled_value, bounds)
        assert np.isclose(value, -4.5)

    def test_values_scaled_to_real_multiple_values(self):
        bounds = [2.0, 4.0]
        scaled_values = [0.0, 1.0]
        values = values_scaled_to_real(scaled_values, bounds)
        assert np.isclose(values, [[2.0], [4.0]]).all()

        scaled_values = [0.0, 1.0, 1.5]
        values = values_scaled_to_real(scaled_values, bounds)
        assert np.isclose(values, [[2.0], [4.0], [5.0]]).all()

    def test_values_scaled_to_real_multiple_bounds(self):
        bounds = [[2.0, 4.0], [2.0, 3.0]]
        scaled_values = [[0.0, 1.0]]
        values = values_scaled_to_real(scaled_values, bounds)
        assert np.isclose(values, [[2.0, 3.0]]).all()

        scaled_values = [[0.0, 1.0], [0.5, 0.0]]
        values = values_scaled_to_real(scaled_values, bounds)
        assert np.isclose(values, [[2.0, 3.0], [3.0, 2.0]]).all()

    def test_variances_scaled_to_real_single(self):
        bounds = [2.0, 4.0]
        scaled_variance = 0.0
        variance = variances_scaled_to_real(scaled_variance, bounds)
        assert np.isclose(variance, 0.0)

        scaled_variance = 1.0
        variance = variances_scaled_to_real(scaled_variance, bounds)
        assert np.isclose(variance, 4.0)

        scaled_variance = 0.5
        variance = variances_scaled_to_real(scaled_variance, bounds)
        assert np.isclose(variance, 2.0)

        scaled_variance = [-0.5]
        with pytest.raises(ValueError, match=r"cannot be less"):
            variance = variances_scaled_to_real(scaled_variance, bounds)

        bounds = [-5.0, -4.0]
        scaled_variance = 0.5
        variance = variances_scaled_to_real(scaled_variance, bounds)
        assert np.isclose(variance, 0.5)

    def test_variances_scaled_to_real_multiple_variances(self):
        bounds = [2.0, 4.0]
        scaled_variances = [0.0, 1.0]
        variances = variances_scaled_to_real(scaled_variances, bounds)
        assert np.isclose(variances, [[0.0], [4.0]]).all()

        scaled_variances = [0.0, 1.0, 1.5]
        variances = variances_scaled_to_real(scaled_variances, bounds)
        assert np.isclose(variances, [[0.0], [4.0], [6.0]]).all()

    def test_variances_scaled_to_real_multiple_bounds(self):
        bounds = [[2.0, 4.0], [2.0, 3.0]]
        scaled_variances = [[1.0, 1.0]]
        variances = variances_scaled_to_real(scaled_variances, bounds)
        assert np.isclose(variances, [[4.0, 1.0]]).all()

        scaled_variances = [[0.0, 1.0], [0.5, 0.0]]
        variances = variances_scaled_to_real(scaled_variances, bounds)
        assert np.isclose(variances, [[0.0, 1.0], [2.0, 0.0]]).all()

    def test_clean_bounds_values(self):
        bounds = [[2.0, 4.0], [3.0, 2.0]]
        values = [[2.0, 3.0]]
        with pytest.raises(ValueError, match=r"Lower bound must"):
            values, bounds = _clean_bounds_values(values, bounds)
        bounds = [3.0, 2.0]
        values = [[2.0, 3.0]]
        with pytest.raises(ValueError, match=r"Lower bound must"):
            values, bounds = _clean_bounds_values(values, bounds)

        bounds = [[2.0, 4.0], [2.0, 3.0]]
        values = [[2.0, 3.0, 2.5]]
        with pytest.raises(ValueError, match=r"Shapes of"):
            values, bounds = _clean_bounds_values(values, bounds)

        values = [[2.0, 3.0], [2.5]]
        with pytest.raises(ValueError, match=r"Shapes of"):
            values, bounds = _clean_bounds_values(values, bounds)

        bounds = [2.0, 4.0]
        values = 3.0
        values, bounds = _clean_bounds_values(values, bounds)
        assert values.shape == (1, 1)

        values = [3.0, 4.0, 5.0]
        values, bounds = _clean_bounds_values(values, bounds)
        assert values.shape == (3, 1)
