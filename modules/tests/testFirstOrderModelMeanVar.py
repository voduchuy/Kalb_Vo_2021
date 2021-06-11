import sys

sys.path.append("../..")
import unittest
import numpy as np
from modules.moments.first_order_networks import find_mean_var


class TestPoissonCase(unittest.TestCase):
    def setUp(self):
        self.rate = 1.0
        self.stoichMatrix = np.array([[1]])
        self.initMeans = np.array([[0]])
        self.initCovariances = np.array([[0]])
        self.timeEval = np.linspace(0, 10, 100)
        self.W = np.array([[0]])
        self.b = np.array([[self.rate]])

        def timeCoeffVec(t):
            return np.array([[1.0]])

        self.timeCoeffVec = timeCoeffVec

    def test_findMeansVars(self):
        means, vars = find_mean_var(
            self.stoichMatrix,
            self.W,
            self.b,
            self.timeCoeffVec,
            self.timeEval,
            self.initMeans,
            self.initCovariances,
        )
        true_means = self.timeEval * self.rate
        true_vars = true_means.copy()
        self.assertLessEqual(
            np.max(means - true_means) + np.max(vars - true_vars), 1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
