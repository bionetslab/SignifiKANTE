from unittest import TestCase
from signifikante.fdr_utils import compute_wasserstein_distance_matrix
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd

class TestWasserstein(TestCase):
    
    def test_wasserstein_against_scipy(self):
        np.random.seed(42)
        a = np.random.normal(0, 1, (100, ))
        b = np.random.normal(1, 1, (100,))
        c = np.random.normal(2, 1, (100,))

        sim_matrix = pd.DataFrame(np.vstack((a, b, c)).T.copy())
        wasserstein_signifikante = compute_wasserstein_distance_matrix(sim_matrix)

        wasserstein_scipy_ab = wasserstein_distance(a, b)
        wasserstein_scipy_ac = wasserstein_distance(a, c)
        wasserstein_scipy_bc = wasserstein_distance(b, c)
        
        self.assertAlmostEqual(wasserstein_signifikante.iloc[0,0], 0.0)
        self.assertAlmostEqual(wasserstein_signifikante.iloc[1,1], 0.0)
        self.assertAlmostEqual(wasserstein_signifikante.iloc[2,2], 0.0)
        self.assertAlmostEqual(wasserstein_signifikante.iloc[0,1], wasserstein_scipy_ab)
        self.assertAlmostEqual(wasserstein_signifikante.iloc[0,2], wasserstein_scipy_ac)
        self.assertAlmostEqual(wasserstein_signifikante.iloc[1,2], wasserstein_scipy_bc)