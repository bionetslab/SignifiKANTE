
from unittest import TestCase
from signifikante.algo import signifikante_fdr, grnboost2
from signifikante.utils import load_tf_names
from signifikante.fdr_utils import compute_wasserstein_distance_matrix
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd

zeisel_small_path = 'tests/sparse/zeisel_small.tsv'
zeisel_tfs_path = 'tests/sparse/zeisel_tfs.txt'
df = pd.read_csv(zeisel_small_path, sep='\t')
tfs = load_tf_names(zeisel_tfs_path)
ref_grn = grnboost2(df,
                    tf_names=tfs,
                    seed=42)

class TestFDR(TestCase):
    
    def test_fdr_output(self):
        fdr_grn = signifikante_fdr(expression_data=df,
                                   tf_names=tfs,
                                   cluster_representative_mode="all_genes",
                                   apply_bh_correction=True,
                                   num_permutations=50,
                                   seed=42
                                   )
        self.assertTrue(isinstance(fdr_grn, pd.DataFrame))
        self.assertGreater(len(fdr_grn), 100)
        self.assertEqual(len(fdr_grn.columns), 5)
        self.assertTrue(set(fdr_grn['TF']).issubset(set(tfs)))
        self.assertTrue(min(fdr_grn['pvalue'])>=(1.0/51))
        self.assertTrue(max(fdr_grn['pvalue'])<=1.0)
        self.assertTrue(min(fdr_grn['pvalue_bh'])>=(1.0/51))
        self.assertTrue(max(fdr_grn['pvalue_bh'])<=1.0)
        self.assertTrue(all(a >= b for a, b in zip(fdr_grn['pvalue_bh'], fdr_grn['pvalue'])))
        
    def test_input_grn(self):
        fdr_grn = signifikante_fdr(expression_data=df,
                                   tf_names=tfs,
                                   input_grn=ref_grn,
                                   cluster_representative_mode="random",
                                   num_target_clusters=5,
                                   num_permutations=50,
                                   seed=42)
        self.assertEqual(len(fdr_grn), len(ref_grn))
        self.assertListEqual(list(fdr_grn['TF']), list(ref_grn['TF']))
        self.assertListEqual(list(fdr_grn['target']), list(ref_grn['target']))
        self.assertListEqual(list(fdr_grn['importance']), list(ref_grn['importance']))
        
    def test_tf_list(self):
        fdr_grn = signifikante_fdr(expression_data=df,
                                   cluster_representative_mode="medoid",
                                   num_target_clusters=4,
                                   num_permutations=50,
                                   seed=42)
        self.assertTrue(set(fdr_grn['TF']).issubset(set(df.columns)))
        
class TestWasserstein(TestCase):
    
    def test_wasserstein_against_scipy(self):
        np.random.seed(42)
        a = np.random.normal(0, 1, (10000, ))
        b = np.random.normal(1, 1, (10000,))
        c = np.random.normal(2, 1, (10000,))

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
