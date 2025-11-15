
from unittest import TestCase
from signifikante.algo import signifikante_fdr, grnboost2
from signifikante.core import SGBM_KWARGS
from signifikante.fdr import count_computation_medoid_representative, count_computation_sampled_representative
from signifikante.fdr_utils import compute_wasserstein_distance_matrix
from distributed import LocalCluster, Client
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd

expression_data = np.random.randn(25, 30)
df = pd.DataFrame(expression_data, columns=[f"Gene{i}" for i in range(30)])
# Simulate three artificial TFs.
tfs = [f"Gene{i}" for i in range(6)]
ref_grn = grnboost2(df,
                    tf_names=tfs,
                    seed=42)

lc = LocalCluster()
client = Client()

class TestFDR(TestCase):
    
    def test_fdr_output(self):
        fdr_grn = signifikante_fdr(expression_data=df,
                                   tf_names=tfs,
                                   cluster_representative_mode="all_genes",
                                   apply_bh_correction=True,
                                   num_permutations=50,
                                   seed=42,
                                   client_or_address=client
                                   )
        self.assertTrue(isinstance(fdr_grn, pd.DataFrame))
        self.assertGreater(len(fdr_grn), 10)
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
                                   normalize_gene_expression=True,
                                   seed=42,
                                   client_or_address=client)
        self.assertEqual(len(fdr_grn), len(ref_grn))
        self.assertListEqual(list(fdr_grn['TF']), list(ref_grn['TF']))
        self.assertListEqual(list(fdr_grn['target']), list(ref_grn['target']))
        self.assertListEqual(list(fdr_grn['importance']), list(ref_grn['importance']))
        
    def test_tf_list(self):
        fdr_grn = signifikante_fdr(expression_data=df,
                                   cluster_representative_mode="medoid",
                                   num_target_clusters=4,
                                   target_cluster_mode="kmeans",
                                   num_permutations=50,
                                   seed=42,
                                   client_or_address=client)
        self.assertTrue(set(fdr_grn['TF']).issubset(set(df.columns)))
    
    def test_wrong_input(self):
        with self.assertRaises(ValueError):
            signifikante_fdr(expression_data=df,
                                   cluster_representative_mode="experimental",
                                   num_target_clusters=4,
                                   num_permutations=50,
                                   seed=42,
                                   client_or_address=client)
        with self.assertRaises(ValueError):
            signifikante_fdr(expression_data=df,
                                   cluster_representative_mode="medoid",
                                   num_target_clusters=4,
                                   num_permutations=50,
                                   target_subset=["target1", 'target2'],
                                   seed=42,
                                   client_or_address=client)
    def test_lasso_fdr(self):
        fdr_grn = signifikante_fdr(expression_data=df,
                                   tf_names=tfs,
                                   input_grn=ref_grn,
                                   cluster_representative_mode="medoid",
                                   num_target_clusters=5,
                                   output_dir="test_output/",
                                   num_permutations=10,
                                   normalize_gene_expression=True,
                                   scale_for_tf_sampling=True,
                                   inference_mode="lasso",
                                   seed=42,
                                   client_or_address=client)
        self.assertGreater(len(fdr_grn), 10)
        self.assertEqual(len(fdr_grn.columns), 6)
        
    def test_genie3_fdr(self):
        fdr_grn = signifikante_fdr(expression_data=df,
                                   tf_names=tfs,
                                   input_grn=ref_grn,
                                   cluster_representative_mode="medoid",
                                   num_target_clusters=5,
                                   num_tf_clusters=3,
                                   num_permutations=3,
                                   normalize_gene_expression=True,
                                   inference_mode="genie3",
                                   seed=42,
                                   client_or_address=client)
        self.assertGreater(len(fdr_grn),10)
        self.assertEqual(len(fdr_grn.columns),4)
        
    def test_tf_with_kmedoids(self):
        expression_data = np.random.randn(70, 100)
        df2 = pd.DataFrame(expression_data, columns=[f"Gene{i}" for i in range(100)])
        # Simulate three artificial TFs.
        tfs2 = [f"Gene{i}" for i in range(70)]
        fdr_grn = signifikante_fdr(expression_data=df2,
                                   tf_names=tfs2,
                                   tf_cluster_mode="kmeans",
                                   cluster_representative_mode="medoid",
                                   num_target_clusters=5,
                                   num_tf_clusters=3,
                                   num_permutations=3,
                                   normalize_gene_expression=True,
                                   seed=42,
                                   client_or_address=client)
        self.assertGreater(len(fdr_grn),10)
        self.assertEqual(len(fdr_grn.columns),4)
        
    def test_xgboost_fdr(self):
        fdr_grn = signifikante_fdr(expression_data=df,
                                   tf_names=tfs,
                                   input_grn=ref_grn,
                                   cluster_representative_mode="medoid",
                                   num_tf_clusters=-1,
                                   num_target_clusters=-1,
                                   num_permutations=3,
                                   normalize_gene_expression=True,
                                   inference_mode="xgboost",
                                   verbose=True,
                                   seed=42,
                                   client_or_address=client)
        self.assertGreater(len(fdr_grn),10)
        self.assertEqual(len(fdr_grn.columns),4)
        
    def test_genie3_fdr(self):
        fdr_grn = signifikante_fdr(expression_data=df,
                                   tf_names=tfs,
                                   input_grn=ref_grn,
                                   cluster_representative_mode="all_genes",
                                   num_permutations=3,
                                   normalize_gene_expression=True,
                                   inference_mode="extra_trees",
                                   seed=42,
                                   client_or_address=client)
        self.assertGreater(len(fdr_grn),10)
        self.assertEqual(len(fdr_grn.columns),4)
        
class TestWasserstein(TestCase):
    
    def test_wasserstein_against_scipy(self):
        np.random.seed(42)
        a = np.random.normal(0, 1, (1000, ))
        b = np.random.normal(1, 1, (1000,))
        c = np.random.normal(2, 1, (1000,))

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
        
class TestParallelizedFunctions(TestCase):
    
    def test_medoid_sampling(self):
        # --- Constants (Implied by function signature) ---
        EARLY_STOP_WINDOW_LENGTH = 10
        DEMON_SEED = 42

        # --- 1. Regression Parameters ---
        regressor_type = 'GBM'
        regressor_kwargs = SGBM_KWARGS

        # --- 2. TF Expression Data ---
        # A matrix with 3 samples (rows) and 3 potential TFs (columns).
        # Note: 'TF3' is intentionally the 'target_gene_name' to test the `clean` step.
        tf_matrix = np.array([
            [1.0, 5.0, 10.1],
            [2.4, 6.1, 11.0],
            [3.0, 7.6, 12.0],
            [2.5, 4.5, 13.2]
        ], dtype=np.float64)
        tf_matrix_gene_names = ['TF1', 'TF2', 'TF3']
        are_tfs_clustered = True  # Set to False to test the cleaning step

        # --- 3. Target Gene Data ---
        target_gene_name = 'target1' # This TF is also the target gene
        target_gene_expression = np.array([9.7, 11.2, 15.2, 2.4], dtype=np.float64)

        # --- 4. Gene Regulatory Network (GRN) Links to be Tested ---
        # This dictionary contains the actual importance values (e.g., from the non-shuffled run).
        # The function will initialize 'count' and 'shuffled_occurences' keys on these.
        partial_input_grn = {
            ('TF1', 'target1'): {'importance': 0.75},
            ('TF2', 'target1'): {'importance': 0.25},
            ('TF3', 'target1'): {'importance': 0.35}
        }

        # --- 5. Clustering and Scaling Parameters (Set to defaults/empty for non-clustered test) ---
        per_target_importance_sums = {'target1' : 0.8, 'target2' : 0.9}
        tf_to_cluster = {'TF1' : 0, 'TF2': 1, 'TF3': 1}
        cluster_to_tf = {0 : ['TF1'], 1: ['TF2', 'TF3']}

        # --- 6. Permutations and Options ---
        n_permutations = 3 # Use a small number for fast testing
        early_stop_window_length = EARLY_STOP_WINDOW_LENGTH
        seed = DEMON_SEED
        scale_for_tf_sampling = False
        
        count_computation_medoid_representative(
            regressor_type=regressor_type,
            regressor_kwargs=regressor_kwargs,
            tf_matrix=tf_matrix,
            are_tfs_clustered=are_tfs_clustered,
            tf_matrix_gene_names=tf_matrix_gene_names,
            target_gene_name=target_gene_name,
            target_gene_expression=target_gene_expression,
            partial_input_grn=partial_input_grn,
            per_target_importance_sums=per_target_importance_sums,
            tf_to_cluster=tf_to_cluster,
            cluster_to_tf=cluster_to_tf,
            n_permutations=n_permutations,
            early_stop_window_length=early_stop_window_length,
            seed=seed,
            output_dir=None,
            scale_for_tf_sampling=scale_for_tf_sampling
        )
        
    def test_random_sampling(self):
        # --- Constants (Implied by function signature) ---
        EARLY_STOP_WINDOW_LENGTH = 10
        DEMON_SEED = 42

        # --- 1. Cluster and Regression Parameters ---
        cluster_id = 0 # ID of the target cluster being processed
        regressor_type = 'GBM'
        regressor_kwargs = SGBM_KWARGS

        # --- 2. TF and Target Expression Data ---
        tf_matrix = np.array([
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [12.0, 22.0, 32.0],
            [13.0, 23.0, 33.0],
            [14.0, 24.0, 34.0]
        ], dtype=np.float64)

        # Gene names corresponding to tf_matrix columns
        tf_matrix_gene_names = ['TF1', 'TF2', 'TF3']

        # --- 3. Target Gene Cluster Information ---
        # Target genes within this cluster, and their indices in the full tf_matrix
        target_gene_names = ['target1', 'target2']
        target_gene_idxs = [0, 1] 
        target_gene_expressions = np.array([
            [10.0, 25.0, 30.0, 40.0],
            [15.0, 21.0, 31.0, 41.0],
            [10.0, 22.0, 34.0, 40.0],
            [10.0, 21.0, 39.0, 48.0],
            [14.0, 24.0, 34.0, 40.0]
        ], dtype=np.float64)

        # --- 4. TF Cluster Information (Required for sampling and scaling) ---
        cluster_to_tfs = {0 : ['TF1'], 1 : ['TF2', 'TF3']}

        # Mapping TFs back to their cluster IDs
        tf_to_cluster = {
            'TF1': 0, 'TF2': 1, 'TF3': 1
        }

        # --- 5. GRN Links to be Tested ---
        partial_input_grn = {
            ('TF1', 'target1'): {'importance': 0.9},
            ('TF1', 'target2'): {'importance': 0.1},
            ('TF2', 'target1'): {'importance': 0.5},
            ('TF3', 'target2'): {'importance': 0.5}
        }

        # --- 6. Importance Sums (Used for scaling in the clustered path) ---
        # Sum of actual importance values for each target in the GRN.
        per_target_importance_sums = {
            'target1': 0.7,
            'target2': 0.4
        }

        # --- 7. Permutations and Options ---
        n_permutations = 3
        early_stop_window_length = EARLY_STOP_WINDOW_LENGTH
        seed = DEMON_SEED
        output_dir = None
        scale_for_tf_sampling = True
        
        count_computation_sampled_representative(
        cluster_id=cluster_id,
        regressor_type=regressor_type,
        regressor_kwargs=regressor_kwargs,
        tf_matrix=tf_matrix,
        tf_matrix_gene_names=tf_matrix_gene_names,
        cluster_to_tfs=cluster_to_tfs,
        target_gene_names=target_gene_names,
        target_gene_idxs=target_gene_idxs,
        target_gene_expressions=target_gene_expressions,
        partial_input_grn=partial_input_grn,
        tf_to_cluster=tf_to_cluster,
        per_target_importance_sums=per_target_importance_sums,
        n_permutations=n_permutations,
        early_stop_window_length=early_stop_window_length,
        seed=seed,
        output_dir=output_dir,
        scale_for_tf_sampling=scale_for_tf_sampling
    )
        
        

