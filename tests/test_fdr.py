
from signifikante.algo import signifikante_fdr
import numpy as np
import pandas as pd

def test_signifikante_medoid_tfs_unclustered():
    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    np.random.seed(42)
    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )
    grn_fdr_df = signifikante_fdr(
        expression_data=expr_mat,
        tf_names=tfs,
        num_target_clusters=3,
        cluster_representative_mode='medoid',
        num_permutations=10,
    )
    print(grn_fdr_df)

def test_signifikante_medoid_tfs_clustered():
    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    np.random.seed(42)
    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )
    grn_fdr_df = signifikante_fdr(
        expression_data=expr_mat,
        tf_names=tfs,
        num_target_clusters=3,
        num_tf_clusters=4,
        cluster_representative_mode='medoid',
        num_permutations=10,
    )
    print(grn_fdr_df)

def test_signifikante_random_tfs_unclustered():
    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    np.random.seed(42)
    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )
    grn_fdr_df = signifikante_fdr(
        expression_data=expr_mat,
        tf_names=tfs,
        num_target_clusters=3,
        cluster_representative_mode='random',
        num_permutations=10,
    )
    print(grn_fdr_df)

def test_signifikante_random_tfs_clustered():
    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    np.random.seed(42)
    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )
    grn_fdr_df = signifikante_fdr(
        expression_data=expr_mat,
        tf_names=tfs,
        num_target_clusters=3,
        num_tf_clusters=4,
        cluster_representative_mode='random',
        num_permutations=10,
    )
    print(grn_fdr_df)

def test_signifikante_full():
    n_tfs = 10
    n_non_tfs = 10
    n_cells = 10
    tfs = [f'TF{i}' for i in range(n_tfs)]
    non_tfs = [f'Gene{i}' for i in range(n_non_tfs)]
    np.random.seed(42)
    expr_mat = pd.DataFrame(
        # np.random.normal(0, 1, (n_cells, n_tfs + n_genes)),
        np.random.poisson(lam=np.random.gamma(shape=2, scale=1, size=(n_cells, n_tfs + n_non_tfs))),
        columns=tfs + non_tfs,
    )
    grn_fdr_df = signifikante_fdr(
        expression_data=expr_mat,
        tf_names=tfs,
        num_target_clusters=3,
        num_tf_clusters=4,
        cluster_representative_mode='all_genes',
        num_permutations=10,
    )
    print(grn_fdr_df)
