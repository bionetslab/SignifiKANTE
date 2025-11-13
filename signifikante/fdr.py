
import copy
from signifikante.core import (
    EARLY_STOP_WINDOW_LENGTH, SGBM_KWARGS, DEMON_SEED, to_tf_matrix, target_gene_indices, clean, fit_model, 
    to_links_df, RF_KWARGS, ET_KWARGS, XGB_KWARGS, LASSO_KWARGS
)
from signifikante.fdr_utils import (
    compute_correlation_distance_matrix, compute_wasserstein_distance_matrix, cluster_genes_to_dict, 
    merge_gene_clusterings, compute_medoids, partition_input_grn, invert_tf_to_cluster_dict, count_helper, 
    subset_tf_matrix, _prepare_client, _prepare_input
)
import numpy as np
import pandas as pd
from dask import delayed, compute
from dask.dataframe import from_delayed
from dask.dataframe.utils import make_meta
import pickle
import os
from statsmodels.stats.multitest import multipletests

FDR_GRN_SCHEMA = make_meta({'TF': str, 'target': str, 'importance': float, 'count' : float, 'shuffled_occurences' : float})

def perform_fdr(
        expression_data : pd.DataFrame,
        input_grn : dict,
        num_non_tf_clusters : int,
        num_tf_clusters : int,
        cluster_representative_mode : str,
        target_cluster_mode : str,
        tf_cluster_mode : str,
        tf_names,
        target_subset,
        client_or_address,
        early_stop_window_length,
        seed,
        verbose,
        num_permutations,
        output_dir,
        scale_for_tf_sampling,
        regressor_type,
        regressor_args,
        apply_bh_correction
):
    # Extract TF name and target name lists from expression matrix object.
    _, gene_names, tf_names = _prepare_input(expression_data, None, tf_names)
    non_tf_names = [gene for gene in gene_names if not gene in tf_names]

    # Check if TF clustering is desired.
    if num_tf_clusters == -1:
        num_tf_clusters = len(tf_names)
        are_tfs_clustered = False
    else:
        are_tfs_clustered = True

    # Check if target clustering is desired.
    if num_non_tf_clusters == -1:
        num_non_tf_clusters = len(non_tf_names)

    # No clustering necessary, just create 'dummy' clustering with singleton clusters.
    tf_representatives = []
    target_representatives = []
    tf_to_clust = None
    target_to_clust = None
    if cluster_representative_mode == 'all_genes':
        tf_representatives = tf_names
        target_representatives = gene_names
        are_tfs_clustered = False
    else:
        # In case of Wasserstein clustering on targets, compute whole distance matrix on all targets.
        if target_cluster_mode == 'wasserstein':
            dist_matrix_all = compute_wasserstein_distance_matrix(expression_data, num_threads=-1)

        if tf_cluster_mode == 'correlation':
            tf_bool = [True if gene in tf_names else False for gene in expression_data.columns]
            exp_matrix_tfs = expression_data.loc[:, tf_bool]
            corr_distances_tfs = compute_correlation_distance_matrix(exp_matrix_tfs)

        # Cluster targets and TFs separately given respective mode.
        if target_cluster_mode == 'kmeans':
            target_to_clust, target_medoids = cluster_genes_to_dict(expression_data, num_clusters=num_non_tf_clusters,
                                                                    mode='kmeans')
        elif target_cluster_mode == 'wasserstein':
            target_to_clust, target_medoids = cluster_genes_to_dict(dist_matrix_all, num_clusters=num_non_tf_clusters,
                                                                    mode='distance')
        else:
            print(f'Unknown target cluster mode: {target_cluster_mode}')

        if tf_cluster_mode == 'correlation':
            tf_to_clust, tf_medoids = cluster_genes_to_dict(corr_distances_tfs, num_clusters=num_tf_clusters, mode='distance')
        elif tf_cluster_mode == 'kmeans':
            tf_to_clust, tf_medoids = cluster_genes_to_dict(exp_matrix_tfs, num_clusters=num_tf_clusters,
                                                            mode='distance')
        else:
            print(f'Unknown TF cluster mode: {tf_cluster_mode}')

        if not output_dir is None:
            with open(os.path.join(output_dir, 'tf_clustering.pkl'), 'wb') as f:
                pickle.dump(tf_to_clust, f)
            with open(os.path.join(output_dir, 'all_genes_clustering.pkl'), 'wb') as f:
                pickle.dump(target_to_clust, f)

        if cluster_representative_mode == 'medoid':
            tf_representatives = tf_medoids
            target_representatives = target_medoids
        else: # In random mode, representatives are drawn from whole set of possible TFs and targets.
            tf_representatives = tf_names
            target_representatives = gene_names

    if not output_dir is None and cluster_representative_mode=='medoid':
        with open(os.path.join(output_dir, 'tf_medoids.pkl'), 'wb') as f:
            pickle.dump(tf_representatives, f)
        with open(os.path.join(output_dir, 'target_medoids.pkl'), 'wb') as f:
            pickle.dump(target_representatives, f)

    fdr_controlled_df = diy_fdr(expression_data=expression_data,
                   regressor_type=regressor_type,
                   regressor_kwargs=regressor_args,
                   gene_names=gene_names,
                   are_tfs_clustered=are_tfs_clustered,
                   tf_representatives=tf_representatives,
                   target_representatives=target_representatives,
                   target_subset=target_subset,
                   tf_to_cluster=tf_to_clust,
                   target_to_cluster=target_to_clust,
                   input_grn=input_grn,
                   client_or_address=client_or_address,
                   early_stop_window_length=early_stop_window_length,
                   seed=seed,
                   verbose=verbose,
                   n_permutations=num_permutations,
                   output_dir=output_dir,
                   scale_for_tf_sampling=scale_for_tf_sampling
                   )
    
    # Transform counts into P-values and remove count column.
    fdr_controlled_df['pvalue'] = (fdr_controlled_df['count']+1)/(num_permutations+1)
    if not scale_for_tf_sampling:
        fdr_controlled_df.drop(columns=['count'], inplace=True)
        fdr_controlled_df.drop(columns=['shuffled_occurences'], inplace=True)
    if apply_bh_correction:
        fdr_controlled_df['pvalue_bh'] = multipletests(fdr_controlled_df['pvalue'], method='fdr_bh')[1]
    
    return fdr_controlled_df


def diy_fdr(expression_data,
            regressor_type,
            regressor_kwargs,
            are_tfs_clustered,
            tf_representatives,
            target_representatives,
            target_subset,
            tf_to_cluster,
            target_to_cluster,
            input_grn,
            scale_for_tf_sampling,
            gene_names=None,
            client_or_address='local',
            early_stop_window_length=None,
            seed=None,
            verbose=False,
            n_permutations=1000,
            output_dir=None,
            ):
    """
    :param are_tfs_clustered: True if TFs have also been clustered for FDR control.
    :param tf_representatives: Either list of pre-chosen TF representatives or simply all TFs.
    :param non_tf_representatives: Either list of pre-chosen non-TF representatives or all non-TFs.
    :param gene_to_cluster: Keys are gene names and values are cluster IDs as integers.
    :param input_grn: Dict storing input GRN for FDR control with keys as edge tuples, and as values dicts with
        {'importance' : <float>} structure.
    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param regressor_type: string. One of: 'RF', 'GBM', 'ET', 'XGB'. Case insensitive.
    :param regressor_kwargs: a dictionary of key-value pairs that configures the regressor.
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param early_stop_window_length: early stopping window length.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param seed: optional random seed for the regressors. Default 666. Use None for random seed.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """
    if verbose:
        print('preparing dask client')

    client, shutdown_callback = _prepare_client(client_or_address)

    try:
        if verbose:
            print('parsing input')

        # TF names do not matter in FDR mode, hence can be set to dummy list.
        tf_names = None
        expression_matrix, gene_names, _ = _prepare_input(expression_data, gene_names, tf_names)

        if verbose:
            print('creating dask graph')

        if input_grn is None:
            raise ValueError(f'Input GRN is None, but needs to be passed in FDR mode.')
        if tf_representatives is None or target_representatives is None:
            raise ValueError(f'TF or non-TF representatives are None, but need to passed in FDR mode.')
        if tf_to_cluster is None and target_to_cluster is None:
            if verbose:
                print("Genes have not been clustered, running full FDR mode.")

        # Compute per-target importance sums for scaling in case TFs have been clustered.
        per_target_importance_sums = dict()
        if are_tfs_clustered:
            for (_, target), import_dict in input_grn.items():
                if not target in per_target_importance_sums:
                    per_target_importance_sums[target] = 0.0
                per_target_importance_sums[target] += import_dict['importance']

        graph = create_graph_fdr(expression_matrix,
                                 gene_names=gene_names,
                                 are_tfs_clustered=are_tfs_clustered,
                                 tf_representatives=tf_representatives,
                                 target_representatives=target_representatives,
                                 target_subset=target_subset,
                                 tf_to_cluster=tf_to_cluster,
                                 target_to_cluster=target_to_cluster,
                                 input_grn=input_grn,
                                 per_target_importance_sums=per_target_importance_sums,
                                 regressor_type=regressor_type,
                                 regressor_kwargs=regressor_kwargs,
                                 client=client,
                                 early_stop_window_length=early_stop_window_length,
                                 seed=seed,
                                 n_permutations=n_permutations,
                                 output_dir=output_dir,
                                 scale_for_tf_sampling=scale_for_tf_sampling)

        if verbose:
            print('{} partitions'.format(graph.npartitions))
            print('computing dask graph')

        return client \
            .compute(graph, sync=True) \
            .sort_values(by='importance', ascending=False)

    finally:
        shutdown_callback(verbose)

        if verbose:
            print('finished')


def create_graph_fdr(expression_matrix: np.ndarray,
                     gene_names: list[str],
                     are_tfs_clustered,
                     tf_representatives: list[str],
                     target_representatives: list[str],
                     target_subset : list[str],
                     tf_to_cluster: dict[str, int],
                     target_to_cluster: dict[str, int],
                     input_grn: dict,
                     per_target_importance_sums: dict,
                     regressor_type,
                     regressor_kwargs,
                     client,
                     scale_for_tf_sampling,
                     early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
                     repartition_multiplier=1,
                     seed=DEMON_SEED,
                     n_permutations=1000,
                     output_dir=None
                     ):
    """
    Main API function for FDR control. Create a Dask computation graph.

    Note: fixing the GC problems was fixed by 2 changes: [1] and [2] !!!

    :param expression_matrix: numpy matrix. Rows are observations and columns are genes.
    :param gene_names : list[str]. List of gene names, in the order as they appear in the expression matrix columns.
    :param fdr_mode: str. One of 'medoid', 'random' to indicate representative drawing mode.
    :param are_tfs_clustered: bool. True if TFs have also been clustered, False otherwise.
    :param tf_representatives: list[str]. List of TFs, either only medoids, or all TFs in 'random' mode.
    :param non_tf_representatives: list[str]. List of non-TFs, either only medoids, or all non-TFs in 'random' mode.
    :param gene_to_cluster: dict[str, int]. Keys are gene names, values are cluster IDs they belong to. If set to
        None, run full 'groundtruth' FDR control on all genes.
    :param input_grn: dict. Input GRN to be FDR-controlled in dict format with (tf,target) as keys.
    :param regressor_type: regressor type. Case insensitive.
    :param regressor_kwargs: dict of key-value pairs that configures the regressor.
    :param client: a dask.distributed client instance.
                   * Used to scatter-broadcast the tf matrix to the workers instead of simply wrapping in a delayed().
    :param early_stop_window_length: window length of the early stopping monitor.
    :param repartition_multiplier: multiplier
    :param n_permutations: int. Number of random permutations to run for FDR control.
    :param seed: (optional) random seed for the regressors. Default 666.
    :return: if include_meta is False, returns a Dask graph that computes the links DataFrame.
             If include_meta is True, returns a tuple: the links DataFrame and the meta DataFrame.
    """
    assert client, "Client not given, but is required in create_graph_fdr!"
    # Extract FDR mode information from TF and non-TF representative lists.
    fdr_mode = None
    if (not tf_to_cluster is None and
            not target_to_cluster is None and
            len(tf_representatives) == len(tf_to_cluster.keys()) and
            len(target_representatives) == len(gene_names)
    ):
        fdr_mode = 'random'
    elif not tf_to_cluster is None and not target_to_cluster is None:
        fdr_mode = 'medoid'
    else:  # Full FDR mode coincides with medoid mode, with all genes assigned to dummy singleton clusters.
        fdr_mode = 'medoid'
        tf_to_cluster = {gene: cluster_id for cluster_id, gene in
                           enumerate(tf_representatives)}
        target_to_cluster = {gene: cluster_id for cluster_id, gene in
                           enumerate(target_representatives)}

    # Check if gene_to_cluster is complete, i.e. if for every gene in expression matrix, a corresponding cluster has
    # been precomputed.
    all_target_genes = {gene for gene, _ in target_to_cluster.items()}
    assert expression_matrix.shape[1] == len(all_target_genes), "Size of expression matrix does not match gene names."
    assert len(gene_names) == len(all_target_genes), "Number of clustered genes and genes in expression matrix do not match."

    # Subset expression matrix to TF representatives ('medoid' mode). Leave as is, if TFs have not been clustered or
    # FDR mode is 'random'.
    tf_matrix, tf_matrix_gene_names = to_tf_matrix(expression_matrix, gene_names, tf_representatives)

    # Partition input GRN into dict storing target-cluster IDs as keys and edge dicts (as in input GRN) as values.
    # Second data structure stores target genes per cluster.
    grn_subsets_per_target, genes_per_target_cluster = partition_input_grn(input_grn, target_to_cluster)

    future_tf_matrix = client.scatter(tf_matrix, broadcast=True)
    # [1] wrap in a list of 1 -> unsure why but Matt. Rocklin does this often...
    [future_tf_matrix_gene_names] = client.scatter([tf_matrix_gene_names], broadcast=True)

    # Broadcast gene-to-cluster dictionaries among all workers.
    # future_gene_to_cluster = client.scatter(gene_to_cluster, broadcast=True) --> gives dict key error...
    [future_tf_to_cluster] = client.scatter([tf_to_cluster], broadcast=True)
    [future_target_to_cluster] = client.scatter([target_to_cluster], broadcast=True)

    # If TFs have been clustered in 'random' mode, then per permutation, one TF per cluster needs to be
    # drawn. Precompute the necessary cluster-TF relationships here such that keys are cluster IDs
    # and values are list of TFs.
    if are_tfs_clustered:
        cluster_to_tfs = invert_tf_to_cluster_dict(tf_representatives, tf_to_cluster)
        [future_cluster_to_tfs] = client.scatter([cluster_to_tfs], broadcast=True)
    else:
        future_cluster_to_tfs = None

    delayed_link_dfs = []  # collection of delayed link DataFrames

    # Use pre-computed medoid representatives for TFs and/or non-TFs.
    if fdr_mode == 'medoid':
        # Loop over all representative targets, i.e. non-TF medoids.
        all_targets = target_representatives
        if target_subset is not None:
            all_targets = target_subset
        for target_gene_index in target_gene_indices(gene_names, all_targets):
            target_gene_name = gene_names[target_gene_index]
            target_gene_expression = delayed(expression_matrix[:, target_gene_index])
            target_subset_grn = delayed(grn_subsets_per_target[target_to_cluster[target_gene_name]])

            # Pass subset of GRN which is represented by the medoids.
            delayed_link_df = delayed(count_computation_medoid_representative, pure=True)(
                regressor_type,
                regressor_kwargs,
                future_tf_matrix,
                are_tfs_clustered,
                future_tf_matrix_gene_names,
                target_gene_name,
                target_gene_expression,
                target_subset_grn,
                per_target_importance_sums,
                future_tf_to_cluster,
                future_cluster_to_tfs,
                n_permutations,
                early_stop_window_length,
                seed,
                output_dir,
                scale_for_tf_sampling
            )

            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)

    # Loop over all genes of cluster, i.e. simulate random drawing of genes from clusters.
    elif fdr_mode == 'random':
        # Loop over all target clusters (that includes TF clusters).
        for cluster_id, cluster_targets in genes_per_target_cluster.items():
            target_cluster_idxs = target_gene_indices(gene_names, cluster_targets)
            # Like this, order of cluster gene names should be consistent with order in cluster_idxs and hence with
            # gene column order in expression matrix.
            target_cluster_gene_names = [gene for index, gene in enumerate(gene_names) if index in target_cluster_idxs]
            cluster_expression = delayed(expression_matrix)

            # Dask does not allow iterating over delayed dictionary, so no delayed() at this point.
            target_subset_grn = grn_subsets_per_target[cluster_id]

            delayed_link_df = delayed(count_computation_sampled_representative, pure=True)(
                cluster_id,
                regressor_type,
                regressor_kwargs,
                future_tf_matrix,
                future_tf_matrix_gene_names,
                future_cluster_to_tfs,
                target_cluster_gene_names,
                target_cluster_idxs,
                cluster_expression,
                target_subset_grn,
                tf_to_cluster,
                per_target_importance_sums,
                n_permutations,
                early_stop_window_length,
                seed,
                output_dir,
                scale_for_tf_sampling
            )


            if delayed_link_dfs is not None:
                delayed_link_dfs.append(delayed_link_df)
    else:
        raise ValueError(f'Unknown FDR mode: {fdr_mode}.')

    # Gather the DataFrames into one distributed DataFrame.
    all_links_df = from_delayed(delayed_link_dfs, meta=FDR_GRN_SCHEMA)

    # [2] repartition to nr of workers -> important to avoid GC problems!
    # see: http://dask.pydata.org/en/latest/dataframe-performance.html#repartition-to-reduce-overhead
    n_parts = len(client.ncores()) * repartition_multiplier

    return all_links_df.repartition(npartitions=n_parts)

def count_computation_medoid_representative(
        regressor_type,
        regressor_kwargs,
        tf_matrix,
        are_tfs_clustered,
        tf_matrix_gene_names: list[str],
        target_gene_name: str,
        target_gene_expression: np.ndarray,
        partial_input_grn: dict,  # {(TF, target): {'importance': float}}
        per_target_importance_sums,
        tf_to_cluster: dict[str, int],
        cluster_to_tf : dict,
        n_permutations: int,
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=DEMON_SEED,
        output_dir=None,
        scale_for_tf_sampling=False
):

    partial_input_grn = copy.deepcopy(partial_input_grn)

    # Remove target from TF-list and TF-expression matrix if target itself is a TF
    if not are_tfs_clustered:
        (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)
    else:
        clean_tf_matrix, clean_tf_matrix_gene_names = tf_matrix, tf_matrix_gene_names

    # Special case in which only a single TF is passed and the target gene
    # here is the same as the TF (clean_tf_matrix is empty after cleaning):
    if clean_tf_matrix.size == 0:
        raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

    # Initialize counts
    for _, val in partial_input_grn.items():
        val.update({'count': 0.0})
        val.update({'shuffled_occurences': 0})

    # Iterate for num permutations
    for i in range(n_permutations):

        # Shuffle target gene expression vector
        np.random.seed(i)
        permuted_target_gene_expression = np.random.permutation(target_gene_expression)

        # Train the random forest regressor
        try:
            trained_regressor = fit_model(
                regressor_type,
                regressor_kwargs,
                clean_tf_matrix,
                permuted_target_gene_expression,
                early_stop_window_length,
                seed
            )
        except ValueError as e:
            raise ValueError(
                "Count_computation_medoid: regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e))
            )

        # Construct the shuffled GRN dataframe from the trained regressor
        shuffled_grn_df = to_links_df(
            regressor_type,
            regressor_kwargs,
            trained_regressor,
            clean_tf_matrix_gene_names,
            target_gene_name
        )

        if are_tfs_clustered:
            shuffled_target_sum = 0.0
            for tf, _, importance in zip(shuffled_grn_df['TF'], shuffled_grn_df['target'], shuffled_grn_df['importance']):
                tf_cluster_size = len(cluster_to_tf[tf_to_cluster[tf]])
                shuffled_target_sum += importance * tf_cluster_size
            scaling_factor = per_target_importance_sums[target_gene_name] / shuffled_target_sum
            shuffled_grn_df['importance'] = shuffled_grn_df['importance'] * scaling_factor

        # Update the count values of the partial input GRN
        count_helper(shuffled_grn_df, partial_input_grn, tf_to_cluster, scale_for_tf_sampling)

    # Change partial input GRN format from dict to df
    partial_input_grn_fdr_df = pd.DataFrame(
        [(TF, target, v['importance'], v['count'], v['shuffled_occurences']) for (TF, target), v in partial_input_grn.items()],
        columns=['TF', 'target', 'importance', 'count', 'shuffled_occurences']
    )

    if not output_dir is None:
        partial_input_grn_fdr_df.to_feather(os.path.join(output_dir, f'target_{target_gene_name}.feather'))

    return partial_input_grn_fdr_df


def count_computation_sampled_representative(
        cluster_id,
        regressor_type,
        regressor_kwargs,
        tf_matrix,
        tf_matrix_gene_names,
        cluster_to_tfs,
        target_gene_names,
        target_gene_idxs,
        target_gene_expressions,
        partial_input_grn: dict,
        tf_to_cluster: dict[str, int],
        per_target_importance_sums : dict,
        n_permutations : int,
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=DEMON_SEED,
        output_dir=None,
        scale_for_tf_sampling=False
):

    partial_input_grn = copy.deepcopy(partial_input_grn)

    are_tfs_clustered = (not cluster_to_tfs is None)
    # Initialize counts on input GRN edges.
    for _, val in partial_input_grn.items():
        val.update({'count': 0.0})
        val.update({'shuffled_occurences': 0})

    for perm in range(n_permutations):
        # Retrieve "random" target gene from cluster.
        perm_index = perm % len(target_gene_names)
        target_gene_name = target_gene_names[perm_index]
        target_gene_index = target_gene_idxs[perm_index]
        target_expression = target_gene_expressions[:, target_gene_index]

        # Remove target from TF-list and TF-expression matrix if target itself is a TF
        if not are_tfs_clustered:
            (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)
        else:
            clean_tf_matrix, clean_tf_matrix_gene_names = tf_matrix, tf_matrix_gene_names

        # Sample one TF per TF-cluster and subset TF expression matrix in case of TFs having been clustered.
        if not cluster_to_tfs is None:
            tf_representatives = []
            for cluster, tf_list in cluster_to_tfs.items():
                cluster_size = len(tf_list)
                representative = tf_list[perm % cluster_size]
                tf_representatives.append(representative)
            # Subset TF expression matrix.
            clean_tf_matrix, clean_tf_matrix_gene_names = subset_tf_matrix(clean_tf_matrix,
                                                                    clean_tf_matrix_gene_names,
                                                                    tf_representatives)

        # Special case in which only a single TF is passed and the target gene
        # here is the same as the TF (clean_tf_matrix is empty after cleaning):
        if clean_tf_matrix.size == 0:
            raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

        # Shuffle target gene expression vector
        np.random.seed(perm)
        permuted_target_gene_expression = np.random.permutation(target_expression)

        # Train the random forest regressor.
        try:
            trained_regressor = fit_model(
                regressor_type,
                regressor_kwargs,
                clean_tf_matrix,
                permuted_target_gene_expression,
                early_stop_window_length,
                seed
            )
        except ValueError as e:
            raise ValueError(
                "Count_computation_sampled: regression for target gene {0} failed. Cause {1}.".format(target_gene_name, repr(e))
            )

        # Construct the shuffled GRN dataframe from the trained regressor
        shuffled_grn_df = to_links_df(
            regressor_type,
            regressor_kwargs,
            trained_regressor,
            clean_tf_matrix_gene_names,
            target_gene_name
        )

        if are_tfs_clustered:
            shuffled_target_sum = 0.0
            for tf, _, importance in zip(shuffled_grn_df['TF'], shuffled_grn_df['target'], shuffled_grn_df['importance']):
                tf_cluster_size = len(cluster_to_tfs[tf_to_cluster[tf]])
                shuffled_target_sum += importance * tf_cluster_size
            scaling_factor = per_target_importance_sums[target_gene_name] / shuffled_target_sum
            shuffled_grn_df['importance'] = shuffled_grn_df['importance'] * scaling_factor

        # Update the count values of the partial input GRN.
        count_helper(shuffled_grn_df, partial_input_grn, tf_to_cluster, scale_for_tf_sampling)

    # Change partial input GRN format from dict to df
    partial_input_grn_fdr_df = pd.DataFrame(
        [(TF, target, v['importance'], v['count'], v['shuffled_occurences']) for (TF, target), v in partial_input_grn.items()],
        columns=['TF', 'target', 'importance', 'count', 'shuffled_occurences']
    )

    if not output_dir is None:
        partial_input_grn_fdr_df.to_feather(os.path.join(output_dir, f'cluster_{cluster_id}.feather'))

    return partial_input_grn_fdr_df

@delayed
def save_df(df, filename):
    df.to_feather(filename)
