
from signifikante.core import (
    EARLY_STOP_WINDOW_LENGTH, DEMON_SEED, to_tf_matrix, target_gene_indices, clean, fit_model, 
    to_links_df
)
from signifikante.fdr_utils import (
    compute_correlation_distance_matrix, compute_wasserstein_distance_matrix, cluster_genes_to_dict, 
    partition_input_grn, invert_tf_to_cluster_dict, new_permutation_null, record_permutation, count_edges,
    subset_tf_matrix, _prepare_client, _prepare_input
)
import numpy as np
import pandas as pd
from dask import delayed, compute
from dask.dataframe.utils import make_meta, pyarrow_strings_enabled
from dask.dataframe._pyarrow import to_pyarrow_string
import pickle
import os
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import AgglomerativeClustering

FDR_GRN_SCHEMA = make_meta({'TF': str, 'target': str, 'importance': float, 'count' : float, 'shuffled_occurences' : float, 'westfall_young' : float, 'cluster_id' : float})

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
        apply_bh_correction,
        apply_westfall_young,
        return_cluster_ids
):
    # Extract TF name and target name lists from expression matrix object.
    _, gene_names, tf_names, _ = _prepare_input(expression_data, None, tf_names, "all")
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
        if target_cluster_mode in ['wasserstein', 'wasserstein-spectral', 'wasserstein-kmedoids']:
            dist_matrix_all = compute_wasserstein_distance_matrix(expression_data, num_threads=-1)

        if tf_cluster_mode == 'correlation' or tf_cluster_mode == "kmeans":
            tf_bool = [True if gene in tf_names else False for gene in expression_data.columns]
            exp_matrix_tfs = expression_data.loc[:, tf_bool]
            if tf_cluster_mode == "correlation":
                corr_distances_tfs = compute_correlation_distance_matrix(exp_matrix_tfs)

        # Cluster targets and TFs separately given respective mode.
        if target_cluster_mode == 'kmeans':
            target_to_clust, target_medoids = cluster_genes_to_dict(expression_data, num_clusters=num_non_tf_clusters,
                                                                    mode='kmeans')
        elif target_cluster_mode == 'wasserstein':
            target_to_clust, target_medoids = cluster_genes_to_dict(dist_matrix_all, num_clusters=num_non_tf_clusters,
                                                                    mode='distance')
        elif target_cluster_mode == 'wasserstein-spectral':
            target_to_clust, target_medoids = cluster_genes_to_dict(dist_matrix_all, num_clusters=num_non_tf_clusters,
                                                                    mode='distance-spectral')
        elif target_cluster_mode == 'wasserstein-kmedoids':
            target_to_clust, target_medoids = cluster_genes_to_dict(dist_matrix_all, num_clusters=num_non_tf_clusters,
                                                                    mode='distance-kmedoids')
        else:
            raise ValueError(f'ERROR: Unknown target cluster mode: {target_cluster_mode}')

        if tf_cluster_mode == 'correlation':
            tf_to_clust, tf_medoids = cluster_genes_to_dict(corr_distances_tfs, num_clusters=num_tf_clusters, mode='distance')
        elif tf_cluster_mode == 'kmeans':
            tf_to_clust, tf_medoids = cluster_genes_to_dict(exp_matrix_tfs, num_clusters=num_tf_clusters,
                                                            mode='kmeans')
        else:
            raise ValueError(f'Unknown TF cluster mode: {tf_cluster_mode}')

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
                   scale_for_tf_sampling=scale_for_tf_sampling,
                   apply_westfall_young=apply_westfall_young
                   )
    
    # Transform counts into P-values and remove count column.
    fdr_controlled_df['pvalue'] = (fdr_controlled_df['count']+1)/(num_permutations+1)
    if not scale_for_tf_sampling:
        fdr_controlled_df.drop(columns=['count'], inplace=True)
        fdr_controlled_df.drop(columns=['shuffled_occurences'], inplace=True)
    if apply_bh_correction:
        fdr_controlled_df['pvalue_bh'] = multipletests(fdr_controlled_df['pvalue'], method='fdr_bh')[1]
    
    if apply_westfall_young:
        fdr_controlled_df["pvalue_westfall_young"] = (fdr_controlled_df['westfall_young']+1)/(num_permutations+1)
    fdr_controlled_df.drop(columns=['westfall_young'], inplace=True)
    
    if return_cluster_ids:
        fdr_controlled_df["cluster_id"] = pd.factorize(fdr_controlled_df["cluster_id"])[0]
    else:  
      fdr_controlled_df.drop(columns=['cluster_id'], inplace=True)
    
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
            apply_westfall_young=False
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
        expression_matrix, gene_names, _, _ = _prepare_input(expression_data, gene_names, tf_names, "all")

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

        delayed_nulls, edge_subsets, tf_to_cluster = create_graph_fdr(expression_matrix,
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
                                 scale_for_tf_sampling=scale_for_tf_sampling,
                                 apply_westfall_young=apply_westfall_young
                                 )

        if verbose:
            print('{} permutation tasks'.format(len(delayed_nulls)))
            print('computing dask graph')

        nulls = client.compute(delayed_nulls, sync=True)
        links_df = pd.concat([
            count_edges(null, partial_input_grn, tf_to_cluster, scale_for_tf_sampling, apply_westfall_young,
                        cluster_id, feather_file)
            for null, (partial_input_grn, cluster_id, feather_file) in zip(nulls, edge_subsets)
        ])
        # same string dtypes as the Dask DataFrame the links were collected in before
        if pyarrow_strings_enabled():
            links_df = to_pyarrow_string(links_df)

        return links_df.sort_values(by='importance', ascending=False)

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
                     output_dir=None,  
                     apply_westfall_young=False
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
    :param repartition_multiplier: not used anymore (the links are no longer collected in a Dask DataFrame).
    :param n_permutations: int. Number of random permutations to run for FDR control.
    :param seed: (optional) random seed for the regressors. Default 666.
    :return: a tuple: the list of delayed permutation nulls (one per permutation task, see permutation_null_medoid
             and permutation_null_sampled), per task the tuple (edges of the input GRN to count against its null,
             cluster ID column value, feather file or None) for count_edges, and the TF clustering.
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

    # The permutation tasks only get their targets and return the null distribution of the TF importances
    # (permutation_null_*); the edges are counted against it afterwards (count_edges, in diy_fdr), so the input GRN
    # does not go into the graph.
    delayed_nulls = []  # collection of delayed permutation nulls
    edge_subsets = []  # per permutation task: (edges to count against its null, cluster ID column, feather file)

    # Use pre-computed medoid representatives for TFs and/or non-TFs.
    if fdr_mode == 'medoid':
        # Loop over all representative targets, i.e. non-TF medoids.
        all_targets = target_representatives
        if target_subset is not None:
            all_targets = target_subset
        for target_gene_index in target_gene_indices(gene_names, all_targets):
            target_gene_name = gene_names[target_gene_index]
            target_gene_expression = delayed(expression_matrix[:, target_gene_index])
            target_importance_sums = {target_gene_name: per_target_importance_sums[target_gene_name]} \
                if target_gene_name in per_target_importance_sums else {}

            delayed_nulls.append(delayed(permutation_null_medoid, pure=True)(
                regressor_type,
                regressor_kwargs,
                future_tf_matrix,
                are_tfs_clustered,
                future_tf_matrix_gene_names,
                target_gene_name,
                target_gene_expression,
                target_importance_sums,
                future_tf_to_cluster,
                future_cluster_to_tfs,
                n_permutations,
                early_stop_window_length,
                seed
            ))
            # Subset of GRN which is represented by the medoid.
            edge_subsets.append((
                grn_subsets_per_target[target_to_cluster[target_gene_name]],
                target_gene_index,
                None if output_dir is None else os.path.join(output_dir, f'target_{target_gene_name}.feather')
            ))

    # Loop over all genes of cluster, i.e. simulate random drawing of genes from clusters.
    elif fdr_mode == 'random':
        # Loop over all target clusters (that includes TF clusters).
        for cluster_id, cluster_targets in genes_per_target_cluster.items():
            target_cluster_idxs = target_gene_indices(gene_names, cluster_targets)
            # Like this, order of cluster gene names should be consistent with order in cluster_idxs and hence with
            # gene column order in expression matrix.
            target_cluster_gene_names = [gene for index, gene in enumerate(gene_names) if index in target_cluster_idxs]
            cluster_expression = delayed(expression_matrix)
            target_importance_sums = {gene: per_target_importance_sums[gene] for gene in target_cluster_gene_names
                                      if gene in per_target_importance_sums}

            delayed_nulls.append(delayed(permutation_null_sampled, pure=True)(
                regressor_type,
                regressor_kwargs,
                future_tf_matrix,
                future_tf_matrix_gene_names,
                future_cluster_to_tfs,
                target_cluster_gene_names,
                target_cluster_idxs,
                cluster_expression,
                tf_to_cluster,
                target_importance_sums,
                n_permutations,
                early_stop_window_length,
                seed
            ))
            edge_subsets.append((
                grn_subsets_per_target[cluster_id],
                cluster_id,
                None if output_dir is None else os.path.join(output_dir, f'cluster_{cluster_id}.feather')
            ))
    else:
        raise ValueError(f'Unknown FDR mode: {fdr_mode}.')

    return delayed_nulls, edge_subsets, tf_to_cluster

def permutation_null_medoid(
        regressor_type,
        regressor_kwargs,
        tf_matrix,
        are_tfs_clustered,
        tf_matrix_gene_names: list[str],
        target_gene_name: str,
        target_gene_expression: np.ndarray,
        target_importance_sums: dict,
        tf_to_cluster: dict[str, int],
        cluster_to_tf : dict,
        n_permutations: int,
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=DEMON_SEED
):
    """
    Fits n_permutations models on the permuted expression of one (medoid) target and returns the importance of each
    TF cluster in each of them, the null distribution to count the edges of the target's cluster against (see
    count_edges). target_importance_sums: {target: importance sum of its input GRN edges}, needed if TFs are clustered.
    """

    # Remove target from TF-list and TF-expression matrix if target itself is a TF
    if not are_tfs_clustered:
        (clean_tf_matrix, clean_tf_matrix_gene_names) = clean(tf_matrix, tf_matrix_gene_names, target_gene_name)
    else:
        clean_tf_matrix, clean_tf_matrix_gene_names = tf_matrix, tf_matrix_gene_names

    # Special case in which only a single TF is passed and the target gene
    # here is the same as the TF (clean_tf_matrix is empty after cleaning):
    if clean_tf_matrix.size == 0:
        raise ValueError("Cleaned TF matrix is empty, skipping inference of target {}.".format(target_gene_name))

    null = new_permutation_null(tf_to_cluster, n_permutations)

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
            scaling_factor = target_importance_sums[target_gene_name] / shuffled_target_sum
            shuffled_grn_df['importance'] = shuffled_grn_df['importance'] * scaling_factor

        record_permutation(null, i, shuffled_grn_df, tf_to_cluster)

    return null


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
        scale_for_tf_sampling=False,
        apply_westfall_young=False,
        target_gene_index=-1
):
    """
    Permutation counts of the edges of partial_input_grn against the permutations of one (medoid) target, see
    permutation_null_medoid and count_edges.
    """
    target_importance_sums = {target_gene_name: per_target_importance_sums[target_gene_name]} \
        if target_gene_name in per_target_importance_sums else {}
    null = permutation_null_medoid(regressor_type, regressor_kwargs, tf_matrix, are_tfs_clustered,
                                   tf_matrix_gene_names, target_gene_name, target_gene_expression,
                                   target_importance_sums, tf_to_cluster, cluster_to_tf, n_permutations,
                                   early_stop_window_length, seed)
    feather_file = None if output_dir is None else os.path.join(output_dir, f'target_{target_gene_name}.feather')
    return count_edges(null, partial_input_grn, tf_to_cluster, scale_for_tf_sampling, apply_westfall_young,
                       target_gene_index, feather_file)


def permutation_null_sampled(
        regressor_type,
        regressor_kwargs,
        tf_matrix,
        tf_matrix_gene_names,
        cluster_to_tfs,
        target_gene_names,
        target_gene_idxs,
        target_gene_expressions,
        tf_to_cluster: dict[str, int],
        target_importance_sums : dict,
        n_permutations : int,
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=DEMON_SEED
):
    """
    Fits n_permutations models on permuted targets drawn in turn from a target cluster and returns the importance of
    each TF cluster in each of them, the null distribution to count the cluster's edges against (see count_edges).
    target_importance_sums: {target: importance sum of its input GRN edges}, needed if TFs are clustered.
    """

    are_tfs_clustered = (not cluster_to_tfs is None)
    null = new_permutation_null(tf_to_cluster, n_permutations)

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
            scaling_factor = target_importance_sums[target_gene_name] / shuffled_target_sum
            shuffled_grn_df['importance'] = shuffled_grn_df['importance'] * scaling_factor

        record_permutation(null, perm, shuffled_grn_df, tf_to_cluster)

    return null


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
        scale_for_tf_sampling=False,
        apply_westfall_young=False
):
    """
    Permutation counts of the edges of partial_input_grn against the permutations of the targets of one target
    cluster, see permutation_null_sampled and count_edges.
    """
    target_importance_sums = {gene: per_target_importance_sums[gene] for gene in target_gene_names
                              if gene in per_target_importance_sums}
    null = permutation_null_sampled(regressor_type, regressor_kwargs, tf_matrix, tf_matrix_gene_names, cluster_to_tfs,
                                    target_gene_names, target_gene_idxs, target_gene_expressions, tf_to_cluster,
                                    target_importance_sums, n_permutations, early_stop_window_length, seed)
    feather_file = None if output_dir is None else os.path.join(output_dir, f'cluster_{cluster_id}.feather')
    return count_edges(null, partial_input_grn, tf_to_cluster, scale_for_tf_sampling, apply_westfall_young,
                       cluster_id, feather_file)

@delayed
def save_df(df, filename):
    df.to_feather(filename)
    
def run_heuristic(expression_mat : pd.DataFrame,
                  max_clusters : int = 100,
                  decrease_fraction : float = 0.1
                  ) -> int:
    """
    Heuristic function for finding optimal number of cluster diameters.

    :param expression_mat: gene expression matrix with genes as columns and samples as rows.
    :return: Best number of clusters according to clustering structure.
    """
    # Compute Wasserstein distance matrix between gene expression columns.
    distance_df = compute_wasserstein_distance_matrix(expression_mat)
    distance_matrix = distance_df.values
    
    # Heuristic: Look for "elbow" in cluster diameters.
    n_genes = distance_matrix.shape[0]
    results = []

    initial_mean_diameter = None
    stopping_threshold = None
    previous_mean_diameter = None

    for num_clusters in range(1, min(max_clusters, n_genes) + 1):

        clustering = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric="precomputed",
            linkage="complete"
        )

        labels = clustering.fit_predict(distance_matrix)

        cluster_diameters = []

        for cluster_id in np.unique(labels):
            cluster_indices = np.where(labels == cluster_id)[0]

            if len(cluster_indices) <= 1:
                cluster_diameters.append(0.0)
                continue

            cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            diameter = np.max(cluster_distances)
            cluster_diameters.append(diameter)

        mean_diameter = float(np.mean(cluster_diameters))

        # Initialize threshold using first clustering.
        if num_clusters == 1:
            initial_mean_diameter = mean_diameter
            stopping_threshold = decrease_fraction * initial_mean_diameter

        # Stop criterion
        if previous_mean_diameter is not None:
            decrease = previous_mean_diameter - mean_diameter
            if decrease < stopping_threshold:
                break

        results.append({
            "n_clusters": num_clusters,
            "mean_cluster_diameter": mean_diameter,
        })

        previous_mean_diameter = mean_diameter

    return num_clusters
