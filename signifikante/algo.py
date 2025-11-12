"""
Top-level functions.
"""

import pandas as pd
from distributed import Client, LocalCluster
# UPDATE FOR NEW GRN METHOD
from signifikante.core import (
    create_graph, SGBM_KWARGS, RF_KWARGS, EARLY_STOP_WINDOW_LENGTH, ET_KWARGS, XGB_KWARGS, LASSO_KWARGS
)
from signifikante.fdr import perform_fdr
import os

def signifikante_fdr(
        expression_data : pd.DataFrame,
        cluster_representative_mode : str,
        num_target_clusters : int = -1,
        num_tf_clusters : int = -1,
        target_cluster_mode : str = 'wasserstein',
        tf_cluster_mode : str = 'correlation',
        input_grn : pd.DataFrame = None,
        tf_names : list[str] = None,
        target_subset : list[str] = None,
        client_or_address='local',
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        seed=None,
        verbose=False,
        num_permutations=1000,
        output_dir=None,
        scale_for_tf_sampling : bool = False,
        inference_mode : str = "grnboost2",
        apply_bh_correction : bool = False,
        normalize_gene_expression : bool = False
):
    """
        :param expression_data: Expression matrix with genes as columns and samples as rows.
        :param cluster_representative_mode: How to draw representatives from target gene clusters. 
            Can be one of "random" or "medoid" for approximate P-value computation, or "all_genes" for exact (DIANE-like) P-values.
        :param num_target_clusters: Number of target gene clusters. If set to -1, no target gene clustering will be applied. Defaults to -1.
        :param num_tf_clusters: Experimental feature. Used for setting the number of desired TF clusters, if set to -1, no TF clustering will be applied. Defaults to -1.
        :param target_cluster_mode: Experimental feature. Indicates, which clustering to use for target gene clustering. Defaults to "wasserstein".
        :param tf_cluster_mode: Experimental feature. Indicates, which clustering mode to use for TF clustering. Defaults to "correlation".
        :param input_grn: Reference GRN to use for FDR control. Needs to possess columns 'TF', 'target', 'importance'. 
            Should only be used, when it is clear that this GRN is inferred using the same method indicated in inference_mode. 
            Defaults to None. If no reference GRN is given, a new one is inferred in the beginning.
        :param target_subset: Subset of target genes to consider for FDR control. Only compatible with "all_genes" FDR mode.
        :param tf_names: List of strings representing TF names. Should be subset of gene names contained in expression_data. Defaults to None. If no list is given, all genes are treated as potential TFs.
        :param client_or_address: Whether to perform computation on given input Dask Cluster object, or to create a new local one ("local"). Defaults to "local".
        :param early_stop_window_length: Window length to use for early stopping. Defaults to 25.
        :param seed: Random seed for regressor models. Defaults to None.
        :param verbose: Whether or not to print detailed additional information. Defaults to False.
        :param num_permutations: How many permutations to perform for random background model for empirical P-value computation. Defaults to 1000.
        :param output_dir: Where to save additional intermediate data to. Defaults to None, i.e. saves no intermediate results.
        :param scale_for_tf_sampling: Experimental feature. Whether or not to keep track of occurences of edges in permuted GRNs. Defaults to False.
        :param inference_mode: Which GRN inference method to use under the hood. Can be one of "grnboost2", "genie3", "xgboost", and "lasso". Defaults to "grnboost2".
        :param apply_bh_correction: Whether or not to additionally return Benjamini-Hochberg adjusted P-values.
        :param normalize_gene_expression: Whether or not to apply z-score normalization on gene columns in input expression matrix.
        :return: Pandas DataFrame with columns 'TF', 'target', 'importance', 'pvalue' representing P-values on each gene regulatory link.
    """
    if cluster_representative_mode not in {'medoid', 'random', 'all_genes'}:
        raise ValueError('cluster_representative_mode must be one of "medoid", "random", "all_genes"')

    if target_subset is not None and cluster_representative_mode != 'all_genes':
        raise ValueError("Target subset is given, but is only compatible with all_genes FDR mode.")

    if num_target_clusters==-1 and num_tf_clusters==-1 and not cluster_representative_mode == "all_genes":
        print("No cluster numbers given, running full FDR mode...")
        cluster_representative_mode="all_genes"

    if verbose and num_tf_clusters == -1:
        print("running FDR without TF clustering")

    if verbose and num_target_clusters == -1:
        print("running FDR without target clustering")

    if output_dir is not None:
        if not os.path.exists(output_dir):
            print('output directory does not exist, creating!')
            os.makedirs(output_dir, exist_ok=True)

    # If desired, apply z-score normalization on input gene expression matrix.
    if normalize_gene_expression:
        expression_data = (expression_data - expression_data.mean()) / expression_data.std(ddof=0)

    # If input GRN has not been given, run one GRN inference call upfront.
    if input_grn is None:
        input_grn = grnboost2(
            expression_data=expression_data,
            tf_names=tf_names,
            client_or_address=client_or_address,
            seed=seed
        )

    # Align the input GRN and expression data w.r.t. the genes
    genes_input_grn = set(input_grn['TF']).union(set(input_grn['target']))
    genes_expression_data = set(expression_data.columns)

    genes_intersection = list(genes_input_grn.intersection(genes_expression_data))

    expression_data_aligned = expression_data[genes_intersection]

    keep_bool = input_grn['TF'].isin(genes_intersection) * input_grn['target'].isin(genes_intersection)
    input_grn_aligned = input_grn[keep_bool]

    # Extract the TFs from the input GRN.
    tf_names_input_grn = list(set(input_grn_aligned['TF']))

    # Transform input GRN into dict format.
    input_grn_dict = dict()
    for tf, target, importance in zip(input_grn_aligned['TF'], input_grn_aligned['target'], input_grn_aligned['importance']):
        input_grn_dict[(tf, target)] = {'importance': importance}

    # UPDATE FOR NEW GRN METHOD
    if inference_mode == "grnboost2":
        regressor_type = "GBM"
        regressor_args = SGBM_KWARGS
    elif inference_mode == "genie3":
        regressor_type = "RF"
        regressor_args = RF_KWARGS
    elif inference_mode == "extra_trees":
        regressor_type = "ET"
        regressor_args = ET_KWARGS
    elif inference_mode == "xgboost":
        regressor_type = "XGB"
        regressor_args = XGB_KWARGS
    elif inference_mode == "lasso":
        regressor_type = "LASSO"
        regressor_args = LASSO_KWARGS
    else:
        raise ValueError(f"Unknown GRN inference mode: {inference_mode}")
        

    return perform_fdr(
        expression_data_aligned,
        input_grn_dict,
        num_target_clusters,
        num_tf_clusters,
        cluster_representative_mode,
        target_cluster_mode,
        tf_cluster_mode,
        tf_names_input_grn,
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
    )


def grnboost2(expression_data,
              gene_names=None,
              tf_names='all',
              target_names = 'all',
              client_or_address='local',
              early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
              limit=None,
              seed=None,
              verbose=False):
    """
    Launch signifikante with [GRNBoost2] profile.

    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :param target_names: optional list of target genes which are used as response variable in the regression model.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param early_stop_window_length: early stop window length. Default 25.
    :param limit: optional number (int) of top regulatory links to return. Default None.
    :param seed: optional random seed for the regressors. Default None.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """

    return diy(expression_data=expression_data, regressor_type='GBM', regressor_kwargs=SGBM_KWARGS,
               gene_names=gene_names, tf_names=tf_names, target_names = target_names, client_or_address=client_or_address,
               early_stop_window_length=early_stop_window_length, limit=limit, seed=seed, verbose=verbose)


def genie3(expression_data,
           gene_names=None,
           tf_names='all',
           target_names = 'all',
           client_or_address='local',
           limit=None,
           seed=None,
           verbose=False):
    """
    Launch signifikante with [GENIE3] profile.

    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param limit: optional number (int) of top regulatory links to return. Default None.
    :param seed: optional random seed for the regressors. Default None.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """

    return diy(expression_data=expression_data, regressor_type='RF', regressor_kwargs=RF_KWARGS,
               gene_names=gene_names, tf_names=tf_names, target_names = target_names, client_or_address=client_or_address,
               limit=limit, seed=seed, verbose=verbose)

def extra_trees(expression_data,
           gene_names=None,
           tf_names='all',
           client_or_address='local',
           limit=None,
           seed=None,
           verbose=False):
    """
    Launch signifikante with [extra-tree (ET)] profile.

    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param limit: optional number (int) of top regulatory links to return. Default None.
    :param seed: optional random seed for the regressors. Default None.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """

    return diy(expression_data=expression_data, regressor_type='ET', regressor_kwargs=ET_KWARGS,
               gene_names=gene_names, tf_names=tf_names, client_or_address=client_or_address,
               limit=limit, seed=seed, verbose=verbose)

def xgboost(expression_data,
           gene_names=None,
           tf_names='all',
           client_or_address='local',
           limit=None,
           seed=None,
           verbose=False):
    """
    Launch signifikante with [xgboost] profile.

    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param limit: optional number (int) of top regulatory links to return. Default None.
    :param seed: optional random seed for the regressors. Default None.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """

    return diy(expression_data=expression_data, regressor_type='XGB', regressor_kwargs=XGB_KWARGS,
               gene_names=gene_names, tf_names=tf_names, client_or_address=client_or_address,
               limit=limit, seed=seed, verbose=verbose)

def lasso(expression_data,
           gene_names=None,
           tf_names='all',
           client_or_address='local',
           limit=None,
           seed=None,
           verbose=False):
    """
    Launch signifikante with [lasso] profile.

    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param limit: optional number (int) of top regulatory links to return. Default None.
    :param seed: optional random seed for the regressors. Default None.
    :param verbose: print info.
    :return: a pandas DataFrame['TF', 'target', 'importance'] representing the inferred gene regulatory links.
    """

    return diy(expression_data=expression_data, regressor_type='LASSO', regressor_kwargs=LASSO_KWARGS,
               gene_names=gene_names, tf_names=tf_names, client_or_address=client_or_address,
               limit=limit, seed=seed, verbose=verbose)


def diy(expression_data,
        regressor_type,
        regressor_kwargs,
        gene_names=None,
        tf_names='all',
        target_names = 'all',
        client_or_address='local',
        early_stop_window_length=EARLY_STOP_WINDOW_LENGTH,
        limit=None,
        seed=None,
        verbose=False):
    """
    :param expression_data: one of:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
    :param regressor_type: string. One of: 'RF', 'GBM', 'ET'. Case insensitive.
    :param regressor_kwargs: a dictionary of key-value pairs that configures the regressor.
    :param gene_names: optional list of gene names (strings). Required when a (dense or sparse) matrix is passed as
                       'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :param early_stop_window_length: early stopping window length.
    :param client_or_address: one of:
           * None or 'local': a new Client(LocalCluster()) will be used to perform the computation.
           * string address: a new Client(address) will be used to perform the computation.
           * a Client instance: the specified Client instance will be used to perform the computation.
    :param limit: optional number (int) of top regulatory links to return. Default None.
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

        expression_matrix, gene_names, tf_names, target_names = _prepare_input(expression_data, gene_names, tf_names, target_names)

        if verbose:
            print('creating dask graph')

        graph = create_graph(expression_matrix,
                             gene_names,
                             tf_names,
                             target_genes = target_names,
                             client=client,
                             regressor_type=regressor_type,
                             regressor_kwargs=regressor_kwargs,
                             early_stop_window_length=early_stop_window_length,
                             limit=limit,
                             seed=seed)

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


def _prepare_client(client_or_address):
    """
    :param client_or_address: one of:
           * None
           * verbatim: 'local'
           * string address
           * a Client instance
    :return: a tuple: (Client instance, shutdown callback function).
    :raises: ValueError if no valid client input was provided.
    """

    if client_or_address is None or str(client_or_address).lower() == 'local':
        local_cluster = LocalCluster(diagnostics_port=None)
        client = Client(local_cluster)

        def close_client_and_local_cluster(verbose=False):
            if verbose:
                print('shutting down client and local cluster')

            client.close()
            local_cluster.close()

        return client, close_client_and_local_cluster

    elif isinstance(client_or_address, str) and client_or_address.lower() != 'local':
        client = Client(client_or_address)

        def close_client(verbose=False):
            if verbose:
                print('shutting down client')

            client.close()

        return client, close_client

    elif isinstance(client_or_address, Client):

        def close_dummy(verbose=False):
            if verbose:
                print('not shutting down client, client was created externally')

            return None

        return client_or_address, close_dummy

    else:
        raise ValueError("Invalid client specified {}".format(str(client_or_address)))


def _prepare_input(expression_data,
                   gene_names,
                   tf_names,
                   target_names):
    """
    Wrangle the inputs into the correct formats.

    :param expression_data: one of:
                            * a pandas DataFrame (rows=observations, columns=genes)
                            * a dense 2D numpy.ndarray
                            * a sparse scipy.sparse.csc_matrix
    :param gene_names: optional list of gene names (strings).
                       Required when a (dense or sparse) matrix is passed as 'expression_data' instead of a DataFrame.
    :param tf_names: optional list of transcription factors. If None or 'all', the list of gene_names will be used.
    :return: a triple of:
             1. a np.ndarray or scipy.sparse.csc_matrix
             2. a list of gene name strings
             3. a list of transcription factor name strings.
    """

    if isinstance(expression_data, pd.DataFrame):
        expression_matrix = expression_data.to_numpy()
        gene_names = list(expression_data.columns)
    else:
        expression_matrix = expression_data
        assert expression_matrix.shape[1] == len(gene_names)

    if tf_names is None:
        tf_names = gene_names
    elif tf_names == 'all':
        tf_names = gene_names
    else:
        if len(tf_names) == 0:
            raise ValueError('Specified tf_names is empty')

        if not set(gene_names).intersection(set(tf_names)):
            raise ValueError('Intersection of gene_names and tf_names is empty.')
    
    if isinstance(target_names, str) and target_names == 'all':
        target_names = gene_names
    else:
        if len(target_names) == 0:
            raise ValueError('Specified target list is empty')

        if not set(gene_names).intersection(set(target_names)):
            raise ValueError('Intersection of gene_names and target_names is empty.')


    return expression_matrix, gene_names, tf_names, target_names
