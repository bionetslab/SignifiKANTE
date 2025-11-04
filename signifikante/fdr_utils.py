
import pandas as pd
import numpy as np

from numba import njit, prange, set_num_threads, types
from sklearn.cluster import AgglomerativeClustering
from distributed import Client, LocalCluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

@njit(nopython=True, nogil=True)
def _merge_sorted_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Merge two sorted 1D NumPy arrays into a single sorted array.

    :param a: a 1D NumPy array sorted in ascending order.
    :param b: a 1D NumPy array sorted in ascending order.
    :return: a 1D NumPy array containing all elements from `a` and `b`, merged in sorted order.
    """

    lenA,lenB = a.shape[0], b.shape[0]
    # Get searchsorted indices
    idx = np.searchsorted(a,b)
    # Offset each searchsorted indices with ranged array to get new positions of b in output array
    b_pos = np.arange(lenB) + idx
    lenTotal = lenA+lenB
    mask = np.ones(lenTotal,dtype=types.boolean)
    out = np.empty(lenTotal,dtype=types.float64)
    mask[b_pos] = False
    out[b_pos] = b
    out[mask] = a
    return out


@njit(nopython=True, parallel=True, nogil=True)
def _pairwise_wasserstein_dists(sorted_matrix: np.ndarray, num_threads: int) -> np.ndarray:
    """
    Compute the pairwise 1D Wasserstein distances between all columns of a sorted matrix.

    Each column in `sorted_matrix` represents a sorted 1D empirical distribution. The function computes the
    Wasserstein distance between each pair of columns, assuming uniform sample weights.

    :param sorted_matrix: a 2D NumPy array where each column is a sorted in ascending order.
    :param num_threads: number of threads to use for parallel execution. If -1, uses the default setting.
    :return: distance matrix, i.e. a 2D NumPy array containing the pairwise Wasserstein distances.
    """

    if num_threads != -1:
        set_num_threads(num_threads)
    num_cols = sorted_matrix.shape[1]
    num_rows = sorted_matrix.shape[0]
    distance_mat = np.zeros((num_cols, num_cols))
    for col1 in prange(num_cols):
        for col2 in range(col1 + 1, num_cols):
            all_values = _merge_sorted_arrays(sorted_matrix[:, col1], sorted_matrix[:, col2])
            # Compute the differences between pairs of successive values of u and v.
            deltas = np.diff(all_values)
            # Get the respective positions of the values of u and v among the values of
            # both distributions.
            col1_cdf_indices = np.searchsorted(sorted_matrix[:, col1], all_values[:-1], 'right')
            col2_cdf_indices = np.searchsorted(sorted_matrix[:, col2], all_values[:-1], 'right')
            # Calculate the CDFs of u and v using their weights, if specified.
            col1_cdf = col1_cdf_indices / num_rows
            col2_cdf = col2_cdf_indices / num_rows
            # Compute the value of the integral based on the CDFs.
            distance = np.sum(np.multiply(np.abs(col1_cdf - col2_cdf), deltas))
            distance_mat[col1, col2] = distance
            distance_mat[col2, col1] = distance
    return distance_mat


def compute_wasserstein_distance_matrix(expression_mat: pd.DataFrame, num_threads: int = -1) -> pd.DataFrame:
    """
    Compute the pairwise 1D Wasserstein distance matrix between columns of a gene expression matrix.

    Each column in the input DataFrame is treated as a 1D empirical distribution.
    The function sorts the values in each column and computes the pairwise Wasserstein distances
    using uniform sample weights.

    :param expression_mat: a pandas DataFrame where each column is a gene's expression profile across samples.
    :param num_threads: number of threads to use for parallel execution. If -1, uses the default setting.
    :return: a pandas DataFrame containing the symmetric matrix of pairwise Wasserstein distances.
    """
    numpy_mat = expression_mat.to_numpy()
    numpy_mat = np.sort(numpy_mat, axis=0)
    distance_mat = _pairwise_wasserstein_dists(numpy_mat, num_threads)
    distance_mat = pd.DataFrame(distance_mat, columns=expression_mat.columns)
    return distance_mat

def compute_correlation_distance_matrix(exp_matrix : pd.DataFrame):
    corr_matrix = np.corrcoef(exp_matrix.to_numpy(), rowvar=False)
    scaled_corr_matrix = (corr_matrix+1.0) / 2.0
    distance_mat = 1.0 - scaled_corr_matrix
    distance_df = pd.DataFrame(distance_mat, columns=exp_matrix.columns)
    return distance_df

def cluster_genes_to_dict(input_matrix : pd.DataFrame, num_clusters : int, mode : str):
    """
    Perform agglomerative clustering on a precomputed distance matrix and return gene-to-cluster mappings.

    :param input_matrix: either a symmetric pandas DataFrame representing pairwise distances between genes,
                         or the full expression matrix in case of KMeans based clustering.
    :param num_clusters: the number of clusters to form.
    :return: a dictionary mapping gene names (column names in the distance matrix) to cluster IDs.
    """
    # Create clusters.
    if mode=='distance':
        agg_clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete')
        cluster_labels = agg_clustering.fit_predict(input_matrix.to_numpy())
        # Map clustering output to dictionary representation.
        gene_names = input_matrix.columns.to_list()
        gene_to_cluster = {name : id for name, id in zip(gene_names, cluster_labels)}
        medoids = compute_medoids(gene_to_cluster, input_matrix)
        return gene_to_cluster, medoids
    elif mode=='kmeans':
        num_samples = len(input_matrix.index)
        # Perform PCA on columns of expression matrix.
        input_matrix_transp = input_matrix.T.copy()
        pca = PCA(n_components=min(50, num_samples))
        pca_result = pca.fit_transform(input_matrix_transp)
        # Run K-Medoids clustering on PCA-reduced data
        kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
        kmedoids.fit(pca_result)
        cluster_labels = kmedoids.labels_
        gene_to_cluster = {gene : cluster for gene, cluster in zip(input_matrix.columns, cluster_labels)}
        gene_list = input_matrix.columns.to_list()
        medoids = [gene_list[i] for i in list(kmedoids.medoid_indices_)]
        return gene_to_cluster, medoids
    else:
        print("Cluster_genes_to_dict: unknown clustering mode...")
        return None

def merge_gene_clusterings(clustering1 : dict, clustering2 : dict):
    num_clusters1 = max({clusterID for _, clusterID in clustering1.items()})+1
    updated_clustering2 = {gene : clusterID+num_clusters1 for gene, clusterID in clustering2.items()}
    return clustering1 | updated_clustering2

def compute_medoids(gene_to_cluster : dict, distance_matrix : pd.DataFrame):
    medoids = []
    clusters = set(gene_to_cluster.values())
    for cluster in clusters:
        # Retrieve genes in current cluster.
        cluster_genes = [gene for gene, c in gene_to_cluster.items() if c==cluster]
        # Extract cluster sub-distance matrix.
        cluster_bool = [True if gene in cluster_genes else False for gene in distance_matrix.columns]
        sub_matrix = distance_matrix.loc[cluster_bool, cluster_bool]
        # Compute average distance for each gene to remaining ones in cluster.
        avg_distances = sub_matrix.mean(axis=0)
        # Medoid is gene with the minimal average distance.
        medoid_gene = avg_distances.idxmin()
        medoids.append(medoid_gene)

    return medoids

def partition_input_grn(input_grn, clustering_dict):
    grn_subsets = dict()
    genes_per_cluster = dict()
    for (tf, target), val in input_grn.items():
        target_cluster = clustering_dict[target]
        if target_cluster in grn_subsets:
            grn_subsets[target_cluster].update({(tf, target): val})
            genes_per_cluster[target_cluster].add(target)
        else:
            grn_subsets[target_cluster] = {(tf, target): val}
            genes_per_cluster[target_cluster] = {target}
    genes_per_cluster = {cluster : list(genes) for cluster, genes in genes_per_cluster.items()}
    return grn_subsets, genes_per_cluster

def invert_tf_to_cluster_dict(tf_representatives : list[str],
                              gene_to_cluster : dict[str, int]):
    """
    Computes per cluster list of corresponding TFs names.
    Args:
        tf_representatives: List of TF names as strings.
        gene_to_cluster: Whole clustering of all input genes.

    Returns:
        Dict with cluster ID as key and list of TF names as values.
    """
    cluster_to_tf = dict()
    # Subset and invert gene-cluster dict to only contain TFs.
    for gene, cluster in gene_to_cluster.items():
        if gene in tf_representatives:
            if not cluster in cluster_to_tf:
                cluster_to_tf[cluster] = []
            cluster_to_tf[cluster].append(gene)
    return cluster_to_tf

def subset_tf_matrix(tf_matrix : np.ndarray,
                      tf_gene_names : list[str],
                      tf_subset_names : list[str]):
    tf_subset_indices = [index for index, tf in enumerate(tf_gene_names) if tf in tf_subset_names]
    tf_subset_matrix = tf_matrix[:, tf_subset_indices]
    tf_subset_names = [tf for tf in tf_gene_names if tf in tf_subset_names]
    return tf_subset_matrix, tf_subset_names

def count_helper(
        shuffled_grn: pd.DataFrame,
        partial_input_grn: dict[str, dict[str, float]],
        tf_to_cluster: dict[str, int],
        scale_for_tf_sampling : bool
) -> None:
    """
    Computes empirical counts for all edges in input GRN based on given decoy edges.
    Args:
        shuffled_grn: Decoy edges based on shuffled expression matrix.
        partial_input_grn: "Groundtruth" edges from input GRN.
        gene_to_clust: Clustering of genes, with gene names as keys and cluster IDs as values.

    Returns: None. Partial_input_GRN is updated in-place.
    """

    # {id of cluster of TF: importance}
    # Note that each TF-cluster-ID (i.e. key in the following dict) still only occurs exactly once, since either
    # TFs have not been clustered (i.e. each TF has one dummy cluster) or if TFs have been clustered, then
    # each TF cluster has exactly one representative and hence can only appear once in shuffled output GRN.
    shuffled_grn_tf_cluster_to_importance = {
        tf_to_cluster[tf]: imp for tf, imp in zip(shuffled_grn['TF'], shuffled_grn['importance'])
    }

    for (tf, _), val in partial_input_grn.items():
        if tf_to_cluster[tf] in shuffled_grn_tf_cluster_to_importance:
            importance_input = val['importance']
            importance_shuffled = shuffled_grn_tf_cluster_to_importance[tf_to_cluster[tf]]
            if scale_for_tf_sampling:
                val['shuffled_occurences'] += 1
            if importance_shuffled >= importance_input:
                val['count'] += 1

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
                   tf_names):
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

    return expression_matrix, gene_names, tf_names