SignifiKANTE
============

.. _arboreto: https://arboreto.readthedocs.io

SignifiKANTE builds upon the arboreto_ software library to enable regression-based gene regulatory network inference and efficient, permutation-based empirical *P*-value computation for predicted regulatory links. Our method is described in detail on bioRxiv (TODO).

.. contents::
   :local:
   :depth: 2


Installation
************

SignifiKANTE is installable via pip from PyPI using

.. code-block:: bash

    pip install signifikante

or locally from this repository with

.. code-block:: bash

    git clone git@github.com:bionetslab/SignifiKANTE.git
    cd SignifiKANTE
    pip install -e .

For installation with pixi, download `pixi <https://pixi.sh/dev/installation/>`_, install and run 

.. code-block:: bash

    git clone git@github.com:bionetslab/SignifiKANTE.git
    cd SignifiKANTE
    pixi install

Create a jupyter kernel using pixi.toml/pyproject.toml, which will install a jupyter kernel using a custom environment (including ipython)

.. code-block:: bash

    git clone git@github.com:bionetslab/SignifiKANTE.git
    cd SignifiKANTE
    pixi run -e kernel install-kernel


Example workflow of SignifiKANTE's FDR control
**********************************************

We provide an efficient FDR control for regulatory links based on any given regression-based GRN inference method. Currently, for GRN inference SignifiKANTE includes GRNBoost2, GENIE3, xgboost, and lasso regression. For the integration of further regression-based GRN inference methods, please see our manual in the section below. Here, we also provide a minimal working example of how to use SignifiKANTE based on GRNBoost2 on a simulated dataset:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from signifikante.algo import signifikante_fdr
    
    if __name__ == "__main__":
    
        # Simulate expression dataset with 100 samples and 10 genes.
        expression_data = np.random.randn(100, 10)
        expression_df = pd.DataFrame(expression_data, columns=[f"Gene{i}" for i in range(10)])
        # Simulate three artificial TFs.
        tf_list = [f"Gene{i}" for i in range(3)]
    
        # Run SignifiKANTE's approximate FDR control.
        fdr_grn = signifikante_fdr(
                    expression_data=expression_df,
                    normalize_gene_expression=True,
                    tf_names=tf_list,
                    cluster_representative_mode="random",
                    num_target_clusters=2,
                    inference_mode="grnboost2",
                    apply_bh_correction=True)
        print(fdr_grn)


Parameter descriptions
**********************

Below, you can find a more detailed description of the parameters of SignifiKANTE's central function for FDR control :code:`signifikante_fdr`. The two absolutely necessary input parameters are:

- :code:`expression_data [pd.DataFrame]`: Expression matrix with genes as columns and samples as rows.
- :code:`cluster_representative_mode [str]`: How to draw representatives from target gene clusters. Can be one of "random" or "medoid" for approximate P-value computation, or "all_genes" for exact (DIANE-like) P-values.

Additional parameters of SignifiKANTE's FDR control:

- :code:`normalize_gene_expression [bool]` :  Whether or not to apply z-score normalization on gene columns in input expression matrix.


- :code:`inference_mode [str]`: Which GRN inference method to use under the hood. Can be one of "grnboost2", "genie3", "xgboost", and "lasso". Defaults to "grnboost2".
- :code:`num_permutations [int]`: How many permutations to perform for random background model for empirical P-value computation. Defaults to 1000.
- :code:`tf_names [list]`: List of strings representing TF names. Should be subset of gene names contained in :code:`expression_data`. Defaults to None. If no list is given, all genes are treated as potential TFs.
- :code:`apply_bh_correction [bool]`: Whether or not to additionally return Benjamini-Hochberg adjusted P-values.
- :code:`input_grn [pd.DataFrame]`: Reference GRN to use for FDR control. Needs to possess columns 'TF', 'target', 'importance'. Should only be used, when it is clear that this GRN is inferred using the same method indicated in :code:`inference_mode`. Defaults to None. If no reference GRN is given, a new one is inferred in the beginning.
- :code:`target_subset [list]`: Subset of target genes to consider for FDR control. Only compatible with "all_genes" FDR mode.
- :code:`num_target_clusters [int]`: Number of target gene clusters. If set to -1, no target gene clustering will be applied. Defaults to -1.
- :code:`num_tf_clusters [int]`: Experimental feature. Used for setting the number of desired TF clusters, if set to -1, no TF clustering will be applied. Defaults to -1.
- :code:`target_cluster_mode [str]`: Experimental feature. Indicates, which clustering to use for target gene clustering. Defaults to "wasserstein".
- :code:`tf_cluster_mode [str]`: Experimental feature. Indicates, which clustering mode to use for TF clustering. Defaults to "correlation".
- :code:`scale_for_tf_sampling [bool]`: Experimental feature. Whether or not to keep track of occurences of edges in permuted GRNs. Defaults to False.

Further more technical parameters:

- :code:`client [str,Dask.Client]`: Whether to perform computation on given input Dask Cluster object, or to create a new local one ("local"). Defaults to "local".
- :code:`early_stop_window_length [int]`: Window length to use for early stopping. Defaults to 25.
- :code:`seed [int]`: Random seed for regressor models. Defaults to None.
- :code:`verbose [bool]`: Whether or not to print detailed additional information. Defaults to False.
- :code:`output_dir [str]`: Where to save additional intermediate data to. Defaults to None, i.e. saves no intermediate results.

The function returns a pandas dataframe representing the reference GRN with columns 'TF', 'target', and 'importance'. The column 'pvalue' stores empirical P-values per edge. If :code:`apply_bh_correction=True`, an additional column 'pvalue_bh' is returned.




Integration of additional regression-based GRN inference methods
****************************************************************

In order to integrate new regression-based GRN inference methods into SignifiKANTE, simply use the following steps, which exemplify the integration of lasso regression as implemented in the `GRENADINE <https://pypi.org/project/grenadine/>`_ package:

1. Give your regression-based method an abbreviated string-based name (:code:`regressor_type`) and name the variable storing its model-specific parameters (:code:`regressor_args`), then add those to the existing accepted values of the :code:`inference_mode` parameter within the function :code:`signifikante_fdr` in the file :code:`algo.py`, directly below the indicated line stating :code:`UPDATE FOR NEW GRN METHOD`. In the case of lasso regression, we simply added the regressor type "LASSO" and the regressor parameters stored in :code:`LASSO_KWARGS` in the respective code block:

.. code-block:: python

    # UPDATE FOR NEW GRN METHOD
    if inference_mode == "grnboost2":
        regressor_type = "GBM"
        regressor_args = SGBM_KWARGS
    # other existing methods...
    elif inference_mode == "lasso":
        regressor_type = "LASSO"
        regressor_args = LASSO_KWARGS

Since the actual parameters of :code:`LASSO_KWARGS` will be defined in another file, you need to make sure to import the variable into :code:`algo.py`. To achieve this, simply add your new regressor's arguments variable at the top of :code:`algo.py`, directly below the indicated line stating :code:`UPDATE FOR NEW GRN METHOD`, just like this:

.. code-block:: python

    # UPDATE FOR NEW GRN METHOD
    from signifikante.core import (
        create_graph, SGBM_KWARGS, RF_KWARGS, EARLY_STOP_WINDOW_LENGTH, ET_KWARGS, XGB_KWARGS, LASSO_KWARGS
    )

2. Now we switch to the file :code:`core.py`. At the top of the file, add any required import-statements for your regression to work (e.g. imports of sklearn). Below import statements, create a dictionary named exactly like the regressor's arguments variable you imported in :code:`algo.py`. You can include it directly below the line stating :code:`# UPDATE FOR NEW GRN METHOD`, analogously to how we did it for the lasso regression:

.. code-block:: python

    from sklearn.linear_model import Lasso
    # ... other code in between
    LASSO_KWARGS = {
    'alpha' : 0.01
    }

The actual logic of your new regression-based inference method will be implemented in the function :code:`fit_model`. There, you should implement a new local function that contains the logic of your new model, given a :code:`tf_matrix` and a :code:`target_gene_expression` vector, such as we did for lasso regression:

.. code-block:: python

    def do_lasso_regression():
        regressor = Lasso(**regressor_kwargs, random_state=seed)
        regressor.fit(tf_matrix, target_gene_expression)
        return regressor

Directly below, add another case distinction for your :code:`regressor_type` which calls your locally defined function. The exact position is indicated by the line stating :code:`# UPDATE FOR NEW GRN METHOD`:

.. code-block:: python

    # UPDATE FOR NEW GRN METHOD
    if is_sklearn_regressor(regressor_type):
        return do_sklearn_regression()
    # other methods...
    elif is_lasso_regressor(regressor_type):
        return do_lasso_regression()

Finally, in the function :code:`to_feature_importances`, you have to implement the extraction of feature importances or model coefficients from your :code:`trained_regressor`, which are supposed to represent edge weights in the inferred GRN. To accomplish that, add another case for your new regressor in the case distinction below the line stating :code:`# UPDATE FOR NEW GRN METHOD`. For lasso regression this looks like:

.. code-block:: python

    # UPDATE FOR NEW GRN METHOD
    if is_oob_heuristic_supported(regressor_type, regressor_kwargs):
        # other code...
    elif regressor_type.upper() == "LASSO":
        scores = np.abs(trained_regressor.coef_)
        return scores

Done, you have successfully added your new desired regression method for GRN inference!

Unit tests
**********

Unit tests for arboreto-based functionalities, as well as additional tests for SignifiKANTE's FDR control functionality and a comparison of our efficiently parallelized Wasserstein-distance computation against SciPy can be found under :code:`tests/`. The tests are based on Python's unittest module, and can be run all-together from the repository's root-directory with

.. code-block:: bash

    python -m unittest discover -s tests -v

    
