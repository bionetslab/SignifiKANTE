SignifiKANTE
***********

.. _arboreto: https://arboreto.readthedocs.io

SignifiKANTE builds upon the arboreto_ software library to enable regression-based gene regulatory network inference and fast, permutation-based empirical *P*-value computation for predicted regulatory links.

Quick install
************

The tool is installable via pip and pixi

.. code-block:: bash

    git clone git@github.com:bionetslab/arboreto_fdr.git
    cd arboreto_fdr
    pip install -e .

To create a pixi environment, download pixi from pixi.sh, install and run 

.. code-block:: bash

    git clone git@github.com:bionetslab/arboreto_fdr.git
    cd arboreto_fdr
    pixi install

Create jupyter kernel using pixi.toml/pyproject.toml, which will install a jupyter kernel using a custom environment (including ipython)

.. code-block:: bash

    git clone git@github.com:bionetslab/arboreto_fdr.git
    cd arboreto_fdr
    pixi run -e kernel install-kernel

FDR control
*******

We provide an efficient FDR control implementation based on GRNBoost2, which computes empirical P-values for each edge in a given or to-be-inferred GRN. Our implementation offers both a full and a (more efficient) approximate way of P-value computation. An example call to our FDR control includes the following steps:

.. code-block:: python

    import pandas as pd
    from arboreto.algo import grnboost2_fdr

    # Load expression matrix - in this case simulate one.
    exp_data = np.random.randn(100, 10)
    exp_df = pd.DataFrame(data, columns=columns)

    # Run approximate FDR control.
    fdr_grn = grnboost2_fdr(
                expression_data=exp_df,
                cluster_representative_mode="random",
                num_target_clusters=5,
                num_tf_clusters=-1
            )

A more detailed description of all parameters of the `grnboost2_fdr` function can be found in the respective docstring.

License
*******
This project is licensed under the GNU General Public `LICENSE <./LICENSE>`_ v3.0.


References
**********
