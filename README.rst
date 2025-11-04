.. image:: img/arboreto.png
    :alt: arboreto
    :scale: 100%
    :align: left

.. image:: https://travis-ci.com/aertslab/arboreto.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.com/aertslab/arboreto

.. image:: https://readthedocs.org/projects/arboreto/badge/?version=latest
    :alt: Documentation Status
    :target: http://arboreto.readthedocs.io/en/latest/?badge=latest

.. image:: https://anaconda.org/bioconda/arboreto/badges/version.svg
    :alt: Bioconda package
    :target: https://anaconda.org/bioconda/arboreto

.. image:: https://img.shields.io/pypi/v/arboreto
    :alt: PyPI package
    :target: https://pypi.org/project/arboreto/

----

.. epigraph::

    *The most satisfactory definition of man from the scientific point of view is probably Man the Tool-maker.*

.. _arboreto: https://arboreto.readthedocs.io
.. _`arboreto documentation`: https://arboreto.readthedocs.io
.. _notebooks: https://github.com/tmoerman/arboreto/tree/master/notebooks
.. _issue: https://github.com/tmoerman/arboreto/issues/new

.. _dask: https://dask.pydata.org/en/latest/
.. _`dask distributed`: https://distributed.readthedocs.io/en/latest/

.. _GENIE3: http://www.montefiore.ulg.ac.be/~huynh-thu/GENIE3.html
.. _`Random Forest`: https://en.wikipedia.org/wiki/Random_forest
.. _ExtraTrees: https://en.wikipedia.org/wiki/Random_forest#ExtraTrees
.. _`Stochastic Gradient Boosting Machine`: https://en.wikipedia.org/wiki/Gradient_boosting#Stochastic_gradient_boosting
.. _`early-stopping`: https://en.wikipedia.org/wiki/Early_stopping

Inferring a gene regulatory network (GRN) from gene expression data is a computationally expensive task, exacerbated by increasing data sizes due to advances
in high-throughput gene profiling technology.

The arboreto_ software library addresses this issue by providing a computational strategy that allows executing the class of GRN inference algorithms
exemplified by GENIE3_ [1] on hardware ranging from a single computer to a multi-node compute cluster. This class of GRN inference algorithms is defined by
a series of steps, one for each target gene in the dataset, where the most important candidates from a set of regulators are determined from a regression
model to predict a target gene's expression profile.

Members of the above class of GRN inference algorithms are attractive from a computational point of view because they are parallelizable by nature. In arboreto,
we specify the parallelizable computation as a dask_ graph [2], a data structure that represents the task schedule of a computation. A dask scheduler assigns the
tasks in a dask graph to the available computational resources. Arboreto uses the `dask distributed`_ scheduler to
spread out the computational tasks over multiple processes running on one or multiple machines.

Arboreto currently supports 2 GRN inference algorithms:

1. **GRNBoost2**: a novel and fast GRN inference algorithm using `Stochastic Gradient Boosting Machine`_ (SGBM) [3] regression with `early-stopping`_ regularization.
2. **GENIE3**: the classic GRN inference algorithm using `Random Forest`_ (RF) or ExtraTrees_ (ET) regression.

Get Started
***********

Arboreto was conceived with the working bioinformatician or data scientist in mind. We provide extensive documentation and examples to help you get up to speed with the library.

* Read the `arboreto documentation`_.
* Browse example notebooks_.
* Report an issue_.

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

BSD 3-Clause License

pySCENIC
========

.. _pySCENIC: https://github.com/aertslab/pySCENIC
.. _SCENIC: https://aertslab.org/#scenic

Arboreto is a component in pySCENIC_: a lightning-fast python implementation of
the SCENIC_ pipeline [5] (Single-Cell rEgulatory Network Inference and Clustering)
which enables biologists to infer transcription factors, gene regulatory networks
and cell types from single-cell RNA-seq data.



References
**********

1. Huynh-Thu VA, Irrthum A, Wehenkel L, Geurts P (2010) Inferring Regulatory Networks from Expression Data Using Tree-Based Methods. PLoS ONE
2. Rocklin, M. (2015). Dask: parallel computation with blocked algorithms and task scheduling. In Proceedings of the 14th Python in Science Conference (pp. 130-136).
3. Friedman, J. H. (2002). Stochastic gradient boosting. Computational Statistics & Data Analysis, 38(4), 367-378.
4. Marbach, D., Costello, J. C., Kuffner, R., Vega, N. M., Prill, R. J., Camacho, D. M., ... & Dream5 Consortium. (2012). Wisdom of crowds for robust gene network inference. Nature methods, 9(8), 796-804.
5. Aibar S, Bravo Gonzalez-Blas C, Moerman T, Wouters J, Huynh-Thu VA, Imrichova H, Kalender Atak Z, Hulselmans G, Dewaele M, Rambow F, Geurts P, Aerts J, Marine C, van den Oord J, Aerts S. SCENIC: Single-cell regulatory network inference and clustering. Nature Methods 14, 1083–1086 (2017). doi: 10.1038/nmeth.4463
