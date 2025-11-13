"""
Tests for signifikante.algo.
"""

from unittest import TestCase, skip
import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix
from distributed import Client, LocalCluster
from os.path import join

from signifikante.algo import _prepare_input, _prepare_client
from signifikante.algo import grnboost2, genie3, extra_trees, xgboost, lasso
from signifikante.utils import *


class PrepareClientTest(TestCase):

    def test_None(self):
        client, shutdown_callback = _prepare_client(None)

        self.assertIn('127.0.0.1', client.scheduler.address)

        shutdown_callback()

    def test_local(self):
        client, shutdown_callback = _prepare_client('local')

        self.assertIn('127.0.0.1', client.scheduler.address)

        shutdown_callback()

    def test_client(self):
        lc = LocalCluster(diagnostics_port=None)
        passed = Client(lc)

        client, shutdown_callback = _prepare_client(passed)

        self.assertEqual(client, passed)

        shutdown_callback()
        passed.close()
        lc.close()
        lc.status

        self.assertEqual(lc.status.value, 'closed')

    def test_address(self):
        with self.assertRaises(Exception) as context:
            address = 'tcp://127.0.0.2:12345'
            _prepare_client(address)

        self.assertIn('Timed out trying to connect to tcp://127.0.0.2:12345', str(context.exception))

    def test_other(self):
        with self.assertRaises(Exception) as context:
            _prepare_client(666)

        self.assertIn('Invalid client specified', str(context.exception))


zeisel_small_path = 'tests/sparse/zeisel_small.tsv'
zeisel_tfs_path = 'tests/sparse/zeisel_tfs.txt'

df = pd.read_csv(zeisel_small_path, sep='\t')
tfs = load_tf_names(zeisel_tfs_path)


class PrepareInputTest(TestCase):

    def test_DataFrame(self):
        expression_matrix, gene_names, tf_names, target_names = _prepare_input(
            expression_data=df,
            gene_names=None,
            tf_names=tfs,
            target_names='all'
        )

        self.assertTrue(isinstance(expression_matrix, np.ndarray))
        self.assertEqual((500, 50), expression_matrix.shape)
        self.assertEqual(50, len(gene_names))
        self.assertEqual(4, len(tf_names))
        self.assertEqual(50, len(target_names))

    def test_numpy_dense_matrix(self):
        expression_matrix, gene_names, tf_names, target_names = _prepare_input(
            expression_data=df.to_numpy(),
            gene_names=list(df.columns),
            tf_names=tfs,
            target_names='all'
        )

        self.assertTrue(isinstance(expression_matrix, np.ndarray))
        self.assertEqual((500, 50), expression_matrix.shape)
        self.assertEqual(50, len(gene_names))
        self.assertEqual(4, len(tf_names))
        self.assertEqual(50, len(target_names))

    def test_scipy_csc_matrix(self):
        csc = csc_matrix(df.to_numpy())

        expression_matrix, gene_names, tf_names, target_names = _prepare_input(
            expression_data=csc,
            gene_names=list(df.columns),
            tf_names=tfs,
            target_names='all'
        )

        self.assertTrue(isinstance(expression_matrix, csc_matrix))
        self.assertEqual((500, 50), expression_matrix.shape)
        self.assertEqual(50, len(gene_names))
        self.assertEqual(4, len(tf_names))
        self.assertEqual(50, len(target_names))


class LaunchTests(TestCase):

    def test_launch_grnboost2(self):
        network_df = grnboost2(df, tf_names=tfs)

        self.assertGreater(len(network_df), 100)

    def test_launch_genie3(self):
        network_df = genie3(df, tf_names=tfs)

        self.assertGreater(len(network_df), 100)
        
    def test_launch_extra_trees(self):
        network_df = extra_trees(df, tf_names=tfs)
        self.assertGreater(len(network_df), 100)
        
    def test_launch_xgboost(self):
        network_df = xgboost(df, tf_names=tfs)
        self.assertGreater(len(network_df), 100)
        
    def test_launch_xgboost(self):
        network_df = lasso(df, tf_names=tfs, verbose=True)
        self.assertGreater(len(network_df), 100)
        
    def test_launch_lasso_with_local_client(self):
        lc = LocalCluster(diagnostics_port=None)
        passed = Client(lc)
        network_df = lasso(df, tf_names=tfs, client_or_address=passed)
        self.assertGreater(len(network_df), 100) 
    
