# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
import nltk # Need to cope with this in a better way...
#import sympy as sp
from GPy.kern import SubsetTreeKernel as SST
from GPy.kern import SymbolAwareSubsetTreeKernel as SASST
#from GPy.kern import PySubsetTreeKernel as PySST
#from GPy.kern import OldSubsetTreeKernel as OldSST
import sys
import datetime

np.set_printoptions(suppress=True)

class SASSTKParallelCheckingTests(unittest.TestCase):
    """
    Tests for the symbol-aware SSTK Parallel version.
    """
    def test_gen_node_list1(self):
        tree = '(S (NP ns) (VP v))'
        k = SASST()
        result = [('NP ns', None), ('S NP VP', [0, 2]), ('VP v', None)]
        self.assertEqual(k.kernel._gen_node_list(tree), result)

    def test_gen_node_list2(self):
        tree = '(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))'
        k = SASST()
        result = [('ADJ colorless', None), ('ADV furiously', None), ('N ideas', None), ('NP ADJ N', [0, 2]), ('S NP VP', [3, 6]), ('V sleep', None), ('VP V ADV', [5, 1])]
        self.assertEqual(k.kernel._gen_node_list(tree), result)


    def test_when_leaves_are_trees_1(self):
        tree = '(NN (appos (NNP (poss Nicole (POS (possessive \'s)))) dad))'
        #print nltk.tree.Tree.fromstring(tree)
        k = SASST()
        #print k.kernel._gen_node_list(tree)

    def test_when_leaves_are_trees_2(self):
        tree = ' (SEG (SENT (NN (null (NNP (nsubj (NNP (nn (Barack))) (Obama))) (VBZ (cop (becomes))) (DT (det (the))) (JJ (amod (fourth))) (JJ (amod (American))) (president) (VB (infmod (TO (aux (to))) (receive) (NNP (dobj (DT (det (the))) (NNP (nn (Nobel))) (NNP (nn (Peace))) (Prize))))))))) '
        #print nltk.tree.Tree.fromstring(tree)
        k = SASST()
        #print k.kernel._gen_node_list(tree)

    @unittest.skip("skip")
    def test_when_leaves_are_trees_3(self):
        tree = '(NN (appos (NNP (poss Nicole (POS (possessive \'s)))) dad (VP is cool)))'
        tree2 = '(NN (appos (NNP (poss Nicole (POS (possessive \'s)))) mom (VP is cool)))'
        k = SASST()
        X = np.array([[tree], [tree2]], dtype=object)
        #print k.K(X)

    def test_when_leaves_are_trees_4(self):
        tree = '(NN (appos (NNP (poss Nicole (POS (possessive \'s)))) dad (VP is cool)))'
        tree2 = '(NN (appos (NNP (poss Nicole (POS (possessive \'s)))) mom (VP is cool)))'
        k = SASST(normalize=False)
        X = np.array([[tree], [tree2]], dtype=object)
        #print "Kdiag:",
        #print k.Kdiag(X)


class SASSTKDiagSmallTests(unittest.TestCase):
    """
    Tests for KDiag only, using a small tree.
    """

    def setUp(self):
        self.tree = '(S (AA a) (B b))'
        self.X = np.array([[self.tree]], dtype=object)

    def test_Kdiag_1(self):
        k = SASST(normalize=False, _lambda=np.array([1.0]))
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 6)

    def test_Kdiag_2(self):
        k = SASST(normalize=False, _lambda=np.array([1.0, 0.5]), _sigma=np.array([0.0, 1.0]))
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 3)

    def test_Kdiag_3(self):
        k = SASST(normalize=False, _lambda=np.array([1.0, 0.5]), _sigma=np.array([0.2, 1.0]))
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 3.44)

    def test_Kdiag_4(self):
        k = SASST(normalize=False, _lambda=np.array([0.6, 0.5]), _sigma=np.array([1.0, 1.0]))
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 2.736)

    def test_Kdiag_5(self):
        k = SASST(normalize=False, _lambda=np.array([0.2, 0.5]), _sigma=np.array([1.0, 1.0]))
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 0.688)

    def test_Kdiag_buckets_1(self):
        k = SASST(normalize=False, _lambda=np.array([1, 0.6]), _sigma=np.array([1.0, 1.0]),
                  lambda_buckets={'AA':1})
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 4.8)

    def test_Kdiag_buckets_2(self):
        k = SASST(normalize=False, _lambda=np.array([1, 0.6, 0.2]), _sigma=np.array([1.0]),
                  lambda_buckets={'AA':1, 'B': 2})
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 2.72)

    def test_Kdiag_buckets_3(self):
        k = SASST(normalize=False, _lambda=np.array([1, 0.6, 0.2, 1]), _sigma=np.array([1.0]),
                  lambda_buckets={'AA':1, 'B': 2, 'S': 3})
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 2.72)


class SASSTKDiagSmallSigmaTests(unittest.TestCase):
    """
    Tests for KDiag only for sigma_buckets, using a small tree.
    """

    def setUp(self):
        self.tree = '(S (AA (AA a)) (B b))'
        self.X = np.array([[self.tree]], dtype=object)

    def test_Kdiag_1(self):
        k = SASST(normalize=False, _lambda=np.array([1.0]))
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 10)

    def test_Kdiag_2(self):
        k = SASST(normalize=False, _lambda=np.array([1.0, 0.5]), _sigma=np.array([0.0, 1.0]))
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 4)

    def test_Kdiag_3(self):
        k = SASST(normalize=False, _lambda=np.array([1.0]), _sigma=np.array([0.2]))
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 4.88)

    def test_Kdiag_4(self):
        k = SASST(normalize=False, _lambda=np.array([0.6, 0.5]), _sigma=np.array([1.0, 1.0]))
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 4.0416)

    def test_Kdiag_buckets_1(self):
        k = SASST(normalize=False, _lambda=np.array([1, 0.6]), _sigma=np.array([1.0, 1.0]),
                  lambda_buckets={'AA':1})
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 6.48)

    def test_Kdiag_buckets_2(self):
        k = SASST(normalize=False, _lambda=np.array([1, 0.6, 0.2]), _sigma=np.array([1.0]),
                  lambda_buckets={'AA':1, 'B': 2})
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 4.112)

    def test_Kdiag_buckets_3(self):
        k = SASST(normalize=False, _lambda=np.array([1, 0.6, 0.2, 1]), _sigma=np.array([1.0]),
                  lambda_buckets={'AA':1, 'B': 2, 'S': 3})
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 4.112)




if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
