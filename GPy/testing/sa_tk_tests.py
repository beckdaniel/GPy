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

#@unittest.skip("skip")
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

    #@unittest.skip("skip")
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


#@unittest.skip("skip")
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
        for i in range(500):
            k.Kdiag(self.X)
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 2.72)


#@unittest.skip("skip")
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

    def test_Kdiag_buckets_4(self):
        k = SASST(normalize=False, _lambda=np.array([1, 0.6, 0.2]), _sigma=np.array([1.0, 0.3]),
                  lambda_buckets={'AA':1, 'B': 2}, sigma_buckets={'S': 1})
        self.assertAlmostEqual(k.Kdiag(self.X)[0], 2.39)


#@unittest.skip("skip")
class SASSTKKernelTests(unittest.TestCase):
    """
    Tests for K, on a small set of trees
    """

    def setUp(self):
        self.tree1 = '(S (AA (AA a)) (B b))'
        self.tree2 = '(S (AA (AA a)) (B c))'
        self.X1 = np.array([[self.tree1]], dtype=object)
        self.X2 = np.array([[self.tree2]], dtype=object)

    #@unittest.skip("skip")
    def test_K_1(self):
        k = SASST(normalize=False, _lambda=np.array([1.0]))
        self.assertAlmostEqual(k.K(self.X1, self.X2), 6)

    #@unittest.skip("skip")
    def test_K_2(self):
        k = SASST(normalize=False, _lambda=np.array([1.0, 1.0]), _sigma=np.array([0.2]))
        self.assertAlmostEqual(k.K(self.X1, self.X2), 2.48)

    #@unittest.skip("skip")
    def test_K_3(self):
        k = SASST(normalize=False, _lambda=np.array([1.0, 0.5]), _sigma=np.array([0.2]),
                  lambda_buckets={'AA':1})
        self.assertAlmostEqual(k.K(self.X1, self.X2), 0.96)

        
    def test_K_4(self):
        k = SASST(normalize=False, _lambda=np.array([1.0, 0.5]), _sigma=np.array([1.0, 0.2]),
                  lambda_buckets={'AA':1}, sigma_buckets={'AA':1})
        #print k.K(self.X1, self.X2)
        #print k.dlambda
        #print k.dsigma
        self.assertAlmostEqual(k.K(self.X1, self.X2), 2.2)


#@unittest.skip("skip")
class SASSTGradientTests(unittest.TestCase):
    """
    Tests for gradients.
    """

    def setUp(self):
        self.tree1 = '(S (AA (AA a)) (B b))'
        self.tree2 = '(S (AA (AA a)) (B c))'
        self.X1 = np.array([[self.tree1]], dtype=object)
        self.X2 = np.array([[self.tree2]], dtype=object)

    #@unittest.skip("skip")
    def test_grad_1(self):
        k = SASST(normalize=False, _lambda=np.array([1.0]))
        result = k.K(self.X1, self.X2)
        #print result
        #print k.dlambda
        #print k.dsigma
        self.assertAlmostEqual(result, 6)
        self.assertAlmostEqual(k.dlambda, 10)
        self.assertAlmostEqual(k.dsigma, 6)

    def test_grad_2(self):
        k = SASST(normalize=False, _lambda=np.array([1.0, 0.4]), _sigma=np.array([1.0, 0.2]),
                  lambda_buckets={'AA':1}, sigma_buckets={'AA':1})
        result = k.K(self.X1, self.X2)
        #print result
        #print k.dlambda
        #print k.dsigma
        self.assertAlmostEqual(result, 1.88)
        self.assertAlmostEqual(k.dlambda[0,0,0], 1.24)
        self.assertAlmostEqual(k.dlambda[0,0,1], 3)
        self.assertAlmostEqual(k.dsigma[0,0,0], 2.24)
        self.assertAlmostEqual(k.dsigma[0,0,1], 0.8)

#@unittest.skip("skip")
class SASSTNormTests(unittest.TestCase):
    """
    Tests for the normalized version
    """
    def setUp(self):
        self.tree1 = '(S (AA (AA a)) (B b))'
        self.tree2 = '(S (AA (AA a)) (B c))'
        self.X1 = np.array([[self.tree1]], dtype=object)
        self.X2 = np.array([[self.tree2]], dtype=object)

    def test_grad_1(self):
        k = SASST(normalize=True, _lambda=np.array([1.0]))
        result = k.K(self.X1, self.X2)

        self.assertAlmostEqual(result, 0.6)
        self.assertAlmostEqual(k.dlambda, -0.2)
        self.assertAlmostEqual(k.dsigma, 0.12)

    def test_grad_2(self):
        k = SASST(normalize=True, _lambda=np.array([1.0, 0.5]))
        result = k.K(self.X1, self.X2)

        self.assertAlmostEqual(result, 0.6)
        self.assertAlmostEqual(k.dlambda[0,0,0], -0.2)
        self.assertAlmostEqual(k.dsigma, 0.12)

    def test_grad_4(self):
        k = SASST(normalize=True, _lambda=np.array([1.0, 0.5]), _sigma=np.array([1.0, 0.4]))
        result = k.K(self.X1, self.X2)
        k2 = SST(normalize=True, _lambda=1.0)
        result2 = k2.K(self.X1, self.X2)

        self.assertAlmostEqual(result, result2)
        self.assertAlmostEqual(k.dlambda[0,0,0], k2.dlambda)
        self.assertAlmostEqual(k.dsigma[0,0,0], k2.dsigma)
    
    @unittest.skip("need to do the maths for this test")
    def test_grad_3(self):
        k = SASST(normalize=True, _lambda=np.array([1.0, 0.4]), _sigma=np.array([1.0, 0.2]),
                  lambda_buckets={'AA':1}, sigma_buckets={'AA':1})
        result = k.K(self.X1, self.X2)
        #print result
        #print k.dlambda
        #print k.dsigma
        self.assertAlmostEqual(result, 1.88)
        self.assertAlmostEqual(k.dlambda[0], 1.24)
        self.assertAlmostEqual(k.dlambda[1], 3)
        self.assertAlmostEqual(k.dsigma[0], 2.24)
        self.assertAlmostEqual(k.dsigma[1], 0.8)

#@unittest.skip("skip")
class SASSTIntegrationTests(unittest.TestCase):
    """
    Tests for integration into GPs.
    """
    def setUp(self):
        self.X = np.array([['(S (NP ns) (VP v))'],
                           ['(S (NP n) (VP v))'],
                           ['(S (NP (N a)) (VP (V c)))'],
                           ['(S (NP (Det a) (N b)) (VP (V c)))'],
                           ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                          dtype=object)
        self.Y = np.array([[(a+10)*5] for a in range(5)])

    def test_integration_grad_1(self):
        k = SASST(normalize=True, _lambda=np.array([1.0, 0.5]), _sigma=np.array([1.0, 0.4]))
        result = k.K(self.X)
        k2 = SST(normalize=True, _lambda=1.0)
        result2 = k2.K(self.X)
        #print result
        #print result2

        self.assertTrue((result == result2).all())
        #self.assertAlmostEqual(k.dlambda[0], k2.dlambda)
        #self.assertAlmostEqual(k.dsigma[0], k2.dsigma)

    def test_integration_1(self):
        k = SASST(_lambda=np.array([1.0]))
        m = GPy.models.GPRegression(self.X, self.Y, kernel=k)
        m.constrain_positive('')
        #print m
        m.optimize(messages=False)
        #print m

    def test_integration_2(self):
        k = SASST(normalize=False, _lambda=np.array([1.0, 0.5]))
        m = GPy.models.GPRegression(self.X, self.Y, kernel=k)
        m.constrain_positive('')
        #print m
        m.optimize(messages=False)
        #print m
        #print m['sasstk.lambda']
        k2 = SST(normalize=False, _lambda=1.0)
        m2 = GPy.models.GPRegression(self.X, self.Y, kernel=k2)
        m2.constrain_positive('')
        #print m2
        m2.optimize(messages=False)
        #print m2
        self.assertAlmostEqual(m['sasstk.lambda'][0], m2['sstk.lambda'])

    @unittest.skip("skip")
    def test_integration_3(self):
        k = SASST(_lambda=np.array([1.0, 0.5]), _sigma=np.array([1.0, 0.4]),
                  lambda_buckets={'NP':1, 'N':1}, sigma_buckets={'NP':1, 'N':1})
        m = GPy.models.GPRegression(self.X, self.Y, kernel=k)
        m.constrain_positive('')
        print m
        print m.predict(self.X)
        m.optimize(messages=False)
        print m
        print m['sasstk.lambda']
        print m['sasstk.sigma']
        print m.predict(self.X)
        #k2 = SST(_lambda=1.0)
        #m2 = GPy.models.GPRegression(self.X, self.Y, kernel=k2)
        #m2.constrain_positive('')
        #print m
        #m2.optimize(messages=False)
        #print m
        #self.assertAlmostEqual(m['sasstk.lambda'][0], m2['sstk.lambda'])

#@unittest.skip("skip")
class SASSTKProfilingTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([['(S (NP ns) (VP v))'],
                           ['(S (NP n) (VP v))'],
                           ['(S (NP (N a)) (VP (V c)))'],
                           ['(S (NP (Det a) (N b)) (VP (V c)))'],
                           ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                          dtype=object)
        self.Y = np.array([[(a+10)*5] for a in range(5)])

    def test_prof_1(self):
        k = SASST(normalize=False, _lambda=np.array([1.0]), num_threads=8)
        start_time = datetime.datetime.now()
        for i in range(100):
            k.K(self.X)
        end_time = datetime.datetime.now()
        print end_time - start_time


class SymbolsDictTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([['(S (NP ns) (VP v))'],
                           ['(S (NP n) (VP v))'],
                           ['(S (NP (N a)) (VP (V c)))'],
                           ['(S (NP (Det a) (N b)) (VP (V c)))'],
                           ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                          dtype=object)

    def test_get_symbols_dict_1(self):
        expected = {'ADJ': 1,
                    'ADV': 2,
                    'Det': 3,
                    'N': 4,
                    'NP': 5,
                    'S': 6,
                    'V': 7,
                    'VP': 8}
        result = SASST().get_symbols_dict(self.X)
        self.assertEqual(result, expected)

    def test_get_symbols_dict_2(self):
        expected = {'NP': 1,
                    'S': 2,
                    'VP': 3}
        result = SASST().get_symbols_dict(self.X, no_pos=True)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
