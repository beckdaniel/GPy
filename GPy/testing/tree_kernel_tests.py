# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
import nltk # Need to cope with this in a better way...
import sympy as sp
from GPy.kern import TreeKernel
from GPy.kern import SubsetTreeKernel as SST
from GPy.kern import PySubsetTreeKernel as PySST
import sys
import datetime

class BasicTreeKernelTests(unittest.TestCase):
    """
    Includes basic tests for Tree Kernels, mock tests and
    naive version tests.
    """

    def test_treekernel_params1(self):
        tk = GPy.kern.TreeKernel()
        tk._set_params(np.array([1, 1]))
        self.assertTrue((tk.parts[0]._get_params() == np.array([1, 1])).all())

    def test_treekernel_params2(self):
        tk = GPy.kern.TreeKernel()
        tk._set_params(np.array([1, 0]))
        self.assertTrue((tk._get_params() == np.array([1, 0])).all())

    def test_treekernel_params3(self):
        tk = GPy.kern.TreeKernel()
        self.assertTrue(tk._get_param_names() == ['tk_decay', 'tk_branch'])
        
    def test_treekernel_mock1(self):
        tk = GPy.kern.TreeKernel(mode="mock")
        X1 = np.array([[nltk.Tree("(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))")]],
                      dtype=nltk.Tree)
        target = tk.K(X1, X1)
        self.assertEqual(target[0], [16])

    def test_treekernel_mock2(self):
        tk = GPy.kern.TreeKernel(mode="mock")
        X = np.array([['a' * i] for i in range(10)], dtype=str)
        Y = np.array([[a] for a in range(10)])
        m = GPy.models.GPRegression(X,Y,kernel=tk)
        m['noise'] = 1

    def test_treekernel_k1(self):
        tk = GPy.kern.TreeKernel()
        tk._set_params([1,0])
        t = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        X1 = np.array([[t]], dtype=object)
        target = tk.K(X1, X1)
        self.assertEqual(target[0], [7])

    def test_treekernel_k2(self):
        tk = GPy.kern.TreeKernel()
        tk._set_params([1,1])
        t = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        X1 = np.array([[t]], dtype=object)
        target = tk.K(X1, X1)
        self.assertEqual(target[0], [37])

    def test_treekernel_delta1(self):
        node1 = 'test'
        node2 = 'test'
        tk = TreeKernel().parts[0]
        self.assertTrue(tk.delta_naive(node1,node2) == 0)

    def test_treekernel_delta2(self):
        node1 = nltk.Tree('(S NP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel().parts[0]
        self.assertTrue(tk.delta_naive(node1,node2) == 0)

    def test_treekernel_delta3(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel().parts[0]
        self.assertTrue(tk.delta_naive(node1,node2) == 1)

    def test_treekernel_delta4(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel(decay=0.5).parts[0]
        self.assertTrue(tk.delta_naive(node1,node2) == 0.5)

    def test_treekernel_delta5(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel().parts[0]
        self.assertTrue(tk.delta_naive(node1,node2) == 4)

    def test_treekernel_delta6(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(branch=0.5).parts[0]
        self.assertTrue(tk.delta_naive(node1,node2) == 2.25)
    
    def test_treekernel_delta7(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5).parts[0]
        self.assertTrue(tk.delta_naive(node1,node2) == 1.125)

    def test_treekernel_delta8(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5, branch=0.5).parts[0]
        self.assertTrue(tk.delta_naive(node1,node2) == 0.5)

    def test_treekernel_delta9(self):
        node1 = nltk.Tree('(S (NP NS) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel().parts[0]
        self.assertTrue(tk.delta_naive(node1,node2) == 2)

    def test_treekernel_deltaparams1(self):
        node1 = 'test'
        node2 = 'test'
        tk = TreeKernel().parts[0]
        self.assertEqual(tk.delta_params_naive(node1,node2), (0,0))

    def test_treekernel_deltaparams2(self):
        node1 = nltk.Tree('(S NP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel().parts[0]
        self.assertEqual(tk.delta_params_naive(node1,node2), (0,0))

    def test_treekernel_deltaparams3(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel().parts[0]
        self.assertEqual(tk.delta_params_naive(node1,node2), (1,0))

    def test_treekernel_deltaparams4(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel(decay=0.5).parts[0]
        self.assertEqual(tk.delta_params_naive(node1,node2), (1,0))

    def test_treekernel_deltaparams5(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel().parts[0]
        self.assertEqual(tk.delta_params_naive(node1,node2), (8,4))

    def test_treekernel_deltaparams6(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(branch=0.5).parts[0]
        self.assertEqual(tk.delta_params_naive(node1,node2), (5.25, 3))
    
    def test_treekernel_deltaparams7(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5).parts[0]
        self.assertEqual(tk.delta_params_naive(node1,node2), (3.75, 1.5))

    def test_treekernel_deltaparams8(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5, branch=0.5).parts[0]
        self.assertEqual(tk.delta_params_naive(node1,node2), (2, 1))

    def test_treekernel_deltaparams9(self):
        node1 = nltk.Tree('(S (NP NS) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel().parts[0]
        self.assertEqual(tk.delta_params_naive(node1,node2), (3, 3))

    def test_treekernel_kernel1(self):
        tk = GPy.kern.TreeKernel()
        X = np.array([['(S (NP a) (VP v))'],
                      ['(S (NP a1) (VP v))'],
                      ['(S (NP (NP a)) (VP (V c)))'],
                      ['(S (VP v2))']],
                     dtype=object)
        k = tk.K(X, X)
        result = np.array([[6,3,2,0],
                           [3,6,1,0],
                           [2,1,15,0],
                           [0,0,0,3]])
        self.assertTrue((k == result).all())

    def test_treekernel_kernel3(self):
        tk = GPy.kern.TreeKernel()
        X = np.array([['(S (NP a) (VP v))']], dtype=object)
        X2 = np.array([['(S (NP (NP a)) (VP (V c)))']], dtype=object)
        k = tk.dK_dtheta(1, X, X2)
        self.assertTrue((k == [2,2]).all())

    def test_treekernel_kernel4(self):
        tk = GPy.kern.TreeKernel()
        X = np.array([['(S (NP a) (VP v))']], dtype=object)
        X2 = np.array([['(S (NP (NP a)) (VP (V c)))']], dtype=object)
        k = tk.K(X, X2)
        self.assertTrue((k == 2))

    def test_treekernel_kernel6(self):
        tk = GPy.kern.TreeKernel(mode="cache")
        X = np.array([['(S (NP a) (VP v))'],
                      ['(S (NP a1) (VP v))'],
                      ['(S (NP (NP a)) (VP (V c)))'],
                      ['(S (VP v2))']],
                     dtype=object)
        k = tk.K(X, X)
        result = np.array([[6,3,2,0],
                           [3,6,1,0],
                           [2,1,15,0],
                           [0,0,0,3]])
        self.assertTrue((k == result).all())


class GradientTreeKernelTests(unittest.TestCase):
    """
    A set of tests that compare results of dK_dtheta with
    numerical approximations.
    """
    def test_treekernel_grad1(self):
        tk = GPy.kern.TreeKernel(mode="naive")
        X = np.array([['(S (NP a) (VP v))']], dtype=object)
        X2 = np.array([['(S (NP (NP a)) (VP (V c)))']], dtype=object)

        h = 0.00001
        #target = [0,0]
        #print tk._get_params()
        tk2 = TreeKernel(mode="naive")
        #print tk2.delta_params_naive(nltk.Tree(X[0][0]), nltk.Tree(X2[0][0]))
        dk_dt = tk.dK_dtheta(1, X, X2)

        tk._set_params([1,1-h])
        k_b1 = tk.K(X, X2)
        tk._set_params([1,1+h])
        k_b2 = tk.K(X, X2)

        tk._set_params([1-h,1])
        k_d1 = tk.K(X, X2)
        tk._set_params([1+h,1])
        k_d2 = tk.K(X, X2)

        #print k_d1
        #print k_d2
        #print k_b1
        #print k_b2

        tk._set_params([1,1])
        #print tk.K(X, X2)
        
        approx = [np.sum((k_d2 - k_d1) / (2 * h)), np.sum((k_b2 - k_b1) / (2 * h))]
        #print approx
        #print dk_dt
        self.assertAlmostEqual(approx[0], dk_dt[0])
        self.assertAlmostEqual(approx[1], dk_dt[1])


class CacheTreeKernelTests(unittest.TestCase):
    """
    Tests using the cached version (with DP).
    We set a fixture with both cached and naive versions
    of Tree Kernels and compare the results, which
    should be the same.
    """

    def setUp(self):
        self.tk_n = TreeKernel(mode="naive")
        self.tk_c = TreeKernel(mode="cache")
        self.tgt_n = np.array([[0.,0,0,0],
                               [0,0,0,0],
                               [0,0,0,0],
                               [0,0,0,0]])
        self.tgt_c = np.copy(self.tgt_n)
        self.tgt_hyp_n = np.array([0., 0])
        self.tgt_hyp_c = np.array([0., 0])
        self.X = np.array([['(S NP)'], ['(S NP VP)'], ['(S (NP N) (VP V))'], ['(S (NP NS) (VP V))']],dtype=object)

    def test_treekernel_deltaparams_cache1(self):
        self.tgt_n = self.tk_n.K(self.X, None)#, self.tgt_n)
        self.tgt_c = self.tk_c.K(self.X, None)#, self.tgt_c)
        self.tgt_hyp_n = self.tk_n.dK_dtheta(1, self.X, None)#, self.tgt_hyp_n)
        self.tgt_hyp_c = self.tk_c.dK_dtheta(1, self.X, None)#, self.tgt_hyp_c)
        self.assertTrue((self.tgt_n == self.tgt_c).all())
        self.assertTrue((self.tgt_hyp_n == self.tgt_hyp_c).all())

    def test_treekernel_deltaparams_cache2(self):
        self.tk_n._set_params([1,0.3])
        self.tk_c._set_params([1,0.3])
        self.tgt_n = self.tk_n.K(self.X, None)#, self.tgt_n)
        self.tgt_c = self.tk_c.K(self.X, None)#, self.tgt_c)
        self.tgt_hyp_n = self.tk_n.dK_dtheta(1, self.X, None)#, self.tgt_hyp_n)
        self.tgt_hyp_c = self.tk_c.dK_dtheta(1, self.X, None)#, self.tgt_hyp_c)
        #self.tk_n.K(self.X, None, self.tgt_n)
        #self.tk_c.K(self.X, None, self.tgt_c)
        #self.tk_n.dK_dtheta(1, self.X, None, self.tgt_hyp_n)
        #self.tk_c.dK_dtheta(1, self.X, None, self.tgt_hyp_c)
        self.assertTrue((self.tgt_n == self.tgt_c).all())
        self.assertTrue((self.tgt_hyp_n == self.tgt_hyp_c).all())

    def test_treekernel_deltaparams_cache3(self):
        self.tk_n._set_params([1,0.1])
        self.tk_c._set_params([1,0.1])
        self.tgt_n = self.tk_n.K(self.X, None)#, self.tgt_n)
        self.tgt_c = self.tk_c.K(self.X, None)#, self.tgt_c)
        self.tgt_hyp_n = self.tk_n.dK_dtheta(1, self.X, None)#, self.tgt_hyp_n)
        self.tgt_hyp_c = self.tk_c.dK_dtheta(1, self.X, None)#, self.tgt_hyp_c)
        #self.tk_n.K(self.X, None, self.tgt_n)
        #self.tk_c.K(self.X, None, self.tgt_c)
        #self.tk_n.dK_dtheta(1, self.X, None, self.tgt_hyp_n)
        #self.tk_c.dK_dtheta(1, self.X, None, self.tgt_hyp_c)
        #print self.tgt_hyp_n
        #print self.tgt_hyp_c
        self.assertTrue((self.tgt_n == self.tgt_c).all())
        self.assertAlmostEqual(self.tgt_hyp_n[0], self.tgt_hyp_c[0])
        self.assertAlmostEqual(self.tgt_hyp_n[1], self.tgt_hyp_c[1])
        #self.assertTrue((self.tgt_hyp_n == self.tgt_hyp_c).all())

    def test_treekernel_deltaparams_cache4(self):
        self.tk_n._set_params([0.3,1])
        self.tk_c._set_params([0.3,1])
        self.tgt_n = self.tk_n.K(self.X, None)#, self.tgt_n)
        self.tgt_c = self.tk_c.K(self.X, None)#, self.tgt_c)
        self.tgt_hyp_n = self.tk_n.dK_dtheta(1, self.X, None)#, self.tgt_hyp_n)
        self.tgt_hyp_c = self.tk_c.dK_dtheta(1, self.X, None)#, self.tgt_hyp_c)
        #self.tk_n.K(self.X, None, self.tgt_n)
        #self.tk_c.K(self.X, None, self.tgt_c)
        #self.tk_n.dK_dtheta(1, self.X, None, self.tgt_hyp_n)
        #self.tk_c.dK_dtheta(1, self.X, None, self.tgt_hyp_c)
        self.assertTrue((self.tgt_n == self.tgt_c).all())
        self.assertTrue((self.tgt_hyp_n == self.tgt_hyp_c).all())


class IntegrationTreeKernelTests(unittest.TestCase):
    """
    The goal of these tests is to apply TreeKernels in GP and
    check if "something" is done. =P
    There are no asserts here, so only an Error is considered a Failure.
    """

    def test_treekernel_real1(self):
        tk = GPy.kern.TreeKernel()
        X = np.array([['(S NP VP)'], ['(S NP ADJ)'], ['(S NP)']], dtype=object)
        Y = np.array([[1],[2],[3]])
        m = GPy.models.GPRegression(X, Y, kernel=tk)

    def test_treekernel_real2(self):
        tk = GPy.kern.TreeKernel()
        X = np.array([['(S (NP N) (VP V))'], ['(S NP ADJ)'], ['(S NP)']], dtype=object)
        Y = np.array([[1],[2],[30]])
        m = GPy.models.GPRegression(X, Y, kernel=tk)
        m.constrain_positive('')
        m.optimize(max_f_eval=50)
    
    def test_treekernel_real3(self):
        tk = GPy.kern.TreeKernel()
        rbf = GPy.kern.rbf(2, ARD=True)
        k = tk.add(rbf, tensor=True)
        k.input_slices = [slice(0,1),slice(1,3)]
        X = np.array([['(S NP VP)', 0.1, 4],
                      ['(S NP ADJ)', 0.4, 5],
                      ['(S NP)', 1.6, 67.8]], dtype=object)
        Y = np.array([[1],[2],[3]])
        m = GPy.models.GPRegression(X, Y, kernel=k)
        m.constrain_positive('')
        m.optimize(max_f_eval=100)

    def test_treekernel_real4(self):
        tk = GPy.kern.TreeKernel()
        X = np.array([['(S NP VP)'],
                      ['(S (NP N) (VP V))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        Y = np.array([[(a+10)*5] for a in range(5)])
        m = GPy.models.GPRegression(X, Y, kernel=tk)
        m.constrain_positive('')
        m.optimize(max_f_eval=10)


class OptTreeKernelTests(unittest.TestCase):
    """
    Tests for the optimized kernel version
    """
    def setUp(self):
        self.tk_n = TreeKernel(mode="naive")
        self.tk_f = TreeKernel(mode="opt")
        self.tgt_n = np.array([[0.,0,0,0],
                               [0,0,0,0],
                               [0,0,0,0],
                               [0,0,0,0]])
        self.tgt_f = np.copy(self.tgt_n)
        self.tgt_hyp_n = np.array([0., 0])
        self.tgt_hyp_f = np.array([0., 0])
        self.X = np.array([['(S NP)'], ['(S NP VP)'], ['(S (NP N) (VP V))'], ['(S (NP NS) (VP V))']], dtype=object)
        #self.X = [['(S (NP (DET a) (N b)) (VP (V c)))'], ['(S (NP (N b)) (VP (V d) (ADV e)))']]

    def test_treekernel_deltaparams_opt1(self):
        self.tgt_n = self.tk_n.K(self.X, None)#, self.tgt_n)
        self.tgt_f = self.tk_f.K(self.X, None)#, self.tgt_c)
        self.tgt_hyp_n = self.tk_n.dK_dtheta(1, self.X, None)#, self.tgt_hyp_n)
        self.tgt_hyp_f = self.tk_f.dK_dtheta(1, self.X, None)#, self.tgt_hyp_c)
        #self.tk_n.K(self.X, None, self.tgt_n)
        #self.tk_f.K(self.X, None, self.tgt_f)
        #self.tk_n.dK_dtheta(1, self.X, None, self.tgt_hyp_n)
        #self.tk_f.dK_dtheta(1, self.X, None, self.tgt_hyp_f)
        #print self.tgt_n
        #print self.tgt_f
        self.assertTrue((self.tgt_n == self.tgt_f).all())
        self.assertTrue((self.tgt_hyp_n == self.tgt_hyp_f).all())

    def test_treekernel_opt_opt1(self):
        tk1 = GPy.kern.TreeKernel(mode="naive")
        rbf1 = GPy.kern.rbf(2, ARD=True)
        k1 = tk1.add(rbf1, tensor=True)
        k1.input_slices = [slice(0,1),slice(1,3)]

        tk2 = GPy.kern.TreeKernel(mode="opt")
        rbf2 = GPy.kern.rbf(2, ARD=True)
        k2 = tk2.add(rbf2, tensor=True)
        k2.input_slices = [slice(0,1),slice(1,3)]

        X = np.array([['(S NP VP)', 0.1, 4],
                      ['(S (NP N) (VP V))', 0.3, 2],
                      ['(S (NP (N a)) (VP (V c)))', 1.9, 12],
                      ['(S (NP (Det a) (N b)) (VP (V c)))', -1.7, -5],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))', 1.8, -9]],
                     dtype=object)
        Y = Y = np.array([[(a+10)*5] for a in range(5)])
        m1 = GPy.models.GPRegression(X, Y, kernel=k1)
        m2 = GPy.models.GPRegression(X, Y, kernel=k2)
        m1.optimize(optimizer="tnc")
        m2.optimize(optimizer="tnc")
        self.assertTrue((m1._get_params() == m2._get_params()).all())

    def test_treekernel_Kdiag_opt(self):
        tk = GPy.kern.TreeKernel(mode="opt", normalize=False)
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        diag = tk.Kdiag(X)
        self.assertTrue(([6,6,15,24,37] == diag).all())

class NormTreeKernelTests(unittest.TestCase):
    """
    Tests for the normalized kernel version.
    """
    def test_treekernel_norm1(self):
        tk = GPy.kern.TreeKernel(mode="opt", normalize=False)
        X = np.array([['(S (NP a) (VP v))'],
                      ['(S (NP a1) (VP v))'],
                      ['(S (NP (NP a)) (VP (V c)))'],
                      ['(S (VP v2))']],
                     dtype=object)
        k = tk.K(X, X)
        result = np.array([[6,3,2,0],
                           [3,6,1,0],
                           [2,1,15,0],
                           [0,0,0,3]])
        self.assertTrue((k == result).all())

    def test_treekernel_norm_grad1(self):
        tk = GPy.kern.TreeKernel(mode="opt", normalize=True)
        X = np.array([['(S (NP a) (VP v))']], dtype=object)
        X2 = np.array([['(S (NP (NP a)) (VP (V c)))']], dtype=object)

        h = 0.00001
        tk2 = TreeKernel(mode="naive")
        k = tk.K(X, X2)
        dk_dt = tk.dK_dtheta(1, X, X2)

        tk._set_params([1,1-h])
        k_b1 = tk.K(X, X2)
        tk._set_params([1,1+h])
        k_b2 = tk.K(X, X2)

        tk._set_params([1-h,1])
        k_d1 = tk.K(X, X2)
        tk._set_params([1+h,1])
        k_d2 = tk.K(X, X2)
        approx = [np.sum((k_d2 - k_d1) / (2 * h)), np.sum((k_b2 - k_b1) / (2 * h))]
        self.assertAlmostEqual(approx[0], dk_dt[0])
        self.assertAlmostEqual(approx[1], dk_dt[1])

    def test_treekernel_norm_grad2(self):
        tk = GPy.kern.TreeKernel(mode="opt", normalize=True)
        X = np.array([['(S (NP a) (VP v))'], ['(S (NP a) (VP c))']], dtype=object)

        h = 0.00001
        k = tk.K(X)
        dk_dt = tk.dK_dtheta(1, X)

        tk._set_params([1,1-h])
        k_b1 = tk.K(X)
        tk._set_params([1,1+h])
        k_b2 = tk.K(X)

        tk._set_params([1-h,1])
        k_d1 = tk.K(X)
        tk._set_params([1+h,1])
        k_d2 = tk.K(X)

        approx = [np.sum((k_d2 - k_d1) / (2 * h)), np.sum((k_b2 - k_b1) / (2 * h))]
        self.assertAlmostEqual(approx[0], dk_dt[0])
        self.assertAlmostEqual(approx[1], dk_dt[1])

    def test_treekernel_norm_grad3(self):
        tk = GPy.kern.TreeKernel(mode="opt", normalize=True)
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        h = 0.00001
        k = tk.K(X)
        dk_dt = tk.dK_dtheta(1, X)
        tk._set_params([1,1-h])
        k_b1 = tk.K(X)
        tk._set_params([1,1+h])
        k_b2 = tk.K(X)
        tk._set_params([1-h,1])
        k_d1 = tk.K(X)
        tk._set_params([1+h,1])
        k_d2 = tk.K(X)
        approx = [np.sum((k_d2 - k_d1) / (2 * h)), np.sum((k_b2 - k_b1) / (2 * h))]
        self.assertAlmostEqual(approx[0], dk_dt[0])
        self.assertAlmostEqual(approx[1], dk_dt[1])

    def test_treekernel_Kdiag_norm(self):
        tk = GPy.kern.TreeKernel(mode="opt", normalize=True)
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        diag = tk.Kdiag(X)
        self.assertTrue(([1,1,1,1,1] == diag).all())





######################
# CYTHON
######################

class SSTTests(unittest.TestCase):

    def test_gen_node_list(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = PySST()
        nodes1, dict1 = k.parts[0]._gen_node_list(repr1)
        result = "[('ADJ colorless', 0, None), ('ADV furiously', 4, None), ('N ideas', 1, None), ('NP ADJ N', 2, [0, 1]), ('S NP VP', 6, [2, 5]), ('V sleep', 3, None), ('VP V ADV', 5, [3, 4])]"
        print nodes1
        self.assertEqual(str(nodes1), result)

    def test_gen_node_list_cy(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = SST()
        nodes1, dict1 = k.parts[0].kernel._gen_node_list(repr1)
        result = "[('ADJ colorless', 0, None), ('ADV furiously', 4, None), ('N ideas', 1, None), ('NP ADJ N', 2, [0, 1]), ('S NP VP', 6, [2, 5]), ('V sleep', 3, None), ('VP V ADV', 5, [3, 4])]"
        print nodes1
        self.assertEqual(str(nodes1), result)

    @unittest.skip("skip")
    def test_get_node_pairs1(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = PySST()
        nodes1, dict1 = k.parts[0]._gen_node_list(repr1)
        nodes2, dict2 = k.parts[0]._gen_node_list(repr2)
        node_list = k.parts[0]._get_node_pairs(nodes1, nodes2)
        result = "[(('ADJ colorless', 0, None), ('ADJ colorless', 0, None)), (('N ideas', 1, None), ('N ideas', 1, None)), (('NP ADJ N', 2, [0, 1]), ('NP ADJ N', 2, [0, 1])), (('V sleep', 3, None), ('V sleep', 3, None)), (('VP V ADV', 5, [3, 4]), ('VP V ADV', 5, [3, 4]))]"
        self.assertEqual(str(node_list), result)

    @unittest.skip("cy")
    def test_get_node_pair_list_cy(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = PySST()
        nodes1 = k.parts[0]._gen_node_list(repr1)
        nodes2 = k.parts[0]._gen_node_list(repr2)
        node_list = k.parts[0]._get_node_pair_list_cy(nodes1, nodes2)
        result = "[((0, 0), (0, 0), 0), ((1, 1), (1, 1), 0), ((0, 1), (0, 1), 0), ((1, 0), (1, 0), 0), ((0,), (0,), 2), ((1,), (1,), 2), ((), (), 2)]"
        self.assertEqual(str(node_list), result)

    def test_get_node_pairs2(self):
        repr1 = '(S (NP ns) (VP v))'
        repr2 = '(S (NP (N a)) (VP (V c)))'
        k = PySST()
        nodes1, dict1 = k.parts[0]._gen_node_list(repr1)
        nodes2, dict2 = k.parts[0]._gen_node_list(repr2)
        node_list = k.parts[0]._get_node_pairs(nodes1, nodes2)
        print node_list
        print dict1
        print dict2

    def test_K(self):
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        k = PySST(_lambda=1)
        target = np.zeros(shape=(len(X), len(X)))
        k.parts[0].K(X, None, target)
        result = [[ 1.,          0.5,         0.10540926,  0.08333333,  0.06711561],
                  [ 0.5,         1.,          0.10540926,  0.08333333,  0.06711561],
                  [ 0.10540926,  0.10540926,  1.,          0.31622777,  0.04244764],
                  [ 0.08333333,  0.08333333,  0.31622777,  1.,          0.0335578 ],
                  [ 0.06711561,  0.06711561,  0.04244764,  0.0335578,   1.        ]]
        self.assertAlmostEqual(np.sum(result), np.sum(target))

    def test_K2(self):
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        k = SST(_lambda=1)
        target = np.zeros(shape=(len(X), len(X)))
        k.parts[0].K(X, None, target)
        result = [[ 1.,          0.5,         0.10540926,  0.08333333,  0.06711561],
                  [ 0.5,         1.,          0.10540926,  0.08333333,  0.06711561],
                  [ 0.10540926,  0.10540926,  1.,          0.31622777,  0.04244764],
                  [ 0.08333333,  0.08333333,  0.31622777,  1.,          0.0335578 ],
                  [ 0.06711561,  0.06711561,  0.04244764,  0.0335578,   1.        ]]
        self.assertAlmostEqual(np.sum(result), np.sum(target))



class SSTProfilingTests(unittest.TestCase):

    @unittest.skip("skip")
    def test_prof_gen_node_pair_list(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = PySST()
        nodes1 = k._gen_node_list(repr1)
        nodes2 = k._gen_node_list(repr2)
        start_time = datetime.datetime.now()
        for i in range(20000):
            node_list = k.parts[0]._get_node_pair_list(nodes1, nodes2)
        end_time = datetime.datetime.now()
        print end_time - start_time

    @unittest.skip("skip")
    def test_prof_gen_node_pair_list_cy(self):
        repr1 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        repr2 = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        k = PySST()
        nodes1 = k._gen_node_list(repr1)
        nodes2 = k._gen_node_list(repr2)
        start_time = datetime.datetime.now()
        for i in range(20000):
            node_list = k.parts[0]._get_node_pair_list_cy(nodes1, nodes2)
        end_time = datetime.datetime.now()
        print end_time - start_time

    @unittest.skip("skip")
    def test_prof_K(self):
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        k = PySST()
        target = np.zeros(shape=(len(X), len(X)))
        ITS = 10
        start_time = datetime.datetime.now()
        for i in range(ITS):
            k.parts[0].K(X, None, target)
        end_time = datetime.datetime.now()
        print target/ITS
        print "PYTHON"
        print end_time - start_time

    def test_prof_K_cy(self):
        X = np.array([['(S (NP ns) (VP v))'],
                      ['(S (NP n) (VP v))'],
                      ['(S (NP (N a)) (VP (V c)))'],
                      ['(S (NP (Det a) (N b)) (VP (V c)))'],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
                     dtype=object)
        k = SST()
        target = np.zeros(shape=(len(X), len(X)))
        ITS = 10
        start_time = datetime.datetime.now()
        for i in range(ITS):
            k.parts[0].K(X, None, target)
        end_time = datetime.datetime.now()
        print target/ITS
        print k.parts[0].dlambda
        print k.parts[0].dsigma
        print "CYTHON"
        print end_time - start_time

    @unittest.skip("skip")
    def test_prof_K_cy2(self):
        #TREES_TRAIN = 'cython_kernels/test/ALL.stanford-np'
        TREES_TRAIN = 'cython_kernels/test/qc_trees.txt'
        TREES = 1000
        with open(TREES_TRAIN) as f:
            X = np.array([[line] for line in f.readlines()], dtype=object)[:TREES]
        k = SST()
        target = np.zeros(shape=(len(X), len(X)))
        ITS = 1
        start_time = datetime.datetime.now()
        for i in range(ITS):
            k.parts[0].K(X, None, target)
        end_time = datetime.datetime.now()
        #print target/ITS
        #print k.dlambdas
        #print k.dsigmas
        print "SSTW"
        print end_time - start_time

    @unittest.skip("skip")
    def test_prof_K_cy5(self):
        #TREES_TRAIN = 'cython_kernels/test/ALL.stanford-np'
        TREES_TRAIN = 'GPy/testing/qc_trees.txt'
        TREES = 1000
        with open(TREES_TRAIN) as f:
            X = np.array([[line] for line in f.readlines()], dtype=object)[:TREES]
        k = SST()
        target = np.zeros(shape=(len(X), len(X)))
        ITS = 1
        start_time = datetime.datetime.now()
        for i in range(ITS):
            k.parts[0].K(X, None, target)
        end_time = datetime.datetime.now()
        #print target/ITS
        #print k.dlambdas
        #print k.dsigmas
        print "SSTW2"
        print end_time - start_time

    @unittest.skip("skip")
    def test_prof_K_cy3(self):
        #TREES_TRAIN = 'cython_kernels/test/ALL.stanford-np'
        TREES_TRAIN = 'GPy/testing/qc_trees.txt'
        TREES = 700
        with open(TREES_TRAIN) as f:
            X = np.array([[line] for line in f.readlines()], dtype=object)[:TREES]
        k = SST()
        target = np.zeros(shape=(len(X), len(X)))
        ITS = 1

        import cProfile, StringIO, pstats
        pr = cProfile.Profile()
        pr.enable()
        for i in range(ITS):
            k.parts[0].K(X, None, target)
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
        ps.print_stats(20)
        print s.getvalue()

    @unittest.skip("skip")
    def test_prof_K_cy4(self):
        #TREES_TRAIN = 'cython_kernels/test/ALL.stanford-np'
        TREES_TRAIN = 'GPy/testing/qc_trees.txt'
        TREES = 2000
        with open(TREES_TRAIN) as f:
            X = np.array([[line] for line in f.readlines()], dtype=object)[:TREES]
        k = SST()
        #k = GPy.kern.SubsetTreeKernel()
        target = np.zeros(shape=(len(X), len(X)))
        ITS = 1

        import cProfile, StringIO, pstats
        pr = cProfile.Profile()
        pr.enable()
        for i in range(ITS):
            k.parts[0].K(X, None, target)
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
        ps.print_stats(20)
        print s.getvalue()



if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()