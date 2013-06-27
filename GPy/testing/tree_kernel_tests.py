# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
import nltk # Need to cope with this in a better way...
from GPy.kern.tree_kernel import TreeKernel
import sys

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
        tk = TreeKernel()
        self.assertTrue(tk.delta_naive(node1,node2) == 0)

    def test_treekernel_delta2(self):
        node1 = nltk.Tree('(S NP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel()
        self.assertTrue(tk.delta_naive(node1,node2) == 0)

    def test_treekernel_delta3(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel()
        self.assertTrue(tk.delta_naive(node1,node2) == 1)

    def test_treekernel_delta4(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel(decay=0.5)
        self.assertTrue(tk.delta_naive(node1,node2) == 0.5)

    def test_treekernel_delta5(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel()
        self.assertTrue(tk.delta_naive(node1,node2) == 4)

    def test_treekernel_delta6(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(branch=0.5)
        self.assertTrue(tk.delta_naive(node1,node2) == 2.25)
    
    def test_treekernel_delta7(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5)
        self.assertTrue(tk.delta_naive(node1,node2) == 1.125)

    def test_treekernel_delta8(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5, branch=0.5)
        self.assertTrue(tk.delta_naive(node1,node2) == 0.5)

    def test_treekernel_delta9(self):
        node1 = nltk.Tree('(S (NP NS) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel()
        self.assertTrue(tk.delta_naive(node1,node2) == 2)

    def test_treekernel_deltaparams1(self):
        node1 = 'test'
        node2 = 'test'
        tk = TreeKernel()
        self.assertEqual(tk.delta_params_naive(node1,node2), (0,0))

    def test_treekernel_deltaparams2(self):
        node1 = nltk.Tree('(S NP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel()
        self.assertEqual(tk.delta_params_naive(node1,node2), (0,0))

    def test_treekernel_deltaparams3(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel()
        self.assertEqual(tk.delta_params_naive(node1,node2), (1,0))

    def test_treekernel_deltaparams4(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel(decay=0.5)
        self.assertEqual(tk.delta_params_naive(node1,node2), (1,0))

    def test_treekernel_deltaparams5(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel()
        self.assertEqual(tk.delta_params_naive(node1,node2), (6,2))

    def test_treekernel_deltaparams6(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(branch=0.5)
        self.assertEqual(tk.delta_params_naive(node1,node2), (3.75, 1.5))
    
    def test_treekernel_deltaparams7(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5)
        self.assertEqual(tk.delta_params_naive(node1,node2), (3, 0.75))

    def test_treekernel_deltaparams8(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5, branch=0.5)
        self.assertEqual(tk.delta_params_naive(node1,node2), (1.5, 0.5))

    def test_treekernel_deltaparams9(self):
        node1 = nltk.Tree('(S (NP NS) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel()
        self.assertEqual(tk.delta_params_naive(node1,node2), (8/3. , 4/3.))

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
        self.assertTrue((k == [2,1]).all())

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
        self.X = [['(S NP)'], ['(S NP VP)'], ['(S (NP N) (VP V))'], ['(S (NP NS) (VP V))']]

    def test_treekernel_deltaparams_cache1(self):
        self.tk_n.K(self.X, None, self.tgt_n)
        self.tk_c.K(self.X, None, self.tgt_c)
        self.tk_n.dK_dtheta(1, self.X, None, self.tgt_hyp_n)
        self.tk_c.dK_dtheta(1, self.X, None, self.tgt_hyp_c)
        self.assertTrue((self.tgt_n == self.tgt_c).all())
        self.assertTrue((self.tgt_hyp_n == self.tgt_hyp_c).all())

    def test_treekernel_deltaparams_cache2(self):
        self.tk_n._set_params([1,0.3])
        self.tk_c._set_params([1,0.3])
        self.tk_n.K(self.X, None, self.tgt_n)
        self.tk_c.K(self.X, None, self.tgt_c)
        self.tk_n.dK_dtheta(1, self.X, None, self.tgt_hyp_n)
        self.tk_c.dK_dtheta(1, self.X, None, self.tgt_hyp_c)
        self.assertTrue((self.tgt_n == self.tgt_c).all())
        self.assertTrue((self.tgt_hyp_n == self.tgt_hyp_c).all())

    def test_treekernel_deltaparams_cache3(self):
        self.tk_n._set_params([1,0.])
        self.tk_c._set_params([1,0.])
        self.tk_n.K(self.X, None, self.tgt_n)
        self.tk_c.K(self.X, None, self.tgt_c)
        self.tk_n.dK_dtheta(1, self.X, None, self.tgt_hyp_n)
        self.tk_c.dK_dtheta(1, self.X, None, self.tgt_hyp_c)
        self.assertTrue((self.tgt_n == self.tgt_c).all())
        self.assertTrue((self.tgt_hyp_n == self.tgt_hyp_c).all())

    def test_treekernel_deltaparams_cache4(self):
        self.tk_n._set_params([0.3,1])
        self.tk_c._set_params([0.3,1])
        self.tk_n.K(self.X, None, self.tgt_n)
        self.tk_c.K(self.X, None, self.tgt_c)
        self.tk_n.dK_dtheta(1, self.X, None, self.tgt_hyp_n)
        self.tk_c.dK_dtheta(1, self.X, None, self.tgt_hyp_c)
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

class FastTreeKernelTests(unittest.TestCase):
    """
    Tests for the fast kernel version
    """
    def setUp(self):
        self.tk_n = TreeKernel(mode="naive")
        self.tk_f = TreeKernel(mode="fast")
        self.tgt_n = np.array([[0.,0,0,0],
                               [0,0,0,0],
                               [0,0,0,0],
                               [0,0,0,0]])
        self.tgt_f = np.copy(self.tgt_n)
        self.tgt_hyp_n = np.array([0., 0])
        self.tgt_hyp_f = np.array([0., 0])
        self.X = [['(S NP)'], ['(S NP VP)'], ['(S (NP N) (VP V))'], ['(S (NP NS) (VP V))']]
        #self.X = [['(S (NP (DET a) (N b)) (VP (V c)))'], ['(S (NP (N b)) (VP (V d) (ADV e)))']]

    def test_treekernel_deltaparams_fast1(self):
        self.tk_n.K(self.X, None, self.tgt_n)
        self.tk_f.K(self.X, None, self.tgt_f)
        self.tk_n.dK_dtheta(1, self.X, None, self.tgt_hyp_n)
        self.tk_f.dK_dtheta(1, self.X, None, self.tgt_hyp_f)
        print self.tgt_n
        print self.tgt_f
        self.assertTrue((self.tgt_n == self.tgt_f).all())
        self.assertTrue((self.tgt_hyp_n == self.tgt_hyp_f).all())


class ProfilingTreeKernelTests(unittest.TestCase):
    """
    A profiling test, to check for performance bottlenecks.
    """
    def test_treekernel_profiling1(self):
        tk = GPy.kern.TreeKernel(mode="fast")
        rbf = GPy.kern.rbf(2, ARD=True)
        k = tk.add(rbf, tensor=True)
        k.input_slices = [slice(0,1),slice(1,3)]
        X = np.array([['(S NP VP)', 0.1, 4],
                      ['(S (NP N) (VP V))', 0.3, 2],
                      ['(S (NP (N a)) (VP (V c)))', 1.9, 12],
                      ['(S (NP (Det a) (N b)) (VP (V c)))', -1.7, -5],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))', 1.8, -9],
                      ['(S NP VP)', 0.1, 4],
                      ['(S (NP N) (VP V))', 0.3, 2],
                      ['(S (NP (N a)) (VP (V c)))', 1.9, 12],
                      ['(S (NP (Det a) (N b)) (VP (V c)))', -1.7, -5],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))', 1.8, -9],
                      ['(S NP VP)', 0.1, 4],
                      ['(S (NP N) (VP V))', 0.3, 2],
                      ['(S (NP (N a)) (VP (V c)))', 1.9, 12],
                      ['(S (NP (Det a) (N b)) (VP (V c)))', -1.7, -5],
                      ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))', 1.8, -9]],
                     dtype=object)
        #X = X[:5]
        Y = np.array([[(a+10)*5] for a in range(15)])
        m = GPy.models.GPRegression(X, Y, kernel=k)
        import cProfile
        m.constrain_positive('')
        cProfile.runctx("m.optimize(optimizer='tnc', max_f_eval=100, messages=True)", 
                        globals(), {'m': m, 'X': X}, sort="cumulative")


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
