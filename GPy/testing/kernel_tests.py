# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
import nltk # Need to cope with this in a better way...
from GPy.kern.tree_kernel import TreeKernel

class KernelTests(unittest.TestCase):
    def test_kerneltie(self):
        K = GPy.kern.rbf(5, ARD=True)
        K.tie_params('.*[01]')
        K.constrain_fixed('2')
        X = np.random.rand(5,5)
        Y = np.ones((5,1))
        m = GPy.models.GPRegression(X,Y,K)
        self.assertTrue(m.checkgrad())

    def test_fixedkernel(self):
        """
        Fixed effect kernel test
        """
        X = np.random.rand(30, 4)
        K = np.dot(X, X.T)
        kernel = GPy.kern.Fixed(4, K)
        Y = np.ones((30,1))
        m = GPy.models.GPRegression(X,Y,kernel=kernel)
        self.assertTrue(m.checkgrad())

    def test_coregionalisation(self):
        X1 = np.random.rand(50,1)*8
        X2 = np.random.rand(30,1)*5
        index = np.vstack((np.zeros_like(X1),np.ones_like(X2)))
        X = np.hstack((np.vstack((X1,X2)),index))
        Y1 = np.sin(X1) + np.random.randn(*X1.shape)*0.05
        Y2 = np.sin(X2) + np.random.randn(*X2.shape)*0.05 + 2.
        Y = np.vstack((Y1,Y2))

        k1 = GPy.kern.rbf(1) + GPy.kern.bias(1)
        k2 = GPy.kern.Coregionalise(2,1)
        k = k1.prod(k2,tensor=True)
        m = GPy.models.GPRegression(X,Y,kernel=k)
        self.assertTrue(m.checkgrad())


class TreeKernelTests(unittest.TestCase):

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
        tk = GPy.kern.TreeKernel(mock=True)
        X1 = np.array([[nltk.Tree("(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))")]], dtype=nltk.Tree)
        #print X1.dtype
        target = tk.K(X1, X1)
        #print target
        self.assertTrue(target[0] == [16])

    def test_treekernel_mock2(self):
        tk = GPy.kern.TreeKernel(mock=True)
        X = np.array([['a' * i] for i in range(10)], dtype=str)
        Y = np.array([[a] for a in range(10)])
        m = GPy.models.GPRegression(X,Y,kernel=tk)
        m['noise'] = 1
        #print m.K
        #print m.predict(X)

    def test_treekernel_mock3(self):
        tk = GPy.kern.TreeKernel(mock=True)
        rbf = GPy.kern.rbf(2, ARD=True)
        k = tk.add(rbf, tensor=True)
        k.input_slices = [slice(0,1),slice(1,3)]
        X = np.array([[nltk.Tree('(S NP VP)'), 0.1, 4],
                      [nltk.Tree('(S NP ADJ)'), 0.4, 5],
                      [nltk.Tree('(S NP)'), 1.6, 67.8]])
        Y = np.array([[1],[2],[3]])
        m = GPy.models.GPRegression(X, Y, kernel=k)
        m.constrain_positive('')
        m.optimize(max_f_eval=10)

    def test_treekernel_mock4(self):
        tk = GPy.kern.TreeKernel(mock=True)
        X = np.array([[nltk.Tree('(S NP VP)')], [nltk.Tree('(S NP ADJ)')], [nltk.Tree('(S NP)')]])
        Y = np.array([[1],[2],[3]])
        m = GPy.models.GPRegression(X, Y, kernel=tk)
        m.constrain_positive('')
        m.optimize(max_f_eval=10)

    def test_treekernel_k1(self):
        tk = GPy.kern.TreeKernel()
        tk._set_params([1,0])
        t = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        X1 = np.array([[t]], dtype=object)
        target = tk.K(X1, X1)
        self.assertTrue(target[0] == [7])

    def test_treekernel_k2(self):
        tk = GPy.kern.TreeKernel()
        tk._set_params([1,1])
        t = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
        X1 = np.array([[t]], dtype=object)
        target = tk.K(X1, X1)
        self.assertTrue(target[0] == [37])

    def test_treekernel_delta1(self):
        node1 = 'test'
        node2 = 'test'
        tk = TreeKernel()
        self.assertTrue(tk.delta(node1,node2) == 0)

    def test_treekernel_delta2(self):
        node1 = nltk.Tree('(S NP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel()
        self.assertTrue(tk.delta(node1,node2) == 0)

    def test_treekernel_delta3(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel()
        self.assertTrue(tk.delta(node1,node2) == 1)

    def test_treekernel_delta4(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel(decay=0.5)
        self.assertTrue(tk.delta(node1,node2) == 0.5)

    def test_treekernel_delta5(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel()
        self.assertTrue(tk.delta(node1,node2) == 4)

    def test_treekernel_delta6(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(branch=0.5)
        self.assertTrue(tk.delta(node1,node2) == 2.25)
    
    def test_treekernel_delta7(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5)
        self.assertTrue(tk.delta(node1,node2) == 1.125)

    def test_treekernel_delta8(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5, branch=0.5)
        self.assertTrue(tk.delta(node1,node2) == 0.5)

    def test_treekernel_delta9(self):
        node1 = nltk.Tree('(S (NP NS) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel()
        self.assertTrue(tk.delta(node1,node2) == 2)


    def test_treekernel_deltaparams1(self):
        node1 = 'test'
        node2 = 'test'
        tk = TreeKernel()
        self.assertEqual(tk.delta_params(node1,node2), (0,0))

    def test_treekernel_deltaparams2(self):
        node1 = nltk.Tree('(S NP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel()
        self.assertEqual(tk.delta_params(node1,node2), (0,0))

    def test_treekernel_deltaparams3(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel()
        self.assertEqual(tk.delta_params(node1,node2), (1,0))

    def test_treekernel_deltaparams4(self):
        node1 = nltk.Tree('(S NP VP)')
        node2 = nltk.Tree('(S NP VP)')
        tk = TreeKernel(decay=0.5)
        self.assertEqual(tk.delta_params(node1,node2), (1,0))

    def test_treekernel_deltaparams5(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel()
        self.assertEqual(tk.delta_params(node1,node2), (6,2))

    def test_treekernel_deltaparams6(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(branch=0.5)
        self.assertEqual(tk.delta_params(node1,node2), (3.75, 1.5))
    
    def test_treekernel_deltaparams7(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5)
        self.assertEqual(tk.delta_params(node1,node2), (3, 0.75))

    def test_treekernel_deltaparams8(self):
        node1 = nltk.Tree('(S (NP N) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel(decay=0.5, branch=0.5)
        self.assertEqual(tk.delta_params(node1,node2), (1.5, 0.5))

    def test_treekernel_deltaparams9(self):
        node1 = nltk.Tree('(S (NP NS) (VP V))')
        node2 = nltk.Tree('(S (NP N) (VP V))')
        tk = TreeKernel()
        self.assertEqual(tk.delta_params(node1,node2), (8/3. , 4/3.))



    def test_treekernel_real1(self):
        tk = GPy.kern.TreeKernel()
        X = np.array([['(S NP VP)'], ['(S NP ADJ)'], ['(S NP)']], dtype=object)
        Y = np.array([[1],[2],[3]])
        m = GPy.models.GPRegression(X, Y, kernel=tk)
        #print m
        #print m.predict(X)

    def test_treekernel_real2(self):
        tk = GPy.kern.TreeKernel()
        X = np.array([['(S (NP N) (VP V))'], ['(S NP ADJ)'], ['(S NP)']], dtype=object)
        Y = np.array([[1],[2],[30]])
        m = GPy.models.GPRegression(X, Y, kernel=tk)
        m.constrain_positive('')
        #print m
        #print m.predict(X)
        m.optimize(max_f_eval=50)
        #print m
        #print m.predict(X)
    
    def test_treekernel_real3(self):
        tk = GPy.kern.TreeKernel()
        rbf = GPy.kern.rbf(2, ARD=True)
        k = tk.add(rbf, tensor=True)
        k.input_slices = [slice(0,1),slice(1,3)]
        X = np.array([['(S NP VP)', 0.1, 4],
                      ['(S NP ADJ)', 0.4, 5],
                      ['(S NP)', 1.6, 67.8]], dtype=object)
        #print X
        Y = np.array([[1],[2],[3]])
        m = GPy.models.GPRegression(X, Y, kernel=k)
        print m
        print m.predict(X)
        m.constrain_positive('')
        m.optimize(max_f_eval=100)
        print m
        print m.predict(X)


if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
