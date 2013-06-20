# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy
import ipdb
import nltk # Need to cope with this in a better way...


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
        #print tk.parts[0]._get_params()
        #print tk._get_params()
        tk._set_params(np.array([1, 1]))
        #ipdb.set_trace()
        self.assertTrue((tk.parts[0]._get_params() == np.array([1, 1])).all())

    def test_treekernel_params2(self):
        tk = GPy.kern.TreeKernel()
        tk._set_params(np.array([1, 0]))
        self.assertTrue((tk._get_params() == np.array([1, 0])).all())

    def test_treekernel_params3(self):
        tk = GPy.kern.TreeKernel()
        #print tk._get_param_names()
        self.assertTrue(tk._get_param_names() == ['tree_kernel_decay', 'tree_kernel_branch'])
        
    def test_treekernel_mock1(self):
        tk = GPy.kern.TreeKernel(mock=True)
        #print tk.parts[0].mock
        X1 = np.array([[nltk.Tree("(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))")]], dtype=nltk.Tree)
        target = tk.K(X1, X1)
        #print target
        #print len(X1)
        #print len(X1[0][0])
        self.assertTrue(target[0] == [4])

    #def test_treekernel_mock2(self):
    #    tk = GPy.kern.TreeKernel(mock=True)
    #    X = np.array([['a' * i] for i in range(10)], dtype=str)
    #    Y = np.ones((10,1))
        #print X
        #print Y
    #    m = GPy.models.GPRegression(X,Y,kernel=tk)



if __name__ == "__main__":
    print "Running unit tests, please be (very) patient..."
    unittest.main()
