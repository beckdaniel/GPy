# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy


class StringKernelTests(unittest.TestCase):
    
    def test_sk_1(self):
        s1 = 'aab'
        s2 = 'aabb'
        k = GPy.kern.StringKernel()
        result = k.calc_k(s1, s2)
        #print result
        self.assertEquals(result, 18)

    def test_sk_2(self):
        s1 = 'gatta'
        s2 = 'cata'
        k = GPy.kern.StringKernel()
        self.assertEquals(k.calc_k(s1, s2), 14)

    def test_sk_3(self):
        s1 = 'gatta'
        s2 = 'cata'
        k = GPy.kern.StringKernel()
        self.assertEquals(k.calc_k(s1, s2), k.calc_k(s2, s1))

    def test_sk_4(self):
        s1 = 'gatta'
        s2 = 'cata'
        k = GPy.kern.StringKernel()
        self.assertEquals(k.calc_k(s1, s2, decay=0.5), 2.75)


if __name__ == "__main__":
    print("Running unit tests, please be (very) patient...")
    unittest.main()
