# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy


class StringKernelTests(unittest.TestCase):
    
    def setUp(self):
        s1 = 'aab'
        s2 = 'aabb'
        
    def test_sk_1(self):
        k = GPy.kern.StringKernel()
        self.assertTrue(k.calc_k(s1, s2), 9)


if __name__ == "__main__":
    print("Running unit tests, please be (very) patient...")
    unittest.main()
