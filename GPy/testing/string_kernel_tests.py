# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import unittest
import numpy as np
import GPy


class AllSubStringKernelTests(unittest.TestCase):
    
    def test_sk_1(self):
        s1 = 'aab'
        s2 = 'aabb'
        k = GPy.kern.AllSubStringKernel()
        result = k.calc_k(s1, s2)
        #print result
        self.assertEquals(result, 18)

    def test_sk_2(self):
        s1 = 'gatta'
        s2 = 'cata'
        k = GPy.kern.AllSubStringKernel()
        self.assertEquals(k.calc_k(s1, s2), 14)

    def test_sk_3(self):
        s1 = 'gatta'
        s2 = 'cata'
        k = GPy.kern.AllSubStringKernel()
        self.assertEquals(k.calc_k(s1, s2), k.calc_k(s2, s1))

    def test_sk_4(self):
        s1 = 'gatta'
        s2 = 'cata'
        k = GPy.kern.AllSubStringKernel()
        self.assertEquals(k.calc_k(s1, s2, decay=0.5), 2.75)

    def test_sk_5(self):
        s1 = 'aab'
        s2 = 'aabb'
        k = GPy.kern.AllSubStringKernel()
        self.assertEquals(k.calc_k(s1, s2, decay=0.5), 3.25)


class FixedLengthSubseqKernelTests(unittest.TestCase):

    def test_fixsubsk_2(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(1)
        self.assertEquals(k.calc_k(s1, s2), 6.0)

    def test_fixsubsk_3(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(2)
        self.assertEquals(k.calc_k(s1, s2), 11.0)

    def test_fixsubsk_4(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(3)
        self.assertEquals(k.calc_k(s1, s2), 13.0)

if __name__ == "__main__":
    print("Running unit tests, please be (very) patient...")
    unittest.main()
