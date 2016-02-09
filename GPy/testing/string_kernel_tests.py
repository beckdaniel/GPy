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
        self.assertEquals(k.calc(s1, s2)[0], 6.0)

    def test_fixsubsk_3(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(2)
        self.assertEquals(k.calc(s1, s2)[0], 11.0)

    def test_fixsubsk_4(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(3)
        self.assertEquals(k.calc(s1, s2)[0], 13.0)

    #@unittest.skip('')
    def test_fixsubsk_5(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(1, decay=1.0, order_coefs=[0.5])
        self.assertEquals(k.calc(s1, s2)[0], 3.0)

    def test_fixsubsk_6(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(2, decay=1.0, order_coefs=[0.5, 0.2])
        self.assertEquals(k.calc(s1, s2)[0], 4.0)

    def test_fixsubsk_7(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(3, decay=1.0, order_coefs=[0.5, 0.2, 0.1])
        self.assertEquals(k.calc(s1, s2)[0], 4.2)

    def test_fixsubsk_grads_1(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(3)
        h = 0.000001
        k.decay += h
        result1, _, _ = k.calc(s1, s2)
        k.decay -= 2*h
        result2, _, _ = k.calc(s1, s2)
        k.decay += h
        result3, dKd, _ = k.calc(s1, s2)
        approx = (result1 - result2) / (2 * h)
        self.assertAlmostEqual(approx, dKd)

    def test_fixsubsk_grads_2(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(3)
        k.decay = 0.5
        h = 0.000001
        k.decay += h
        result1, _, _ = k.calc(s1, s2)
        k.decay -= 2*h
        result2, _, _ = k.calc(s1, s2)
        k.decay += h
        result3, dKd, _ = k.calc(s1, s2)
        approx = (result1 - result2) / (2 * h)
        self.assertAlmostEqual(approx, dKd)

    def test_fixsubsk_grads_3(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(3)
        k.decay = 0.01
        h = 0.000001
        k.decay += h
        result1, _, _ = k.calc(s1, s2)
        k.decay -= 2*h
        result2, _, _ = k.calc(s1, s2)
        k.decay += h
        result3, dKd, _ = k.calc(s1, s2)
        approx = (result1 - result2) / (2 * h)
        self.assertAlmostEqual(approx, dKd)

    def test_fixsubsk_grads_4(self):
        s2 = 'gatta'
        s1 = 'cata'
        k = GPy.kern.FixedLengthSubseqKernel(3)
        h = 0.000001
        k.order_coefs[2] = 0.3
        k.order_coefs[2] += h
        result1, _, _ = k.calc(s1, s2)
        k.order_coefs[2] -= 2*h
        result2, _, _ = k.calc(s1, s2)
        k.order_coefs[2] += h
        result3, _, dKcoefs = k.calc(s1, s2)
        print result1
        print result2
        print result3
        approx = (result1 - result2) / (2 * h)
        print approx
        print k.order_coefs.gradient[2]
        self.assertAlmostEqual(approx, dKcoefs[2])

    def test_fixsubsk_grads_5(self):
        s2 = 'gataacaatcaaata'
        s1 = 'ggtagcgaaatgca'
        k = GPy.kern.FixedLengthSubseqKernel(3)
        k.decay = 0.01
        h = 0.000001
        k.decay += h
        result1, _, _ = k.calc(s1, s2)
        k.decay -= 2*h
        result2, _, _ = k.calc(s1, s2)
        k.decay += h
        result3, dKd, _ = k.calc(s1, s2)
        approx = (result1 - result2) / (2 * h)
        self.assertAlmostEqual(approx, dKd)

    @unittest.skip('')
    def test_profiling_1(self):
        data = np.loadtxt('trial2', dtype=object, delimiter='\t')[:1]
        labels = np.array(data[:,0], dtype=np.float64)[:, None]
        import sklearn.preprocessing as pp
        scaler = pp.StandardScaler()
        scaler.fit(labels)
        #print labels
        labels = scaler.transform(labels)
        #print labels
        X = data[:, 1:]
        #inputs = np.array(range(len(X)))[:, None]
        inputs = X
        print inputs
        k = GPy.kern.FixedLengthSubseqKernel(1, decay=0.5)
        k.K(inputs)
        #print inputs
        #print labels
        m = GPy.models.GPRegression(inputs, labels, kernel=k)
        #m['fixsubsk.decay'].constrain_positive(0.1)
        print m
        print m['.*coefs']
        print m.checkgrad(verbose=True)
        m.optimize(messages=True)
        print m
        print m['.*coefs']
        for elem in zip(scaler.inverse_transform(m.predict(inputs)[0]), 
                        scaler.inverse_transform(labels)):
            print elem

    @unittest.skip('')
    def test_profiling_2(self):
        #data = np.loadtxt('trial2', dtype=object, delimiter='\t')[:5]
        data = np.array([['cata', 1.0],
                         ['gatta', -1.0],
                         ['acta', -2.0],
                         ['gattaca', 2.0],
                         ['tcaat', 0.0]])
        labels = np.array(data[:,1], dtype=np.float64)[:, None]
        import sklearn.preprocessing as pp
        scaler = pp.StandardScaler()
        scaler.fit(labels)
        #print labels
        labels = scaler.transform(labels)
        #print labels
        X = data[:, 0:1]
        #inputs = np.array(range(len(X)))[:, None]
        k = GPy.kern.FixedLengthSubseqKernel(2, decay=0.1)
        k.K(X)
        #print inputs
        #print labels
        print X
        m = GPy.models.GPRegression(X, labels, kernel=k)
        m['fixsubsk.decay'].constrain_bounded(0.0001, 1.0)
        m['.*noise.*'].constrain_fixed(0.3)
        print m
        print m['.*coefs']
        print m.checkgrad(verbose=True)
        m.optimize_restarts(messages=True, num_restarts=5)
        print m.checkgrad(verbose=True)
        print m
        print m['.*coefs']
        
        for elem in zip(scaler.inverse_transform(m.predict(X)[0]), 
                        scaler.inverse_transform(labels)):
            print elem

if __name__ == "__main__":
    print("Running unit tests, please be (very) patient...")
    unittest.main()
