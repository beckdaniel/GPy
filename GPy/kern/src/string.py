import numpy as np
from .kern import Kern
import copy
from ...core.parameterization import Param
from paramz.transformations import Logexp


class AllSubStringKernel(Kern):
    """
    String Kernel
    """
    def __init__(self, name='sk'):
        super(AllSubStringKernel, self).__init__(1, 1, name)

    def calc_k(self, s1, s2, decay=1.0):
        """
        Do the actual kernel calculation
        """
        n = len(s1)
        m = len(s2)
        dp = [[decay] * (m + 1)] * (n + 1)
        p = [0.0] * (m + 1)
        for i in xrange(1, n + 1):
            #last = 0
            p[0] = 0
            for k in xrange(1, m + 1):
                if s2[k - 1] == s1[i - 1]:
                    sim = 1
                else:
                    sim = 0
                p[k] = p[k - 1] + (dp[i - 1][k - 1] * sim * decay)
                #p[k] = p[last]
                #if s2[k - 1] == s1[i - 1]:
                #    p[k] = p[last] + (dp[i - 1][k - 1] * sim * decay)
                #    last = k
            for k in xrange(1, m + 1):
                dp[i][k] = dp[i - 1][k] + p[k]
            #print dp
        return dp[n][m]
        

class FixedLengthSubseqKernel(Kern):
    """
    Fixed length subsequences Kernel.
    """
    def __init__(self, length, X_data=None, name='fixsubsk', decay=1.0, order_coefs=None):
        super(FixedLengthSubseqKernel, self).__init__(1, None, name)
        self.length = length
        self.decay = Param('decay', decay, Logexp())
        self.link_parameter(self.decay)
        if order_coefs is None:
            order_coefs = [1.0] * length
        self.order_coefs = Param('order_coefs', order_coefs, Logexp())
        self.link_parameter(self.order_coefs)
        self.X_data = X_data

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
            symm = True
        else:
            symm = False
        result = np.zeros(shape=(len(X), len(X2)))
        for i, index1 in enumerate(X):
            for j, index2 in enumerate(X2):
                x1 = self.X_data[int(index1[0])]
                x2 = self.X_data[int(index2[0])]
                if symm and (j > i):
                    result[i, j] = result[j, i]
                else:
                    result[i, j] = self.calc_k(x1, x2)
        return result

    def Kdiag(self, X):
        result = np.zeros(shape=(len(X)))
        for i, index1 in enumerate(X):
            x1 = self.X_data[int(index1[0])]
            result[i] = self.calc_k(x1, x1)
        return result

    def update_gradients_full(self, dL_dK, X, X2=None):
        pass
                    
    def calc_k2(self, s1, s2):
        """
        Kernel calculation (based on Lodhi, Cancedda)
        and Shogun implementation
        """
        n = len(s1)
        m = len(s2)
        Kp = np.zeros(shape=(self.length + 1, n, m))
        decay = self.decay
        for j in xrange(n):
            for k in xrange(m):
                Kp[0][j][k] = 1.0
        result = 0.0
        for i in xrange(self.length):
            for j in xrange(n - 1):
                Kpp = 0.0
                for k in xrange(m - 1):
                    term = (decay * (s1[j] == s2[k]) * Kp[i][j][k])
                    Kpp = decay * (Kpp + term)
                    Kp[i + 1][j + 1][k + 1] = decay * Kp[i + 1][j][k + 1] + Kpp
                    result += decay * decay * term
                result += decay * decay * (s1[j] == s2[m - 1]) * Kp[i][j][m - 1]
            for k in xrange(m):
                result += decay * decay * (s1[n - 1] == s2[k]) * Kp[i][n - 1][k]
        #for i in xrange(self.length):
        #    for j in xrange(n):
        #        for k in xrange(m):
        #            result += decay * decay * (s1[j] == s2[k]) * Kp[i][j][k]
        return result

    def calc_k(self, s1, s2):
        """
        Kernel calculation (based on Lodhi, Cancedda)
        and Shogun implementation
        """
        n = len(s1)
        m = len(s2)
        Kp = np.zeros(shape=(self.length + 1, n, m))
        decay = self.decay
        for j in xrange(n):
            for k in xrange(m):
                Kp[0][j][k] = 1.0

        for i in xrange(self.length):
            for j in xrange(n - 1):
                Kpp = 0.0
                for k in xrange(m - 1):
                    Kpp = decay * Kpp + decay * decay * (s1[j] == s2[k]) * Kp[i][j][k]
                    Kp[i + 1][j + 1][k + 1] = decay * Kp[i + 1][j][k + 1] + Kpp
        result = 0.0
        for i in xrange(self.length):
            result_i = 0.0
            for j in xrange(n):
                for k in xrange(m):
                    result_i += decay * decay * (s1[j] == s2[k]) * Kp[i][j][k]
            #print result_i
            result += self.order_coefs[i] * result_i
        #print Kp
        return result
