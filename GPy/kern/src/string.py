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
        #self.X_data = X_data
        self.sim = self.hard_match
        self.calc = self.calc_k2_all

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
            symm = True
        else:
            symm = False
        result = np.zeros(shape=(len(X), len(X2)))
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                #x1 = self.X_data[int(index1[0])]
                #x2 = self.X_data[int(index2[0])]
                if symm and (j > i):
                    result[i, j] = result[j, i]
                else:
                    result[i, j] = self.calc(x1, x2)
        return result

    def Kdiag(self, X):
        result = np.zeros(shape=(len(X)))
        for i, x1 in enumerate(X):
            #x1 = self.X_data[int(index1[0])]
            result[i] = self.calc(x1, x1)
        return result

    def calc_k2_all(self, s1, s2):
        result_i = self.calc_k3(s1, s2)
        print result_i
        return result_i.dot(self.order_coefs)

    def calc_k2(self, s1, s2):
        """
        Kernel calculation (based on Lodhi, Cancedda)
        and Shogun implementation
        """
        n = len(s1)
        m = len(s2)
        Kp = np.zeros(shape=(self.length, n, m))
        decay = self.decay
        Ki = np.zeros(shape=(self.length))

        for j in xrange(n):
            for k in xrange(m):
                Kp[0][j][k] = 1.0

        for i in xrange(self.length - 1): # Kp is not needed for p == self.length
            for j in xrange(n - 1):
                Kpp = 0.0
                for k in xrange(m - 1):
                    Kpp = decay * Kpp + decay * decay * self.sim(s1[j], s2[k]) * Kp[i][j][k]
                    print Kpp
                    Kp[i + 1][j + 1][k + 1] = decay * Kp[i + 1][j][k + 1] + Kpp

        for i in xrange(self.length):
            for j in xrange(i, n): # s1[:i-1] is always zero.
                for k in xrange(m):
                    Ki[i] += decay * decay * self.sim(s1[j], s2[k]) * Kp[i][j][k]
        return Ki

    def hard_match(self, s1, s2):
        return int(s1 == s2)

    def calc_k3(self, s1, s2):
        """
        Kernel calculation (based on Lodhi, Cancedda)
        and Shogun implementation
        """
        n = len(s1)
        m = len(s2)
        Kp = np.zeros(shape=(self.length, n, m))
        dKp = np.zeros(shape=(self.length, n, m))
        decay = self.decay
        Ki = np.zeros(shape=(self.length))

        for j in xrange(n):
            for k in xrange(m):
                Kp[0][j][k] = 1.0
                dKp[0][j][k] = 0.0

        for i in xrange(self.length - 1): # Kp is not needed for p == self.length
            for j in xrange(n - 1):
                #Kpp = 0.0
                #dKpp = 0.0
                Kpp = np.zeros(shape=(m))
                for k in xrange(1, m):
                #for k in xrange(m - 1):
                    Kpp[k] = decay * Kpp[k - 1] + decay * decay * self.sim(s1[j], s2[k - 1]) * Kp[i][j][k - 1]
                    #Kpp = decay * Kpp + decay * decay * self.sim(s1[j], s2[k]) * Kp[i][j][k]
                    #Kp[i + 1][j + 1][k + 1] = decay * Kp[i + 1][j][k + 1] + Kpp
                    
                #print Kpp
                Kp[i + 1][j + 1] = decay * Kp[i + 1][j] + Kpp

        for i in xrange(self.length):
            for j in xrange(i, n): # s1[:i-1] is always zero.
                for k in xrange(m):
                    Ki[i] += decay * decay * self.sim(s1[j], s2[k]) * Kp[i][j][k]
        
        # Gradients for coeficients are simply the fixed kernel evals.
        self.order_coefs.gradient = Ki.copy()
        return Ki


    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        We assume the gradients were already calculated inside K.
        """
        self.decay.gradient *= dL_dK
        self.order_coefs *= dL_dK