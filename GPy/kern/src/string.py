import numpy as np
from .kern import Kern
import copy


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
    def __init__(self, length, name='fixsubsk', decay=1.0, l_coef=None):
        super(FixedLengthSubseqKernel, self).__init__(1, 1, name)
        self.length = length
        self.decay = decay
        if l_coef:
            self.l_coef = l_coef
        else:
            self.l_coef = [1.0] * length


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

                
        
