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
    def __init__(self, length, name='fixsubsk', decay=1.0, order_coefs=None):
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
        #self.calc = self.calc_vectorized

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
            symm = True
        else:
            symm = False
        result = np.zeros(shape=(len(X), len(X2)))
        ddecay = np.zeros(shape=(len(X), len(X2)))
        dcoefs = np.zeros(shape=(len(self.order_coefs), len(X), len(X2)))
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                #print x1
                #print x2
                #x1 = self.X_data[int(index1[0])]
                #x2 = self.X_data[int(index2[0])]
                if symm and (j > i):
                    result[i, j] = result[j, i]
                    ddecay[i, j] = ddecay[j, i]
                    dcoefs[:, i, j] = dcoefs[:, j, i]
                else:
                    result[i, j], ddecay[i, j], dcoefs[:, i, j] = self.calc(x1[0], x2[0])
                    #print result[i, j], ddecay[i, j], dcoefs[:, i, j]
        
        self.decay_grad = ddecay
        #print "COEFS GRADS AT END OF K:"
        #print dcoefs
        #print result
        self.order_coefs_grad = dcoefs
        return result

    def Kdiag(self, X):
        result = np.zeros(shape=(len(X)))
        for i, x1 in enumerate(X):
            #x1 = self.X_data[int(index1[0])]
            result[i] = self.calc(x1, x1)[0]
        return result

    def calc_k2_all(self, s1, s2):
        #Ki, dKi = self.calc_k3(s1, s2)
        Ki, dKi = self.calc_vectorized(s1, s2)
        ddecay = dKi.sum()
        dcoefs = Ki.copy()
        #print "Ki"
        #print Ki
        #print self.order_coefs
        return Ki.dot(self.order_coefs), ddecay, dcoefs

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
        dKi = np.zeros(shape=(self.length))

        for j in xrange(n):
            for k in xrange(m):
                Kp[0][j][k] = 1.0
                dKp[0][j][k] = 0.0

        for i in xrange(self.length - 1): # Kp is not needed for p == self.length
            for j in xrange(n - 1):
                #Kpp = 0.0
                #dKpp = 0.0
                Kpp = np.zeros(shape=(m))
                dKpp = np.zeros(shape=(m))
                for k in xrange(1, m):
                #for k in xrange(m - 1):
                    Kpp[k] = decay * Kpp[k - 1] + decay * decay * self.sim(s1[j], s2[k - 1]) * Kp[i][j][k - 1]
                    dKpp[k] = (Kpp[k - 1] + (decay * dKpp[k - 1]) +
                               (2 * decay * self.sim(s1[j], s2[k - 1]) * Kp[i][j][k - 1]) +
                               (decay * decay * self.sim(s1[j], s2[k - 1]) * dKp[i][j][k - 1])
                           )
                    #Kpp = decay * Kpp + decay * decay * self.sim(s1[j], s2[k]) * Kp[i][j][k]
                    #Kp[i + 1][j + 1][k + 1] = decay * Kp[i + 1][j][k + 1] + Kpp
                #print Kpp
                Kp[i + 1][j + 1] = decay * Kp[i + 1][j] + Kpp
                dKp[i + 1][j + 1] = Kp[i + 1][j] + decay * dKp[i + 1][j] + dKpp

        for i in xrange(self.length):
            for j in xrange(i, n): # s1[:i-1] is always zero.
                for k in xrange(m):
                    Ki[i] += decay * decay * self.sim(s1[j], s2[k]) * Kp[i][j][k]
                    dKi[i] += (2 * decay * self.sim(s1[j], s2[k]) * Kp[i][j][k] +
                            decay * decay * self.sim(s1[j], s2[k]) * dKp[i][j][k])
                    #print dKn
        
        # Gradients for coeficients are simply the fixed kernel evals.
        #print dKi
        #self.order_coefs.gradient = Ki.copy()
        #self.decay.gradient = dKi.sum()
        #print "DECAY GRAD AT END OF K: %.5f" % self.decay.gradient
        #print s1
        #print s2
        #print Ki
        return Ki, dKi.sum()


    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        We assume the gradients were already calculated inside K.
        """
        pass
        #self.decay.gradient = np.sum(self.decay_grad * dL_dK)
        #for i in xrange(self.length):
        #    self.order_coefs.gradient[i] = np.sum(self.order_coefs_grad[i] * dL_dK)


    def calc_vectorized(self, s1, s2):
        """
        Vectorized version.
        """
        n = len(s1)
        m = len(s2)
        length = len(self.order_coefs)
        Kp = np.zeros(shape=(2, length + 1, n, m))
        Kp[0,0,:,:] = 1.0

        # store sim(j, k) values
        S = np.zeros(shape=(n,m))
        for j in xrange(n):
            for k in xrange(m):
                S[j,k] = self.sim(s1[j],s2[k])
        
        # store triangular matrix with powers of decay (to side-step recursion)
        max_len = max(n, m)
        D = np.zeros((max_len,max_len))
        d1, d2 = np.indices(D.shape)
        for k in xrange(max_len):
            D[d2-k == d1] = self.decay ** k 
        #print D

        # Some precomputation, I expect we'll need this as tensor
        # This stores the similarities multiplied by the decays
        DS = self.decay * self.decay * S[:, :-1]
        
        for i in xrange(length):
            Kp[1, i, :, 1:] = (DS * Kp[0, i, :, :-1]).dot(D[1:m, 1:m])
            Kp[0, i + 1, 1:, :] = Kp[1, i, :-1, :].T.dot(D[1:n, 1:n]).T #Last colons are useless, put them for clarity

        Ki = np.sum(np.sum(S * Kp[0, :-1], axis=1), axis=1) * self.decay * self.decay
        return Ki, self.order_coefs.copy()
        #print 'FINAL', result
