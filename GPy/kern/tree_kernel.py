
from kernpart import Kernpart
import numpy as np
import sys
import nltk

class TreeKernel(Kernpart):
    """A convolution kernel that compares two trees. See Moschitti(2006).
    """
    
    def __init__(self, decay=1, branch=1, mock=False):
        """blah
        """
        try:
            import nltk
        except ImportError:
            sys.stderr.write("Tree Kernels need NLTK. Install it using \"pip install nltk\"")
            raise
        self.input_dim = 1 # A hack. Actually tree kernels have lots of dimensions.
        self.num_params = 2
        self.name = 'tk'
        self.decay = decay
        self.branch = branch
        self.mock = mock
        
    def _get_params(self):
        return np.hstack((self.decay, self.branch))

    def _set_params(self, x):
        self.decay = x[0]
        self.branch = x[1]

    def _get_param_names(self):
        return ['decay', 'branch']

    def K(self, X, X2, target):
        """
        The mock parameter is mainly for testing and debugging.
        """
        if X2 == None:
            X2 = X
        if self.mock:
            # we have to ensure positive semi-definiteness, so we build a triangular matrix
            # and them multiply it by its transpose (like a "reverse" Cholesky)
            result = np.array([[(self.decay + self.branch + len(x1) + len(x2)) for x1 in X] for x2 in X2])
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if i > j:
                        result[i][j] = 0
            #print result
            target += result.T.dot(result)
            #target += np.array([[(self.decay + self.branch) for x1 in X] for x2 in X2])
        else:
            #print X
            for i, x1 in enumerate(X):
                for j, x2 in enumerate(X2):
                    x1 = nltk.Tree(x1[0])
                    x2 = nltk.Tree(x2[0])
                    result = 0
                    for node1 in x1.treepositions():
                        for node2 in x2.treepositions():
                            result += self.delta(x1[node1], x2[node2])
                    target[i][j] += result

    def Kdiag(self, X, target):
        if self.mock:
            result = np.array([[(self.decay + self.branch + len(x1) + len(x2)) for x1 in X] for x2 in X])
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if i > j:
                        result[i][j] = 0
            target += np.diag(result.T.dot(result))
            #target += np.array([self.decay + self.branch for i in range(X.shape[0])])
                               
    def dK_dtheta(self, dL_dK, X, X2, target):
        if self.mock:
            #print dL_dK
            #print X
            s = np.sum(dL_dK)
            target += [s, s]

    def delta(self, node1, node2):
        result = np.abs(len(node1) - len(node2))
        print result
        return result
        
