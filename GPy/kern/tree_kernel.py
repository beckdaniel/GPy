
from kernpart import Kernpart
import numpy as np
import sys


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
        self.name = 'tree_kernel'
        self.decay = decay
        self.branch = branch
        #print "Inside TK init:" + str(mock)
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
        if self.mock:
            #print "mock"
            target += np.array([(self.decay + self.branch + len(x[0])) for x in X])
        else:
            #print "not mock"
            pass

    def Kdiag(self, X, target):
        pass
