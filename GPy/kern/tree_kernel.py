
from kernpart import Kernpart
import numpy as np
import sys
import nltk

class TreeKernel(Kernpart):
    """A convolution kernel that compares two trees. See Moschitti(2006).
    """
    
    def __init__(self, decay=1, branch=1, mode="naive"):
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
        if mode == "mock":
            self.K = self.K_mock
            self.Kdiag = self.Kdiag_mock
            self.dK_dtheta = self.dK_dtheta_mock
        elif mode == "naive":
            self.K = self.K_naive
            self.Kdiag = self.Kdiag_naive
            self.dK_dtheta = self.dK_dtheta_naive
        elif mode == "cache":
            self.K = self.K_cache
            self.Kdiag = self.Kdiag_cache
            self.dK_dtheta = self.dK_dtheta_cache
        self.cache = {}
        self.cache_ddecay = {}
        self.cache_dbranch = {}
        
    def _get_params(self):
        return np.hstack((self.decay, self.branch))

    def _set_params(self, x):
        self.decay = x[0]
        self.branch = x[1]

    def _get_param_names(self):
        return ['decay', 'branch']


    ##############
    # MOCK METHODS
    #
    # These were designed to mainly test how to couple
    # TreeKernels to the GPy kernel API.
    ##############

    def K_mock(self, X, X2, target):
        """
        The mock parameter is mainly for testing and debugging.
        """
        if X2 == None:
            X2 = X
        # we have to ensure positive semi-definiteness, so we build a triangular matrix
        # and them multiply it by its transpose (like a "reverse" Cholesky)
        result = np.array([[(self.decay + self.branch + len(x1) + len(x2)) for x1 in X] for x2 in X2])
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if i > j:
                    result[i][j] = 0
        target += result.T.dot(result)

    def Kdiag_mock(self, X, target):
        result = np.array([[(self.decay + self.branch + len(x1) + len(x2)) for x1 in X] for x2 in X])
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if i > j:
                    result[i][j] = 0
        target += np.diag(result.T.dot(result))

    def dK_dtheta_mock(self, dL_dK, X, X2, target):
        if X2 == None:
            X2 = X
        s = np.sum(dL_dK)
        target += [s, s]

    ###############
    # NAIVE METHODS
    #
    # These are implemented by just plugging the kernel formulas
    # into code. They have exponential complexity. They are still useful
    # for testing and comparinng results with the other implementations
    # using small toy data.
    ###############

    def K_naive(self, X, X2, target):
        if X2 == None:
            X2 = X
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                t1 = nltk.Tree(x1[0])
                t2 = nltk.Tree(x2[0])
                result = 0
                for pos1 in t1.treepositions():
                    node1 = t1[pos1]
                    for pos2 in t2.treepositions():
                        result += self.delta_naive(node1, t2[pos2])
                target[i][j] += result

    def Kdiag_naive(self, X, target):
        for i, x1 in enumerate(X):
            t1 = nltk.Tree(x1[0])
            result = 0
            for pos1 in t1.treepositions():
                node1 = t1[pos1]
                for pos2 in t1.treepositions():
                    result += self.delta_naive(node1, t1[pos2])
            target[i] += result

    def dK_dtheta_naive(self, dL_dK, X, X2, target):
        if X2 == None:
            X2 = X
        s = np.sum(dL_dK)
        dK_ddecay = 0
        dK_dbranch = 0
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                if i <= j:
                    t1 = nltk.Tree(x1[0])
                    t2 = nltk.Tree(x2[0])
                    for pos1 in t1.treepositions():
                        node1 = t1[pos1]
                        for pos2 in t2.treepositions():
                            d, b = self.delta_params_naive(node1, t2[pos2])
                            dK_ddecay += d 
                            dK_dbranch += b 
                            if i < j:
                                dK_ddecay += d
                                dK_dbranch += b 
        target += [dK_ddecay * s, dK_dbranch * s]

    def delta_naive(self, node1, node2):
        # zeroth case -> leaves
        if type(node1) == str or type(node2) == str:
            return 0
        # first case
        if node1.node != node2.node:
            return 0
        if node1.productions()[0] != node2.productions()[0]:
            return 0
        # second case -> preterms
        if node1.height() == 2 and node2.height() == 2:
            return self.decay
        # third case
        result = self.decay
        for i, child in enumerate(node1): #node2 has the same children
            result *= (self.branch + self.delta_naive(node1[i], node2[i]))
        return result

    def delta_params_naive(self, node1, node2):
        # zeroth case -> leaves
        if type(node1) == str or type(node2) == str:
            return (0, 0)
        # first case
        if node1.node != node2.node:
            return (0, 0)
        if node1.productions()[0] != node2.productions()[0]:
            return (0, 0)
        # second case -> preterms
        if node1.height() == 2 and node2.height() == 2:
            return (1, 0)
        # third case
        prod = 1
        sum_delta = 0
        sum_decay = 0
        sum_branch = 0
        for i, child in enumerate(node1): #node2 has the same children
            g = self.branch + self.delta_naive(node1[i], node2[i])
            prod *= g
            sum_delta += g
            d, b = self.delta_params_naive(node1[i], node2[i])
            sum_decay += d
            sum_branch += (1 + b)
        h = (prod * self.decay) / float(sum_delta)
        ddecay = prod + (h * sum_decay)
        dbranch = h * sum_branch
        return (ddecay, dbranch)

    ###############
    # CACHE METHODS
    #
    # 
    ###############

    def K_cache(self, X, X2, target):
        if X2 == None:
            X2 = X
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                t1 = nltk.Tree(x1[0])
                t2 = nltk.Tree(x2[0])
                result = 0
                self.cache = {} # DP
                for pos1 in t1.treepositions(order="postorder"):
                    node1 = t1[pos1]
                    for pos2 in t2.treepositions(order="postorder"):
                        node2 = t2[pos2]
                        key = (pos1, pos2)
                        result += self.delta_cache(node1, node2, key)
                target[i][j] += result

    def Kdiag_cache(self, X, target):
        for i, x1 in enumerate(X):
            t1 = nltk.Tree(x1[0])
            result = 0
            self.cache = {} # DP
            for pos1 in t1.treepositions(order="postorder"):
                node1 = t1[pos1]
                for pos2 in t1.treepositions(order="postorder"):
                    node2 = t1[pos2]
                    key = (pos1, pos2)
                    result += self.delta_cache(node1, node2, key)
            target[i] += result

    def dK_dtheta_cache(self, dL_dK, X, X2, target):
        if X2 == None:
            X2 = X
        s = np.sum(dL_dK)
        dK_ddecay = 0
        dK_dbranch = 0
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                if i <= j:
                    t1 = nltk.Tree(x1[0])
                    t2 = nltk.Tree(x2[0])
                    self.cache
                    self.cache_ddecay = {}
                    self.cache_dbranch = {}
                    for pos1 in t1.treepositions(order="postorder"):
                        node1 = t1[pos1]
                        for pos2 in t2.treepositions(order="postorder"):
                            node2 = t2[pos2]
                            key = (pos1, pos2)
                            d, b = self.delta_params_cache(node1, node2, key)
                            dK_ddecay += d 
                            dK_dbranch += b 
                            if i < j:
                                dK_ddecay += d
                                dK_dbranch += b 
        target += [dK_ddecay * s, dK_dbranch * s]

    def delta_cache(self, node1, node2, key):
        # zeroth case -> leaves
        if type(node1) == str or type(node2) == str:
            return 0
        # first case
        if node1.node != node2.node: #optimization
            self.cache[key] = 0
            return 0
        if node1.productions()[0] != node2.productions()[0]:
            self.cache[key] = 0
            return 0
        # second case -> preterms
        if node1.height() == 2 and node2.height() == 2:
            self.cache[key] = self.decay
            return self.decay
        # third case
        result = self.decay
        for i, child in enumerate(node1): #node2 has the same children
            #result *= (self.branch + self.delta(node1[i], node2[i])) 
            child_key = (tuple(list(key[0]) + [i]),
                         tuple(list(key[1]) + [i]))
            result *= (self.branch + self.cache[child_key])
        self.cache[key] = result
        return result

    def delta_params_cache(self, node1, node2, key):
        # zeroth case -> leaves
        if type(node1) == str or type(node2) == str:
            return (0, 0)
        # first case
        if node1.node != node2.node:
            self.cache[key] = 0
            self.cache_ddecay[key] = 0
            self.cache_dbranch[key] = 0
            return (0, 0)
        if node1.productions()[0] != node2.productions()[0]:
            self.cache[key] = 0
            self.cache_ddecay[key] = 0
            self.cache_dbranch[key] = 0
            return (0, 0)
        # second case -> preterms
        if node1.height() == 2 and node2.height() == 2:
            self.cache[key] = self.decay
            self.cache_ddecay[key] = 1
            self.cache_dbranch[key] = 0
            return (1, 0)
        # third case
        prod = 1
        sum_delta = 0
        sum_decay = 0
        sum_branch = 0
        ###
        # Now this is ugly: to cope with DP, we need to calculate delta again...
        # There should be a better way to do this...
        delta_result = self.decay
        # End of uglyness =P
        ###
        for i, child in enumerate(node1): #node2 has the same children
            child_key = (tuple(list(key[0]) + [i]),
                         tuple(list(key[1]) + [i]))
            #g = self.branch + self.delta(node1[i], node2[i])
            g = self.branch + self.cache[child_key]
            prod *= g
            sum_delta += g
            #d, b = self.delta_params(node1[i], node2[i])
            d = self.cache_ddecay[child_key]
            b = self.cache_dbranch[child_key]
            sum_decay += d
            sum_branch += (1 + b)
            ###
            # Uglyness begin
            delta_result *= g
            # Uglyness end
            ###
        h = (prod * self.decay) / float(sum_delta)
        ddecay = prod + (h * sum_decay)
        dbranch = h * sum_branch
        self.cache[key] = delta_result
        self.cache_ddecay[key] = ddecay
        self.cache_dbranch[key] = dbranch
        return (ddecay, dbranch)
