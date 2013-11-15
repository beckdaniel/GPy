
from kernpart import Kernpart
import numpy as np
from collections import defaultdict
import sys
import sympy as sp
from sympy.utilities.lambdify import lambdastr
import nltk
import marshal

class TreeKernel(Kernpart):
    """A convolution kernel that compares two trees. See Moschitti(2006).
    """
    
    def __init__(self, decay=1, branch=1, mode="naive", normalize=False):
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
        self.normalize = normalize
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
        elif mode == "opt":
            self.K = self.K_fast
            self.Kdiag = self.Kdiag_fast
            self.dK_dtheta = self.dK_dtheta_fast
            self.tree_cache = {}
        
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
        #sum_delta = 0
        sum_decay = 0
        sum_branch = 0
        for i, child in enumerate(node1): #node2 has the same children
            g = float(self.branch + self.delta_naive(node1[i], node2[i]))
            prod *= g
            #sum_delta += g
            d, b = self.delta_params_naive(node1[i], node2[i])
            sum_decay += (d / g)
            sum_branch += ((1 + b) / g)
        #h = (prod * self.decay) / float(sum_delta)
        h = self.decay * prod
        ddecay = prod + (h * sum_decay)
        dbranch = h * sum_branch
        return (ddecay, dbranch)

    ###############
    # CACHE METHODS
    #
    # These methods use DP to reach polynomial complexity.
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
                    self.cache = {}
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
        #sum_delta = 0
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
            g = float(self.branch + self.cache[child_key])
            prod *= g
            #sum_delta += g
            #d, b = self.delta_params(node1[i], node2[i])
            d = self.cache_ddecay[child_key]
            b = self.cache_dbranch[child_key]
            sum_decay += (d / g)
            sum_branch += ((1 + b) / g)
            ###
            # Uglyness begin
            delta_result *= g
            # Uglyness end
            ###
        h = (prod * self.decay)
        ddecay = prod + (h * sum_decay)
        dbranch = h * sum_branch
        self.cache[key] = delta_result
        self.cache_ddecay[key] = ddecay
        self.cache_dbranch[key] = dbranch
        return (ddecay, dbranch)

    ##############
    # FAST METHODS
    #
    # These methods also use DP but have an implementation
    # that rely less on NLTK methods (which are slow because
    # they traverse the entire tree multiple times).
    ##############

    def K_fast(self, X, X2, target):
        # If X == X2, K is symmetric
        if X2 == None:
            X2 = X
            symmetrize = True
        else:
            symmetrize = False
        # We are going to store everything first and in the end we
        # normalize it before adding to target.
        K_results = np.zeros(shape=(len(X), len(X2)))
        ddecays = np.zeros(shape=(len(X), len(X2)))
        dbranches = np.zeros(shape=(len(X), len(X2)))
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                if symmetrize and i > j:
                    continue
                t1, t2 = self._get_trees(x1, x2)                
                self.delta_result = 0
                self.ddecay = 0
                self.dbranch = 0
                self.cache = {} # DP
                self.cache_ddecay = {}
                self.cache_dbranch = {}
                # Recursive calculation happens here.
                self.delta_fast(t1, t2, ((),()))
                # End recursive calculation
                K_results[i][j] = self.delta_result
                ddecays[i][j] = self.ddecay
                dbranches[i][j] = self.dbranch
                if symmetrize and i != j:
                    K_results[j][i] = self.delta_result
                    ddecays[j][i] = self.ddecay
                    dbranches[j][i] = self.dbranch
        # Now we normalize everything
        # hack: we are going to calculate the gradients too
        # and store them for later use.
        self.ddecay_results = np.zeros(shape=(len(X), len(X2)))
        self.dbranch_results = np.zeros(shape=(len(X), len(X2)))
        if self.normalize:
            if symmetrize:
                self._normalize_K_sym(X, K_results, ddecays, dbranches, target)
            else:
                self._normalize_K(X, X2, K_results, ddecays, dbranches, target)
        else:
            target += K_results
            self.ddecay_results = ddecays
            self.dbranch_results = dbranches
                
    def Kdiag_fast(self, X, target):
        # We are going to calculate gradients too and
        # save them for later use
        if self.normalize:
            target += np.ones(shape=(len(X),))
            self.ddecay_diag = np.zeros(shape=(len(X),))
            self.dbranch_diag = np.zeros(shape=(len(X),))
        else:
            K_vec, ddecay_vec, dbranch_vec = self._diag_calculations(X)
            target += K_vec
            self.ddecay_diag = ddecay_vec
            self.dbranch_diag = dbranch_vec
    
    def _diag_calculations(self, X):
        K_vec = np.zeros(shape=(len(X),))
        ddecay_vec = np.zeros(shape=(len(X),))
        dbranch_vec = np.zeros(shape=(len(X),))
        for i, x1 in enumerate(X):
            t1, t2 = self._get_trees(x1, x1)
            self.delta_result = 0
            self.ddecay = 0
            self.dbranch = 0
            self.cache = {} # DP
            self.cache_ddecay = {}
            self.cache_dbranch = {}
            # Recursive calculation happens here.
            self.delta_fast(t1, t2, ((),()))
            # End recursive calculation
            K_vec[i] = self.delta_result
            ddecay_vec[i] = self.ddecay
            dbranch_vec[i] = self.dbranch
        return (K_vec, ddecay_vec, dbranch_vec)

    def dK_dtheta_fast(self, dL_dK, X, X2, target):
        s_like = np.sum(dL_dK)
        s_decay = np.sum(self.ddecay_results)
        s_branch = np.sum(self.dbranch_results)
        target += [s_like * s_decay, s_like * s_branch]

    def delta_fast(self, t1, t2, key):
        for i1, child1 in enumerate(t1):
            if type(child1) != str:
                child_key = (tuple(list(key[0]) + [i1]), key[1])
                self.delta_fast(child1, t2, child_key)
        self.fill_cache(t1, t2, key)
            
    def fill_cache(self, t1, t2, key):
        for i2, child2 in enumerate(t2):
            if type(child2) != str:
                child_key = (key[0], tuple(list(key[1]) + [i2]))
                self.fill_cache(t1, child2, child_key)

        # first case: diff nodes or diff lengths imply diff productions
        if (t1.node != t2.node or len(t1) != len(t2)):
            self.cache[key] = 0
            self.cache_ddecay[key] = 0
            self.cache_dbranch[key] = 0
            return

        # second case: preterms
        if (type(t1[0]) == str or type(t2[0]) == str):
            if t1[0] == t2[0]:
                self.cache[key] = self.decay
                self.delta_result += self.decay
                self.cache_ddecay[key] = 1
                self.ddecay += 1
                self.cache_dbranch[key] = 0
            else:
                self.cache[key] = 0
                self.cache_ddecay[key] = 0
                self.cache_dbranch[key] = 0
            return
        
        # third case, non-preterms with different children
        for i, ch in enumerate(t1):
            if ch.node != t2[i].node: 
                self.cache[key] = 0
                self.cache_ddecay[key] = 0
                self.cache_dbranch[key] = 0
                return

        # fourth case, non-preterms with same children, we do the recursion
        prod = 1
        sum_decay = 0
        sum_branch = 0
        d_result = self.decay
        for i, child in enumerate(t1): #t2 has the same children
            child_key = (tuple(list(key[0]) + [i]),
                         tuple(list(key[1]) + [i]))
            g = float(self.branch + self.cache[child_key])
            prod *= g
            d = self.cache_ddecay[child_key]
            b = self.cache_dbranch[child_key]
            sum_decay += (d / g)
            sum_branch += ((1 + b) / g)
            d_result *= g
        #print "TK PROD: %f" % prod
        h = (prod * self.decay)
        # update delta
        self.cache[key] = d_result
        self.delta_result += d_result
        # update ddecay
        ddecay = prod + (h * sum_decay)
        self.cache_ddecay[key] = ddecay
        self.ddecay += ddecay
        # update dbranch
        dbranch = h * sum_branch
        self.cache_dbranch[key] = dbranch
        self.dbranch += dbranch

    def _get_trees(self, x1, x2):
        """
        Return the trees corresponding to input strings
        in the tree cache. If one do not exist, it builds
        a new one and store it in the cache.
        """
        try:
            t1 = self.tree_cache[x1[0]]
        except KeyError:
            t1 = nltk.Tree(x1[0])
            self.tree_cache[x1[0]] = t1
        try:
            t2 = self.tree_cache[x2[0]]
        except KeyError:
            t2 = nltk.Tree(x2[0])
            self.tree_cache[x2[0]] = t2
        return (t1, t2)

    def _normalize_K_sym(self, X, K_results, ddecays, dbranches, target):
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X):
                if i > j:
                    continue
                # diagonals, gradients are zero
                if i == j:
                    target[i][j] += 1
                    continue
                # calculate some intermediate values
                fac = K_results[i][i] * K_results[j][j]
                root = np.sqrt(fac)
                denom = 2 * fac
                K_norm = K_results[i][j] / root
                # update K
                target[i][j] += K_norm
                # update ddecay
                self.ddecay_results[i][j] = ((ddecays[i][j] / root) -
                                             ((K_norm / denom) *
                                              ((ddecays[i][i] * K_results[j][j]) +
                                               (K_results[i][i] * ddecays[j][j]))))
                self.dbranch_results[i][j] = ((dbranches[i][j] / root) -
                                              ((K_norm / denom) *
                                               ((dbranches[i][i] * K_results[j][j]) +
                                                (K_results[i][i] * dbranches[j][j]))))
                target[j][i] += K_norm
                self.ddecay_results[j][i] = self.ddecay_results[i][j]
                self.dbranch_results[j][i] = self.dbranch_results[i][j]

    def _normalize_K(self, X, X2, K_results, ddecays, dbranches, target):
        X_K, X_ddecay, X_dbranch = self._diag_calculations(X)
        X2_K, X2_ddecay, X2_dbranch = self._diag_calculations(X2)
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                # calculate some intermediate values
                fac = X_K[i] * X2_K[j]
                root = np.sqrt(fac)
                denom = 2 * fac
                K_norm = K_results[i][j] / root
                # update K
                target[i][j] += K_norm
                # update ddecay
                self.ddecay_results[i][j] = ((ddecays[i][j] / root) -
                                             ((K_norm / denom) *
                                              ((X_ddecay[i] * X2_K[j]) +
                                               (X_K[i] * X2_ddecay[j]))))
                self.dbranch_results[i][j] = ((dbranches[i][j] / root) -
                                              ((K_norm / denom) *
                                               ((X_dbranch[i] * X2_K[j]) +
                                                (X_K[i] * X2_dbranch[j]))))

        
class SimpleFastTreeKernel(Kernpart):
    """
    Moschitti's FTK but only with decay hyperparameter.
    """
    
    def __init__(self, decay=1, has_root=False):
        try:
            import nltk
        except ImportError:
            sys.stderr.write("Tree Kernels need NLTK. Install it using \"pip install nltk\"")
            raise
        self.input_dim = 1 # A hack. Actually tree kernels have lots of dimensions.
        self.num_params = 1
        self.name = 'sftk'
        self.decay = decay
        self.cache = {}
        self.has_root = has_root
        
    def _get_params(self):
        return self.decay

    def _set_params(self, x):
        self.decay = x[0]

    def _get_param_names(self):
        return ['decay']

    def build_cache(self, X, X2=None):
        if X2 == None:
            self._build_cache_sym(X)
        else:
            self._build_cache_nsym(X, X2)

    def _build_cache_sym(self, X):
        self.cache["tree_ids"] = {}
        self.cache["node_pair_lists"] = {}
        # Temporary node cache
        node_cache = {}
        # Store trees
        for tree in X:
            tree_id = len(self.cache["tree_ids"])
            self.cache["tree_ids"].setdefault(tree[0], tree_id)
            node_cache[tree_id] = self._get_nodes(tree[0])
        # Store node pairs
        for tree1 in X:
            id1 = self.cache["tree_ids"][tree1[0]]
            self.cache["node_pair_lists"][id1] = {}
            for tree2 in X:
                id2 = self.cache["tree_ids"][tree2[0]]
                if id1 > id2: #symmetry
                    continue
                nodes1 = node_cache[id1]
                nodes2 = node_cache[id2]
                node_pairs = self._get_node_pair_list(nodes1, nodes2)
                self.cache["node_pair_lists"][id1][id2] = node_pairs

    def _get_node_pair_list(self, nodes1, nodes2):
        """
        Get two trees represented by strings and
        return the node pair list
        """
        node_list = []
        i1 = 0
        i2 = 0
        while True:
            try:
                if nodes1[i1][0] > nodes2[i2][0]:
                    i2 += 1
                elif nodes1[i1][0] < nodes2[i2][0]:
                    i1 += 1
                else:
                    while nodes1[i1][0] == nodes2[i2][0]:
                        reset2 = i2
                        while nodes1[i1][0] == nodes2[i2][0]:
                            if type(nodes1[i1][0].rhs()[0]) == str:
                                # We consider preterms as leaves
                                tup = (nodes1[i1][1], nodes2[i2][1], 0)
                            else:
                                tup = (nodes1[i1][1], nodes2[i2][1], len(nodes1[i1][0].rhs()))
                            node_list.append(tup)
                            i2 += 1
                        i1 += 1
                        i2 = reset2
            except IndexError:
                break
        node_list.sort(key=lambda x: len(x[0]), reverse=True)
        return node_list

    def _get_node_pairs(self, tree1, tree2):
        """
        Get two trees represented by strings and
        return the node pair list
        """
        try:
            id1 = self.cache["tree_ids"][tree1]
            id2 = self.cache["tree_ids"][tree2]
            return self.cache["node_pair_lists"][id1][id2]
        except KeyError:
            nodes1 = self._get_nodes(tree1)
            nodes2 = self._get_nodes(tree2)
            return self._get_node_pair_list(nodes1, nodes2)
            
    def _get_nodes(self, x):
        t = nltk.Tree(x)
        if self.has_root:
            t = t[0]
        pos = t.treepositions()
        for l in t.treepositions(order="leaves"):
            pos.remove(l)
        z = zip(t.productions(), pos)
        #import ipdb
        #ipdb.set_trace()
        z.sort()
        return z

    def Kdiag(self, X, target):
        """
        Since this is a normalized kernel, diag values are all equal to 1.
        """
        target += np.ones(shape=(len(X),))

    def K(self, X, X2, target):
        # A check to ensure that ddecays cache will always change when K changes
        self.ddecays = None
        if X2 == None:
            self.K_sym(X, target)
        else:
            self.K_nsym(X, X2, target)

    def K_sym(self, X, target):
        if self.cache == {}:
            self.build_cache(X)

        # First, we are going to calculate K for diagonal values
        # because we will need them later to normalize.
        diag_deltas, diag_ddecays = self._diag_calculations(X)

        # Second, we are going to initialize the ddecay values
        # because we are going to calculate them at the same time as K.
        K_results = np.zeros(shape=(len(X), len(X)))
        ddecays = np.zeros(shape=(len(X), len(X)))
        
        # Now we proceed for the actual calculation
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X):
                if i > j:
                    K_results[i][j] = K_results[j][i]
                    ddecays[i][j] = ddecays[j][i]
                    continue
                if i == j:
                    K_results[i][j] = 1
                    continue
                # It will always be a 1-element array
                node_list = self._get_node_pairs(x1[0], x2[0])
                try:
                    K_result, ddecay_result = self.delta(node_list)
                except:
                    print node_list
                    raise
                norm = diag_deltas[i] * diag_deltas[j]
                sqrt_norm = np.sqrt(norm)
                K_norm = K_result / sqrt_norm
                
                diff_term = ((diag_ddecays[i] * diag_deltas[j]) +
                             (diag_deltas[i] * diag_ddecays[j]))
                diff_term /= float(2 * norm)
                ddecay_norm = ((ddecay_result / sqrt_norm) -
                               (K_norm * diff_term))

                K_results[i][j] = K_norm
                ddecays[i][j] = ddecay_norm
        
        target += K_results
        self.ddecays = ddecays

    def K_nsym(self, X, X2, target):
        # First, we are going to calculate K for diagonal values
        # because we will need them later to normalize.
        diag_deltas_X, diag_ddecays_X = self._diag_calculations(X)
        diag_deltas_X2, diag_ddecays_X2 = self._diag_calculations(X2)

        # Second, we are going to initialize the ddecay values
        # because we are going to calculate them at the same time as K.
        K_results = np.zeros(shape=(len(X), len(X2)))
        ddecays = np.zeros(shape=(len(X), len(X2)))
        
        # Now we proceed for the actual calculation
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                # It will always be a 1-element array
                node_list = self._get_node_pairs(x1[0], x2[0])
                try:
                    K_result, ddecay_result = self.delta(node_list)
                except:
                    print node_list
                    raise
                norm = diag_deltas_X[i] * diag_deltas_X2[j]
                sqrt_norm = np.sqrt(norm)
                K_norm = K_result / sqrt_norm
                
                diff_term = ((diag_ddecays_X[i] * diag_deltas_X2[j]) +
                             (diag_deltas_X[i] * diag_ddecays_X2[j]))
                diff_term /= float(2 * norm)
                ddecay_norm = ((ddecay_result / sqrt_norm) -
                               (K_norm * diff_term))

                K_results[i][j] = K_norm
                ddecays[i][j] = ddecay_norm        
        target += K_results
        self.ddecays = ddecays

    def dK_dtheta(self, dL_dK, X, X2, target):
        s_like = np.sum(dL_dK)
        if self.ddecays != None:
            s_decay = np.sum(self.ddecays)
        else:
            raise IndexError("No ddecays cached")
        target += [s_like * s_decay]
                
    def _diag_calculations(self, X):
        K_vec = np.zeros(shape=(len(X),))
        ddecay_vec = np.zeros(shape=(len(X),))
        for i, x in enumerate(X):
            node_list = self._get_node_pairs(x[0], x[0])
            delta_result = 0
            ddecay = 0
            cache = {} # DP
            cache_ddecay = {}
            # Calculation happens here.
            delta_result, ddecay = self.delta(node_list)
            K_vec[i] = delta_result
            ddecay_vec[i] = ddecay
        return (K_vec, ddecay_vec)

    def delta(self, node_list):
        cache_delta = defaultdict(int) # DP
        cache_ddecay = defaultdict(int)
        for node_pair in node_list:
            node1, node2, child_len = node_pair
            key = (node1, node2)
            if child_len == 0:
                cache_delta[key] = self.decay
                cache_ddecay[key] = 1
            else:
                prod = 1
                sum_decay = 0
                for i in xrange(child_len):
                    child_key = (tuple(list(node1) + [i]),
                                 tuple(list(node2) + [i]))
                    ch_delta = cache_delta[child_key]
                    ch_ddecay = cache_ddecay[child_key]
                    prod *= 1 + ch_delta
                    sum_decay += ch_ddecay / (1 + float(ch_delta))
                delta_result = self.decay * prod
                cache_delta[key] = delta_result
                cache_ddecay[key] = prod + (delta_result * sum_decay)
        return (sum(cache_delta.values()),
                sum(cache_ddecay.values()))


class SympySimpleFastTreeKernel(Kernpart):
    """
    Moschitti's FTK but only with decay hyperparameter. A sympy version.
    """

    def __init__(self, decay=1, has_root=False):
        try:
            import nltk
        except ImportError:
            sys.stderr.write("Tree Kernels need NLTK. Install it using \"pip install nltk\"")
            raise
        self.input_dim = 1 # A hack. Actually tree kernels have lots of dimensions.
        self.num_params = 1
        self.name = 'symftk'
        self.decay = decay
        self.cache = {}
        self.has_root = has_root

    def _get_params(self):
        return self.decay

    def _set_params(self, x):
        self.decay = x[0]

    def _get_param_names(self):
        return ['decay']

    def build_cache(self, X):
        self.cache["tree_ids"] = {}
        self.cache["tree_pair_ks"] = {}
        self.cache["tree_pair_ddecays"] = {}
        # Temporary node cache
        node_cache = {}
        # Store trees
        for tree in X:
            tree_id = len(self.cache["tree_ids"])
            self.cache["tree_ids"].setdefault(tree[0], tree_id)
            node_cache[tree_id] = self._get_nodes(tree[0])
        # Store node pairs
        for tree1 in X:
            id1 = self.cache["tree_ids"][tree1[0]]
            self.cache["tree_pair_ks"][id1] = {}
            self.cache["tree_pair_ddecays"][id1] = {}
            for tree2 in X:
                id2 = self.cache["tree_ids"][tree2[0]]
                if id1 > id2: #symmetry
                    continue
                nodes1 = node_cache[id1]
                nodes2 = node_cache[id2]
                k, ddecay = self._formulate(nodes1, nodes2)
                self.cache["tree_pair_ks"][id1][id2] = k
                self.cache["tree_pair_ddecays"][id1][id2] = ddecay

    def _get_nodes(self, x):
        t = nltk.Tree(x)
        if self.has_root:
            t = t[0]
        pos = t.treepositions()
        for l in t.treepositions(order="leaves"):
            pos.remove(l)
        z = zip(t.productions(), pos)
        z.sort(key=lambda x: len(x[1]), reverse=True)
        return z

    def _formulate(self, nodes1, nodes2):
        """
        Get two trees represented by strings and
        return the node pair list as sympy formula
        """
        cache = defaultdict(int)
        l = sp.Symbol("l")
        formula = sp.sympify(0)
        for n1 in nodes1:
            for n2 in nodes2:
                if n1[0] == n2[0]:
                    key = (n1[1], n2[1])
                    if type(n1[0].rhs()[0]) == str:
                        formula += l
                        cache[key] = l
                    else:
                        prod = 1
                        for i in xrange(len(n1[0].rhs())):
                            child_key = (tuple(list(n1[1]) + [i]),
                                         tuple(list(n2[1]) + [i]))
                            ch_delta = cache[child_key]
                            prod *= 1 + ch_delta
                        res = (prod * l).expand()
                        formula += res
                        cache[key] = res
        return (lambdastr(l, formula),
                lambdastr(l, formula.diff(l)))

    def _get_formulas(self, tree1, tree2):
        """
        Get two trees represented by strings and
        return the node pair list
        """
        try:
            id1 = self.cache["tree_ids"][tree1]
            id2 = self.cache["tree_ids"][tree2]
            return (self.cache["tree_pair_ks"][id1][id2],
                    self.cache["tree_pair_ddecays"][id1][id2])
        except KeyError:
            nodes1 = self._get_nodes(tree1)
            nodes2 = self._get_nodes(tree2)
            return self._formulate(nodes1, nodes2)

    def Kdiag(self, X, target):
        """
        Since this is a normalized kernel, diag values are all equal to 1.
        """
        target += np.ones(shape=(len(X),))

    def K(self, X, X2, target):
        if X2 == None:
            if self.cache == {}:
                self.build_cache(X)
            self.K_sym(X, target)
        else:
            self.K_nsym(X, X2, target)

    def K_sym(self, X, target):

        # First, we are going to calculate K for diagonal values
        # because we will need them later to normalize.
        diag_deltas, diag_ddecays = self._diag_calculations(X)

        # Second, we are going to initialize the ddecay values
        # because we are going to calculate them at the same time as K.
        K_results = np.zeros(shape=(len(X), len(X)))
        ddecays = np.zeros(shape=(len(X), len(X)))
        
        # Now we proceed for the actual calculation
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X):
                if i > j:
                    K_results[i][j] = K_results[j][i]
                    ddecays[i][j] = ddecays[j][i]
                    continue
                if i == j:
                    K_results[i][j] = 1
                    continue
                # It will always be a 1-element array
                k, ddecay = self._get_formulas(x1[0], x2[0])
                try:
                    K_result = eval(k)(self.decay)
                    ddecay_result = eval(ddecay)(self.decay)
                except:
                    print formula
                    raise
                norm = diag_deltas[i] * diag_deltas[j]
                sqrt_norm = np.sqrt(norm)
                K_norm = K_result / sqrt_norm
                
                diff_term = ((diag_ddecays[i] * diag_deltas[j]) +
                             (diag_deltas[i] * diag_ddecays[j]))
                diff_term /= float(2 * norm)
                ddecay_norm = ((ddecay_result / sqrt_norm) -
                               (K_norm * diff_term))

                K_results[i][j] = K_norm
                ddecays[i][j] = ddecay_norm
        
        target += K_results
        self.ddecays = ddecays

    def K_nsym(self, X, X2, target):
        # First, we are going to calculate K for diagonal values
        # because we will need them later to normalize.
        diag_deltas_X, diag_ddecays_X = self._diag_calculations(X)
        diag_deltas_X2, diag_ddecays_X2 = self._diag_calculations(X2)

        # Second, we are going to initialize the ddecay values
        # because we are going to calculate them at the same time as K.
        K_results = np.zeros(shape=(len(X), len(X2)))
        ddecays = np.zeros(shape=(len(X), len(X2)))
        
        # Now we proceed for the actual calculation
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                # It will always be a 1-element array
                node_list = self._get_node_pairs(x1[0], x2[0])
                try:
                    K_result, ddecay_result = self.delta(node_list)
                except:
                    print node_list
                    raise
                norm = diag_deltas_X[i] * diag_deltas_X2[j]
                sqrt_norm = np.sqrt(norm)
                K_norm = K_result / sqrt_norm
                
                diff_term = ((diag_ddecays_X[i] * diag_deltas_X2[j]) +
                             (diag_deltas_X[i] * diag_ddecays_X2[j]))
                diff_term /= float(2 * norm)
                ddecay_norm = ((ddecay_result / sqrt_norm) -
                               (K_norm * diff_term))

                K_results[i][j] = K_norm
                ddecays[i][j] = ddecay_norm        
        target += K_results
        self.ddecays = ddecays

    def dK_dtheta(self, dL_dK, X, X2, target):
        s_like = np.sum(dL_dK)
        if self.ddecays != None:
            s_decay = np.sum(self.ddecays)
        else:
            raise IndexError("No ddecays cached")
        target += [s_like * s_decay]
                
    def _diag_calculations(self, X):
        K_vec = np.zeros(shape=(len(X),))
        ddecay_vec = np.zeros(shape=(len(X),))
        for i, x in enumerate(X):
            k, ddecay = self._get_formulas(x[0], x[0])
            K_vec[i] = eval(k)(self.decay)
            ddecay_vec[i] = eval(ddecay)(self.decay)
        return (K_vec, ddecay_vec)

    def delta(self, formula):
        #l = sp.Symbol('l')
        #d = formula.evalf(subs={l: self.decay})
        #ddecay = formula.diff(l).evalf(subs={l: self.decay})
        d = eval(formula)(self.decay)
        #d = formula(self.decay)
        ddecay = 0
        return (d, ddecay)
        
