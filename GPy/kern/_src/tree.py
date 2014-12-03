
#from kernpart import Kernpart
from kern import Kern
from ...core.parameterization import Param
import numpy as np
from collections import defaultdict
import sys
import cy_tree
import nltk

MAX_NODES = 300


####################################

class SubsetTreeKernel(Kern):
    """
    The SST kernel by Moschitti(2006), with two hyperparameters.
    This is mainly a wrapper for a Cython implementation (see cy_tree.pyx).
    The Cython kernel is stored on the "kernel" attribute.
    """

    def __init__(self, _lambda=0.1, _sigma=1, normalize=True, active_dims=None, parallel=False,
                 num_threads=1):
        try:
            import nltk
        except ImportError:
            sys.stderr.write("Tree Kernels need NLTK. Install it using \"pip install nltk\"")
            raise
        super(SubsetTreeKernel, self).__init__(1, active_dims, 'sstk')
        self._lambda = Param('lambda', _lambda)
        self._sigma = Param('sigma', _sigma)
        self.add_parameters(self._lambda, self._sigma)
        self.normalize = normalize
        #print num_threads
        if parallel:
            self.kernel = cy_tree.ParSubsetTreeKernel(_lambda, _sigma, normalize, num_threads=num_threads)
        else:
            self.kernel = cy_tree.CySubsetTreeKernel(_lambda, _sigma, normalize)
        
    def _get_params(self):
        return np.hstack((self.kernel._lambda, self.kernel._sigma))

    def _set_params(self, x):
        self.kernel._lambda = x[0]
        self.kernel._sigma = x[1]

    def _get_param_names(self):
        return ['lambda', 'sigma']

    def K(self, X, X2):#, target):
        self.kernel._lambda = self._lambda[0]
        self.kernel._sigma = self._sigma[0]
        result, dl, ds = self.kernel.K(X, X2)
        self.dlambda = dl
        self.dsigma = ds
        return result

    def Kdiag(self, X):#), target):
        self.kernel._lambda = self._lambda[0]
        self.kernel._sigma = self._sigma[0]
        if self.normalize:
            #target += np.ones(X.shape[0])
            return np.ones(X.shape[0])
        else:
            #target += self.kernel.Kdiag(X)
            return self.kernel.Kdiag(X)

    def dK_dtheta(self, dL_dK, X, X2):#, target):
        return [np.sum(self.dlambda * dL_dK),
                np.sum(self.dsigma * dL_dK)]

    def update_gradients_full(self, dL_dK, X, X2):
        self._lambda.gradient = np.sum(self.dlambda * dL_dK)
        self._sigma.gradient = np.sum(self.dsigma * dL_dK)

####################################

class SymbolAwareSubsetTreeKernel(Kern):
    """
    An extension of SST, including specific lambdas/sigmas for each symbol.
    """
    def __init__(self, _lambda=np.array([0.5]), _sigma=np.array([1.0]), lambda_buckets={}, sigma_buckets={},
                 normalize=True, active_dims=None, num_threads=1):
        try:
            import nltk
        except ImportError:
            sys.stderr.write("Tree Kernels need NLTK. Install it using \"pip install nltk\"")
            raise
        super(SymbolAwareSubsetTreeKernel, self).__init__(1, active_dims, 'sasstk')
        self.normalize = normalize
        # SASST will be parallel by default.
        self._lambda = Param('_lambda', _lambda)
        self._sigma = Param('_sigma', _sigma)
        self.kernel = cy_tree.SymbolAwareSubsetTreeKernel(_lambda, _sigma, lambda_buckets, sigma_buckets,
                                                          normalize, num_threads=num_threads)
        
    def _get_params(self):
        return np.hstack((self.kernel._lambda, self.kernel._sigma))

    def _set_params(self, x):
        self.kernel._lambda = x[0]
        self.kernel._sigma = x[1]

    def _get_param_names(self):
        return ['lambda', 'sigma']

    def K(self, X, X2):#, target):
        self.kernel._lambda = self._lambda
        self.kernel._sigma = self._sigma
        result, dl, ds = self.kernel.K(X, X2)
        #print dl
        self.dlambda = dl
        #print self.dlambda
        self.dsigma = ds
        return result

    def Kdiag(self, X):#), target):
        self.kernel._lambda = self._lambda
        self.kernel._sigma = self._sigma
        if self.normalize:
            #target += np.ones(X.shape[0])
            return np.ones(X.shape[0])
        else:
            #target += self.kernel.Kdiag(X)
            return self.kernel.Kdiag(X)

    def dK_dtheta(self, dL_dK, X, X2):#, target):
        return [np.sum(self.dlambda * dL_dK),
                np.sum(self.dsigma * dL_dK)]

    def update_gradients_full(self, dL_dK, X, X2):
        self._lambda.gradient = np.sum(self.dlambda * dL_dK)
        self._sigma.gradient = np.sum(self.dsigma * dL_dK)

####################################

class Node(object):
    """
    A node object, used by the Python implementation.
    """
    def __init__(self, production, node_id, children_ids):
        self.production = production
        self.node_id = node_id
        self.children_ids = children_ids

    def __repr__(self):
        return str((self.production, self.node_id, self.children_ids))

####################################

class PySubsetTreeKernel(Kern):
    """
    An SST Kernel, as defined by Moschitti, with
    two hyperparameters. A pure python version of the cython SSTK. This
    is mainly for profiling purposes and check how much performance we gain by
    using Cython.
    """
    
    def __init__(self, _lambda=0.1, _sigma=1, active_dims=None):
        try:
            import nltk
        except ImportError:
            sys.stderr.write("Tree Kernels need NLTK. Install it using \"pip install nltk\"")
            raise
        super(PySubsetTreeKernel, self).__init__(1, active_dims, 'sstk')
        #self.input_dim = 1
        #self.num_params = 2
        #self.name = 'sstk'
        self._lambda = _lambda
        self._sigma = _sigma
        self._tree_cache = {}

    def _gen_node_list(self, tree_repr):
        #tree = nltk.Tree(tree_repr)
        tree = nltk.tree.Tree.fromstring(tree_repr)
        c = 0
        node_list = []
        self._get_node(tree, node_list)
        node_list.sort(key=lambda x: x.production)
        node_dict = dict([(node.node_id, node) for node in node_list])
        return node_list, node_dict

    def _get_node(self, tree, node_list):
        if type(tree[0]) != str: #non preterm
            #prod = [tree.node]
            prod = [tree.label()]
            children = []
            for ch in tree:
                ch_id = self._get_node(ch, node_list)
                #prod.append(ch.node)
                prod.append(ch.label())
                children.append(ch_id)
            node_id = len(node_list)
            node = Node(' '.join(prod), node_id, children)
            node_list.append(node)
            return node_id
        else:
            #prod = ' '.join([tree.node, tree[0]])
            prod = ' '.join([tree.label(), tree[0]])
            node_id = len(node_list)
            node = Node(prod, node_id, None)
            node_list.append(node)
            return node_id            

    def _get_node_pairs(self, nodes1, nodes2):
        node_pair_list = []
        i1 = 0
        i2 = 0
        while True:
            try:
                if nodes1[i1].production > nodes2[i2].production:
                    i2 += 1
                elif nodes1[i1].production < nodes2[i2].production:
                    i1 += 1
                else:
                    while nodes1[i1].production == nodes2[i2].production:
                        reset2 = i2
                        while nodes1[i1].production == nodes2[i2].production:
                            node_pair_list.append((nodes1[i1], nodes2[i2]))
                            i2 += 1
                        i1 += 1
                        i2 = reset2
            except IndexError:
                break
        return node_pair_list

    #def _get_node_pair_list_cy(self, nodes1, nodes2):
    #    return cy_sst.cy_get_node_pair_list(nodes1, nodes2)

    def _build_cache(self, X):
        #print X
        for tree_repr in X:
            t_repr = tree_repr[0]
            node_list, node_dict = self._gen_node_list(t_repr)
            self._tree_cache[t_repr] = (node_list, node_dict)
        
    def K(self, X, X2, target):
        # A check to ensure that ddecays cache will always change when K changes
        self.ddecays = None
        if X2 == None:
            self.K_sym(X, target)
        else:
            self.K_nsym(X, X2, target)

    def K_sym(self, X, target):
        if self._tree_cache == {}:
            self._build_cache(X)

        # First, we are going to calculate K for diagonal values
        # because we will need them later to normalize.
        diag_deltas, diag_ddecays = self._diag_calculations(X)

        # Second, we are going to initialize the ddecay values
        # because we are going to calculate them at the same time as K.
        K_results = np.zeros(shape=(len(X), len(X)))
        ddecays = np.zeros(shape=(len(X), len(X)))
        
        # Now we proceed for the actual calculation
        #import ipdb
        #ipdb.set_trace()

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
                nodes1, dict1 = self._tree_cache[x1[0]]
                nodes2, dict2 = self._tree_cache[x2[0]]
                node_pairs = self._get_node_pairs(nodes1, nodes2)
                try:
                    K_result, ddecay_result = self.calc_K(node_pairs, dict1, dict2)
                    #import ipdb
                    #ipdb.set_trace()
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
        
    def _diag_calculations(self, X):
        K_vec = np.zeros(shape=(len(X),))
        ddecay_vec = np.zeros(shape=(len(X),))
        for i, x in enumerate(X):
            nodes, dicts = self._tree_cache[x[0]]
            node_pairs = self._get_node_pairs(nodes, nodes)
            delta_result = 0
            ddecay = 0
            # Calculation happens here.
            delta_result, ddecay = self.calc_K(node_pairs, dicts, dicts)
            K_vec[i] = delta_result
            ddecay_vec[i] = ddecay
        return (K_vec, ddecay_vec)

    def calc_K(self, node_pairs, dict1, dict2):
        K_total = 0
        ddecay_total = 0
        delta_matrix = np.zeros(shape=(MAX_NODES, MAX_NODES))
        for node_pair in node_pairs:
            K_result, ddecay_result = self.delta(node_pair[0], node_pair[1], delta_matrix, dict1, dict2)
            K_total += K_result
            ddecay_total += ddecay_result
        return (K_total, ddecay_total)

    def delta(self, node1, node2, delta_matrix, dict1, dict2):
        id1 = node1.node_id
        id2 = node2.node_id
        val = delta_matrix[id1, id2]
        if val > 0:
            return val, val
        if node1.children_ids == None:
            delta_matrix[id1, id2] = self._lambda
            return (self._lambda, 1)
        prod = 1
        for ch1, ch2 in zip(node1.children_ids, node2.children_ids):
            if dict1[ch1].production == dict2[ch2].production:
                K_result, ddecay_result = self.delta(dict1[ch1], dict2[ch2], 
                                                     delta_matrix, dict1, dict2)
                prod *= (self._sigma + K_result)
        result = self._lambda * prod
        delta_matrix[id1, id2] = result
        return result, result

##############################
#
# OLD VERSIONS
#
# These are old, pure python versions of the Subset Tree Kernel.
# The reason they are kept here is mainly for testing purposes:
# results for these implementations should match the cython ones.
# They should not be used in real applications because they are too slow.
#
# This version has "modes", referring to different implementations of K
# and dK_dtheta: "naive", "cache" and "fast". Only the "fast" mode has
# normalization.
#
# In the future, when we have more tests for the cython version, the
# plan is to remove this code.
#
##############################

class OldSubsetTreeKernel(Kern):
    """A convolution kernel that compares two trees. See Moschitti(2006).
    """
    
    def __init__(self, _lambda=0.1, _sigma=1, mode="naive", normalize=False, active_dims=None):
        """blah
        """
        try:
            import nltk
        except ImportError:
            sys.stderr.write("Tree Kernels need NLTK. Install it using \"pip install nltk\"")
            raise
        super(OldSubsetTreeKernel, self).__init__(1, active_dims, 'sstk')
        #self.input_dim = 1 # A hack. Actually tree kernels have lots of dimensions.
        #self.num_params = 2
        #self.name = 'tk'
        self.decay = _lambda
        self.branch = _sigma
        self.normalize = normalize
        if mode == "naive":
            self.K = self.K_naive
            self.Kdiag = self.Kdiag_naive
            self.dK_dtheta = self.dK_dtheta_naive
        elif mode == "cache":
            self.K = self.K_cache
            self.Kdiag = self.Kdiag_cache
            self.dK_dtheta = self.dK_dtheta_cache
        elif mode == "fast":
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


    ###############
    # NAIVE METHODS
    #
    # These are implemented by just plugging the kernel formulas
    # into code. They have exponential complexity. They are still useful
    # for testing and comparing results with the other implementations
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
        #if node1.node != node2.node:
        if node1.label() != node2.label():
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
        #if node1.node != node2.node:
        if node1.label() != node2.label():
            return (0, 0)
        if node1.productions()[0] != node2.productions()[0]:
            return (0, 0)
        # second case -> preterms
        if node1.height() == 2 and node2.height() == 2:
            return (1, 0)
        # third case
        prod = 1
        sum_decay = 0
        sum_branch = 0
        for i, child in enumerate(node1): #node2 has the same children
            g = float(self.branch + self.delta_naive(node1[i], node2[i]))
            prod *= g
            d, b = self.delta_params_naive(node1[i], node2[i])
            sum_decay += (d / g)
            sum_branch += ((1 + b) / g)
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

    def K_fast(self, X, X2=None):
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
                return self._normalize_K_sym(X, K_results, ddecays, dbranches)#, target)
                #return self._normalize_K(X, X, K_results, ddecays, dbranches)#, target)
            else:
                return self._normalize_K(X, X2, K_results, ddecays, dbranches)#, target)
        else:
            #target += K_results
            self.ddecay_results = ddecays
            self.dbranch_results = dbranches
            return K_results
                
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
        #if (t1.node != t2.node or len(t1) != len(t2)):
        if (t1.label() != t2.label() or len(t1) != len(t2)):
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
            #if ch.node != t2[i].node: 
            if ch.label() != t2[i].label(): 
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
            t1 = nltk.tree.Tree.fromstring(x1[0])
            self.tree_cache[x1[0]] = t1
        try:
            t2 = self.tree_cache[x2[0]]
        except KeyError:
            t2 = nltk.tree.Tree.fromstring(x2[0])
            self.tree_cache[x2[0]] = t2
        return (t1, t2)

    def _normalize_K_sym(self, X, K_results, ddecays, dbranches):#, target):
        target = np.zeros(shape=(X.shape[0],X.shape[0]))
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
        return target

    def _normalize_K(self, X, X2, K_results, ddecays, dbranches):#, target):
        target = np.zeros(shape=(X.shape[0],X2.shape[0]))
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
        return target

