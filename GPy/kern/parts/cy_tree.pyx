# distutils: language = c++
# cython: profile=True
import nltk
import numpy as np
cimport numpy as np
from cython.parallel import prange
from collections import defaultdict
from libcpp.string cimport string

cdef extern from "math.h":
    double sqrt(double x)

DTYPE = np.double
ctypedef np.double_t DTYPE_t
cdef unsigned int MAX_NODES = 300


cdef class Node(object):
    """
    A node object.
    """

    cdef string production
    cdef int node_id
    cdef list children_ids

    def __init__(self, production, node_id, children_ids):
        self.production = production
        self.node_id = node_id
        self.children_ids = children_ids

    def __repr__(self):
        return str((self.production, self.node_id, self.children_ids))


class CySubsetTreeKernel(object):
    
    def __init__(self, _lambda=1., _sigma=1.):
        self._lambda = _lambda
        self._sigma = _sigma
        self._tree_cache = {}

        cdef np.ndarray[DTYPE_t, ndim=2] delta_matrix = np.zeros([MAX_NODES, MAX_NODES], dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] dlambda_matrix = np.zeros([MAX_NODES, MAX_NODES], dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] dsigma_matrix = np.zeros([MAX_NODES, MAX_NODES], dtype=DTYPE)
        self.delta_matrix = delta_matrix
        self.dlambda_matrix = dlambda_matrix
        self.dsigma_matrix = dsigma_matrix

    def _gen_node_list(self, tree_repr):
        """
        Generates an ordered list of nodes from a tree.
        The list is used to generate the node pairs when
        calculating K.
        It also returns a nodes dict for fast node access.
        """
        tree = nltk.Tree(tree_repr)
        c = 0
        node_list = []
        self._get_node(tree, node_list)
        node_list.sort(key=lambda Node x: x.production)
        cdef Node node
        node_dict = dict([(node.node_id, node) for node in node_list])
        #print node_list
        return node_list, node_dict

    def _get_node(self, tree, node_list):
        """
        Recursive method for generating the node lists.
        """
        cdef string cprod
        if type(tree[0]) != str: #non preterm
            prod_list = [tree.node]
            children = []
            for ch in tree:
                ch_id = self._get_node(ch, node_list)
                prod_list.append(ch.node)
                children.append(ch_id)
            node_id = len(node_list)
            prod = ' '.join(prod_list)
            cprod = prod
            #print prod
            #print cprod
            node = Node(cprod, node_id, children)
            node_list.append(node)
            return node_id
        else:
            prod = ' '.join([tree.node, tree[0]])
            cprod = prod
            #print prod
            #print cprod
            node_id = len(node_list)
            node = Node(cprod, node_id, None)
            node_list.append(node)
            return node_id            

    def _get_node_pairs(self, nodes1, nodes2):
        """
        The node pair detection method devised by Moschitti.
        """
        node_pairs = []
        i1 = 0
        i2 = 0
        cdef Node n1, n2
        while True:
            try:
                n1 = nodes1[i1]
                n2 = nodes2[i2]
                if n1.production > n2.production:
                    i2 += 1
                elif n1.production < n2.production:
                    i1 += 1
                else:
                    while n1.production == n2.production:
                        reset2 = i2
                        while n1.production == n2.production:
                            node_pairs.append((n1, n2))
                            self.delta_matrix[n1.node_id, n2.node_id] = 0.
                            self.dlambda_matrix[n1.node_id, n2.node_id] = 0.
                            self.dsigma_matrix[n1.node_id, n2.node_id] = 0.
                            i2 += 1
                            n2 = nodes2[i2]
                        i1 += 1
                        i2 = reset2
                        n1 = nodes1[i1]
                        n2 = nodes2[i2]
            except IndexError:
                break
        return node_pairs

    def _build_cache(self, X):
        """
        Caches the node lists, usually at the first time
        K_sym is called.
        """
        for tree_repr in X:
            t_repr = tree_repr[0]
            node_list, node_dict = self._gen_node_list(t_repr)
            self._tree_cache[t_repr] = (node_list, node_dict)
        
    def K(self, X, X2):
        """
        A wrapper around K, resetting the derivatives cache and choosing
        between symmetric gram matrix and non-symmetric.
        """
        # A check to ensure that derivatives cache will always change when K changes
        #self.dlambdas = None
        #self.dsigmas = None
        #print "YAY"
        if X2 == None:
            return self.K_sym(X)
        else:
            return self.K_nsym(X, X2)

    def K_sym(self, X):
        """
        The symmetric version of the gram matrix method.
        """
        if self._tree_cache == {}:
            self._build_cache(X)

        # First, we are going to calculate K for diagonal values
        # because we will need them later to normalize.
        diag_Ks, diag_dlambdas, diag_dsigmas = self._diag_calculations(X)

        # Second, we are going to initialize the ddecay values
        # because we are going to calculate them at the same time as K.
        Ks = np.zeros(shape=(len(X), len(X)))
        dlambdas = np.zeros(shape=(len(X), len(X)))
        dsigmas = np.zeros(shape=(len(X), len(X)))
        
        # Now we proceed for the actual calculation

        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X):
                if i > j:
                    Ks[i][j] = Ks[j][i]
                    dlambdas[i][j] = dlambdas[j][i]
                    dsigmas[i][j] = dsigmas[j][i]
                    continue
                if i == j:
                    Ks[i][j] = 1
                    continue
                # It will always be a 1-element array so we just index by 0
                nodes1, dict1 = self._tree_cache[x1[0]]
                nodes2, dict2 = self._tree_cache[x2[0]]
                node_pairs = self._get_node_pairs(nodes1, nodes2)
                #K_result, dlambda, dsigma = self.calc_K(node_pairs, dict1, dict2)
                K_result, dlambda, dsigma = calc_K_ext(node_pairs, dict1, dict2, self._lambda, self._sigma)
                #result = pool.apply_async(calc_K_ext, (node_pairs, dict1, dict2, self._lambda, self._sigma))
                #K_result, dlambda, dsigma = result.get()

                K_norm, dlambda_norm, dsigma_norm = self._normalize(K_result, dlambda, dsigma,
                                                                   diag_Ks[i], diag_Ks[j],
                                                                   diag_dlambdas[i], diag_dlambdas[j],
                                                                   diag_dsigmas[i], diag_dsigmas[j])
                # Normalization happens here
                Ks[i][j] = K_norm
                dlambdas[i][j] = dlambda_norm
                dsigmas[i][j] = dsigma_norm

        return (Ks, dlambdas, dsigmas)

    def _normalize(self, double K_result, double dlambda, double dsigma, double diag_Ks_i, 
                   double diag_Ks_j, double diag_dlambdas_i, double diag_dlambdas_j, 
                   double diag_dsigmas_i, double diag_dsigmas_j):
        cdef double norm, sqrt_nrorm, K_norm, diff_lambda, dlambda_norm, diff_sigma, dsigma_norm

        norm = diag_Ks_i * diag_Ks_j
        sqrt_norm = sqrt(norm)
        K_norm = K_result / sqrt_norm
                
        diff_lambda = ((diag_dlambdas_i * diag_Ks_j) +
                       (diag_Ks_i * diag_dlambdas_j))
        diff_lambda /= 2 * norm
        dlambda_norm = ((dlambda / sqrt_norm) -
                        (K_norm * diff_lambda))
        
        diff_sigma = ((diag_dsigmas_i * diag_Ks_j) +
                      (diag_Ks_i * diag_dsigmas_j))
        diff_sigma /= 2 * norm
        dsigma_norm = ((dsigma / sqrt_norm) -
                       (K_norm * diff_sigma))
        return K_norm, dlambda_norm, dsigma_norm
        
    def _diag_calculations(self, X):
        """
        Calculate the gram matrix diagonal first because
        it is used in normalization.
        """
        K_vec = np.zeros(shape=(len(X),))
        dlambda_vec = np.zeros(shape=(len(X),))
        dsigma_vec = np.zeros(shape=(len(X),))
        for i, x in enumerate(X):
            nodes, dicts = self._tree_cache[x[0]]
            node_pairs = self._get_node_pairs(nodes, nodes)
            #K_result = 0
            #dlambda = 0
            #dsigma = 0
            # Calculation happens here.
            K_result, dlambda, dsigma = self.calc_K(node_pairs, dicts, dicts)
            K_vec[i] = K_result
            dlambda_vec[i] = dlambda
            dsigma_vec[i] = dsigma
        return (K_vec, dlambda_vec, dsigma_vec)

    def calc_K(self, list node_pairs, dict1, dict2):
        """
        This method calculates the kernel between two trees.
        It also calculates the derivatives wrt lambda and sigma.
        """
        cdef double K_total = 0
        cdef double dlambda_total = 0
        cdef double dsigma_total = 0
        cdef double K_result, dlambda, dsigma
        #self.delta_matrix[:,:] = 0.
        #self.dlambda_matrix[:,:] = 0. 
        #self.dsigma_matrix[:,:] = 0.

        #self.delta_matrix = defaultdict(int)
        #self.dlambda_matrix = defaultdict(int)
        #self.dsigma_matrix = defaultdict(int)

        for node_pair in node_pairs:
            K_result, dlambda, dsigma = self.delta(node_pair[0], node_pair[1], dict1, dict2)
            K_total += K_result
            dlambda_total += dlambda
            dsigma_total += dsigma
        return (K_total, dlambda_total, dsigma_total)

    def delta(self, Node node1, Node node2, dict1, dict2):
        """
        Recursive method used in kernel calculation.
        It also calculates the derivatives wrt lambda and sigma.
        """
        cdef int id1, id2, ch1, ch2, i
        cdef DTYPE_t val, prod, K_result, dlambda, dsigma, sum_lambda, sum_sigma, denom
        cdef DTYPE_t delta_result, dlambda_result, dsigma_result
        cdef Node n1, n2
        #cdef list children1, children2
        id1 = node1.node_id
        id2 = node2.node_id
        #tup = (id1, id2)
        val = self.delta_matrix[id1][id2]
        #val = self.delta_matrix[tup]
        if val > 0:
            return val, self.dlambda_matrix[id1, id2], self.dsigma_matrix[id1, id2]
            #return val, self.dlambda_matrix[tup], self.dsigma_matrix[tup]

        if node1.children_ids == None:
            self.delta_matrix[id1, id2] = self._lambda
            self.dlambda_matrix[id1, id2] = 1
            #self.delta_matrix[tup] = self._lambda
            #self.dlambda_matrix[tup] = 1
            return (self._lambda, 1, 0)
        prod = 1
        sum_lambda = 0
        sum_sigma = 0
        children1 = node1.children_ids
        children2 = node2.children_ids
        #print node1
        #print node2
        for i in range(len(children1)):
            ch1 = children1[i]
            ch2 = children2[i]
            n1 = dict1[ch1]
            n2 = dict2[ch2]
            if n1.production == n2.production:
                K_result, dlambda, dsigma = self.delta(n1, n2, 
                                                       dict1, dict2)
                denom = self._sigma + K_result
                prod *= denom
                sum_lambda += dlambda / denom
                sum_sigma += (1 + dsigma) / denom
        delta_result = self._lambda * prod
        dlambda_result = prod + (delta_result * sum_lambda)
        dsigma_result = delta_result * sum_sigma
        self.delta_matrix[id1, id2] = delta_result
        self.dlambda_matrix[id1, id2] = dlambda_result
        self.dsigma_matrix[id1, id2] = dsigma_result
        #self.delta_matrix[tup] = delta_result
        #self.dlambda_matrix[tup] = dlambda_result
        #self.dsigma_matrix[tup] = dsigma_result
        return (delta_result, dlambda_result, dsigma_result)



def calc_K_ext(list node_pairs, dict1, dict2, _lambda, _sigma):
    """
    This method calculates the kernel between two trees.
    It also calculates the derivatives wrt lambda and sigma.
    """
    cdef double K_total = 0
    cdef double dlambda_total = 0
    cdef double dsigma_total = 0
    cdef double K_result, dlambda, dsigma
    #cdef np.ndarray[DTYPE_t, ndim=2] delta_matrix = np.zeros([MAX_NODES, MAX_NODES], dtype=DTYPE)
    #cdef np.ndarray[DTYPE_t, ndim=2] dlambda_matrix = np.zeros([MAX_NODES, MAX_NODES], dtype=DTYPE)
    #cdef np.ndarray[DTYPE_t, ndim=2] dsigma_matrix = np.zeros([MAX_NODES, MAX_NODES], dtype=DTYPE)
    delta_matrix = defaultdict(int)
    dlambda_matrix = defaultdict(int)
    dsigma_matrix = defaultdict(int)

    for node_pair in node_pairs:
        K_result, dlambda, dsigma = delta_ext(node_pair[0], node_pair[1], dict1, dict2,
                                              delta_matrix, dlambda_matrix, dsigma_matrix,
                                              _lambda, _sigma)
        K_total += K_result
        dlambda_total += dlambda
        dsigma_total += dsigma
    return (K_total, dlambda_total, dsigma_total)

def delta_ext(Node node1, Node node2, dict1, dict2, 
              delta_matrix, dlambda_matrix, dsigma_matrix,
              _lambda, _sigma):
    """
    Recursive method used in kernel calculation.
    It also calculates the derivatives wrt lambda and sigma.
    """
    cdef int id1, id2, ch1, ch2, i
    cdef DTYPE_t val, prod, K_result, dlambda, dsigma, sum_lambda, sum_sigma, denom
    cdef DTYPE_t delta_result, dlambda_result, dsigma_result
    cdef Node n1, n2
    #cdef list children1, children2
    id1 = node1.node_id
    id2 = node2.node_id
    tup = (id1, id2)
    val = delta_matrix[tup]
    #val = delta_matrix[id1][id2]
    if val > 0:
        #return val, dlambda_matrix[id1, id2], dsigma_matrix[id1, id2]
        return val, dlambda_matrix[tup], dsigma_matrix[tup]
    
    if node1.children_ids == None:
        #delta_matrix[id1, id2] = _lambda
        #dlambda_matrix[id1, id2] = 1
        delta_matrix[tup] = _lambda
        dlambda_matrix[tup] = 1
        return (_lambda, 1, 0)
    prod = 1
    sum_lambda = 0
    sum_sigma = 0
    children1 = node1.children_ids
    children2 = node2.children_ids
    for i in range(len(children1)):
        ch1 = children1[i]
        ch2 = children2[i]
        n1 = dict1[ch1]
        n2 = dict2[ch2]
        if n1.production == n2.production:
            K_result, dlambda, dsigma = delta_ext(n1, n2, dict1, dict2,
                                                  delta_matrix, dlambda_matrix, dsigma_matrix,
                                                  _lambda, _sigma)
            denom = _sigma + K_result
            prod *= denom
            sum_lambda += dlambda / denom
            sum_sigma += (1 + dsigma) / denom
    delta_result = _lambda * prod
    dlambda_result = prod + (delta_result * sum_lambda)
    dsigma_result = delta_result * sum_sigma
    delta_matrix[tup] = delta_result
    dlambda_matrix[tup] = dlambda_result
    dsigma_matrix[tup] = dsigma_result
    #delta_matrix[id1, id2] = delta_result
    #dlambda_matrix[id1, id2] = dlambda_result
    #dsigma_matrix[id1, id2] = dsigma_result
    return (delta_result, dlambda_result, dsigma_result)