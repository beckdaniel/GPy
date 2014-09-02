#ifndef CY_TREE_H
#define CY_TREE_H
# distutils: language = c++
# cython: profile=True
import nltk
import numpy as np
cimport numpy as np
from cython.parallel import prange
from collections import defaultdict
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.list cimport list as clist
from libcpp.vector cimport vector
from libc.stdio cimport printf
import multiprocessing as mp

cdef extern from "math.h":
    double sqrt(double x)

DTYPE = np.double
ctypedef np.double_t DTYPE_t
cdef unsigned int MAX_NODES = 300


cdef class Node(object):
    """
    A node object, containing a grammar production, an id and the children ids.
    These are the nodes stored in the node lists implementation of the SSTK
    (the "Fast Tree Kernel" of Moschitti (2006))
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

#ctypedef Node NODE_t

class CySubsetTreeKernel(object):
    
    def __init__(self, _lambda=1., _sigma=1., normalize=True):
        self._lambda = _lambda
        self._sigma = _sigma
        self._tree_cache = {}
        self.normalize = normalize

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
            node = Node(cprod, node_id, children)
            node_list.append(node)
            return node_id
        else:
            prod = ' '.join([tree.node, tree[0]])
            cprod = prod
            node_id = len(node_list)
            node = Node(cprod, node_id, None)
            node_list.append(node)
            return node_id            



    def _build_cache(self, X):
        """
        Caches the node lists, for each tree that it is not
        in the cache. If all trees in X are already cached, this
        method does nothing.
        """
        for tree_repr in X:
            t_repr = tree_repr[0]
            if t_repr not in self._tree_cache:
                node_list, node_dict = self._gen_node_list(t_repr)
                self._tree_cache[t_repr] = (node_list, node_dict)

    def Kdiag(self, X):
        X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas = self._diag_calculations(X)
        return X_diag_Ks
        
    def K(self, X, X2):
        """
        The method that calls the SSTK for each tree pair. Some shortcuts are used
        when X2 == None (when calculating the Gram matrix for X).
        IMPORTANT: this is the normalized version.
        """
        # A check to ensure that derivatives cache will always change when K changes
        #self.dlambdas = None
        #self.dsigmas = None
        #print "YAY"

        # Put any new trees in the cache. If the trees are already cached, this code
        # won't do anything.
        self._build_cache(X)
        if X2 == None:
            symmetric = True
            X2 = X
        else:
            symmetric = False
            self._build_cache(X2)
            
        # Calculate K for diagonal values
        # because we will need them later to normalize.
        if self.normalize:
            X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas = self._diag_calculations(X)
            if not symmetric:
                X2_diag_Ks, X2_diag_dlambdas, X2_diag_dsigmas = self._diag_calculations(X2)
            
        # Initialize the derivatives here 
        # because we are going to calculate them at the same time as K.
        # It is a bit ugly but way more efficient. We store the derivatives
        # and just return the stored values when dK_dtheta is called.
        Ks = np.zeros(shape=(len(X), len(X2)))
        dlambdas = np.zeros(shape=(len(X), len(X2)))
        dsigmas = np.zeros(shape=(len(X), len(X2)))

        ##########
        cdef double _lambda = self._lambda
        cdef double _sigma = self._sigma
        ###########

        #pool = mp.Pool(1)

        # Iterate over the trees in X and X2 (or X and X in the symmetric case).
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                # Shortcut: no calculation is needed for the upper
                # part of the Gram matrix because it is symmetric
                if symmetric:
                    if i > j:
                        Ks[i][j] = Ks[j][i]
                        dlambdas[i][j] = dlambdas[j][i]
                        dsigmas[i][j] = dsigmas[j][i]
                        continue
                    # Another shortcut: because this is the normalized SSTK
                    # diagonal values will always be equal to 1.
                    if i == j and self.normalize:
                        Ks[i][j] = 1
                        continue
                
                # It will always be a 1-element array so we just index by 0
                nodes1, dict1 = self._tree_cache[x1[0]]
                nodes2, dict2 = self._tree_cache[x2[0]]
                #node_pairs = self._get_node_pairs(nodes1, nodes2)
                #node_pairs = self._get_node_pairs(nodes1, nodes2)
                #K_result, dlambda, dsigma = self.calc_K(node_pairs, dict1, dict2)
                K_result, dlambda, dsigma = self.calc_K(nodes1, nodes2, dict1, dict2)
                #K_result, dlambda, dsigma = calc_K_ext(nodes1, nodes2, dict1, dict2, self._lambda, self._sigma)
                #K_result, dlambda, dsigma = self.calc_K(node_pairs, dict1, dict2, _lambda, _sigma)
                #result = pool.apply_async(calc_K_ext, (nodes1, nodes2, dict1, dict2, _lambda, _sigma))
                #K_result, dlambda, dsigma = result.get()
                #K_result, dlambda, dsigma = super_wrapper(nodes1, nodes2, dict1, dict2, _lambda, _sigma)

                # Normalization happens here.
                if self.normalize:
                    if symmetric:
                        K_norm, dlambda_norm, dsigma_norm = self._normalize(K_result, dlambda, dsigma,
                                                                            X_diag_Ks[i], X_diag_Ks[j],
                                                                            X_diag_dlambdas[i], X_diag_dlambdas[j],
                                                                            X_diag_dsigmas[i], X_diag_dsigmas[j])
                    else:
                        K_norm, dlambda_norm, dsigma_norm = self._normalize(K_result, dlambda, dsigma,
                                                                            X_diag_Ks[i], X2_diag_Ks[j],
                                                                            X_diag_dlambdas[i], X2_diag_dlambdas[j],
                                                                            X_diag_dsigmas[i], X2_diag_dsigmas[j])

                # Store everything, including derivatives.
                    Ks[i][j] = K_norm
                    dlambdas[i][j] = dlambda_norm
                    dsigmas[i][j] = dsigma_norm
                else:
                    Ks[i][j] = K_result
                    dlambdas[i][j] = dlambda
                    dsigmas[i][j] = dsigma

        return (Ks, dlambdas, dsigmas)

    def _normalize(self, double K_result, double dlambda, double dsigma, double diag_Ks_i, 
                   double diag_Ks_j, double diag_dlambdas_i, double diag_dlambdas_j, 
                   double diag_dsigmas_i, double diag_dsigmas_j):
        """
        Normalize the result from SSTK, including derivatives.
        """
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
        Calculate the K(x,x) values first because
        they are used in normalization.
        """
        K_vec = np.zeros(shape=(len(X),))
        dlambda_vec = np.zeros(shape=(len(X),))
        dsigma_vec = np.zeros(shape=(len(X),))
        for i, x in enumerate(X):
            nodes, dicts = self._tree_cache[x[0]]
            #node_pairs = self._get_node_pairs(nodes, nodes)
            K_result, dlambda, dsigma = self.calc_K(nodes, nodes, dicts, dicts)
            K_vec[i] = K_result
            dlambda_vec[i] = dlambda
            dsigma_vec[i] = dsigma
        return (K_vec, dlambda_vec, dsigma_vec)

    def _get_node_pairs(self, nodes1, nodes2):
        """
        The node pair detection method devised by Moschitti (2006).
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
                            i2 += 1
                            n2 = nodes2[i2]
                        i1 += 1
                        i2 = reset2
                        n1 = nodes1[i1]
                        n2 = nodes2[i2]
            except IndexError:
                break
        return node_pairs

    def calc_K(self, nodes1, nodes2, dict1, dict2):
        """
        The actual SSTK kernel, evaluated over two node lists.
        It also calculates the derivatives wrt lambda and sigma.
        """
        cdef double K_total = 0
        cdef double dlambda_total = 0
        cdef double dsigma_total = 0
        cdef double K_result, dlambda, dsigma

        # Initialize the DP structure. Python dicts are quite
        # efficient already but maybe a specialized C structure
        # would be better?
        delta_matrix = defaultdict(float)
        dlambda_matrix = defaultdict(float)
        dsigma_matrix = defaultdict(float)

        # We store the hypers inside C doubles and pass them as 
        # parameters for "delta", for efficiency.
        cdef double _lambda = self._lambda
        cdef double _sigma = self._sigma

        node_pairs = self._get_node_pairs(nodes1, nodes2)
        for node_pair in node_pairs:
            K_result, dlambda, dsigma = self.delta(node_pair[0], node_pair[1], dict1, dict2,
                                                   delta_matrix, dlambda_matrix, dsigma_matrix,
                                                   _lambda, _sigma)
            K_total += K_result
            dlambda_total += dlambda
            dsigma_total += dsigma
        return (K_total, dlambda_total, dsigma_total)

    def delta(self, Node node1, Node node2, dict1, dict2,
              delta_matrix, dlambda_matrix, dsigma_matrix,
              double _lambda, double _sigma):
        """
        Recursive method used in kernel calculation.
        It also calculates the derivatives wrt lambda and sigma.
        """
        cdef int id1, id2, ch1, ch2, i
        cdef double val, prod, K_result, dlambda, dsigma, sum_lambda, sum_sigma, denom
        cdef double delta_result, dlambda_result, dsigma_result
        
        cdef Node n1, n2
        id1 = node1.node_id
        id2 = node2.node_id
        tup = (id1, id2)
        val = delta_matrix[tup]
        if val > 0:
            return val, dlambda_matrix[tup], dsigma_matrix[tup]
        if node1.children_ids == None:
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
                K_result, dlambda, dsigma = self.delta(n1, n2, dict1, dict2, 
                                                       delta_matrix,
                                                       dlambda_matrix,
                                                       dsigma_matrix,
                                                       _lambda, _sigma)
                denom = _sigma + K_result
                prod *= denom
                sum_lambda += dlambda / denom
                sum_sigma += (1 + dsigma) / denom
            else:
                prod *= _sigma
                sum_sigma += 1 /_sigma

        delta_result = _lambda * prod
        dlambda_result = prod + (delta_result * sum_lambda)
        dsigma_result = delta_result * sum_sigma

        delta_matrix[tup] = delta_result
        dlambda_matrix[tup] = dlambda_result
        dsigma_matrix[tup] = dsigma_result
        return (delta_result, dlambda_result, dsigma_result)

################
# PARALLEL CLASS
################

cdef struct Node_struct:
    string production
    int node_id
    clist[int] children_ids


class ParSubsetTreeKernel(object):
    
    def __init__(self, _lambda=1., _sigma=1., normalize=True):
        self._lambda = _lambda
        self._sigma = _sigma
        self._tree_cache = {}
        self.normalize = normalize

    def _gen_node_list(self, tree_repr):
        """
        Generates an ordered list of nodes from a tree.
        The list is used to generate the node pairs when
        calculating K.
        It also returns a nodes dict for fast node access.
        """
        tree = nltk.Tree(tree_repr)
        c = 0
        cdef list node_list = []
        self._get_node(tree, node_list)
        node_list.sort(key=lambda Node x: x.production)
        cdef Node node
        node_dict = dict([(node.node_id, node) for node in node_list])
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
            node = Node(cprod, node_id, children)
            node_list.append(node)
            return node_id
        else:
            prod = ' '.join([tree.node, tree[0]])
            cprod = prod
            node_id = len(node_list)
            node = Node(cprod, node_id, None)
            node_list.append(node)
            return node_id            

    def _build_cache(self, X):
        """
        Caches the node lists, for each tree that it is not
        in the cache. If all trees in X are already cached, this
        method does nothing.
        """
        for tree_repr in X:
            t_repr = tree_repr[0]
            if t_repr not in self._tree_cache:
                node_list, node_dict = self._gen_node_list(t_repr)
                self._tree_cache[t_repr] = (node_list, node_dict)

    def K(self, X, X2):
        """
        The method that calls the SSTK for each tree pair. Some shortcuts are used
        when X2 == None (when calculating the Gram matrix for X).
        IMPORTANT: this is the normalized version.

        The final goal is to be able to deal only with C objects inside the main
        gram matrix loop.
        """
        # A check to ensure that derivatives cache will always change when K changes
        #self.dlambdas = None
        #self.dsigmas = None
        #print "YAY"

        # First, as normal, we build the cache. This is exactly as in the normal
        # version, except that we will use C++ STD structures here.
        self._build_cache(X)
        if X2 == None:
            symmetric = True
            X2 = X
        else:
            symmetric = False
            self._build_cache(X2)

        # Caches are python dicts that assign a node_list with a string.
        # We have to convert these node_lists to C++ lists, so we can use them
        # inside the nogil loop.
        cdef clist[Node_struct] X_list
        cdef clist[Node_struct] X2_list
        cdef Node_struct ns1
        for tree in X:
            node_list, node_dict = self._tree_cache[tree[0]]
            #print "AAAAAAA"
            #print n1
            #ns1.production = n1.production
            #print ns1.production
            #printf(ns1.production.c_str())
            ns1.node_id = n1.node_id
            if n1.children_ids is not None:
                ns1.children_ids.clear()
                for ch in n1.children_ids:
                    ns1.children_ids.push_back(ch)
            X_list.push_back(ns1)
        if symmetric:
            X2_list = X_list
        else:
            for tree in X2:
                n1 = self._tree_cache[tree[0]]
                ns1.production = n1.production
                ns1.node_id = n1.node_id
                if n1.children_ids is not None:
                    ns1.children_ids.clear()
                    for ch in n1.children_ids:
                        ns1.children_ids.push_back(ch)
                X2_list.push_back(ns1)

        for node in X_list:
            #print node[0]
            printf(node.production.c_str())

        return
        # Let's forget normalization for now
        #if self.normalize:
        #    X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas = self._diag_calculations(X)
        #    if not symmetric:
        #        X2_diag_Ks, X2_diag_dlambdas, X2_diag_dsigmas = self._diag_calculations(X2)
            
        # Initialize the derivatives here 
        # because we are going to calculate them at the same time as K.
        # It is a bit ugly but way more efficient. We store the derivatives
        # and just return the stored values when dK_dtheta is called.
        Ks = np.zeros(shape=(len(X), len(X2)))
        dlambdas = np.zeros(shape=(len(X), len(X2)))
        dsigmas = np.zeros(shape=(len(X), len(X2)))

        # Iterate over the trees in X and X2 (or X and X in the symmetric case).
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                # Shortcut: no calculation is needed for the upper
                # part of the Gram matrix because it is symmetric
                if symmetric:
                    if i > j:
                        Ks[i][j] = Ks[j][i]
                        dlambdas[i][j] = dlambdas[j][i]
                        dsigmas[i][j] = dsigmas[j][i]
                        continue
                    # Another shortcut: because this is the normalized SSTK
                    # diagonal values will always be equal to 1.
                    if i == j and self.normalize:
                        Ks[i][j] = 1
                        continue
                
                # It will always be a 1-element array so we just index by 0
                nodes1, dict1 = self._tree_cache[x1[0]]
                nodes2, dict2 = self._tree_cache[x2[0]]
                #node_pairs = self._get_node_pairs(nodes1, nodes2)
                #node_pairs = self._get_node_pairs(nodes1, nodes2)
                #K_result, dlambda, dsigma = self.calc_K(node_pairs, dict1, dict2)
                K_result, dlambda, dsigma = self.calc_K(nodes1, nodes2, dict1, dict2)
                #K_result, dlambda, dsigma = calc_K_ext(nodes1, nodes2, dict1, dict2, self._lambda, self._sigma)
                #K_result, dlambda, dsigma = self.calc_K(node_pairs, dict1, dict2, _lambda, _sigma)
                #result = pool.apply_async(calc_K_ext, (nodes1, nodes2, dict1, dict2, _lambda, _sigma))
                #K_result, dlambda, dsigma = result.get()
                #K_result, dlambda, dsigma = super_wrapper(nodes1, nodes2, dict1, dict2, _lambda, _sigma)

                # Normalization happens here.
                if self.normalize:
                    if symmetric:
                        K_norm, dlambda_norm, dsigma_norm = self._normalize(K_result, dlambda, dsigma,
                                                                            X_diag_Ks[i], X_diag_Ks[j],
                                                                            X_diag_dlambdas[i], X_diag_dlambdas[j],
                                                                            X_diag_dsigmas[i], X_diag_dsigmas[j])
                    else:
                        K_norm, dlambda_norm, dsigma_norm = self._normalize(K_result, dlambda, dsigma,
                                                                            X_diag_Ks[i], X2_diag_Ks[j],
                                                                            X_diag_dlambdas[i], X2_diag_dlambdas[j],
                                                                            X_diag_dsigmas[i], X2_diag_dsigmas[j])

                # Store everything, including derivatives.
                    Ks[i][j] = K_norm
                    dlambdas[i][j] = dlambda_norm
                    dsigmas[i][j] = dsigma_norm
                else:
                    Ks[i][j] = K_result
                    dlambdas[i][j] = dlambda
                    dsigmas[i][j] = dsigma

        return (Ks, dlambdas, dsigmas)

##############
# EXTERNAL METHODS
##############


#endif //CY_TREE_H
