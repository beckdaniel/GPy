#ifndef CY_TREE_H
#define CY_TREE_H
# distutils: language = c++
# cython: profile=False
import nltk
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from collections import defaultdict
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.list cimport list as clist
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
import multiprocessing as mp
cimport cython
from cython cimport view

cdef extern from "math.h" nogil:
    double sqrt(double x)

cdef extern from "unistd.h" nogil:
    unsigned int sleep(unsigned int seconds)

cdef extern from "stdio.h":
    int printf(char *format, ...) nogil

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
        #tree = nltk.Tree(tree_repr)
        tree = nltk.tree.Tree.fromstring(tree_repr)
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
            #prod_list = [tree.node]
            prod_list = [tree.label()]
            children = []
            for ch in tree:
                ch_id = self._get_node(ch, node_list)
                #prod_list.append(ch.node)
                prod_list.append(ch.label())
                children.append(ch_id)
            node_id = len(node_list)
            prod = ' '.join(prod_list)
            cprod = prod
            node = Node(cprod, node_id, children)
            node_list.append(node)
            return node_id
        else:
            #prod = ' '.join([tree.node, tree[0]])
            prod = ' '.join([tree.label(), tree[0]])
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
                K_result, dlambda, dsigma = self.calc_K(nodes1, nodes2, dict1, dict2)

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
                            if i2 >= len(nodes2):
                                break
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
            #K_result = 4
            #dlambda = 6
            #dsigma = 9
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

ctypedef vector[int] IntList
ctypedef pair[string, IntList] CNode
ctypedef pair[CNode, CNode] NodePair
ctypedef vector[CNode] VecNode
ctypedef pair[int, int] IntPair
ctypedef vector[IntPair] VecIntPair
#ctypedef pair[double, double] DoublePair
#ctypedef pair[double, DoublePair] Triple
#ctypedef unordered_map[int, double] DPCell
#ctypedef unordered_map[int, DPCell] DPStruct
#ctypedef map[int, double] DPStruct
ctypedef map[int, double] DPCell
ctypedef map[int, DPCell] DPStruct
ctypedef map[IntPair, double] DPStruct2
ctypedef map[string, double] SAStruct
ctypedef vector[double*] SAMatrix
#ctypedef view.array DPStruct2
#ctypedef double** DPStruct2
#ctypedef double** DPStruct3
#ctypedef np.ndarray[DTYPE_t, ndim=2] NParray
ctypedef struct Result:
    double k
    double dlambda
    double dsigma
ctypedef struct SAResult:
    double k
    double* dlambda
    double* dsigma

ctypedef pair[NodePair, IntPair] PairTup
ctypedef vector[PairTup] VecPairTup

class ParSubsetTreeKernel(object):
    
    def __init__(self, _lambda=1., _sigma=1., normalize=True, num_threads=1):
        self._lambda = _lambda
        self._sigma = _sigma
        self._tree_cache = {}
        self.normalize = normalize
        self.num_threads = num_threads
        #print self.num_threads

    def _gen_node_list(self, tree_repr):
        """
        Generates an ordered list of nodes from a tree.
        The list is used to generate the node pairs when
        calculating K.
        It also returns a nodes dict for fast node access.
        """
        #tree = nltk.Tree(tree_repr)
        tree = nltk.tree.Tree.fromstring(tree_repr)
        c = 0
        cdef list node_list = []
        self._get_node2(tree, node_list)
        node_list.sort(key=lambda Node x: x.production)
        cdef Node node
        node_dict = dict([(node.node_id, node) for node in node_list])
        final_list = []
        cdef list ch_ids
        for node in node_list:
            if node.children_ids == None:
                final_list.append((node.production, None))
            else:
                ch_ids = []
                for ch in node.children_ids:
                    ch_node = node_dict[ch]
                    index = node_list.index(ch_node)
                    ch_ids.append(index)
                final_list.append((node.production, ch_ids))
        return final_list

    def _get_node(self, tree, node_list):
        """
        Recursive method for generating the node lists.
        """
        cdef Node node
        cdef string cprod
        if type(tree[0]) != str and len(tree[0]) > 0: #non preterm
            #prod_list = [tree.node]
            prod_list = [tree.label()]
            children = []
            for ch in tree:
                ch_id = self._get_node(ch, node_list)
                #prod_list.append(ch.node)
                prod_list.append(ch.label())
                children.append(ch_id)
            node_id = len(node_list)
            prod = ' '.join(prod_list)
            cprod = prod
            node = Node(cprod, node_id, children)
            node_list.append(node)
            return node_id
        else:
            if type(tree[0]) == str:
                #prod = ' '.join([tree.node, tree[0]])
                prod = ' '.join([tree.label(), tree[0]])
            else:
                #prod = ' '.join([tree.node, tree[0].node])
                prod = ' '.join([tree.label(), tree[0].label()])
            cprod = prod
            node_id = len(node_list)
            node = Node(cprod, node_id, None)
            node_list.append(node)
            return node_id            

    def _get_node2(self, tree, node_list):
        """
        Recursive method for generating the node lists.
        """
        cdef Node node
        cdef string cprod
        if type(tree) == str:
            return -1
        if len(tree) == 0:
            return -2
        #prod_list = [tree.node]
        prod_list = [tree.label()]
        children = []
        for ch in tree:
            ch_id = self._get_node2(ch, node_list)
            if ch_id == -1:
                prod_list.append(ch)
            elif ch_id == -2:
                #prod_list.append(ch.node)
                prod_list.append(ch.label())
            else:
                #prod_list.append(ch.node)
                prod_list.append(ch.label())
                children.append(ch_id)
        node_id = len(node_list)
        prod = ' '.join(prod_list)
        cprod = prod
        if children == []:
            children = None
        node = Node(cprod, node_id, children)
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
                self._tree_cache[t_repr] = self._gen_node_list(t_repr)

    def _diag_calculations(self, X_list):
        """
        Calculate the K(x,x) values first because
        they are used in normalization.
        """
        cdef double _lambda = self._lambda
        cdef double _sigma = self._sigma
        cdef Result result
        K_vec = np.zeros(shape=(len(X_list),))
        dlambda_vec = np.zeros(shape=(len(X_list),))
        dsigma_vec = np.zeros(shape=(len(X_list),))
        for i in range(len(X_list)):
            result = calc_K(X_list[i], X_list[i], _lambda, _sigma)
            K_vec[i] = result.k
            dlambda_vec[i] = result.dlambda
            dsigma_vec[i] = result.dsigma
        return (K_vec, dlambda_vec, dsigma_vec)

    def Kdiag(self, X):
        cdef vector[VecNode] X_list
        cdef VecNode vecnode
        cdef CNode cnode
        for tree in X:
            node_list = self._tree_cache[tree[0]]
            vecnode.clear()
            for node in node_list:
                cnode.first = node[0]
                cnode.second.clear()
                if node[1] != None:
                    for ch in node[1]:
                        cnode.second.push_back(ch)
                vecnode.push_back(cnode)
            X_list.push_back(vecnode)
        X_diag_Ks, _, _ = self._diag_calculations(X_list)
        return X_diag_Ks

    @cython.boundscheck(False)
    def K(self, X, X2):
        """
        The method that calls the SSTK for each tree pair. Some shortcuts are used
        when X2 == None (when calculating the Gram matrix for X).
        IMPORTANT: this is the normalized version.

        The final goal is to be able to deal only with C objects inside the main
        gram matrix loop.
        """
        # First, as normal, we build the cache.
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
        cdef vector[VecNode] X_list
        cdef vector[VecNode] X2_list
        cdef VecNode vecnode
        cdef CNode cnode
        #cdef Node node
        for tree in X:
            node_list = self._tree_cache[tree[0]]
            vecnode.clear()
            for node in node_list:
                cnode.first = node[0]
                cnode.second.clear()
                if node[1] != None:
                    for ch in node[1]:
                        cnode.second.push_back(ch)
                #cnode.second = node[1]
                vecnode.push_back(cnode)
            X_list.push_back(vecnode)
        if symmetric:
            X2_list = X_list
        else:
            for tree in X2:
                node_list = self._tree_cache[tree[0]]
                vecnode.clear()
                for node in node_list:
                    cnode.first = node[0]
                    cnode.second.clear()
                    if node[1] != None:
                        for ch in node[1]:
                            cnode.second.push_back(ch)
                    vecnode.push_back(cnode)
                X2_list.push_back(vecnode)


        cdef np.ndarray[DTYPE_t, ndim=1] X_diag_Ks = np.zeros(shape=(len(X),))
        cdef np.ndarray[DTYPE_t, ndim=1] X_diag_dlambdas = np.zeros(shape=(len(X),))
        cdef np.ndarray[DTYPE_t, ndim=1] X_diag_dsigmas = np.zeros(shape=(len(X),))
        cdef np.ndarray[DTYPE_t, ndim=1] X2_diag_Ks, X2_diag_dlambdas, X2_diag_dsigmas
        # Start the diag values for normalization
        if self.normalize:
            X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas = self._diag_calculations(X_list)
            if not symmetric:
                X2_diag_Ks, X2_diag_dlambdas, X2_diag_dsigmas = self._diag_calculations(X2_list)
            
        # Initialize the derivatives here 
        # because we are going to calculate them at the same time as K.
        # It is a bit ugly but way more efficient. We store the derivatives
        # and just return the stored values when dK_dtheta is called.
        cdef np.ndarray[DTYPE_t, ndim=2] Ks = np.zeros(shape=(len(X), len(X2)))
        cdef np.ndarray[DTYPE_t, ndim=2] dlambdas = np.zeros(shape=(len(X), len(X2)))
        cdef np.ndarray[DTYPE_t, ndim=2] dsigmas = np.zeros(shape=(len(X), len(X2)))
        cdef int X_len = len(X)
        cdef int X2_len = len(X2)
        cdef int do_normalize
        cdef int i, j
        cdef VecNode vecnode2
        cdef double _lambda = self._lambda
        cdef double _sigma = self._sigma
        cdef Result result, norm_result
        if self.normalize:
            do_normalize = 1
        else:
            do_normalize = 0
        #print self.normalize
        #print symmetric
        # Iterate over the trees in X and X2 (or X and X in the symmetric case).
        cdef int num_threads = self.num_threads
        #print "NUM THREADS: %d" % num_threads
        with nogil, parallel(num_threads=num_threads):
            for i in prange(X_len, schedule='dynamic'):
                #j = 0
                for j in range(X2_len):
                    if symmetric:
                        if i < j:
                            continue
                        if i == j and do_normalize:
                            Ks[i,j] = 1
                            continue
                        #Ks[i,j] = i * j
                        
                    vecnode = X_list[i]
                    vecnode2 = X2_list[j]
                    result = calc_K(vecnode, vecnode2, _lambda, _sigma)

                    # Normalization happens here.
                    if do_normalize:
                        if symmetric:
                            #K_norm, dlambda_norm, dsigma_norm = normalize(K_result, dlambda, dsigma,
                            norm_result = normalize(result.k, result.dlambda, result.dsigma,               
                                                    X_diag_Ks[i], X_diag_Ks[j],
                                                    X_diag_dlambdas[i], X_diag_dlambdas[j],
                                                    X_diag_dsigmas[i], X_diag_dsigmas[j])
                        else:
                            #K_norm, dlambda_norm, dsigma_norm = self._normalize(K_result, dlambda, dsigma,
                            norm_result = normalize(result.k, result.dlambda, result.dsigma,
                                                    X_diag_Ks[i], X2_diag_Ks[j],
                                                    X_diag_dlambdas[i], X2_diag_dlambdas[j],
                                                    X_diag_dsigmas[i], X2_diag_dsigmas[j])
                    else:
                        norm_result = result
                    
                    # Store everything
                    Ks[i,j] = norm_result.k
                    dlambdas[i,j] = norm_result.dlambda
                    dsigmas[i,j] = norm_result.dsigma
                    if symmetric:
                        Ks[j,i] = norm_result.k
                        dlambdas[j,i] = norm_result.dlambda
                        dsigmas[j,i] = norm_result.dsigma

        return (Ks, dlambdas, dsigmas)

##############
# SYMBOL AWARE
##############


class SymbolAwareSubsetTreeKernel(object):
    
    def __init__(self, _lambda=np.array([0.5]), _sigma=np.array([1.0]),
                 lambda_buckets={}, sigma_buckets={}, normalize=True, num_threads=1):
        #super(SymbolAwareSubsetTreeKernel, self).__init__(_lambda, _sigma, normalize, num_threads)
        self._lambda = _lambda
        self._sigma = _sigma
        self.lambda_buckets = lambda_buckets
        self.sigma_buckets = sigma_buckets
        self._tree_cache = {}
        self.normalize = normalize
        self.num_threads = num_threads

    def _dict_to_map(self, dic):
        cdef SAStruct result_map
        for item in dic:
            result_map[item] = dic[item]
        return result_map

    def _gen_node_list(self, tree_repr):
        """
        Generates an ordered list of nodes from a tree.
        The list is used to generate the node pairs when
        calculating K.
        It also returns a nodes dict for fast node access.
        """
        #tree = nltk.Tree(tree_repr)
        tree = nltk.tree.Tree.fromstring(tree_repr)
        c = 0
        cdef list node_list = []
        self._get_node2(tree, node_list)
        node_list.sort(key=lambda Node x: x.production)
        cdef Node node
        node_dict = dict([(node.node_id, node) for node in node_list])
        final_list = []
        cdef list ch_ids
        for node in node_list:
            if node.children_ids == None:
                final_list.append((node.production, None))
            else:
                ch_ids = []
                for ch in node.children_ids:
                    ch_node = node_dict[ch]
                    index = node_list.index(ch_node)
                    ch_ids.append(index)
                final_list.append((node.production, ch_ids))
        return final_list

    def _get_node2(self, tree, node_list):
        """
        Recursive method for generating the node lists.
        """
        cdef Node node
        cdef string cprod
        if type(tree) == str:
            return -1
        if len(tree) == 0:
            return -2
        #prod_list = [tree.node]
        prod_list = [tree.label()]
        children = []
        for ch in tree:
            ch_id = self._get_node2(ch, node_list)
            if ch_id == -1:
                prod_list.append(ch)
            elif ch_id == -2:
                #prod_list.append(ch.node)
                prod_list.append(ch.label())
            else:
                #prod_list.append(ch.node)
                prod_list.append(ch.label())
                children.append(ch_id)
        node_id = len(node_list)
        prod = ' '.join(prod_list)
        cprod = prod
        if children == []:
            children = None
        node = Node(cprod, node_id, children)
        node_list.append(node)
        return node_id

    def _diag_calculations(self, X_list):
        """
        Calculate the K(x,x) values first because
        they are used in normalization.
        """
        cdef SAResult result
        cdef SAStruct lambda_buckets = self._dict_to_map(self.lambda_buckets)
        cdef SAStruct sigma_buckets = self._dict_to_map(self.sigma_buckets)
        
        K_vec = np.zeros(shape=(len(X_list),))
        dlambda_mat = np.zeros(shape=(len(X_list), len(self._lambda)))
        dsigma_mat = np.zeros(shape=(len(X_list), len(self._sigma)))
        for i in range(len(X_list)):
            result = calc_K_sa(X_list[i], X_list[i], self._lambda, self._sigma, 
                               lambda_buckets, sigma_buckets)
            K_vec[i] = result.k
            for j in range(len(self._lambda)):
                dlambda_mat[i][j] = result.dlambda[j]
            for j in range(len(self._sigma)):
                dsigma_mat[i][j] = result.dsigma[j]

        return (K_vec, dlambda_mat, dsigma_mat)

    def _build_cache(self, X):
        """
        Caches the node lists, for each tree that it is not
        in the cache. If all trees in X are already cached, this
        method does nothing.
        """
        for tree_repr in X:
            t_repr = tree_repr[0]
            if t_repr not in self._tree_cache:
                self._tree_cache[t_repr] = self._gen_node_list(t_repr)

    def Kdiag(self, X):
        cdef vector[VecNode] X_list
        cdef VecNode vecnode
        cdef CNode cnode

        self._build_cache(X)
        for tree in X:
            node_list = self._tree_cache[tree[0]]
            vecnode.clear()
            for node in node_list:
                cnode.first = node[0]
                cnode.second.clear()
                if node[1] != None:
                    for ch in node[1]:
                        cnode.second.push_back(ch)
                vecnode.push_back(cnode)
            X_list.push_back(vecnode)
        X_diag_Ks, _, _ = self._diag_calculations(X_list)
        return X_diag_Ks

    @cython.boundscheck(False)
    def K(self, X, X2):
        """
        The method that calls the SSTK for each tree pair. Some shortcuts are used
        when X2 == None (when calculating the Gram matrix for X).
        IMPORTANT: this is the normalized version.

        The final goal is to be able to deal only with C objects inside the main
        gram matrix loop.
        """
        # First, as normal, we build the cache.
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
        cdef vector[VecNode] X_list
        cdef vector[VecNode] X2_list
        cdef VecNode vecnode
        cdef CNode cnode
        #cdef Node node
        for tree in X:
            node_list = self._tree_cache[tree[0]]
            vecnode.clear()
            for node in node_list:
                cnode.first = node[0]
                cnode.second.clear()
                if node[1] != None:
                    for ch in node[1]:
                        cnode.second.push_back(ch)
                #cnode.second = node[1]
                vecnode.push_back(cnode)
            X_list.push_back(vecnode)
        if symmetric:
            X2_list = X_list
        else:
            for tree in X2:
                node_list = self._tree_cache[tree[0]]
                vecnode.clear()
                for node in node_list:
                    cnode.first = node[0]
                    cnode.second.clear()
                    if node[1] != None:
                        for ch in node[1]:
                            cnode.second.push_back(ch)
                    vecnode.push_back(cnode)
                X2_list.push_back(vecnode)


        cdef np.ndarray[DTYPE_t, ndim=1] X_diag_Ks = np.zeros(shape=(len(X),))
        cdef np.ndarray[DTYPE_t, ndim=1] X_diag_dlambdas = np.zeros(shape=(len(X),))
        cdef np.ndarray[DTYPE_t, ndim=1] X_diag_dsigmas = np.zeros(shape=(len(X),))
        cdef np.ndarray[DTYPE_t, ndim=1] X2_diag_Ks, X2_diag_dlambdas, X2_diag_dsigmas
        # Start the diag values for normalization
        if self.normalize:
            X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas = self._diag_calculations(X_list)
            if not symmetric:
                X2_diag_Ks, X2_diag_dlambdas, X2_diag_dsigmas = self._diag_calculations(X2_list)
            
        # Initialize the derivatives here 
        # because we are going to calculate them at the same time as K.
        # It is a bit ugly but way more efficient. We store the derivatives
        # and just return the stored values when dK_dtheta is called.
        cdef np.ndarray[DTYPE_t, ndim=2] Ks = np.zeros(shape=(len(X), len(X2)))
        cdef np.ndarray[DTYPE_t, ndim=2] dlambdas = np.zeros(shape=(len(X), len(X2)))
        cdef np.ndarray[DTYPE_t, ndim=2] dsigmas = np.zeros(shape=(len(X), len(X2)))
        cdef int X_len = len(X)
        cdef int X2_len = len(X2)
        cdef int do_normalize
        cdef int i, j
        cdef VecNode vecnode2
        cdef double _lambda = self._lambda
        cdef double _sigma = self._sigma
        cdef Result result, norm_result
        if self.normalize:
            do_normalize = 1
        else:
            do_normalize = 0
        #print self.normalize
        #print symmetric
        # Iterate over the trees in X and X2 (or X and X in the symmetric case).
        cdef int num_threads = self.num_threads
        #print "NUM THREADS: %d" % num_threads
        with nogil, parallel(num_threads=num_threads):
            for i in prange(X_len, schedule='dynamic'):
                #j = 0
                for j in range(X2_len):
                    if symmetric:
                        if i < j:
                            continue
                        if i == j and do_normalize:
                            Ks[i,j] = 1
                            continue
                        #Ks[i,j] = i * j
                        
                    vecnode = X_list[i]
                    vecnode2 = X2_list[j]
                    result = calc_K(vecnode, vecnode2, _lambda, _sigma)

                    # Normalization happens here.
                    if do_normalize:
                        if symmetric:
                            #K_norm, dlambda_norm, dsigma_norm = normalize(K_result, dlambda, dsigma,
                            norm_result = normalize(result.k, result.dlambda, result.dsigma,               
                                                    X_diag_Ks[i], X_diag_Ks[j],
                                                    X_diag_dlambdas[i], X_diag_dlambdas[j],
                                                    X_diag_dsigmas[i], X_diag_dsigmas[j])
                        else:
                            #K_norm, dlambda_norm, dsigma_norm = self._normalize(K_result, dlambda, dsigma,
                            norm_result = normalize(result.k, result.dlambda, result.dsigma,
                                                    X_diag_Ks[i], X2_diag_Ks[j],
                                                    X_diag_dlambdas[i], X2_diag_dlambdas[j],
                                                    X_diag_dsigmas[i], X2_diag_dsigmas[j])
                    else:
                        norm_result = result
                    
                    # Store everything
                    Ks[i,j] = norm_result.k
                    dlambdas[i,j] = norm_result.dlambda
                    dsigmas[i,j] = norm_result.dsigma
                    if symmetric:
                        Ks[j,i] = norm_result.k
                        dlambdas[j,i] = norm_result.dlambda
                        dsigmas[j,i] = norm_result.dsigma

        return (Ks, dlambdas, dsigmas)

##############
# EXTERNAL METHODS
##############

cdef void print_node(CNode node) nogil:
    printf("(%s, [", node.first.c_str())
    for ch in node.second:
        printf("%d, ", ch)
    printf("])")

cdef void print_vec_node(VecNode vecnode) nogil:
    printf("[")
    for node in vecnode:
        print_node(node)
        printf(", ")
    printf("]\n")

cdef void print_node_pair(NodePair node_pair) nogil:
    printf("(")
    print_node(node_pair.first)
    printf(" ,")
    print_node(node_pair.second)
    printf(" )")

cdef void print_int_pair(IntPair int_pair) nogil:
    printf("(%d, %d)", int_pair.first, int_pair.second)

cdef void print_int_pairs(VecIntPair int_pairs) nogil:
    printf("[")
    cdef IntPair int_pair
    for int_pair in int_pairs:
        print_int_pair(int_pair)
    printf("]\n")

cdef VecIntPair get_node_pairs(VecNode& vecnode1, VecNode& vecnode2) nogil:
    """
    The node pair detection method devised by Moschitti (2006).
    """
    cdef VecIntPair int_pairs
    cdef int i1 = 0
    cdef int i2 = 0
    cdef CNode n1, n2
    cdef int len1 = vecnode1.size()
    cdef int len2 = vecnode2.size()
    cdef IntPair tup
    cdef int reset2
    while True:
        if (i1 >= len1) or (i2 >= len2):
            return int_pairs
        n1 = vecnode1[i1]
        n2 = vecnode2[i2]
        if n1.first > n2.first:
            i2 += 1
        elif n1.first < n2.first:
            i1 += 1
        else:
            while n1.first == n2.first:
                reset2 = i2
                while n1.first == n2.first:
                    tup.first = i1
                    tup.second = i2
                    int_pairs.push_back(tup)
                    i2 += 1
                    if i2 >= len2:
                        #return int_pairs
                        break
                    n2 = vecnode2[i2]
                i1 += 1
                if i1 >= len1:
                    return int_pairs
                i2 = reset2
                n1 = vecnode1[i1]
                n2 = vecnode2[i2]
    return int_pairs


cdef Result calc_K(VecNode& vecnode1, VecNode& vecnode2, double _lambda, double _sigma) nogil:
    """
    The actual SSTK kernel, evaluated over two node lists.
    It also calculates the derivatives wrt lambda and sigma.
    """
    cdef double K_total = 0
    cdef double dlambda_total = 0
    cdef double dsigma_total = 0
    cdef Result result
    cdef VecIntPair node_pairs
    node_pairs = get_node_pairs(vecnode1, vecnode2)

    cdef int len1 = vecnode1.size()
    cdef int len2 = vecnode2.size()
    cdef double* delta_matrix = <double*> malloc(len1 * len2 * sizeof(double))
    cdef double* dlambda_matrix = <double*> malloc(len1 * len2 * sizeof(double))
    cdef double* dsigma_matrix = <double*> malloc(len1 * len2 * sizeof(double))
    
    cdef int i, j
    cdef int index

    for i in range(len1):
        for j in range(len2):
            index = i * len2 + j
            delta_matrix[index] = 0
            dlambda_matrix[index] = 0
            dsigma_matrix[index] = 0

    cdef double *k = <double*> malloc(sizeof(double))
    cdef double *dlambda = <double*> malloc(sizeof(double))
    cdef double *dsigma = <double*> malloc(sizeof(double))

    #printf("ALLOCATED\n")
    for int_pair in node_pairs:
        delta(int_pair.first, int_pair.second, 
              vecnode1, vecnode2,
              delta_matrix, dlambda_matrix,
              dsigma_matrix, _lambda, _sigma,
              k, dlambda, dsigma)
        K_total += k[0]
        dlambda_total += dlambda[0]
        dsigma_total += dsigma[0]

    result.k = K_total
    result.dlambda = dlambda_total
    result.dsigma = dsigma_total

    free(delta_matrix)
    free(dlambda_matrix)
    free(dsigma_matrix)
    free(k)
    free(dlambda)
    free(dsigma)

    return result


cdef void delta(int id1, int id2, VecNode& vecnode1, VecNode& vecnode2,
                  double* delta_matrix,
                  double* dlambda_matrix,
                  double* dsigma_matrix,
                  double _lambda, double _sigma,
                  double* k, double* dlambda, double* dsigma) nogil:
    """
    Recursive method used in kernel calculation.
    It also calculates the derivatives wrt lambda and sigma.
    """
    cdef int ch1, ch2, i
    cdef double val, prod, 
    cdef double sum_lambda, sum_sigma, denom
    cdef IntList children1, children2
    cdef CNode node1, node2
    cdef int len2 = vecnode2.size()
    cdef int index = id1 * len2 + id2
    val = delta_matrix[index]
    if val > 0:
        k[0] = val
        dlambda[0] = dlambda_matrix[index]
        dsigma[0] = dsigma_matrix[index]
        return

    node1 = vecnode1[id1]
    if node1.second.empty():
        delta_matrix[index] = _lambda
        dlambda_matrix[index] = 1
        k[0] = _lambda
        dlambda[0] = 1
        dsigma[0] = 0
        return

    node2 = vecnode2[id2]
    prod = 1
    sum_lambda = 0
    sum_sigma = 0
    children1 = node1.second
    children2 = node2.second
    for i in range(children1.size()):
        ch1 = children1[i]
        ch2 = children2[i]
        if vecnode1[ch1].first == vecnode2[ch2].first:
            delta(ch1, ch2, vecnode1, vecnode2,
                  delta_matrix, dlambda_matrix,
                  dsigma_matrix, _lambda, _sigma,
                  k, dlambda, dsigma)

            denom = _sigma + k[0]
            prod *= denom
            sum_lambda += dlambda[0] / denom
            sum_sigma += (1 + dsigma[0]) / denom
        else:
            prod *= _sigma
            sum_sigma += 1 /_sigma

    delta_result = _lambda * prod
    dlambda_result = prod + (delta_result * sum_lambda)
    dsigma_result = delta_result * sum_sigma

    delta_matrix[index] = delta_result
    dlambda_matrix[index] = dlambda_result
    dsigma_matrix[index] = dsigma_result

    k[0] = delta_result
    dlambda[0] = dlambda_result
    dsigma[0] = dsigma_result

    #return result

cdef Result normalize(double K_result, double dlambda, double dsigma, double diag_Ks_i, 
                      double diag_Ks_j, double diag_dlambdas_i, double diag_dlambdas_j, 
                      double diag_dsigmas_i, double diag_dsigmas_j) nogil:
    """
    Normalize the result from SSTK, including derivatives.
    """
    cdef double norm, sqrt_nrorm, K_norm, diff_lambda, dlambda_norm, diff_sigma, dsigma_norm
    cdef Result result

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

    result.k = K_norm
    result.dlambda = dlambda_norm
    result.dsigma = dsigma_norm
    return result


cdef SAResult calc_K_sa(VecNode& vecnode1, VecNode& vecnode2, double[:] _lambda, double[:] _sigma, 
                        SAStruct sa_lambda, SAStruct sa_sigma) nogil:
    """
    The actual SSTK kernel, evaluated over two node lists.
    It also calculates the derivatives wrt lambda and sigma.
    """
    cdef int lambda_size = _lambda.shape[0]
    cdef int sigma_size = _sigma.shape[0]

    cdef double k_total = 0
    cdef double* dlambda_total = <double*> malloc(lambda_size * sizeof(double))
    cdef double* dsigma_total = <double*> malloc(sigma_size * sizeof(double))
    cdef SAResult result
    cdef VecIntPair node_pairs
    node_pairs = get_node_pairs(vecnode1, vecnode2)

    cdef int len1 = vecnode1.size()
    cdef int len2 = vecnode2.size()
    cdef double* delta_matrix = <double*> malloc(len1 * len2 * sizeof(double))    
    cdef int i, j
    cdef int index

    cdef SAMatrix dlambda_vecmatrix
    cdef SAMatrix dsigma_vecmatrix
    cdef double* symbol_matrix
    for i in range(lambda_size):
        symbol_matrix = <double*> malloc(len1 * len2 * sizeof(double))
        dlambda_vecmatrix.push_back(symbol_matrix)
    for i in range(sigma_size):
        symbol_matrix = <double*> malloc(len1 * len2 * sizeof(double))
        dsigma_vecmatrix.push_back(symbol_matrix)

    for i in range(len1):
        for j in range(len2):
            index = i * len2 + j
            delta_matrix[index] = 0

    for sid in range(lambda_size):
        for i in range(len1):
            for j in range(len2):
                index = i * len2 + j
                dlambda_vecmatrix[sid][index] = 0

    for sid in range(sigma_size):
        for i in range(len1):
            for j in range(len2):
                index = i * len2 + j
                dsigma_vecmatrix[sid][index] = 0

    #cdef double k = 0
    #cdef double* dlambda = <double*> malloc(lambda_size * sizeof(double))
    #cdef double* dsigma = <double*> malloc(sigma_size * sizeof(double))

    for i in range(lambda_size):
        dlambda_total[i] = 0
    for i in range(sigma_size):
        dsigma_total[i] = 0


    cdef SAResult delta_result
    #printf("ALLOCATED\n")
    for int_pair in node_pairs:
        delta_result = sa_delta(int_pair.first, int_pair.second, 
                                vecnode1, vecnode2,
                                delta_matrix, dlambda_vecmatrix,
                                dsigma_vecmatrix, _lambda, _sigma)

        k_total += delta_result.k
        for i in range(lambda_size):
            dlambda_total[i] += delta_result.dlambda[i]
        for i in range(sigma_size):
            dsigma_total[i] += delta_result.dsigma[i]

    result.k = k_total
    result.dlambda = dlambda_total
    result.dsigma = dsigma_total

    free(delta_matrix)
    for i in range(lambda_size):
        free(dlambda_vecmatrix[i])
    for i in range(sigma_size):
        free(dsigma_vecmatrix[i])
    #free(k)
    #free(dlambda)
    #free(dsigma)

    return result

cdef SAResult sa_delta(int id1, int id2, VecNode& vecnode1, VecNode& vecnode2,
                       double* delta_matrix,
                       SAMatrix dlambda_vecmatrix,
                       SAMatrix dsigma_vecmatrix,
                       double[:] _lambda, double[:] _sigma) nogil:
    """
    Recursive method used in kernel calculation.
    It also calculates the derivatives wrt lambda and sigma.
    """
    cdef int ch1, ch2, i
    cdef double val, prod, 
    cdef double sum_lambda, sum_sigma, denom
    cdef IntList children1, children2
    cdef CNode node1, node2
    cdef int len2 = vecnode2.size()
    cdef int index = id1 * len2 + id2

    cdef int lambda_size = _lambda.shape[0]
    cdef int sigma_size = _sigma.shape[0]
    cdef SAResult result
    result.dlambda = <double*> malloc(lambda_size * sizeof(double))
    result.dsigma = <double*> malloc(sigma_size * sizeof(double))

    val = delta_matrix[index]
    if val > 0:
        result.k = val
        for i in range(lambda_size):
            result.dlambda[i] = dlambda_vecmatrix[i][index]
        for i in range(sigma_size):
            result.dsigma[i] = dsigma_vecmatrix[i][index]
        return result

    # node1 = vecnode1[id1]
    # if node1.second.empty():
    #     delta_matrix[index] = _lambda
    #     dlambda_matrix[index] = 1
    #     k[0] = _lambda
    #     dlambda[0] = 1
    #     dsigma[0] = 0
    #     return

    # node2 = vecnode2[id2]
    # prod = 1
    # sum_lambda = 0
    # sum_sigma = 0
    # children1 = node1.second
    # children2 = node2.second
    # for i in range(children1.size()):
    #     ch1 = children1[i]
    #     ch2 = children2[i]
    #     if vecnode1[ch1].first == vecnode2[ch2].first:
    #         delta(ch1, ch2, vecnode1, vecnode2,
    #               delta_matrix, dlambda_matrix,
    #               dsigma_matrix, _lambda, _sigma,
    #               k, dlambda, dsigma)

    #         denom = _sigma + k[0]
    #         prod *= denom
    #         sum_lambda += dlambda[0] / denom
    #         sum_sigma += (1 + dsigma[0]) / denom
    #     else:
    #         prod *= _sigma
    #         sum_sigma += 1 /_sigma

    # delta_result = _lambda * prod
    # dlambda_result = prod + (delta_result * sum_lambda)
    # dsigma_result = delta_result * sum_sigma

    # delta_matrix[index] = delta_result
    # dlambda_matrix[index] = dlambda_result
    # dsigma_matrix[index] = dsigma_result

    # k[0] = delta_result
    # dlambda[0] = dlambda_result
    # dsigma[0] = dsigma_result

    return result

#endif //CY_TREE_H
