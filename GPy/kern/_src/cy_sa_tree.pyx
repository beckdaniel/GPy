#ifndef CY_SA_TREE_H
#define CY_SA_TREE_H
# distutils: language = c++
# cython: profile=False
import nltk
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
cimport cython
from cython cimport view


cdef extern from "math.h" nogil:
    double sqrt(double x)

cdef extern from "stdio.h":
    int printf(char *format, ...) nogil

DTYPE = np.double
ctypedef np.double_t DTYPE_t


ctypedef vector[int] IntList
ctypedef pair[string, string] StrPair
ctypedef pair[int, int] IntPair
ctypedef pair[string, IntList] NodeInf
ctypedef pair[NodeInf, IntPair] CNode
ctypedef pair[CNode, CNode] NodePair
ctypedef vector[CNode] VecNode
ctypedef vector[VecNode] VecVecNode
ctypedef vector[IntPair] VecIntPair
ctypedef vector[double] Vector
ctypedef vector[Vector] Matrix
ctypedef vector[Matrix] Tensor
ctypedef struct SAResult:
    double k
    Vector dlambda
    Vector dsigma


cdef class Node(object):
    """
    A node object, containing a grammar production, an id and the children ids.
    These are the nodes stored in the node lists implementation of the SSTK
    (the "Fast Tree Kernel" of Moschitti (2006))
    """

    cdef basestring production
    cdef int node_id
    cdef list children_ids

    def __init__(self, production, node_id, children_ids):
        self.production = production
        self.node_id = node_id
        self.children_ids = children_ids

    def __repr__(self):
        return str((self.production, self.node_id, self.children_ids))

##############
# SYMBOL AWARE
##############


class SymbolAwareSubsetTreeKernel(object):
    
    def __init__(self, _lambda=np.array([0.5]), _sigma=np.array([1.0]),
                 lambda_buckets={}, sigma_buckets={}, normalize=True, num_threads=1,
                 parallel=True, no_grads=False):
        self._lambda = _lambda
        self._sigma = _sigma
        self.lambda_buckets = lambda_buckets
        self.sigma_buckets = sigma_buckets
        self._tree_cache = {}
        self.normalize = normalize
        self.num_threads = num_threads
        self.no_grads = no_grads
        #self.parallel = parallel

    def _gen_node_list(self, tree_repr):
        """
        Generates an ordered list of nodes from a tree.
        The list is used to generate the node pairs when
        calculating K.
        It also returns a nodes dict for fast node access.
        """
        cdef Node node
        cdef list ch_ids
        cdef list node_list = []
        
        tree = nltk.tree.Tree.fromstring(unicode(tree_repr, encoding='utf-8'))
        self._get_node(tree, node_list)
        node_list.sort(key=lambda Node x: x.production)
        node_dict = dict([(node.node_id, node) for node in node_list])
        final_list = []
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
        cdef basestring production
        
        if type(tree) == str or type(tree) == unicode: # leaf
            return -1
        if len(tree) == 0: # leaf, but non string
            return -2
        prod_list = [tree.label()]
        children = []
        for ch in tree:
            ch_id = self._get_node(ch, node_list)
            if ch_id == -1:
                prod_list.append(ch)
            elif ch_id == -2:
                prod_list.append(ch.label())
            else:
                prod_list.append(ch.label())
                children.append(ch_id)
        node_id = len(node_list)
        production = ' '.join(prod_list)
        if children == []:
            children = None
        node = Node(production, node_id, children)
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

    def Kdiag(self, X):
        """
        Obtain the Gram matrix diagonal, calculating
        the kernel between each input and itself.
        """
        # To ensure all inputs are cached.
        self._build_cache(X)
        X_diag_Ks, _, _ = self._diag_calculations(X)
        return X_diag_Ks
    
    def _diag_calculations(self, X):
        """
        Calculate the kernel between elements
        of X and themselves. Used in Kdiag but also
        in K when normalization is enabled.
        """
        # Convert the python input into a C++ one.
        cdef VecVecNode X_cpp
        X_cpp = convert_input(X, self._tree_cache,
                              self.lambda_buckets, self.sigma_buckets)
        K_vec = np.zeros(shape=(len(X),))
        dlambda_mat = np.zeros(shape=(len(X), len(self._lambda)))
        dsigma_mat = np.zeros(shape=(len(X), len(self._sigma)))
        cdef double[:] K_vec_view = K_vec
        cdef double[:,:] dlambda_view = dlambda_mat
        cdef double[:,:] dsigma_view = dsigma_mat
        cdef int no_grads = self.no_grads
        for i in range(len(X)):
            calc_K(X_cpp[i], X_cpp[i], self._lambda, self._sigma, 
                   K_vec_view[i], dlambda_view[i], dsigma_view[i],
                   no_grads)
        return (K_vec, dlambda_mat, dsigma_mat)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def K(self, X, X2=None):
        """
        Obtain the kernel evaluations between two input vectors.
        If X2 is None, we assume the Gram matrix calculation between
        X and itself.
        """
        # First we build the cache and check if we calculating the Gram matrix.
        self._build_cache(X)
        if X2 == None:
            gram = True
            X2 = X
        else:
            gram = False
            self._build_cache(X2)

        # We have to convert a bunch of stuff to C++ objects
        # since the actual kernel computation happens without the GIL.
        cdef VecVecNode X_cpp = convert_input(X, self._tree_cache,
                                              self.lambda_buckets, self.sigma_buckets)
        cdef VecVecNode X2_cpp = convert_input(X2, self._tree_cache,
                                              self.lambda_buckets, self.sigma_buckets)

        # Start the diag values for normalization
        cdef double[:] X_diag_Ks, X2_diag_Ks
        cdef double[:,:] X_diag_dlambdas, X_diag_dsigmas
        cdef double[:,:] X2_diag_dlambdas, X2_diag_dsigmas
        X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas = self._diag_calculations(X)
        X2_diag_Ks, X2_diag_dlambdas, X2_diag_dsigmas = self._diag_calculations(X2)
            
        # Gradients are calculated at the same time as K.
        cdef int lambda_size = len(self._lambda)
        cdef int sigma_size = len(self._sigma)
        cdef np.ndarray[DTYPE_t, ndim=2] Ks = np.zeros(shape=(len(X), len(X2)))
        cdef np.ndarray[DTYPE_t, ndim=3] dlambdas
        cdef np.ndarray[DTYPE_t, ndim=3] dsigmas
        dlambdas = np.zeros(shape=(len(X), len(X2), lambda_size))
        dsigmas = np.zeros(shape=(len(X), len(X2), sigma_size))
        cdef double[:,:] Ks_view = Ks
        cdef double[:,:,:] dlambdas_view = dlambdas
        cdef double[:,:,:] dsigmas_view = dsigmas

        cdef int normalize, i, j, k, no_grads
        cdef int num_threads = self.num_threads
        cdef VecNode vecnode, vecnode2
        cdef double[:] _lambda = self._lambda
        cdef double[:] _sigma = self._sigma
        normalize = self.normalize
        no_grads = self.no_grads
        if no_grads:
            print "NO_GRADS!"
        
        # Iterate over the trees in X and X2 (or X and X in the symmetric case).
        with nogil, parallel(num_threads=num_threads):
            for i in prange(X_cpp.size(), schedule='dynamic'):
        #for i in range(X_cpp.size()):
                for j in range(X2_cpp.size()):
                    K_wrapper(X_cpp, X2_cpp, i, j, _lambda, _sigma,
                              Ks_view, dlambdas_view, dsigmas_view, gram, normalize, 
                              X_diag_Ks[i], X2_diag_Ks[j], X_diag_dlambdas[i],
                              X2_diag_dlambdas[j], X_diag_dsigmas[i], X2_diag_dsigmas[j],
                              no_grads) 
        return (Ks, dlambdas, dsigmas)

    
######################
# EXTERNAL METHODS
######################


cdef VecVecNode convert_input(X, dict tree_cache, dict lambda_buckets, dict sigma_buckets):
    """
    Convert an input vector (a Python object) to a 
    STL Vector of Nodes (a C++ object). This is to enable
    their use inside no-GIL code, to enable parallelism.
    """
    cdef VecVecNode X_cpp
    cdef VecNode vecnode
    cdef CNode cnode
        
    for tree in X:
        node_list = tree_cache[tree[0]]
        vecnode.clear()
        for node in node_list:
            root = node[0].split()[0]
            if root in lambda_buckets:
                cnode.second.first = lambda_buckets[root]
            else:
                cnode.second.first = 0
            if root in sigma_buckets:
                cnode.second.second = sigma_buckets[root]
            else:
                cnode.second.second = 0
            cnode.first.first = node[0].encode('utf-8')
            cnode.first.second.clear()
            if node[1] != None:
                for ch in node[1]:
                    cnode.first.second.push_back(ch)
            vecnode.push_back(cnode)
        X_cpp.push_back(vecnode)
    return X_cpp


@cython.wraparound(False)
@cython.boundscheck(False)
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
    cdef int reset
    
    while True:
        if (i1 >= len1) or (i2 >= len2):
            return int_pairs
        n1 = vecnode1[i1]
        n2 = vecnode2[i2]
        if n1.first.first > n2.first.first:
            i2 += 1
        elif n1.first.first < n2.first.first:
            i1 += 1
        else:
            while n1.first.first == n2.first.first:
                reset = i2
                while n1.first.first == n2.first.first:
                    tup.first = i1
                    tup.second = i2
                    int_pairs.push_back(tup)
                    i2 += 1
                    if i2 >= len2:
                        break
                    n2 = vecnode2[i2]
                i1 += 1
                if i1 >= len1:
                    return int_pairs
                i2 = reset
                n1 = vecnode1[i1]
                n2 = vecnode2[i2]
    return int_pairs


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void K_wrapper(VecVecNode& X_cpp, VecVecNode& X2_cpp, int i,
                    int j, double[:] _lambda, double[:] _sigma,
                    double[:,:] Ks, double[:,:,:] dlambdas, double[:,:,:] dsigmas,
                    int gram, int normalize, double X_diag_Ks_i,
                    double X2_diag_Ks_j, double[:] X_diag_dlambdas_i,
                    double[:] X2_diag_dlambdas_j, double[:] X_diag_dsigmas_i,
                    double[:] X2_diag_dsigmas_j, int no_grads) nogil:
    """
    Wrapper around K calculation.
    """ 
    if gram:
        if i < j:
            return
        if i == j and normalize:
            Ks[i,j] = 1
            return
    vecnode = X_cpp[i]
    vecnode2 = X2_cpp[j]
    calc_K(vecnode, vecnode2, _lambda, _sigma,
           Ks[i,j], dlambdas[i,j], dsigmas[i,j], no_grads)
    if normalize == 1:
        _normalize(Ks[i,j], dlambdas[i,j], dsigmas[i,j],
                   X_diag_Ks_i, X2_diag_Ks_j, X_diag_dlambdas_i,
                   X2_diag_dlambdas_j, X_diag_dsigmas_i, X2_diag_dsigmas_j,
                   no_grads)    
    if gram:
        Ks[j,i] = Ks[i,j]
        if not no_grads:
            dlambdas[j,i] = dlambdas[i,j]
            dsigmas[j,i] = dsigmas[i,j]


@cython.wraparound(False)
@cython.boundscheck(False)            
cdef void calc_K(VecNode& vecnode1, VecNode& vecnode2,
                 double[:] _lambda, double[:] _sigma, 
                 double &K_result, double[:] dlambdas, 
                 double[:] dsigmas, int no_grads) nogil:
    """
    The actual SSTK kernel, evaluated over two node lists.
    It also calculates the derivatives wrt lambda and sigma.
    """
    cdef int lambda_size = _lambda.shape[0]
    cdef int sigma_size = _sigma.shape[0]
    cdef int i, j, k, index, index2
    cdef int len1 = vecnode1.size()
    cdef int len2 = vecnode2.size()
    cdef double* delta_matrix = <double*> malloc(len1 * len2 * sizeof(double))
    #cdef Matrix delta_matrix = Matrix(len1)
    cdef double* dlambda_tensor
    cdef double* dsigma_tensor
    if not no_grads:
        dlambda_tensor = <double*> malloc(len1 * len2 * lambda_size
                                          * sizeof(double))
        dsigma_tensor = <double*> malloc(len1 * len2 * sigma_size
                                         * sizeof(double))

    cdef VecIntPair node_pairs
    cdef SAResult pair_result
    pair_result.dlambda = Vector(lambda_size)
    pair_result.dsigma = Vector(sigma_size)

    for i in range(len1):
        #delta_matrix.push_back(Vector(len2))
        #delta_matrix[i].assign(len2, 0)
        for j in range(len2):
            index = i * len2 + j
            delta_matrix[index] = 0
            if not no_grads:
                for k in range(lambda_size):
                    index2 = index * lambda_size + k
                    dlambda_tensor[index2] = 0
                for k in range(sigma_size):
                    index2 = index * sigma_size + k
                    dsigma_tensor[index2] = 0

    node_pairs = get_node_pairs(vecnode1, vecnode2)
    for int_pair in node_pairs:
        delta(K_result, dlambdas, dsigmas, pair_result, int_pair, vecnode1, vecnode2,
              delta_matrix, dlambda_tensor, dsigma_tensor, _lambda, _sigma, no_grads)
    
    free(delta_matrix)
    if not no_grads:
        free(dlambda_tensor)
        free(dsigma_tensor)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void delta(double &K_result, double[:] dlambdas, double[:] dsigmas,
                SAResult& pair_result, IntPair int_pair,
                VecNode& vecnode1, VecNode& vecnode2, double* delta_matrix,
                double* dlambda_tensor, double* dsigma_tensor,
                double[:] _lambda, double[:] _sigma, int no_grads) nogil:
    """
    Recursive method used in kernel calculation.
    It also calculates the derivatives wrt lambda and sigma.
    """
    cdef double val, g, denom
    cdef double delta_result, dlambda_result, dsigma_result
    cdef IntList children1, children2
    cdef CNode node1, node2
    cdef int id1, id2, len2, index, index2, i, space
    cdef int lambda_index, sigma_index
    cdef int lambda_size = _lambda.shape[0]
    cdef int sigma_size = _sigma.shape[0]
    cdef IntPair ch_pair
    cdef string production, root

    # RECURSIVE CASE: get value from DP matrix if it was already calculated
    id1 = int_pair.first
    id2 = int_pair.second
    len2 = vecnode2.size()
    index = id1 * len2 + id2
    val = delta_matrix[index]
    #val = delta_matrix[id1][id2]
    if val > 0:
        pair_result.k = val
        if not no_grads:
            for i in range(lambda_size):
                index2 = index * lambda_size + i
                pair_result.dlambda[i] = dlambda_tensor[index2]
            for i in range(sigma_size):
                index2 = index * sigma_size + i
                pair_result.dsigma[i] = dsigma_tensor[index2]
        return

    # BASE CASE: found a preterminal
    node1 = vecnode1[id1]
    lambda_index = node1.second.first
    if node1.first.second.empty():
        #delta_matrix[id1][id2] = _lambda[lambda_index]
        delta_matrix[index] = _lambda[lambda_index] 
        pair_result.k = _lambda[lambda_index]
        (&K_result)[0] = (&K_result)[0] + _lambda[lambda_index]
        if not no_grads:
            for i in range(lambda_size):
                index2 = index * lambda_size + i
                if i == lambda_index:
                    pair_result.dlambda[i] = 1
                    dlambda_tensor[index2] = 1
                    dlambdas[i] += 1
                else:
                    pair_result.dlambda[i] = 0
                    #dlambda_tensor[index2] = 0
            for i in range(sigma_size):
                index2 = index * sigma_size + i
                pair_result.dsigma[i] = 0
                #dsigma_tensor[index2] = 0
        return

    # RECURSIVE CASE: if val == 0, then we proceed to do recursion
    node2 = vecnode2[id2]
    prod = 1
    sum_lambda = 0
    sum_sigma = 0
    g = 1
    cdef Vector vec_lambda = Vector(lambda_size)
    cdef Vector vec_sigma = Vector(sigma_size)
    if not no_grads:
        vec_lambda.assign(lambda_size, 0)
        vec_sigma.assign(sigma_size, 0)
    
    children1 = node1.first.second
    children2 = node2.first.second
    sigma_index = node1.second.second
    for i in range(children1.size()):
        ch_pair.first = children1[i]
        ch_pair.second = children2[i]
        if vecnode1[ch_pair.first].first.first == vecnode2[ch_pair.second].first.first:
            delta(K_result, dlambdas, dsigmas,
                  pair_result, ch_pair, vecnode1, vecnode2,
                  delta_matrix, dlambda_tensor, dsigma_tensor, _lambda,
                  _sigma, no_grads)
            denom = _sigma[sigma_index] + pair_result.k
            g *= denom
            if not no_grads:
                for j in range(lambda_size):
                    vec_lambda[j] += pair_result.dlambda[j] / denom
                for j in range(sigma_size):
                    if j == sigma_index:
                        vec_sigma[j] += (1 + pair_result.dsigma[j]) / denom
                    else:
                        vec_sigma[j] += pair_result.dsigma[j] / denom
        else:
            g *= _sigma[sigma_index]
            if not no_grads:
                vec_sigma[sigma_index] += 1 / _sigma[sigma_index]

    delta_result = _lambda[lambda_index] * g
    delta_matrix[index] = delta_result
    #delta_matrix[id1][id2] = delta_result
    pair_result.k = delta_result
    (&K_result)[0] = (&K_result)[0] + delta_result
    if not no_grads:
        for i in range(lambda_size):
            index2 = index * lambda_size + i
            dlambda_result = delta_result * vec_lambda[i]
            if i == lambda_index:
                dlambda_result += g
            dlambda_tensor[index2] = dlambda_result
            pair_result.dlambda[i] = dlambda_result
            dlambdas[i] += dlambda_result
     
        for i in range(sigma_size):
            index2 = index * sigma_size + i
            dsigma_result = delta_result * vec_sigma[i]
            dsigma_tensor[index2] = dsigma_result
            pair_result.dsigma[i] = dsigma_result
            dsigmas[i] += dsigma_result


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _normalize(double& K_result, double[:] dlambdas, double[:] dsigmas,
                     double diag_Ks_i, double diag_Ks_j, 
                     double[:] diag_dlambdas_i, double[:] diag_dlambdas_j, 
                     double[:] diag_dsigmas_i, double[:] diag_dsigmas_j,
                     int no_grads) nogil:
    """
    Normalize the result from SSTK, including gradients.
    """
    cdef double norm, sqrt_nrorm, K_norm, diff_lambda 
    cdef double dlambda_norm, diff_sigma, dsigma_norm
    cdef int i
    cdef int lambda_size = dlambdas.shape[0]
    cdef int sigma_size = dsigmas.shape[0]

    norm = diag_Ks_i * diag_Ks_j
    sqrt_norm = sqrt(norm)
    K_norm = (&K_result)[0] / sqrt_norm
    (&K_result)[0] = K_norm
    if not no_grads:
        for i in range(lambda_size):
            diff_lambda = ((diag_dlambdas_i[i] * diag_Ks_j) +
                           (diag_Ks_i * diag_dlambdas_j[i]))
            diff_lambda /= 2 * norm
            dlambdas[i] = ((dlambdas[i] / sqrt_norm) - (K_norm * diff_lambda))
        for i in range(sigma_size):
            diff_sigma = ((diag_dsigmas_i[i] * diag_Ks_j) +
                          (diag_Ks_i * diag_dsigmas_j[i]))
            diff_sigma /= 2 * norm
            dsigmas[i] = ((dsigmas[i] / sqrt_norm) - (K_norm * diff_sigma))

#endif //CY_SA_TREE_H
