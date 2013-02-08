# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core.parameterised import parameterised
from kernpart import kernpart
import itertools
from product_orthogonal import product_orthogonal
from .kern import kern

class coreg_kern(kern):
    def __init__(self,D,parts=[], input_slices=None,index=None):
        assert index is not None, "A column of indices must be specified"
        self.index = index
        self.parts = parts
        self.Nparts = len(parts)
        self.Nparam = sum([p.Nparam for p in self.parts])
        self.D = D

        #deal with input_slices
        if input_slices is None:
            self.input_slices = [slice(None) for p in self.parts]
        else:
            assert len(input_slices)==len(self.parts)
            self.input_slices = [sl if type(sl) is slice else slice(None) for sl in input_slices]
        for p in self.parts:
            assert isinstance(p,kernpart), "bad kernel part"
        self.compute_param_slices()
        parameterised.__init__(self)

    def _extract_index(self,X):
        # Indices for X
        X_i = X[:,self.index]
        assert all([int(x) == x for x in X_i]), "Index cannot contain non-integer values"
        _range = range(X.shape[1])
        _range.pop(self.index)
        X = X[:,_range]
        return X,X_i

    def K(self,X,X2=None,slices1=None,slices2=None):
        assert X.shape[1]==self.D + 1
        slices1, slices2 = self._process_slices(slices1,slices2)
        if X2 is None: X2 = X
        target = np.zeros((X.shape[0],X2.shape[0]))
        X,X_i = self._extract_index(X)
        X2,X2_i = self._extract_index(X2)
        [p.K(X[s1,i_s],X2[s2,i_s],target=target[s1,s2],X_i=X_i[s1],X2_i=X2_i[s2]) for p,i_s,s1,s2 in zip(self.parts,self.input_slices,slices1,slices2)]
        return target

    def Kdiag(self,X,slices=None):
        assert X.shape[1]==self.D + 1
        slices = self._process_slices(slices,False)
        X,X_i = self._extract_index(X)
        target = np.zeros(X.shape[0])
        [p.Kdiag(X[s,i_s],target=target[s],X_i=X_i[s]) for p,i_s,s in zip(self.parts,self.input_slices,slices)]
        return target

    def dK_dtheta(self,partial,X,X2=None,slices1=None,slices2=None):
        """
        :param partial: An array of partial derivaties, dL_dK
        :type partial: Np.ndarray (N x M)
        :param X: Observed data inputs
        :type X: np.ndarray (N x D)
        :param X2: Observed dara inputs (optional, defaults to X)
        :type X2: np.ndarray (M x D)
        :param slices1: a slice object for each kernel part, describing which data are affected by each kernel part
        :type slices1: list of slice objects, or list of booleans
        :param slices2: slices for X2
        """
        assert X.shape[1]==self.D + 1
        slices1, slices2 = self._process_slices(slices1,slices2)
        if X2 is None:
            X2 = X
        target = np.zeros(self.Nparam)
        X,X_i = self._extract_index(X)
        X2,X2_i = self._extract_index(X2)
        [p.dK_dtheta(partial[s1,s2],X[s1,i_s],X2[s2,i_s],target[ps],X_i=X_i[s1],X2_i=X2_i[s2]) for p,i_s,ps,s1,s2 in zip(self.parts, self.input_slices, self.param_slices, slices1, slices2)]
        return target

    def dKdiag_dtheta(self,partial,X,slices=None):
        assert X.shape[1]==self.D + 1
        assert len(partial.shape)==1
        assert partial.size==X.shape[0]
        slices = self._process_slices(slices,False)
        target = np.zeros(self.Nparam)
        X,X_i = self._extract_index(X)
        [p.dKdiag_dtheta(partial[s],X[s,i_s],target[ps],X_i=X_i[s]) for p,i_s,s,ps in zip(self.parts,self.input_slices,slices,self.param_slices)]
        return target

    def dK_dX(self,partial,X,X2=None,slices1=None,slices2=None): #FIXME
        if X2 is None:
            X2 = X
        slices1, slices2 = self._process_slices(slices1,slices2)
        X,X_i = self._extract_index(X)
        X2,X2_i = self._extract_index(X2)
        target = np.zeros_like(X)
        [p.dK_dX(partial[s1,s2],X[s1,i_s],X2[s2,i_s],target[s1,i_s],X_i=X_i[s1],X2_i=X2_i[s2]) for p, i_s, s1, s2 in zip(self.parts, self.input_slices, slices1, slices2)]
        return target

    def dKdiag_dX(self, partial, X, slices=None): #NOTE so far not used
        assert X.shape[1]==self.D + 1
        slices = self._process_slices(slices,False)
        X,X_i = self._extract_index(X)
        target = np.zeros_like(X)
        [p.dKdiag_dX(partial[s],X[s,i_s],target[s,i_s],X_i=X_i[s]) for p,i_s,s in zip(self.parts,self.input_slices,slices)]
        return target

    def psi0(self,Z,mu,S,slices=None):
        raise NotImplementedError
        """
        slices = self._process_slices(slices,False)
        Z,Z_i = self._extract_index(Z)
        mu,mu_i = self._extract_index(mu)
        S,S_i = self._extract_index(S)
        target = np.zeros(mu.shape[0])
        [p.psi0(Z,mu[s],S[s],target[s],Z_i=Z_i[s],mu_i=mu_i[s],S_i=S_i[s]) for p,s in zip(self.parts,slices)]
        return target
        """




