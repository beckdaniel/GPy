# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import kernpart
import numpy as np
from GPy.util.linalg import mdot, pdinv

class cor_white(kernpart):
    """
    White noise kernel for Corregionalization Models

    :param D: the number of input dimensions
    :type D: int
    :param variance: the variance of the Matern kernel
    :type variance: float
    :param lengthscale: the lengthscale of the Matern kernel
    :type lengthscale: np.ndarray of size (D,)
    :param period: the period
    :type period: float
    :param n_freq: the number of frequencies considered for the periodic subspace
    :type n_freq: int
    :rtype: kernel object

    """
    def __init__(self,base_kern,R=1.,Dw=1):
        self.name = 'cor_white'
        self.base_kern = base_kern
        self.D = self.base_kern.D
        self.R = R
        self.Dw = Dw
        #self.kappa = 1.
        self.B = np.eye(self.R)
        self.Nparam = self.base_kern.Nparam
        #self._set_params(np.hstack([self.base_kern._get_params()]))
        self._set_params(self.base_kern._get_params())

        #initialize cache NOTE ???
        #self._X, self._X2, self._params = np.empty(shape=(3,1))

    def _get_params(self):
        #return np.hstack([self.base_kern._get_params(),self.W.flatten(),self.kappa])
        return self.base_kern._get_params()

    def _set_params(self,x):
        assert x.size == self.Nparam
        #self.kappa = x[-1]
        #self.W = x[-(self.Dw*self.R+1):-1].reshape(self.Dw,self.R)
        #self.base_kern._set_params(x[:-(self.Dw*self.R+1)])
        #self.B = np.dot(self.W.T,self.W) + self.kappa*np.eye(self.R)
        self.base_kern._set_params(x)

    def _get_param_names(self):
        names = self.base_kern._get_param_names()
        #temp = np.meshgrid(range(self.R),range(self.Dw))
        #_i = temp[1].flatten()
        #_j = temp[0].flatten()
        #names += [ 'W_%s_%s' %(i,j) for i,j in zip(_i,_j)] + ['kappa']
        return names

    def K(self,X,X2,target,X_i,X2_i):
        i_cov, I_cov = self._cross_ref(X_i)
        j_cov, J_cov = self._cross_ref(X2_i)
        BK = []
        for i,I in zip(i_cov,I_cov):
            BK.append([])
            for j,J in zip(j_cov,J_cov):
                BK[-1].append(self.B[i,j]*self.base_kern.K(X[I,:],X2[J,:])) #NOTE target?
            BK[-1] = np.hstack(BK[-1])
        BK = np.vstack(BK)
        target +=BK

    def Kdiag(self,X,target,X_i):
        i_cov, I_cov = self._cross_ref(X_i)
        BK = []
        for i,I in zip(i_cov,I_cov):
                BK.append(self.B[i,i]*self.base_kern.Kdiag(X[I,:])) #NOTE target?
        target += np.hstack(BK)

    def dK_dtheta(self,partial,X,X2,target,X_i,X2_i):
        i_cov, I_cov = self._cross_ref(X_i)
        if X2 is None: X2 = X
        j_cov, J_cov = self._cross_ref(X2_i)
        dtheta_base = 0
        dkappa = 0
        PK = np.zeros((self.R,self.R))
        for i,I in zip(i_cov,I_cov):
            for j,J in zip(j_cov,J_cov):
                dtheta_base += self.B[i,j]*self.base_kern.dK_dtheta(partial[I,J],X[I,:],X2[J,:]) #NOTE target?
                #PK[i,j] += np.sum(partial[I,J]*self.base_kern.K(X[I,:],X2[J,:])) #ARD?
                #if i == j:
                #    dkappa += np.sum(partial[I,J]*self.base_kern.K(X[I,:],X2[J,:])) #ARD?
        #dW = 2*np.sum(PK[:,None,:]*self.W[None,:,:],-1).flatten(1)
        #target += np.hstack([dtheta_base,dW,dkappa])
        target += dtheta_base

    def dKdiag_dtheta(self,partial,X,target,X_i):
        i_cov, I_cov = self._cross_ref(X_i)
        dtheta = 0
        dkappa = 0
        dW = np.zeros((self.Dw,self.R))
        PK = np.zeros(self.R)[:,None]
        for i,I in zip(i_cov,I_cov):
            dtheta += self.B[i,i] * self.base_kern.dKdiag_dtheta(partial[I],X[I,:])
            #dkappa += np.sum(partial[I]*self.base_kern.Kdiag(X[I,:]))
            #PK_ij = partial[I]*self.base_kern.Kdiag(X[I,:]) #ARD?
            #PK[i] += np.sum(partial[I]*self.base_kern.Kdiag(X[I,:])) #ARD?
            #for k in range(self.Dw):
            #    dW[k,i] += 2*np.sum(self.W[k,i] * PK_ij)
        #dW2 = (PK * dW.T).flatten(1)
        #target +=  np.hstack([dtheta,dW.flatten(),dkappa])
        target +=  dtheta

    def dK_dX(self,partial,X,X2,target,X_i,X2_i): #NOTE only gradients wrt X, or also X2?
        i_cov, I_cov = self._cross_ref(X_i)
        j_cov, J_cov = self._cross_ref(X2_i)
        BK = []
        for i,I in zip(i_cov,I_cov):
            BK.append(0)
            for j,J in zip(j_cov,J_cov):
                BK[-1] += self.B[i,j]*self.base_kern.dK_dX(partial[I,J],X[I,:],X2[J,:]) #NOTE target?
        BK = np.vstack(BK)
        target +=BK

    def dKdiag_dX(self,partial,X,target,X_i):
        pass

    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self,Z,mu,S,target,Z_i,mu_i,S_i): #NOTE WTF???
        raise NotImplementedError
        """
        zi_cov, zI_cov = self._cross_ref(Z_i)
        mui_cov, muI_cov = self._cross_ref(mu_i)
        si_cov, sI_cov = self._cross_ref(S_i)
        for zi,zI,mui,muI,si,sI in zip(zi_cov,zI_cov,mui_cov,muI_cov,si_cov,sI_cov):
            target[muI] += self.B[zi,zi]*self.base_kern.psi0(Z[zI,:],mu[muI,:],S[sI,:]) #NOTE target?
        """

    def dpsi0_dtheta(self,partial,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi1(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dtheta(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dZ(self,partial,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi2(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dZ(self,partial,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dtheta(self,partial,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        raise NotImplementedError


    #---------------------------------------#
    #            Other functions            #
    #---------------------------------------#
    def _cross_ref(self,X_i):
        coreg_i = self._remove_duplicates(X_i)
        assert all([self.R >= s for s in coreg_i]), "Coregionalization matrix rank is smaller than number of outputs"
        nx = np.arange(X_i.size)
        #X_slices = [nx[X_i == s].tolist() for s in coreg_i]
        X_slices = [slice(nx[X_i == s][0],1+nx[X_i == s][-1]) for s in coreg_i]
        return coreg_i, X_slices


    def _remove_duplicates(self,index):
        used_terms = []
        return np.array([s for s in index if s not in used_terms and not used_terms.append(s)])
