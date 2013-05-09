# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from .. import kern
from ..core import model
from ..util.linalg import pdinv, mdot
from ..util.plot import gpplot, x_frame1D, x_frame2D, Tango
from ..likelihoods import EP

class toy_GP(model):
    def __init__(self, X, likelihood, kernel, normalize_X=False):

        # parse arguments
        self.X = X
        assert len(self.X.shape) == 2
        self.N, self.Q = self.X.shape
        assert isinstance(kernel, kern.kern)
        self.kern = kernel
        self.likelihood = likelihood
        assert self.X.shape[0] == self.likelihood.data.shape[0]
        self.N, self.D = self.likelihood.data.shape

        # here's some simple normalization for the inputs
        if normalize_X:
            self._Xmean = X.mean(0)[None, :]
            self._Xstd = X.std(0)[None, :]
            self.X = (X.copy() - self._Xmean) / self._Xstd
            if hasattr(self, 'Z'):
                self.Z = (self.Z - self._Xmean) / self._Xstd
        else:
            self._Xmean = np.zeros((1, self.X.shape[1]))
            self._Xstd = np.ones((1, self.X.shape[1]))

        if not hasattr(self,'has_uncertain_inputs'):
            self.has_uncertain_inputs = False
        model.__init__(self)

    def dL_dZ(self):
        """
        TODO: one day we might like to learn Z by gradient methods?
        """
        return np.zeros_like(self.Z)

    def _set_params(self, p):
        self.kern._set_params_transformed(p[:self.kern.Nparam_transformed()])
        self.likelihood._set_params(p[self.kern.Nparam_transformed():])  # test by Nicolas

        self.K = self.kern.K(self.X)
        self.K += self.likelihood.covariance_matrix

        self.Ki, self.L, self.Li, self.K_logdet = pdinv(self.K)

        # the gradient of the likelihood wrt the covariance matrix
        if self.likelihood.YYT is None:
            alpha = np.dot(self.Ki, self.likelihood.Y)
            self.dL_dK = 0.5 * (np.dot(alpha, alpha.T) - self.D * self.Ki)
        else:
            tmp = mdot(self.Ki, self.likelihood.YYT, self.Ki)
            self.dL_dK = 0.5 * (tmp - self.D * self.Ki)

    def _get_params(self):
        return np.hstack((self.kern._get_params_transformed(), self.likelihood._get_params()))

    def _get_param_names(self):
        return self.kern._get_param_names_transformed() + self.likelihood._get_param_names()

    def update_likelihood_approximation(self):
        """
        Approximates a non-gaussian likelihood using Expectation Propagation

        For a Gaussian likelihood, no iteration is required:
        this function does nothing
        """
        self.likelihood.fit_full(self.kern.K(self.X))
        self._set_params(self._get_params())  # update the GP

    def _model_fit_term(self):
        """
        Computes the model fit using YYT if it's available
        """
        if self.likelihood.YYT is None:
            return -0.5 * np.sum(np.square(np.dot(self.Li, self.likelihood.Y)))
        else:
            return -0.5 * np.sum(np.multiply(self.Ki, self.likelihood.YYT))

    def log_likelihood(self):
        """
        The log marginal likelihood of the GP.

        For an EP model,  can be written as the log likelihood of a regression
        model for a new variable Y* = v_tilde/tau_tilde, with a covariance
        matrix K* = K + diag(1./tau_tilde) plus a normalization term.
        """

        self.Ki, self.L, self.Li, self.K_logdet = pdinv(self.K)

        #Case 1
        A = -.5*self.K_logdet
        B = -0.5 * mdot(self.likelihood.Y.T,self.Ki,self.likelihood.Y)
        B_ = B.copy()

        #Case 2
        A = -.5*self.K_logdet
        B = -0.5 * mdot(self.likelihood.Y.T,np.diag(1./np.diag(self.K)),self.likelihood.Y)

        #Case 3
        A = -.5*self.K_logdet
        self.psi1 = np.eye(30)
        Qnn = mdot(self.psi1.T,self.Ki,self.psi1)
        B = -0.5 * mdot(self.likelihood.Y.T,Qnn,self.likelihood.Y)
        self.Kmmi = self.Ki
        self.Z = self.X

        #Case 4
        A = -.5*self.K_logdet
        self.Z = self.X
        self.Kmmi = self.Ki
        self.psi1 = np.ones((30,30))
        self.Qnn = mdot(self.psi1.T,self.Kmmi,self.psi1)
        B = -0.5 * mdot(self.likelihood.Y.T,self.Qnn,self.likelihood.Y)

        #Case 5
        A = -.5*self.K_logdet
        self.Z = np.random.rand(4)[:,None]
        self.Kmm = self.kern.K(self.Z)
        self.Kmmi,Lm,Lmi,Kmm_logdet = pdinv(self.Kmm)
        self.psi1 = np.ones((4,30))
        self.Qnn = mdot(self.psi1.T,self.Kmmi,self.psi1) + np.eye(30)
        B = -0.5 * mdot(self.likelihood.Y.T,self.Qnn,self.likelihood.Y)

        #A = -.5*self.K_logdet
        #self.Z = np.random.rand(4)[:,None]
        #self.Kmm = self.kern.K(self.Z)
        #self.Kmmi,Lm,Lmi,Kmm_logdet = pdinv(self.Kmm)
        #self.psi1 = np.ones((4,30))
        #self.Qnn = mdot(self.psi1.T,self.Kmmi,self.psi1) + np.eye(30)
        #self.Qi, Lq, Lqi, Qlogdet = pdinv(self.Qnn)
        #B = -0.5 * mdot(self.likelihood.Y.T,self.Qi,self.likelihood.Y)
        return A + B
        #return -0.5 * self.D * self.K_logdet + self._model_fit_term() + self.likelihood.Z


    def _log_likelihood_gradients(self):
        """
        The gradient of all parameters.

        Note, we use the chain rule: dL_dtheta = dL_dK * d_K_dtheta
        """
        #return np.hstack((self.kern.dK_dtheta(dL_dK=self.dL_dK, X=self.X), self.likelihood._gradients(partial=np.diag(self.dL_dK))))
        #Case 1
        dA_dK = -.5 * self.Ki
        YYT = np.dot(self.likelihood.Y,self.likelihood.Y.T)
        dB_dK = .5 * mdot(self.Ki,YYT,self.Ki)
        #Kiy = np.dot(self.Ki,self.likelihood.Y)
        #dB_dK = .5 * np.dot(Kiy,Kiy.T)
        dL_dtheta = self.kern.dK_dtheta(dA_dK + dB_dK,X=self.X)

        #Case 2
        dA_dK = -.5 * self.Ki
        Kdiag_inv = (1./np.diag(self.K))[:,None]
        Kiy = Kdiag_inv*self.likelihood.Y
        dB_dK = .5 * Kiy*Kiy
        dL_dtheta_1 = self.kern.dK_dtheta(dA_dK,X=self.X)
        dL_dtheta_2 = self.kern.dKdiag_dtheta(dB_dK,X=self.X)
        dL_dtheta = dL_dtheta_1 + dL_dtheta_2

        #Case 3
        dA_dK = -.5 * self.Ki
        Kmmipsi1 = np.dot(self.Kmmi,self.psi1)
        YYT = np.dot(self.likelihood.Y,self.likelihood.Y.T)
        dB_dKmm = .5 * mdot(Kmmipsi1,YYT,Kmmipsi1.T)
        dL_dtheta_1 = self.kern.dK_dtheta(dA_dK,X=self.X)
        dL_dtheta_2 = self.kern.dK_dtheta(dB_dKmm,self.Z)
        dL_dtheta = dL_dtheta_1 + dL_dtheta_2

        #Case 4
        dA_dK = -.5 * self.Ki
        Kmmipsi1 = np.dot(self.Kmmi,self.psi1)
        YYT = np.dot(self.likelihood.Y,self.likelihood.Y.T)
        dB_dKmm = .5 * mdot(Kmmipsi1,YYT,Kmmipsi1.T)
        dL_dtheta_1 = self.kern.dK_dtheta(dA_dK,X=self.X)
        dL_dtheta_2 = self.kern.dK_dtheta(dB_dKmm,self.Z)
        dL_dtheta = dL_dtheta_1 + dL_dtheta_2

        #Case 5
        #dA_dK = -.5 * self.Ki
        #Kmmipsi1Qi = mdot(self.Kmmi,self.psi1,self.Qi)
        #YYT = np.dot(self.likelihood.Y,self.likelihood.Y.T)
        #dB_dKmm = -.5 * mdot(Kmmipsi1Qi,YYT,Kmmipsi1Qi.T)
        #dL_dtheta_1 = self.kern.dK_dtheta(dA_dK,X=self.X)
        #dL_dtheta_2 = self.kern.dK_dtheta(dB_dKmm,self.Z)
        #dL_dtheta = dL_dtheta_1 + dL_dtheta_2




        #dB_dpsi1 = -.5 *np.dot(Kmmipsi1,YYT)
        #dL_dtheta_2 = self.kern.dK_dtheta(dB_dpsi1,self.Z,self.X)

        return np.hstack((dL_dtheta, self.likelihood._gradients(partial=np.diag(self.dL_dK))))

    def _raw_predict(self, _Xnew, which_parts='all', full_cov=False):
        """
        Internal helper function for making predictions, does not account
        for normalization or likelihood

         #TODO: which_parts does nothing


        """
        Kx = self.kern.K(self.X, _Xnew,which_parts=which_parts)
        mu = np.dot(np.dot(Kx.T, self.Ki), self.likelihood.Y)
        KiKx = np.dot(self.Ki, Kx)
        if full_cov:
            Kxx = self.kern.K(_Xnew, which_parts=which_parts)
            var = Kxx - np.dot(KiKx.T, Kx)
        else:
            Kxx = self.kern.Kdiag(_Xnew, which_parts=which_parts)
            var = Kxx - np.sum(np.multiply(KiKx, Kx), 0)
            var = var[:, None]
        return mu, var


    def predict(self, Xnew, which_parts='all', full_cov=False):
        """
        Predict the function(s) at the new point(s) Xnew.

        Arguments
        ---------
        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.Q
        :param which_parts:  specifies which outputs kernel(s) to use in prediction
        :type which_parts: ('all', list of bools)
        :param full_cov: whether to return the folll covariance matrix, or just the diagonal
        :type full_cov: bool
        :rtype: posterior mean,  a Numpy array, Nnew x self.D
        :rtype: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise
        :rtype: lower and upper boundaries of the 95% confidence intervals, Numpy arrays,  Nnew x self.D


           If full_cov and self.D > 1, the return shape of var is Nnew x Nnew x self.D. If self.D == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        """
        # normalize X values
        Xnew = (Xnew.copy() - self._Xmean) / self._Xstd
        mu, var = self._raw_predict(Xnew, which_parts, full_cov)

        # now push through likelihood
        mean, var, _025pm, _975pm = self.likelihood.predictive_values(mu, var, full_cov)

        return mean, var, _025pm, _975pm


    def plot_f(self, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, full_cov=False):
        """
        Plot the GP's view of the world, where the data is normalized and the likelihood is Gaussian

        :param samples: the number of a posteriori samples to plot
        :param which_data: which if the training data to plot (default all)
        :type which_data: 'all' or a slice object to slice self.X, self.Y
        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :param which_parts: which of the kernel functions to plot (additively)
        :type which_parts: 'all', or list of bools
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D

        Plot the posterior of the GP.
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - In higher dimensions, we've no implemented this yet !TODO!

        Can plot only part of the data and part of the posterior functions using which_data and which_functions
        Plot the data's view of the world, with non-normalized values and GP predictions passed through the likelihood
        """
        if which_data == 'all':
            which_data = slice(None)

        if self.X.shape[1] == 1:
            Xnew, xmin, xmax = x_frame1D(self.X, plot_limits=plot_limits)
            if samples == 0:
                m, v = self._raw_predict(Xnew, which_parts=which_parts)
                gpplot(Xnew, m, m - 2 * np.sqrt(v), m + 2 * np.sqrt(v))
                pb.plot(self.X[which_data], self.likelihood.Y[which_data], 'kx', mew=1.5)
            else:
                m, v = self._raw_predict(Xnew, which_parts=which_parts, full_cov=True)
                Ysim = np.random.multivariate_normal(m.flatten(), v, samples)
                gpplot(Xnew, m, m - 2 * np.sqrt(np.diag(v)[:, None]), m + 2 * np.sqrt(np.diag(v))[:, None])
                for i in range(samples):
                    pb.plot(Xnew, Ysim[i, :], Tango.colorsHex['darkBlue'], linewidth=0.25)
            pb.plot(self.X[which_data], self.likelihood.Y[which_data], 'kx', mew=1.5)
            pb.xlim(xmin, xmax)
            ymin, ymax = min(np.append(self.likelihood.Y, m - 2 * np.sqrt(np.diag(v)[:, None]))), max(np.append(self.likelihood.Y, m + 2 * np.sqrt(np.diag(v)[:, None])))
            ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
            pb.ylim(ymin, ymax)
            if hasattr(self, 'Z'):
                pb.plot(self.Z, self.Z * 0 + pb.ylim()[0], 'r|', mew=1.5, markersize=12)

        elif self.X.shape[1] == 2:
            resolution = resolution or 50
            Xnew, xmin, xmax, xx, yy = x_frame2D(self.X, plot_limits, resolution)
            m, v = self._raw_predict(Xnew, which_parts=which_parts)
            m = m.reshape(resolution, resolution).T
            pb.contour(xx, yy, m, vmin=m.min(), vmax=m.max(), cmap=pb.cm.jet)
            pb.scatter(Xorig[:, 0], Xorig[:, 1], 40, Yorig, linewidth=0, cmap=pb.cm.jet, vmin=m.min(), vmax=m.max())
            pb.xlim(xmin[0], xmax[0])
            pb.ylim(xmin[1], xmax[1])
        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def plot(self, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, levels=20):
        """
        TODO: Docstrings!
        :param levels: for 2D plotting, the number of contour levels to use

        """
        # TODO include samples
        if which_data == 'all':
            which_data = slice(None)

        if self.X.shape[1] == 1:

            Xu = self.X * self._Xstd + self._Xmean  # NOTE self.X are the normalized values now

            Xnew, xmin, xmax = x_frame1D(Xu, plot_limits=plot_limits)
            m, var, lower, upper = self.predict(Xnew, which_parts=which_parts)
            gpplot(Xnew, m, lower, upper)
            pb.plot(Xu[which_data], self.likelihood.data[which_data], 'kx', mew=1.5)
            if self.has_uncertain_inputs:
                pb.errorbar(Xu[which_data, 0], self.likelihood.data[which_data, 0],
                            xerr=2 * np.sqrt(self.X_variance[which_data, 0]),
                            ecolor='k', fmt=None, elinewidth=.5, alpha=.5)

            ymin, ymax = min(np.append(self.likelihood.data, lower)), max(np.append(self.likelihood.data, upper))
            ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
            pb.xlim(xmin, xmax)
            pb.ylim(ymin, ymax)
            if hasattr(self, 'Z'):
                Zu = self.Z * self._Xstd + self._Xmean
                pb.plot(Zu, Zu * 0 + pb.ylim()[0], 'r|', mew=1.5, markersize=12)
                    # pb.errorbar(self.X[:,0], pb.ylim()[0]+np.zeros(self.N), xerr=2*np.sqrt(self.X_variance.flatten()))

        elif self.X.shape[1] == 2:  # FIXME
            resolution = resolution or 50
            Xnew, xx, yy, xmin, xmax = x_frame2D(self.X, plot_limits, resolution)
            x, y = np.linspace(xmin[0], xmax[0], resolution), np.linspace(xmin[1], xmax[1], resolution)
            m, var, lower, upper = self.predict(Xnew, which_parts=which_parts)
            m = m.reshape(resolution, resolution).T
            pb.contour(x, y, m, levels, vmin=m.min(), vmax=m.max(), cmap=pb.cm.jet)
            Yf = self.likelihood.Y.flatten()
            pb.scatter(self.X[:, 0], self.X[:, 1], 40, Yf, cmap=pb.cm.jet, vmin=m.min(), vmax=m.max(), linewidth=0.)
            pb.xlim(xmin[0], xmax[0])
            pb.ylim(xmin[1], xmax[1])
            if hasattr(self, 'Z'):
                pb.plot(self.Z[:, 0], self.Z[:, 1], 'wo')

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"
