# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from GP import GP
from .. import likelihoods
from .. import kern
from ..util.plot import gpplot,x_frame1D,x_frame2D, Tango
from ..likelihoods import EP

class mGP(GP):
    """
    Multiple output Gaussian Process with different likelihoods

    This is a thin wrapper around the GP class, with a set of sensible defalts

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf+white
    :param normalize_X: whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y: whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :param Xslices: how the X,Y data co-vary in the kernel (i.e. which "outputs" they correspond to). See (link:slicing)
    :rtype: model object

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """
    #TODO allow mixed likelihoods (i.e. non-gaussian)

    def __init__(self,X_list,likelihood_list,kernel=None,normalize_X=False, Xslices_list=None):

        X = np.vstack (X_list)
        Y = np.vstack([l.Y for l in likelihood_list])
        likelihood = likelihoods.Gaussian(Y)
        #distribution = likelihoods.likelihood_functions.Poisson()
        #likelihood = likelihoods.EP(Y,distribution)

        if kernel is None:
            base = kern.rbf(X.shape[1]-1)
            kernel = kern.icm(base,R=len(X_list),index=0)

        Xslices = None #FIXME


        GP.__init__(self, X, likelihood, kernel, normalize_X=False)#, Xslices)

    def _get_output_slices(self,X):
        self.R = self.kern.parts[0].R
        self.index = self.kern.index
        _range = range(self.D + 1)
        _range.pop(self.index)
        self.input_cols = np.array(_range)
        self._I = self.X[:,self.index]
        used_terms = []
        self.output_nums = np.array([s for s in self._I if s not in used_terms and not used_terms.append(s)])
        assert all([self.R >= s for s in self.output_nums]), "Coregionalization matrix rank is smaller than number of outputs"
        nx = np.arange(self._I.size)
        return [slice(nx[self._I == s][0],1+nx[self._I == s][-1]) for s in self.output_nums]

    def _index_off(self,X):
        return X[:,self.input_cols], X[:,self.index][:,None]

    def _index_on(self,X,index):
        return np.hstack([X[:,self.input_cols<self.index],index,X[:,self.input_cols>self.index]])


    def predict(self,Xnew, slices=None, full_cov=False):
        """
        Predict the function(s) at the new point(s) Xnew.

        Arguments
        ---------
        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.Q
        :param slices:  specifies which outputs kernel(s) the Xnew correspond to (see below)
        :type slices: (None, list of slice objects, list of ints)
        :param full_cov: whether to return the folll covariance matrix, or just the diagonal
        :type full_cov: bool
        :rtype: posterior mean,  a Numpy array, Nnew x self.D
        :rtype: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise
        :rtype: lower and upper boundaries of the 95% confidence intervals, Numpy arrays,  Nnew x self.D

        .. Note:: "slices" specifies how the the points X_new co-vary wich the training points.

             - If None, the new points covary throigh every kernel part (default)
             - If a list of slices, the i^th slice specifies which data are affected by the i^th kernel part
             - If a list of booleans, specifying which kernel parts are active

           If full_cov and self.D > 1, the return shape of var is Nnew x Nnew x self.D. If self.D == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalisations of the output dimensions.

        """
        #normalise X values
        Xnew, index_ = self._index_off(Xnew)
        Xnew = (Xnew.copy() - self._Xmean) / self._Xstd
        Xnew = self._index_on(Xnew,index_)

        #predict
        mu, var = self._raw_predict(Xnew, slices, full_cov)

        #now push through likelihood TODO
        mean, _025pm, _975pm = self.likelihood.predictive_values(mu, var)

        return mean, var, _025pm, _975pm

    def plot_f(self, samples=0, plot_limits=None, which_data='all', which_functions='all', resolution=None, full_cov=False):
        """
        Plot the GP's view of the world, where the data is normalised and the likelihood is Gaussian

        :param samples: the number of a posteriori samples to plot
        :param which_data: which if the training data to plot (default all)
        :type which_data: 'all' or a slice object to slice self.X, self.Y
        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :param which_functions: which of the kernel functions to plot (additively)
        :type which_functions: list of bools
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D

        Plot the posterior of the GP.
          - In one dimension, the function is plotted with a shaded region identifying two standard deviations.
          - In two dimsensions, a contour-plot shows the mean predicted function
          - In higher dimensions, we've no implemented this yet !TODO!

        Can plot only part of the data and part of the posterior functions using which_data and which_functions
        Plot the data's view of the world, with non-normalised values and GP predictions passed through the likelihood
        """
        if which_functions=='all':
            which_functions = [True]*self.kern.Nparts
        if which_data=='all':
            which_data = slice(None)

        if self.D == 1:
            output_slices = self._get_output_slices(self.X)
            Xnew = []
            for os,on in zip(output_slices,self.output_nums):
                X_, index_ = self._index_off(self.X[os,:])
                Xnew, xmin, xmax = x_frame1D(X_, plot_limits=plot_limits)
                I_ = np.repeat(on,resolution or 200)[:,None]
                Xnew = self._index_on(Xnew,I_)
                #Xnew_ = self._index_on(Xnew_,I_)
                #Xnew.append(Xnew_)
            #Xnew = np.vstack(Xnew)
            #xmin,xmax = Xnew.min(),Xnew.max()
                pb.figure()
                if samples == 0:
                    m,v = self._raw_predict(Xnew, slices=which_functions)
                    gpplot(Xnew[:,self.input_cols],m,m-2*np.sqrt(v),m+2*np.sqrt(v))
                else:
                    m,v = self._raw_predict(Xnew, slices=which_functions,full_cov=True)
                    Ysim = np.random.multivariate_normal(m.flatten(),v,samples)
                    gpplot(Xnew[:,self.input_cols],m,m-2*np.sqrt(np.diag(v)[:,None]),m+2*np.sqrt(np.diag(v))[:,None])
                    for i in range(samples):
                        pb.plot(Xnew[:,self.input_cols],Ysim[i,:],Tango.coloursHex['darkBlue'],linewidth=0.25)

                pb.plot(self.X[which_data,self.input_cols],self.likelihood.Y[which_data],'kx',mew=1.5)
                pb.xlim(xmin,xmax)

        elif self.D == 2: #FIXME
            resolution = resolution or 50
            Xnew, xmin, xmax, xx, yy = x_frame2D(self.X, plot_limits,resolution)
            m,v = self._raw_predict(Xnew, slices=which_functions)
            m = m.reshape(resolution,resolution).T
            pb.contour(xx,yy,m,vmin=m.min(),vmax=m.max(),cmap=pb.cm.jet)
            pb.scatter(Xorig[:,0],Xorig[:,1],40,Yorig,linewidth=0,cmap=pb.cm.jet,vmin=m.min(), vmax=m.max())
            pb.xlim(xmin[0],xmax[0])
            pb.ylim(xmin[1],xmax[1])
        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def plot(self,samples=0,plot_limits=None,which_data='all',which_functions='all',resolution=None,full_cov=False):
        # TODO include samples
        if which_functions=='all':
            which_functions = [True]*self.kern.Nparts
        if which_data=='all':
            which_data = slice(None)

        if self.D == 1:

            output_slices = self._get_output_slices(self.X)
            Xnew = []
            for os,on in zip(output_slices,self.output_nums):
                X_, index_ = self._index_off(self.X[os,:])
                Xnew, xmin, xmax = x_frame1D(X_, plot_limits=plot_limits)
                I_ = np.repeat(on,resolution or 200)[:,None]
                Xnew = self._index_on(Xnew,I_)
                #Xnew.append(Xnew_)
                #Xnew = np.vstack(Xnew)
                #xmin,xmax = Xnew.min(),Xnew.max()
                pb.figure()
                m, var, lower, upper = self.predict(Xnew, slices=which_functions)
                gpplot(Xnew[:,self.input_cols],m, lower, upper)
                pb.plot(self.X[which_data,self.input_cols],self.likelihood.data[which_data],'kx',mew=1.5)
                ymin,ymax = min(np.append(self.likelihood.data,lower)), max(np.append(self.likelihood.data,upper))
                ymin, ymax = ymin - 0.1*(ymax - ymin), ymax + 0.1*(ymax - ymin)
                pb.xlim(xmin,xmax)
                pb.ylim(ymin,ymax)

        elif self.X.shape[1]==2:
            resolution = resolution or 50
            Xnew, xx, yy, xmin, xmax = x_frame2D(self.X, plot_limits,resolution)
            x, y = np.linspace(xmin[0],xmax[0],resolution), np.linspace(xmin[1],xmax[1],resolution)
            m, var, lower, upper = self.predict(Xnew, slices=which_functions)
            m = m.reshape(resolution,resolution).T
            pb.contour(x,y,m,vmin=m.min(),vmax=m.max(),cmap=pb.cm.jet)
            Yf = self.likelihood.Y.flatten()
            pb.scatter(self.X[:,0], self.X[:,1], 40, Yf, cmap=pb.cm.jet,vmin=m.min(),vmax=m.max(), linewidth=0.)
            pb.xlim(xmin[0],xmax[0])
            pb.ylim(xmin[1],xmax[1])
        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

