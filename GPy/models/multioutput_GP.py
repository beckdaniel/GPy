# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from GP import GP
from sparse_GP import sparse_GP
from .. import likelihoods
from .. import kern
from ..util.plot import gpplot,x_frame1D,x_frame2D, Tango
from ..likelihoods import EP

class multioutput_GP(sparse_GP):
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

    def __init__(self,X_list,likelihood_list,kernel=None,Z=None,M=10,normalize_X=False, normalize_Y=False, Xslices_list=None,Zslices_list=None):

        #Aggregate X_list and normalize values
        X = np.vstack (X_list)
        self.Xos = self._get_output_slices(X,kernel) #Slices per output
        Xslices = None #TODO how to handle this?
        if normalize_X: #Better to normalize before passing to GP/sparse_GP
            _Xmean = X.mean(0)[None,:]
            _Xstd = X.std(0)[None,:]
            _Xmean[:,self.index] = 0 # Index column shouldn't be normilized
            _Xstd[:,self.index] = 1 # Index column shouldn't be normilized
            X = (X.copy() - _Xmean) / _Xstd
        else:
            _Xmean = np.zeros((1,X.shape[1]))
            _Xstd = np.ones((1,X.shape[1]))

        #Aggregate Y_list and normalize values
        self.likelihoods = [likelihoods.Gaussian(l.Y,normalize=normalize_Y) for l in likelihood_list] #Needed for handling mixed likelihoods
        Y = np.vstack([l.Y for l in self.likelihoods]) #Needed for using GP/sparse_GP
        likelihood = likelihoods.Gaussian(Y,normalize=False)

        #Default kernel
        if kernel is None:
            base = kern.rbf(X.shape[1]-1)
            kernel = kern.icm(base,R=len(X_list),index=0)

        #Inducing inputs
        if Z is None:
            Z = np.vstack([np.random.permutation(Xi.copy())[:M] for Xi in X_list]) #FIXME inducing inputs are used across all outputs
        else:
            self._M = Z.shape[0]
            Z = np.vstack([np.hstack([np.repeat(i,Z.shape[0])[:,None], Z]) for i in range(len(X_list))])
        assert Z.shape[1]==X.shape[1]
        self.Zos = self._get_output_slices(Z,kernel) #Slices per output
        Zslices = None #TODO how to handle this?
        if normalize_X: #Better to normalize before passing to GP/sparse_GP
            _Zmean = Z.mean(0)[None,:]
            _Zstd = Z.std(0)[None,:]
            _Zmean[:,self.index] = 0 # Index column shouldn't be normilized
            _Zstd[:,self.index] = 1 # Index column shouldn't be normilized
            Z = (Z.copy() - _Zmean) / _Zstd
        else:
            _Zmean = np.zeros((1,Z.shape[1]))
            _Zstd = np.ones((1,Z.shape[1]))


        #Pass through GP/sparse_GP
        sparse_GP.__init__(self, X, likelihood, kernel, Z, normalize_X=False, Xslices=Xslices)

        #Overwrite _Xmean and _Xstd from GP/sparseGP
        self._Xmean = _Xmean
        self._Xstd = _Xstd
        self._Zmean = _Zmean
        self._Zstd = _Zstd

    def _get_output_slices(self,X,kernel):
        """
        Return the slices from X that correspond to each output
        """
        self.R = kernel.parts[0].R
        self.index = kernel.index
        _range = range(X.shape[1])
        _range.pop(self.index)
        self.input_cols = np.array(_range)
        self._I = X[:,self.index]
        used_terms = []
        self.output_nums = np.array([int(s) for s in self._I if s not in used_terms and not used_terms.append(s)])
        assert all([self.R >= s for s in self.output_nums]), "Coregionalization matrix rank is smaller than the number of outputs"
        nx = np.arange(self._I.size)
        return [slice(nx[self._I == s][0],1+nx[self._I == s][-1]) for s in self.output_nums]

    def _index_off(self,X):
        """
        Remove the output-index column from X
        """
        return X[:,self.input_cols], X[:,self.index][:,None]

    def _index_on(self,X,index):
        """
        Add an output-index column to X
        """
        return np.hstack([X[:,self.input_cols<self.index],index,X[:,self.input_cols>self.index]])

    def _set_params(self, p):
        """
        _Z,_I = self._index_off(self.Z)
        _Z = p[:self.M*(self.Q-1)].reshape(self.M,self.Q-1)
        self.Z = self._index_on(_Z,_I)
        self.kern._set_params(p[_Z.size:_Z.size+self.kern.Nparam])
        self.likelihood._set_params(p[_Z.size+self.kern.Nparam:])
        self._computations()
        """
        _Z = p[:self._M*(self.Q-1)].reshape(self._M,self.Q-1)
        self.Z = np.vstack([np.hstack([np.repeat(i,self._M)[:,None],_Z]) for i in range(len(self.Zos))])
        self.kern._set_params(p[_Z.size:_Z.size+self.kern.Nparam])
        self.likelihood._set_params(p[_Z.size+self.kern.Nparam:])
        self._computations()

    def _get_params(self):
        _Z,_I = self._index_off(self.Z[:self._M,:]) #Remove index and repetitions for each output
        return np.hstack([_Z.flatten(),GP._get_params(self)])

    def _get_param_names(self):
        _Z,_I = self._index_off(self.Z[:self._M,:])
        return sum([['iip_%i_%i'%(i,j) for i in range(_Z.shape[0])] for j in range(_Z.shape[1])],[]) + GP._get_param_names(self)


    def _log_likelihood_gradients(self):#FIXME number of parameters for Z
        #return np.hstack((self.dL_dZ().flatten(), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))
        return np.hstack((np.zeros(self._M*(self.Q-1)), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))


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
        #Normalise X values
        Xnew = (Xnew.copy() - self._Xmean) / self._Xstd
        mu, var = self._raw_predict(Xnew, slices, full_cov)

        #Push through the corresponding likelihood
        mean = []
        _025pm = []
        _975pm = []
        for on in self.output_nums: #Each output can be related to a different likelihood
            _index = Xnew[:,0] == on
            mean_on,_025pm_on,_975pm_on = self.likelihoods[on].predictive_values(mu[_index],var[_index])
            mean.append(mean_on)
            _025pm.append(_025pm_on)
            _975pm.append(_975pm_on)
        mean = np.vstack(mean)
        _025pm = np.vstack(_025pm)
        _975pm = np.vstack(_975pm)
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
            for os,on,oz in zip(self.Xos,self.output_nums,self.Zos):
                Xu, index_ = self._index_off(self.X[os,:])
                Xnew, xmin, xmax = x_frame1D(Xu, plot_limits=plot_limits)
                I_ = np.repeat(on,resolution or 200)[:,None]
                Xnew = self._index_on(Xnew,I_)
                pb.figure()
                if samples == 0:
                    m,v = self._raw_predict(Xnew, slices=which_functions)
                    gpplot(Xnew[which_data,self.input_cols],m,m-2*np.sqrt(v),m+2*np.sqrt(v))
                else: #FIXME
                    m,v = self._raw_predict(Xnew, slices=which_functions,full_cov=True)
                    Ysim = np.random.multivariate_normal(m.flatten(),v,samples)
                    gpplot(Xnew[which_data,self.input_cols],m[which_data,:],m[which_data,:]-2*np.sqrt(np.diag(v)[which_data,:]),m+2*np.sqrt(np.diag(v))[which_data,:])
                    for i in range(samples):
                        pb.plot(Xnew[:,self.input_cols],Ysim[i,:],Tango.coloursHex['darkBlue'],linewidth=0.25)

                pb.plot(self.X[which_data,self.input_cols],self.likelihood.Y[which_data],'kx',mew=1.5)
                pb.xlim(xmin,xmax)
                ymin,ymax = min(np.append(self.likelihood.Y,m-2*np.sqrt(np.diag(v)[:,None]))), max(np.append(self.likelihood.Y,m+2*np.sqrt(np.diag(v)[:,None])))
                ymin, ymax = ymin - 0.1*(ymax - ymin), ymax + 0.1*(ymax - ymin)
                pb.ylim(ymin,ymax)
                pb.plot(self.Z,self.Z*0+pb.ylim()[0],'r|',mew=1.5,markersize=12)
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
            Xnew = [] #needed?
            for os,on,oz in zip(self.Xos,self.output_nums,self.Zos):
                Xu = self.X[os,:] * self._Xstd + self._Xmean #NOTE self.X are the normalized values now
                Xu, index_ = self._index_off(Xu)
                Xnew, xmin, xmax = x_frame1D(Xu, plot_limits=plot_limits)
                I_ = np.repeat(on,resolution or 200)[:,None]
                Xnew = self._index_on(Xnew,I_)
                pb.figure()
                m, var, lower, upper = self.predict(Xnew, slices=which_functions)
                gpplot(Xnew[:,self.input_cols],m, lower, upper)
                pb.plot(Xu,self.likelihood.data[os],'kx',mew=1.5)
                ymin,ymax = min(np.append(self.likelihood.data[os],lower)), max(np.append(self.likelihood.data[os],upper))
                ymin, ymax = ymin - 0.1*(ymax - ymin), ymax + 0.1*(ymax - ymin)
                pb.xlim(xmin,xmax)
                pb.ylim(ymin,ymax)
                Zu = self.Z * self._Xstd + self._Xmean
                Zu, index_ = self._index_off(Zu)
                pb.plot(Zu,Zu*0+pb.ylim()[0],'r|',mew=1.5,markersize=12)

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

    def plot_HD(self,input_col,output_num,plot_limits=None,which_data='all',which_functions='all',resolution=None,full_cov=False):
        # input_col \in self.input_cols
        if which_functions=='all':
            which_functions = [True]*self.kern.Nparts
        if which_data=='all':
            which_data = slice(None)
        os = self.Xos[output_num]
        oz = self.Zos[output_num]

        Xu = self.X[os,:] * self._Xstd + self._Xmean #NOTE self.X are the normalized values now
        Xu, index_ = self._index_off(Xu)
        Xnew, xmin, xmax = self.x_frameHD(Xu,input_col, plot_limits)
        I_ = np.repeat(output_num,resolution or 200)[:,None]
        Xnew = self._index_on(Xnew,I_)
        m, var, lower, upper = self.predict(Xnew, slices=which_functions)
        Xu = self._index_on(Xu,index_)
        gpplot(Xnew[:,input_col],m, lower, upper)
        pb.plot(Xu[:,input_col],self.likelihood.data[os],'kx',mew=1.5)
        #ymin,ymax = min(np.append(self.likelihood.data[os],lower)), max(np.append(self.likelihood.data[os],upper))
        #ymin, ymax = ymin - 0.1*(ymax - ymin), ymax + 0.1*(ymax - ymin)
        #pb.xlim(xmin,xmax)
        #pb.ylim(ymin,ymax)
        Zu = self.Z * self._Xstd + self._Xmean
        pb.plot(Zu[:,input_col],Zu[:,input_col]*0+pb.ylim()[0],'r|',mew=1.5,markersize=12)


    def x_frameHD(self,X,input_col,plot_limits=None,resolution=None):
        """
        Internal helper function for making plots, returns a set of input values to plot as well as lower and upper limits
        """
        if plot_limits is None:
            xmean,xmean = X.mean(0),X.mean(0)
            xmin,xmax = X.min(0),X.max(0)
            xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
        elif len(plot_limits)==2:
            xmin, xmax = plot_limits
        else:
            raise ValueError, "Bad limits for plotting"

        Xnew = np.hstack([np.repeat(mean_i,resolution or 200)[:,None] for mean_i in xmean.flatten()])
        Xnew_i = np.linspace(xmin[0],xmax[0],resolution or 200)[:,None] #FIXME xmin[input_col -1]
        Xnew[:,0] = Xnew_i.flatten() #FIXME
        return Xnew, xmin[input_col], xmax[input_col]

