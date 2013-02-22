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

    :param X_list: list of input observations, one element for each task/output
    :param Y: list of observed values, one element for each task/output
    :param kernel: a GPy coregionalization-kernel, defaults to icm with rbf+white
    :param normalize_X: whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y: whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :param Xslices: how the X,Y data co-vary in the kernel (i.e. which "outputs" they correspond to). See (link:slicing)
    :rtype: model object
    """
    #TODO allow mixed likelihoods (i.e. non-gaussian)

    def __init__(self,X_list,likelihood_list,kernel=None,normalize_X=False, normalize_Y=False, Xslices_list=None):

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
        #distribution = likelihoods.likelihood_functions.Poisson()
        #likelihood = likelihoods.EP(Y,distribution)

        #Default kernel
        if kernel is None:
            base = kern.rbf(X.shape[1]-1)
            kernel = kern.icm(base,R=len(X_list),index=0)

        #Pass through GP/sparse_GP
        GP.__init__(self, X, likelihood, kernel, normalize_X=False, Xslices=Xslices)

        #Overwrite _Xmean and _Xstd from GP/sparseGP
        self._Xmean = _Xmean
        self._Xstd = _Xstd

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
        #TODO: which_data
        if which_functions=='all':
            which_functions = [True]*self.kern.Nparts
        if which_data=='all':
            which_data = slice(None)

        if self.X.shape[1] == 2:
            #TODO: make sure different outputs don't get mixed
            Xu, _index = self._index_off(self.X[which_data,:])
            index = int(_index[0,0])
            Xnew, xmin, xmax = x_frame1D(Xu, plot_limits=plot_limits)
            I_ = np.repeat(index,resolution or 200)[:,None] #repeat the first index number
            Xnew = self._index_on(Xnew,I_)
            if samples == 0:
                m,v = self._raw_predict(Xnew, slices=which_functions)
                gpplot(Xnew[:,self.input_cols],m,m-2*np.sqrt(v),m+2*np.sqrt(v))
            else:
                m,v = self._raw_predict(Xnew, slices=which_functions,full_cov=True)
                Ysim = np.random.multivariate_normal(m.flatten(),v,samples)
                gpplot(Xnew[:,self.input_cols],m,m-2*np.sqrt(np.diag(v)[:,None]),m+2*np.sqrt(np.diag(v)[:,None]))
                for i in range(samples):
                    pb.plot(Xnew[:,self.input_cols],Ysim[i,:],Tango.coloursHex['darkBlue'],linewidth=0.25)
            pb.plot(self.X[which_data,self.input_cols],self.likelihoods[index].Y,'kx',mew=1.5) #repeat the first index number
            pb.xlim(xmin,xmax)
            ymin,ymax = min(np.append(self.likelihoods[index].Y,m-2*np.sqrt(np.diag(v)[:,None]))), max(np.append(self.likelihoods[index].Y,m+2*np.sqrt(np.diag(v)[:,None])))
            ymin, ymax = ymin - 0.1*(ymax - ymin), ymax + 0.1*(ymax - ymin)
            pb.ylim(ymin,ymax)
            #pb.plot(self.Z,self.Z*0+pb.ylim()[0],'r|',mew=1.5,markersize=12)

        elif self.X.shape[1] == 3:
            resolution = resolution or 50
            #for os,on in zip(self.Xos,self.output_nums):
            Xu, _index = self._index_off(self.X[which_data,:])
            index = int(_index[0,0]) #FIXME
            Xnew, xx, yy, x, y, xmin, xmax = x_frame2D(Xu, plot_limits,resolution)
            I_ = np.repeat(index,resolution**2)[:,None]
            Xnew = self._index_on(Xnew,I_)
            m,v = self._raw_predict(Xnew, slices=which_functions)
            m = m.reshape(resolution,resolution)
            pb.contour(x,y,m,vmin=m.min(),vmax=m.max(),cmap=pb.cm.jet)
            pb.scatter(Xu[:,0],Xu[:,1],40,self.likelihoods[index].Y,linewidth=0,cmap=pb.cm.jet,vmin=m.min(), vmax=m.max())
            pb.xlim(xmin[0],xmax[0])
            pb.ylim(xmin[1],xmax[1])
            #if hasattr(self,'Z'):
            #    pb.scatter(self.Z[:,0],self.Z[:,1],'kx',mew=1.5,markersize=12)

        else:
            raise NotImplementedError, "Cannot define a frame with more than two input dimensions"

    def plot(self,samples=0,plot_limits=None,which_data='all',which_functions='all',resolution=None,full_cov=False):
        # TODO include samples
        if which_functions=='all':
            which_functions = [True]*self.kern.Nparts
        if which_data=='all':
            which_data = slice(None)

        if self.X.shape[1] == 2:
            #TODO: make sure different outputs don't get mixed
            Xu = self.X[which_data,:] * self._Xstd + self._Xmean #NOTE self.X are the normalized values now
            Xu, _index = self._index_off(Xu)
            index = int(_index[0,0])
            Xnew, xmin, xmax = x_frame1D(Xu, plot_limits=plot_limits)
            I_ = np.repeat(index,resolution or 200)[:,None] #repeat the first index number
            Xnew = self._index_on(Xnew,I_)
            m, var, lower, upper = self.predict(Xnew, slices=which_functions)
            gpplot(Xnew[:,self.input_cols],m, lower, upper)
            pb.plot(Xu,self.likelihoods[index].data,'kx',mew=1.5)
            ymin,ymax = min(np.append(self.likelihoods[index].data,lower)), max(np.append(self.likelihoods[index].data,upper))
            ymin, ymax = ymin - 0.1*(ymax - ymin), ymax + 0.1*(ymax - ymin)
            pb.xlim(xmin,xmax)
            pb.ylim(ymin,ymax)
            #Zu = self.Z * self._Xstd + self._Xmean
            #Zu, _index = self._index_off(Zu)
            #pb.plot(Zu,Zu*0+pb.ylim()[0],'r|',mew=1.5,markersize=12)

        elif self.X.shape[1] == 3: #FIXME
            resolution = resolution or 50
            #for os,on in zip(self.Xos,self.output_nums):
            Xu = self.X[which_data,:] * self._Xstd + self._Xmean #NOTE self.X are the normalized values now
            Xu, _index = self._index_off(Xu)
            index = int(_index[0,0])
            Xnew, xx, yy, x, y, xmin, xmax = x_frame2D(Xu, plot_limits,resolution)
            I_ = np.repeat(index,resolution**2)[:,None]
            Xnew = self._index_on(Xnew,I_)
            m,v,lower,upper = self.predict(Xnew, slices=which_functions)
            m = m.reshape(resolution,resolution)
            pb.contour(x,y,m,vmin=m.min(),vmax=m.max(),cmap=pb.cm.jet)
            pb.scatter(Xu[:,0],Xu[:,1],40,self.likelihoods[index].data,linewidth=0,cmap=pb.cm.jet,vmin=m.min(), vmax=m.max())
            pb.xlim(xmin[0],xmax[0])
            pb.ylim(xmin[1],xmax[1])
            #if hasattr(self,'Z'):
            #    pb.scatter(self.Z[:,0],self.Z[:,1],'kx',mew=1.5,markersize=12)

    def x_frameHD(self,X,which_input,plot_limits=None,resolution=None):
        """
        Internal helper function for making plots, returns a set of input values to plot as well as lower and upper limits

        which_input: column within X matrix (index column doesn't count)
        """
        if plot_limits is None:
            xmean,xmean = X.mean(0),X.mean(0)
            xmin,xmax = X.min(0),X.max(0)
            xmin, xmax = xmin-0.2*(xmax-xmin), xmax+0.2*(xmax-xmin)
        elif len(plot_limits)==2:
            xmin, xmax = plot_limits
        else:
            raise ValueError, "Bad limits for plotting"

        resolution = resolution or 200
        Xnew = np.hstack([np.repeat(mean_i,resolution)[:,None] for mean_i in xmean.flatten()])
        Xnew_i = np.linspace(xmin[which_input],xmax[which_input],resolution)[:,None]
        Xnew[:,which_input] = Xnew_i.flatten()
        return Xnew, xmin[which_input], xmax[which_input]


    def plot_HD(self, which_input=None, plot_limits=None, which_data='all', which_functions='all', resolution=None, full_cov=False):
        #TODO: which_data
        if which_functions=='all':
            which_functions = [True]*self.kern.Nparts
        if which_data=='all':
            which_data = slice(None)
        wi = self.input_cols[which_input]

        #TODO: make sure different outputs don't get mixed
        Xu = self.X[which_data,:] * self._Xstd + self._Xmean #NOTE self.X are the normalized values now
        Xu, _index = self._index_off(Xu)
        index = int(_index[0,0])
        Xnew, xmin, xmax = self.x_frameHD(Xu,which_input=which_input, plot_limits=plot_limits)
        I_ = np.repeat(index,resolution or 200)[:,None] #repeat the first index number
        Xnew = self._index_on(Xnew,I_)
        m, var, lower, upper = self.predict(Xnew, slices=which_functions)
        gpplot(Xnew[:,wi],m, lower, upper)
        Xu = self._index_on(Xu,_index)
        pb.plot(Xu[:,wi],self.likelihoods[index].data,'kx',mew=1.5) #repeat the first index number
        ymin,ymax = min(np.append(self.likelihoods[index].data,lower)), max(np.append(self.likelihoods[index].data,upper))
        ymin, ymax = ymin - 0.1*(ymax - ymin), ymax + 0.1*(ymax - ymin)
        pb.ylim(ymin,ymax)
        pb.xlim(xmin,xmax)
        #pb.plot(self.Z,self.Z*0+pb.ylim()[0],'r|',mew=1.5,markersize=12)


